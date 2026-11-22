#!/usr/bin/env python3
"""Offline input audit and hypothetical projection sensitivity for topic review.

Does not train quality regressors or test localization algorithms. Frame metrics
must come from an existing, identified inference run. Injected pitch changes are
assumptions, not measured calibration errors. All outputs stay in --output-dir.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from drivable_bev.calibration import calibrations_from_dict


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values):
    x = np.asarray(values, dtype=float)
    assert len(x) and np.isfinite(x).all() and ((0 <= x) & (x <= 1)).all()
    return {'n': len(x), 'frame_mean_iou': float(x.mean()),
            'minimum': float(x.min()), 'p05': float(np.quantile(x, .05)),
            'median': float(np.median(x)), 'p95': float(np.quantile(x, .95)),
            'maximum': float(x.max()),
            **{'below_' + str(t): int((x < t).sum()) for t in (.6, .7, .8)}}


def ray_intersection(camera, pixels):
    """Independent flat-ground ray calculation to check inverse homography."""
    plane = camera.ground_plane
    origin = np.asarray(camera.translation_m)
    rays_optical = np.linalg.solve(camera.intrinsic, np.c_[pixels, np.ones(len(pixels))].T).T
    rays_body = rays_optical @ camera.optical_from_body_rotation
    denom = rays_body[:, 2] - plane.dz_dx * rays_body[:, 0] - plane.dz_dy * rays_body[:, 1]
    scale = (plane.z_at_origin_m + plane.dz_dx * origin[0] + plane.dz_dy * origin[1] - origin[2]) / denom
    assert (scale > 0).all(), 'Ray does not meet ground in front of camera'
    return (origin + scale[:, None] * rays_body)[:, :2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane-map', type=Path, required=True)
    parser.add_argument('--frame-metrics', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    sources = {k: getattr(args, k).resolve() for k in ('lane_map', 'frame_metrics', 'calibration')}
    for path in sources.values():
        if not path.is_file():
            parser.error('Missing input: ' + str(path))
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Freeze definitions before computing or inspecting the resulting statistics.
    protocol = {
        'purpose': 'Topic-review auxiliary analysis, not prior project progress',
        'inputs': {k: {'path': str(p), 'sha256': sha256(p)} for k, p in sources.items()},
        'implementation': {str(p.relative_to(REPO)): sha256(p) for p in
                           (Path(__file__).resolve(), REPO / 'src/drivable_bev/calibration.py')},
        'quality': {'metric': 'saved front-frame white/yellow union IoU',
                    'thresholds': [.6, .7, .8], 'split': 'existing historical validation',
                    'causal_weather_comparison': False, 'new_inference': False},
        'geometry': {'camera': 'front', 'reference_x_m': [5, 10, 20, 30],
                     'reference_y_m': [-1.75, 0, 1.75], 'pitch_delta_deg': [-1, -.5, -.25, 0, .25, .5, 1],
                     'fixed': 'same nominal image pixel; only assumed camera pitch changes',
                     'interpretation': 'hypothetical projection sensitivity, not measured sensor error'},
        'map': {'broken_rule': 'lane_shape contains Broken, case insensitive',
                'interval_check': 'presence and positive numeric L1/L2; no accuracy claim'},
    }
    write_json(out / 'protocol.json', protocol)

    boundaries = json.loads(sources['lane_map'].read_text())
    assert isinstance(boundaries, list) and boundaries
    required = ('lane_shape', 'lane_width', 'dash_interval_L1', 'dash_interval_L2', 'points')
    def broken(record):
        shapes = record.get('lane_shape', [])
        if isinstance(shapes, str):
            shapes = [shapes]
        return any('broken' in str(s).lower() for s in shapes)
    dashed = [x for x in boundaries if broken(x)]
    def positive(x, k):
        return isinstance(x.get(k), (float, int)) and np.isfinite(x[k]) and x[k] > 0
    interval_pairs = Counter((x.get('dash_interval_L1'), x.get('dash_interval_L2')) for x in dashed)
    audit = {'boundary_records': len(boundaries),
             'fields_present': {k: sum(k in x for x in boundaries) for k in required},
             'lane_type_definitions': dict(Counter(str(x.get('lane_type_def')) for x in boundaries)),
             'broken_records': len(dashed),
             'broken_with_both_intervals_positive': sum(positive(x, 'dash_interval_L1') and positive(x, 'dash_interval_L2') for x in dashed),
             'broken_interval_pairs': [{'L1': a, 'L2': b, 'count': count} for (a, b), count in interval_pairs.items()],
             'limitations': ['Field presence does not establish alignment with rendered road markings.',
                             'No road coverage, missing-lane, map-position or Semantic GT correctness audit was performed.']}
    write_json(out / 'map_audit.json', audit)

    with sources['frame_metrics'].open() as f:
        frames = list(csv.DictReader(f))
    assert len({x['sample_id'] for x in frames}) == len(frames)
    valid = [x for x in frames if np.isfinite(float(x['lane_union'])) and int(x['gt_lane_pixels']) > 0]
    groups = defaultdict(list)
    for row in valid:
        groups[row['run_id']].append(row)
    per_run = []
    for run, rows in sorted(groups.items()):
        assert len({(x['weather'], x['hour']) for x in rows}) == 1
        per_run.append({'run_id': run, 'weather': rows[0]['weather'], 'hour': rows[0]['hour'],
                        **summarize([float(x['lane_union']) for x in rows])})
    quality = {'source_rows': len(frames), 'excluded_nonfinite_or_empty_gt': len(frames) - len(valid),
               'overall': summarize([float(x['lane_union']) for x in valid]), 'sessions': per_run,
               'limitations': ['Descriptive reanalysis of saved historical validation; no new model execution.',
                               'Session, route, weather and time vary together; cannot isolate weather causality.',
                               'Frame arithmetic mean is not pixel-aggregate IoU.']}
    write_json(out / 'frame_quality.json', quality)
    write_csv(out / 'frame_quality_by_session.csv', per_run)

    camera = calibrations_from_dict(yaml.safe_load(sources['calibration'].read_text()))['front']
    ground = np.asarray([(x, y) for x in protocol['geometry']['reference_x_m'] for y in protocol['geometry']['reference_y_m']])
    pixels, depth = camera.project_ground(ground)
    assert ((pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height) & (depth > 0)).all()
    cases = []
    for delta in protocol['geometry']['pitch_delta_deg']:
        roll, pitch, yaw = camera.rotation_deg
        changed = replace(camera, rotation_deg=(roll, pitch + delta, yaw))
        homogeneous = (np.linalg.inv(changed.ground_to_image_homography) @ np.c_[pixels, np.ones(len(pixels))].T).T
        estimated = homogeneous[:, :2] / homogeneous[:, 2:3]
        ray_estimate = ray_intersection(changed, pixels)
        np.testing.assert_allclose(estimated, ray_estimate, atol=1e-9)
        if delta == 0:
            np.testing.assert_allclose(estimated, ground, atol=1e-9)
        for xy, uv, result in zip(ground, pixels, estimated):
            error = result - xy
            cases.append({'reference_x_m': float(xy[0]), 'reference_y_m': float(xy[1]),
                          'image_u': float(uv[0]), 'image_v': float(uv[1]), 'pitch_delta_deg': delta,
                          'estimated_x_m': float(result[0]), 'estimated_y_m': float(result[1]),
                          'delta_x_m': float(error[0]), 'delta_y_m': float(error[1]),
                          'displacement_m': float(np.linalg.norm(error))})
    write_csv(out / 'hypothetical_pitch_sensitivity.csv', cases)
    geometry = {'nominal_pitch_deg': camera.rotation_deg[1], 'ground_z_m': camera.ground_plane.z_at_origin_m,
                'camera_height_m': camera.translation_m[2], 'cases': len(cases),
                'centerline_delta_plus_half_degree': [x for x in cases if x['reference_y_m'] == 0 and x['pitch_delta_deg'] == .5],
                'checks': ['zero perturbation reconstructs original points', 'inverse homography equals independent ray-plane intersection', 'all nominal points inside front image'],
                'limitations': ['Synthetic reference points on the configured ground plane.',
                                'Injected pitch offsets are not actual measured calibration errors.',
                                'Displacement is projected-point shift, not vehicle localization error or algorithm ranking.']}
    write_json(out / 'geometry_sensitivity.json', geometry)
    assert all(sha256(p) == protocol['inputs'][k]['sha256'] for k, p in sources.items())
    manifest = {'status': 'completed_auxiliary_analysis', 'source_inputs_unchanged': True,
                'quality_training_performed': False, 'localization_experiment_performed': False,
                'hypothetical_projection_sensitivity_performed': True,
                'output_hashes': {p.name: sha256(p) for p in out.iterdir() if p.is_file()}}
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({'map': audit, 'quality': quality['overall'], 'geometry': geometry}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
