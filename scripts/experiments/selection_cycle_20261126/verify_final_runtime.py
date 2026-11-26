"""Verify RGB-only review, exact version binding, GT reattachment and both profiles."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess
import sys
import numpy as np
from quality_estimation.versioned_review import VersionedReview
from quality_estimation.perception_identity import baseline_identity
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.portable_forest import PortableForest
from prepare import dump, jsonl
sys.path.insert(0, str(Path(__file__).parents[1] / 'roi9_gp_20261125'))
from study import load


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['study', 'repo']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-role',choices=['development','heldout_diagnostic'],default='heldout_diagnostic')
    args = parser.parse_args()
    study, root = args.study, args.study / 'verification'
    root.mkdir(exist_ok=True)
    decision = json.loads((study / 'deployment/candidate_pair.json').read_text())
    run = study / decision['candidate_perception_run']
    bundle = study / decision['candidate_review_bundle']
    review=VersionedReview(bundle)
    index = json.loads((study / 'protocol/quality_index_frozen.json').read_text())[args.source_role]['rows']
    prefix='drive_confirmation_test_' if args.source_role=='heldout_diagnostic' else 'drive_development_'
    rows = []
    for site in sorted({r['site'] for r in index if r['source_key'].startswith(prefix)}):
        group = [r for r in index if r['site'] == site and r['source_key'].startswith(prefix)]
        rows.extend(group[i] for i in np.linspace(0, len(group) - 1, 32, dtype=int))
    assert len(rows) == len({r['sample_id'] for r in rows}) == 96
    # The inference must succeed despite deliberately unavailable GT references.
    rgb_rows = [dict(r, semantic='deliberately_unavailable_GT/unused.png', y=-999.) for r in rows]
    manifest = study / 'protocol/g1_runtime_rgb96_no_gt.jsonl'
    jsonl(manifest, rgb_rows)
    mounts = ['-v', str(args.repo) + ':/repo', '-v', '/mnt/e:/mnt/e', '-v', '/mnt/d:/mnt/d:ro',
              '-w', '/repo', '-e', 'PYTHONPATH=/repo/src', '-e', 'OMP_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1']
    for profile, device in [('rtx50-cu128', 'cuda'), ('rtx40-cu118', 'cpu')]:
        output = root / ('g1_rgb96_' + profile)
        if (output / '_SUCCESS').exists():
            continue
        command = ['docker', 'run', '--name', 'ml-cycle-runtime-verification', '--rm']
        if device == 'cuda':
            command += ['--gpus', 'all']
        command += ['--shm-size', '1g', *mounts, 'ml-perception:' + profile,
                    'python', '-m', 'quality_estimation.infer_versioned_review',
                    '--bundle', str(bundle), '--manifest', str(manifest), '--data-root', '/mnt',
                    '--adaptation-run', str(run), '--output', str(output),
                    '--batch-size', '4', '--device', device, '--budget', '12']
        with (study / 'logs' / ('verify_rgb96_' + profile + '.log')).open('a') as log:
            subprocess.run(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT, check=True)
    gpu_root, cpu_root = root / 'g1_rgb96_rtx50-cu128', root / 'g1_rgb96_rtx40-cu118'
    gpu = np.load(gpu_root / 'features.npz')
    cpu = np.load(cpu_root / 'features.npz')
    assert np.array_equal(gpu['ids'], cpu['ids']) and np.array_equal(gpu['ordered_features'], cpu['ordered_features'])
    pack, schema = load(study / 'g1_quality/packs', [args.source_role])
    lookup = {sid: i for i, sid in enumerate(pack['ids'])}
    indices = [lookup[sid] for sid in gpu['ids']]
    expected = feature_matrix(pack, schema, gpu['ordered_features'].tolist())[indices]
    parity = float(abs(gpu['X'] - expected).max())
    cross = float(abs(gpu['X'] - cpu['X']).max())
    assert parity < 1e-5, 'RGB-only extraction differs from version-matched X/y pack'
    gp_rows = [json.loads(line) for line in (gpu_root / 'predictions.jsonl').read_text().splitlines()]
    cp_rows = [json.loads(line) for line in (cpu_root / 'predictions.jsonl').read_text().splitlines()]
    mean_delta = max(abs(g['estimated_iou'] - c['estimated_iou']) for g, c in zip(gp_rows, cp_rows))
    assert mean_delta < .001, 'cross-profile quality predictions differ materially'
    assert all(not r['uses_ground_truth'] and not r['admission_approved'] for r in gp_rows + cp_rows)
    candidates = [json.loads(line) for line in (gpu_root / 'candidates.jsonl').read_text().splitlines()]
    audit = [r for r in candidates if r['selection_component'] == 'random_audit']
    assert len(audit) == int(np.ceil(len(candidates) * .1))
    assert all(0 < r['conditional_inclusion_probability'] <= 1 for r in candidates)
    mismatch_rejected = False
    try:
        VersionedReview(bundle, perception_identity=baseline_identity())
    except ValueError:
        mismatch_rejected = True
    assert mismatch_rejected
    forest=PortableForest.load(bundle/'quality_rf_portable.npz')
    forest.validate_context(review.columns,review.profile.version,review.profile.regions_sha256,
                            decision['candidate_perception_identity']['checkpoint_sha256'])
    assert forest.metadata['context']['perception_identity']==review.spec['perception_identity']
    take=[gpu['ordered_features'].tolist().index(name) for name in forest.feature_names]
    expected_forest=forest.predict(gpu['X'][:,take])
    code='''from pathlib import Path
import json,sys,numpy as np
from quality_estimation.portable_forest import PortableForest
root=Path(sys.argv[1]);f=PortableForest.load(root/'quality_rf_portable.npz');m=json.loads((root/'bundle_manifest.json').read_text())
f.validate_context(f.feature_names,m['profile'],m['regions_sha256'],m['checkpoint_sha256'])
assert f.metadata['context']['perception_identity']==m['perception_identity']
z=np.load(sys.argv[2]);columns=z['ordered_features'].tolist();x=z['X'][:,[columns.index(n) for n in f.feature_names]]
np.save(sys.argv[3],f.predict(x))
'''
    forest_deltas={}
    for profile in ['rtx40-cu118','rtx50-cu128']:
        result=root/('portable_rf_'+profile+'.npy')
        subprocess.run(['docker','run','--rm',*mounts,'ml-perception:'+profile,'python','-c',code,str(bundle),str(gpu_root/'features.npz'),str(result)],cwd=args.repo,check=True)
        forest_deltas[profile]=float(abs(np.load(result)-expected_forest).max())
        assert forest_deltas[profile]<1e-12
    output = root / 'g1_rgb96_selected_gt'
    if not (output / '_SUCCESS').exists():
        command = ['docker', 'run', '--rm', *mounts, 'ml-perception:rtx50-cu128', 'python', '-m',
                   'quality_estimation.join_spatial_gt', '--queue-dir', str(gpu_root),
                   '--packs', str(study / 'g1_quality/packs'), '--source-index', str(study / 'protocol/quality_index_frozen.json'),
                   '--key',args.source_role,'--data-root', '/mnt', '--output-dir', str(output)]
        with (study / 'logs/verify_g1_gt_attachment.log').open('a') as log:
            subprocess.run(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT, check=True)
    joined=[json.loads(line) for line in (output/'gt_review.jsonl').read_text().splitlines()]
    assert len(joined)==len(candidates) and all(not r['admission_approved'] for r in joined)
    assert all(r['source_context']['adaptation_role']==args.source_role and r['admission_block_reason']=='reserved_role_cannot_train_in_this_cycle' for r in joined)
    dump(study / 'reports/final_runtime_verification.json', dict(
         completed_utc=datetime.now(timezone.utc).isoformat(), evaluation_role=args.source_role,rows=96, selected=len(candidates),
         audit_slots=len(audit), feature_columns=len(gpu['ordered_features']),
         rgb_only_vs_matched_pack_max_abs=parity, cross_profile_features_max_abs=cross,
         cross_profile_iou_prediction_max_abs=mean_delta,
         portable_rf_cross_profile_max_abs=forest_deltas,
         uses_ground_truth_for_inference=False, intentionally_unavailable_gt_ignored=True,
         mismatched_f0_identity_rejected=mismatch_rejected,
         actual_gpu='RTX5070Ti with rtx50-cu128', second_profile='rtx40-cu118 on CPU; actual40GPU not tested',
         gt_attached_only_after_selection=True, admission_approved=False,
         reserved_source_roles_preserved_after_gt=True,
         deployment_decision_sha256=sha(study / 'deployment/candidate_pair.json'),
         bundle_manifest_sha256=sha(bundle / 'bundle_manifest.json'),
         source_sha256=sha(__file__)))
    print('RGB96/GT/profile verification complete', parity, cross, mean_delta, flush=True)


if __name__ == '__main__':
    main()
