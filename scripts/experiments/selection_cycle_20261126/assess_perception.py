"""Report fixed neural endpoints after opening heldout; never revise choices here."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import numpy as np
from prepare import dump


def cohort(source):
    if source.startswith('drive_confirmation_test_'):
        return 'confirmation'
    if source.startswith('drive_calibration_shift_'):
        return 'condition_shift'
    if source.startswith('drive_test_'):
        return 'main'
    raise ValueError('unexpected heldout cohort')


def metrics(rows):
    by_road = {}
    for road in sorted({r['site'] for r in rows}):
        records = [r for r in rows if r['site'] == road]
        values = [r['counts']['lane'][0] / sum(r['counts']['lane']) for r in records
                  if r['counts']['lane'][0] + r['counts']['lane'][2] >= 50]
        if values:
            by_road[road] = float(np.mean(values))
    counts = np.array([r['counts']['lane'] for r in rows], np.int64).sum(0)
    return dict(rows=len(rows), supported_rows=sum(r['counts']['lane'][0] + r['counts']['lane'][2] >= 50 for r in rows),
                road_mean_iou=by_road,
                macro_road_iou=float(np.mean(list(by_road.values()))) if by_road else None,
                pooled_iou=float(counts[0] / counts.sum()) if counts.sum() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    study = parser.parse_args().study
    plan = json.loads((study / 'protocol/heldout_evaluation_frozen.json').read_text())
    dev = json.loads((study / 'reports/perception_development_comparison.json').read_text())
    gates = json.loads((study / 'protocol/evaluation_gate_v2_frozen.json').read_text())['perception_adoption']
    records = {}
    baseline_ids = None
    for model in plan['models']:
        path = study / 'evaluations' / model['name'] / 'heldout_diagnostic'
        assert (path / '_SUCCESS').exists()
        rows = [json.loads(line) for line in (path / 'per_frame.jsonl').read_text().splitlines()]
        ids = [r['sample_id'] for r in rows]
        if baseline_ids is None:
            baseline_ids = ids
        assert ids == baseline_ids
        records[model['name']] = {key: metrics([r for r in rows if cohort(r['source_key']) == key])
                                 for key in ['confirmation', 'main', 'condition_shift']}
    baseline = records['f0']
    outcomes = {}
    for arm, choice in plan['development_choices'].items():
        runs = []
        for item in choice['records']:
            name = Path(item['run']).name
            current = records[name]['confirmation']
            old = baseline['confirmation']
            deltas = {road: value - old['road_mean_iou'][road] for road, value in current['road_mean_iou'].items()}
            gain = current['macro_road_iou'] - old['macro_road_iou']
            passed = bool(item['passes'] and gain >= gates['minimum_macro_road_lane_iou_gain']
                          and sum(v > 0 for v in deltas.values()) >= gates['minimum_improved_roads']
                          and min(deltas.values()) >= -gates['maximum_road_lane_iou_loss'])
            runs.append(dict(name=name, seed=item['seed'], alpha=item.get('alpha', 1.),
                             development_passes=item['passes'], confirmation_gain=gain,
                             confirmation_road_deltas=deltas, passes=passed,
                             retention=item['metrics']['retention']))
        outcomes[arm] = dict(variant=choice['variant'], runs=runs,
            mean_confirmation_gain=float(np.mean([r['confirmation_gain'] for r in runs])),
            all_seeds_pass=bool(len(runs) == 2 and all(r['passes'] for r in runs)))
    paired = {}
    for arm in ['QualityRank', 'ClusterBO']:
        paired[arm] = dict(vs_random_confirmation_gain=outcomes[arm]['mean_confirmation_gain'] - outcomes['Random']['mean_confirmation_gain'],
                          vs_replay_confirmation_gain=outcomes[arm]['mean_confirmation_gain'] - outcomes['ReplayOnly']['mean_confirmation_gain'])
    output = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        primary='confirmation three roads; pixel-GT support >=50 for frame mean, then equal road mean',
        baseline=baseline, fixed_model_metrics=records, development_selected_methods=outcomes,
        selection_comparisons=paired,
        limitations=['Two optimizer seeds share one fixed admission bank per method.',
                     'Previously opened f0 simulator cohorts are paired heldout diagnostics, not untouched final test.',
                     'Only three primary roads; do not interpret correlated frames as independent repetitions.',
                     'Raw equal-compute endpoints and development-selected common-alpha variants are distinct comparisons.',
                     'Continuation controls isolate optimizer work; no GT threshold or policy is selected on these results.'])
    dump(study / 'reports/perception_heldout_assessment.json', output)
    print(json.dumps(outcomes, indent=2), flush=True)


if __name__ == '__main__':
    main()
