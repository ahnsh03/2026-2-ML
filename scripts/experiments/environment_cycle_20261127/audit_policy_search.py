"""Audit frozen Dev search; propose a union candidate without opening confirmation.

This post-study tool does not export a deployable bundle or modify the original
policy. Any proposed candidate needs a newly registered independent evaluation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('search loss must be a finite number')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('search loss must be a finite number')
    return result


def audit_policy_search(policy):
    common = finite_number(policy['common_objective'])
    reports = {}
    eligible = []
    for name, study in sorted(policy['studies'].items()):
        bo = study['bo_indices']
        random = study['random_indices']
        if not bo or len(bo) != len(random):
            raise ValueError('BO and random query budgets must match and be nonempty')
        trials = study['trials']
        for indices, losses in [(bo, study['bo_losses']), (random, study['random_losses'])]:
            if len(indices) != len(losses) or len(set(indices)) != len(indices):
                raise ValueError('search history is truncated or repeats a query')
            for index, loss in zip(indices, losses):
                if type(index) is not int or str(index) not in trials:
                    raise ValueError('search trial missing from frozen receipt')
                observed = finite_number(trials[str(index)]['loss'])
                if not math.isclose(observed, finite_number(loss), rel_tol=0, abs_tol=1e-12):
                    raise ValueError('search history disagrees with frozen trial loss')

        def key(index):
            trial = trials[str(index)]
            params = trial['parameters']
            return (finite_number(trial['loss']),
                    finite_number(params.get('balanced_floor', 0)),
                    finite_number(params.get('uncertainty_weight', 0)), index)

        chosen = {label: min(indices, key=key) for label, indices in
                  [('bo', bo), ('random', random), ('evaluated_union', sorted(set(bo + random)))]}
        summaries = {}
        for label, index in chosen.items():
            trial = trials[str(index)]
            groups = trial['reference']['groups']
            summaries[label] = dict(index=index, loss=finite_number(trial['loss']),
                                    parameters=copy.deepcopy(trial['parameters']),
                                    thresholds={group: value['threshold'] for group, value in groups.items()})
        frozen_loss = finite_number(study['selected']['loss'])
        if not math.isclose(frozen_loss, summaries['bo']['loss'], rel_tol=0, abs_tol=1e-12):
            raise ValueError('expected an original BO-only selected policy receipt')
        reports[name] = dict(queries_per_method=len(bo), original_selected_loss=frozen_loss,
                             best=summaries, random_beats_bo=summaries['random']['loss'] < summaries['bo']['loss'])
        trial = trials[str(chosen['evaluated_union'])]
        if (policy['positive_support_passes'] is True and not name.startswith('sim_')
                and finite_number(trial['loss']) <= common - .01):
            eligible.append((finite_number(trial['loss']), name, chosen['evaluated_union'], trial))

    runtime = copy.deepcopy(policy['runtime_policy'])
    candidate = None
    if eligible:
        loss, name, index, trial = min(eligible, key=lambda item: item[:3])
        params = trial['parameters']
        runtime.update(grouping=name, reference=copy.deepcopy(trial['reference']),
                       balanced_floor=params['balanced_floor'], uncertainty_weight=params['uncertainty_weight'],
                       environment_thresholds_adopted=False,
                       decision='Development-only BO/random union candidate; requires new confirmation, not adopted.')
        candidate = dict(grouping=name, trial_index=index, development_loss=loss,
                         development_gain=common-loss, runtime_policy=runtime)
    return dict(version='development-policy-search-audit-1', status='development_analysis_only',
                common_objective=common, studies=reports, proposed_candidate=candidate,
                perception_identity=copy.deepcopy(policy.get('perception_identity')),
                quality_selection_sha256=policy.get('quality_selection_sha256'),
                selection_rule='Equal-budget evaluated BO/random union; minimize Dev loss, then balance/uncertainty complexity.',
                minimum_development_gain=.01, requires_new_confirmation=True,
                fresh_assessment_read=False, raw_gt_read=False, original_policy_modified=False,
                inference_bundle_created=False,
                interpretation='Frozen Dev aggregates only. No new evidence of independent transfer or downstream retraining benefit.')


def write_audit(source: Path, output: Path):
    source_bytes = source.read_bytes()
    policy = json.loads(source_bytes)
    result = audit_policy_search(policy)
    result['source_policy_sha256'] = hashlib.sha256(source_bytes).hexdigest()
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON path; existing files are never overwritten.')
    args = parser.parse_args()
    result = write_audit(args.policy, args.output)
    print(json.dumps(dict(status=result['status'], proposed_candidate=result['proposed_candidate']), ensure_ascii=False))


if __name__ == '__main__':
    main()
