"""Record the research pair decision and a predeclared review fallback.

This does not deploy to a simulator, change the default checkpoint, publish
weights, or permit the next training cycle. All source bundles remain immutable.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shutil
from prepare import dump
from quality_estimation.versioned_review import VersionedReview


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    study = parser.parse_args().study
    output = study / 'deployment'
    if output.exists():
        raise ValueError('pair decision already exists')
    output.mkdir()
    perception = json.loads((study / 'reports/perception_heldout_assessment.json').read_text())
    quality = json.loads((study / 'reports/quality_heldout_assessment.json').read_text())
    provisional = json.loads((study / 'protocol/provisional_f1_frozen.json').read_text())
    rules = json.loads((study / 'protocol/review_acceptance_frozen.json').read_text())
    table = quality['same_road_budget_review']['confirmation']
    rows = {r['policy']: r for r in table if r['fraction'] == .1}
    mixed, common, random = rows['g1_policy'], rows['global_probability_audit'], rows['random_100_seeds']
    gate = rules['review_mix_gate']
    supported = quality['risk']['confirmation']['positives'] >= gate['minimum_positive_rows']
    plan = json.loads((study / 'protocol/quality_index_frozen.json').read_text())['heldout_diagnostic']['rows']
    # Require positive support on at least two roads, using already-fixed frame
    # semantics. This only accepts/rejects the fixed mix; no tuning is permitted.
    import numpy as np
    pack = np.load(study / 'g1_quality/packs/heldout_diagnostic.npz')
    positive_roads = {r['site'] for i, r in enumerate(plan)
                      if r['source_key'].startswith('drive_confirmation_test_')
                      and np.isfinite(pack['y'][i]) and pack['y'][i] < .7
                      and (pack['counts'][i, :, 0] + pack['counts'][i, :, 2]).sum() >= 100}
    supported &= len(positive_roads) >= gate['minimum_positive_roads']
    mix_passes = bool(supported and mixed['macro_road_recall'] is not None
                      and mixed['macro_road_recall'] >= common['macro_road_recall'] - gate['maximum_macro_road_recall_loss_to_global_probability_audit']
                      and mixed['precision'] >= random['mean']['precision'])
    source = study / 'g1_quality/bundle'
    target = output / 'review_bundle'
    target.mkdir()
    manifest = json.loads((source / 'bundle_manifest.json').read_text())
    for name in manifest['artifact_sha256']:
        if name == 'policy.json':
            continue
        try:
            os.link(source / name, target / name)
        except OSError:
            shutil.copy2(source / name, target / name)
    policy = json.loads((source / 'policy.json').read_text())
    if not mix_passes:
        policy.update(rules['fallback'])
    policy['selection_decision'] = 'development_mix_accepted' if mix_passes else 'predeclared_common_probability_audit_fallback'
    dump(target / 'policy.json', policy)
    manifest.update(source_bundle_manifest_sha256=sha(source / 'bundle_manifest.json'),
                    fixed_review_mix_accepted=mix_passes,
                    review_acceptance_rules_sha256=sha(study / 'protocol/review_acceptance_frozen.json'))
    manifest['artifact_sha256'] = {name: sha(target / name) for name in manifest['artifact_sha256']}
    dump(target / 'bundle_manifest.json', manifest)
    VersionedReview(target)
    method = perception['development_selected_methods'][provisional['selected']['arm']]
    neural_passes = bool(provisional['eligible_for_adoption'] and method['all_seeds_pass'])
    pair_passes = bool(neural_passes and quality['quality_adoption_gate_passes'])
    decision = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='offline_research_pair_accepted' if pair_passes else 'candidate_only_keep_f0_g0',
        perception_gate=neural_passes, quality_gate=quality['quality_adoption_gate_passes'],
        fixed_review_mix_accepted=mix_passes, positive_support_roads=len(positive_roads),
        candidate_perception_run=os.path.relpath(provisional['selected']['run'], study),
        candidate_perception_identity=quality['model_perception_identity'],
        candidate_review_bundle=os.path.relpath(target, study),
        runtime_bundle_manifest_sha256=sha(target / 'bundle_manifest.json'),
        default_checkpoint_changed=False, automatic_admission=False, live_connection_completed=False,
        relative_environment_thresholds_adopted=False, absolute_quality_benchmark=.7,
        evidence_sha256={name: sha(study / name) for name in [
            'reports/perception_heldout_assessment.json', 'reports/quality_heldout_assessment.json',
            'protocol/provisional_f1_frozen.json', 'protocol/heldout_evaluation_frozen.json']},
        scope='Offline paired simulator research result. f1-to-f2 cumulative lineage, ROS shadow replay and independent verification remain next units.')
    dump(output / 'candidate_pair.json', decision)
    (output / '_SUCCESS').write_text(sha(output / 'candidate_pair.json') + '\n')
    print('PAIR', decision['status'], 'mix accepted', mix_passes, flush=True)


if __name__ == '__main__':
    main()
