"""Export aggregate evidence and selected figures without private paths/sample IDs."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
import shutil
from prepare import dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    study, output = args.study, args.output
    output.mkdir(parents=True, exist_ok=True)
    read = lambda name: json.loads((study / name).read_text())
    dev = read('reports/perception_development_comparison.json')
    held = read('reports/perception_heldout_assessment.json')
    quality = read('reports/quality_heldout_assessment.json')
    decision = read('deployment/candidate_pair.json')
    provisional = read('protocol/provisional_f1_frozen.json')
    selection = read('g1_quality/refit/quality_selection_frozen.json')
    runtime = study / decision['candidate_review_bundle']
    model_manifest = json.loads((runtime / 'bundle_manifest.json').read_text())
    policy = json.loads((runtime / 'policy.json').read_text())
    # Model names are arm/optimizer-seed/alpha only. Road aliases are stable
    # within each role; source/session names never leave the private volume.
    def sanitized_metrics(metrics):
        return {cohort: dict(rows=v['rows'], supported_rows=v['supported_rows'],
                    macro_road_iou=v['macro_road_iou'], pooled_iou=v['pooled_iou'],
                    road_mean_iou={f'Road {i+1}': value for i, (_, value) in enumerate(sorted(v['road_mean_iou'].items()))})
                for cohort, v in metrics.items()}
    records = []
    for item in dev['records']:
        records.append(dict(arm=item['arm'], seed=item['seed'], alpha=item.get('alpha', 1.),
             development_iou=item['metrics']['development_macro'], development_gain=item['development_gain'],
             retention=item['metrics']['retention'], retention_views=item['metrics']['retention_views'],
             passes=item['passes'], failed_gates=item['failed_gates']))
    methods = {}
    for arm, value in held['development_selected_methods'].items():
        methods[arm] = dict(variant=value['variant'], all_seeds_pass=value['all_seeds_pass'],
            mean_confirmation_gain=value['mean_confirmation_gain'],
            runs=[{k: r[k] for k in ['name', 'seed', 'alpha', 'development_passes', 'confirmation_gain', 'passes', 'retention']} for r in value['runs']])
    dump(output / 'perception_comparison.json', dict(
         baseline_development=dev['baseline']['development_macro'], baseline_retention=dev['baseline']['retention'],
         baseline_heldout=sanitized_metrics(held['baseline']),
         fixed_models={name: sanitized_metrics(value) for name, value in held['fixed_model_metrics'].items()},
         development_records=records, development_selected_methods=methods,
         selection_comparisons=held['selection_comparisons'],matched_coefficient_comparisons=held['matched_coefficient_comparisons'],limitations=held['limitations']))
    dump(output / 'quality_comparison.json', {k: quality[k] for k in [
         'point','risk','same_road_budget_review','same_full_population_union_supported_review','environment_confirmation',
         'quality_adoption_gate_passes', 'limitations']})
    dump(output / 'environment_diagnostics.json', read('reports/environment_diagnostics.json'))
    dump(output / 'admission_summary.json', read('reports/admission_summary.json'))
    cluster = read('clusters/study.json')
    dump(output / 'clustering_comparison.json', dict(records=cluster['records'],
         selected={k: v for k, v in cluster['selected'].items() if k not in ['path', 'archive']}))
    dump(output / 'feature_roles.json', dict(profile=selection['profile'], regions_sha256=selection['regions_sha256'],
         quality=selection['selected']['columns'], target_names=selection['selected']['targets'],
         environment=model_manifest['artifact_sha256']['environment.json'],
         quality_fit_rows=selection['quality_fit_rows'], development_rows=selection['development_rows'],
         primary_development_rows=selection['primary_development_rows'], calibration_rows=selection['calibration_rows'],
         extra_features_adopted=selection['extra_features_adopted'], near_f1_joint_adopted=selection['near_f1_joint_adopted']))
    dump(output / 'environment_centers.json', json.loads((runtime / 'environment.json').read_text()))
    dump(output / 'runtime_evidence.json', dict(
         status=decision['status'], perception_gate=decision['perception_gate'], quality_gate=decision['quality_gate'],
         fixed_review_mix_accepted=decision['fixed_review_mix_accepted'],
         candidate_perception_identity=decision['candidate_perception_identity'], policy=policy,
         runtime_bundle_manifest_sha256=decision['runtime_bundle_manifest_sha256'],
         artifact_sha256=model_manifest['artifact_sha256'], default_checkpoint_changed=False,
         automatic_admission=False, live_connection_completed=False, scope=decision['scope']))
    dump(output / 'runtime_verification.json', read('reports/final_runtime_verification.json'))
    dump(output / 'complete_study_verification.json',read('reports/complete_study_verification_summary.json'))
    search = read('g1_quality/refit/bo_hyperparameters.json')
    dump(output / 'quality_bo.json', search)
    dump(output / 'rf_quality_bo.json', read('g1_quality/refit/rf_bo_hyperparameters.json'))
    dump(output / 'policy_bo.json', read('g1_quality/bundle/policy_search.json'))
    dump(output / 'environment_reference_f1.json', dict(
         development=read('g1_quality/relative_reference/relative_study_frozen.json'),
         heldout=read('reports/relative_reference_heldout_assessment.json')))
    dump(output / 'acceptance_gates.json', dict(perception_quality=read('protocol/evaluation_gate_v2_frozen.json'),
         review=read('protocol/review_acceptance_frozen.json'),
         interpolation=read('protocol/interpolation_extension_frozen.json')))
    summaries = []
    for directory in sorted((study / 'runs').iterdir()):
        if not (directory / '_SUCCESS').exists():
            continue
        metadata = json.loads((directory / 'summary.json').read_text())
        log = [json.loads(line) for line in (directory / 'training_steps.jsonl').read_text().splitlines()]
        summaries.append(dict(run=directory.name, arm=metadata['arm'], seed=metadata['seed'],
             new_rows=metadata['new_rows'], replay_rows=metadata['replay_rows'],
             replacement_replay_rows=metadata.get('replacement_replay_rows', 0),
             steps=metadata['steps'], elapsed_seconds=metadata['elapsed_seconds'],
             checkpoint_sha256=metadata['checkpoint_sha256'],
             max_allocated_bytes=max(r['memory_allocated'] for r in log),
             initial_loss=log[0]['losses']['loss'], final_loss=log[-1]['losses']['loss'],
             source_sha256=metadata['source_sha256'],
             batch_norm_buffers_unchanged=metadata['batch_norm_buffers_unchanged']))
    dump(output / 'training_budget.json', dict(runs=summaries, selected_arm=provisional['selected']['arm'],
         selected_seed=provisional['selected']['seed'], selected_alpha=provisional['selected'].get('alpha', 1.),
         original_train_samples=26001, original_front_train_samples=8667, original_validation_samples=1404,
         candidate_pool=7442, new_budget_per_arm=3072, common_replay=4096,
         training_runs=len(summaries), derived_evaluations=24,
         original_rgb_overlap=read('reports/original_exposure_audit.json'),
         quality_fit_proximity=read('reports/quality_fit_proximity_diagnostic.json'),
         original_vs_full_registry_rgb_audit=read('reports/original_vs_full_registry_pixel_audit.json'),
         original_target_parity={k:v for k,v in read('reports/input_and_control_mask_audit.json').items() if k!='verified_pack_keys'},
         scope='Two optimizer seeds share fixed admissions per method; controls use only original train. No new raw capture.'))
    # The exposure audit can carry a local manifest filename: export only its
    # numerical/hash facts, never source filesystem locations.
    budget = json.loads((output / 'training_budget.json').read_text())
    audit = budget['original_rgb_overlap']
    budget['original_rgb_overlap'] = {k: v for k, v in audit.items() if not isinstance(v, (str, list, dict)) or 'sha256' in k}
    budget['original_rgb_overlap']['candidate_pixel_overlap_count']=len(audit['candidate_pixel_overlap'])
    dump(output / 'training_budget.json', budget)
    pictures = output / 'figures'
    pictures.mkdir(exist_ok=True)
    for folder in ['figures_initial', 'figures_results']:
        for path in sorted((study / folder).glob('*.png')):
            shutil.copy2(path, pictures / path.name)
    for pattern in ['case_A_maximum_change.png','case_C_minimum_change.png','case_D_largest_prediction_error.png']:
        path = study / 'figures_cases' / pattern
        shutil.copy2(path, pictures / pattern)
    # Explicit denylist complements field-level export. Private sample IDs and
    # source-index rows are excluded by construction, not by text replacement.
    for path in output.glob('*.json'):
        if re.search(r'/home/|/mnt/|aim06|ASMC|asmc|kcity_comp_sample', path.read_text()):
            raise ValueError('private path/identifier in aggregate evidence: ' + path.name)
    evidence = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'evidence_inventory.json'}
    dump(output / 'evidence_inventory.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
         files=evidence, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('EXPORTED PUBLIC AGGREGATES', output, len(evidence), flush=True)


if __name__ == '__main__':
    main()
