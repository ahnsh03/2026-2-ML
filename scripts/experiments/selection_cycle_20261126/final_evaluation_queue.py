"""Open heldout diagnostics only after development choices and the f1/g1 pair freeze.

Long training workers may run independently. This coordinator never selects a
model, an interpolation coefficient, or a policy using heldout results.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess
import time

from prepare import dump


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['study', 'repo', 'python', 'checkpoints']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    study = args.study
    requirements = ['reports/control_development_evaluations_complete.json',
                    'reports/interpolation_complete.json',
                    'g1_quality/bundle/bundle_manifest.json']
    while not all((study / path).exists() for path in requirements):
        time.sleep(15)
    subprocess.run([str(args.python),
                    'scripts/experiments/selection_cycle_20261126/compare_perception.py',
                    '--study', str(study)], cwd=args.repo, check=True)
    report = json.loads((study / 'reports/perception_development_comparison.json').read_text())
    provisional = json.loads((study / 'protocol/provisional_f1_frozen.json').read_text())
    plan_path = study / 'protocol/heldout_evaluation_frozen.json'
    if not plan_path.exists():
        records = report['records']
        chosen = {}
        for arm in ['Random', 'QualityRank', 'ClusterBO', 'ReplayOnly']:
            variants = {k: v for k, v in report['methods'].items() if v['base_arm'] == arm}
            passing = {k: v for k, v in variants.items() if v['all_seeds_pass']}
            pool = passing or variants
            top = max(v['mean_development_iou'] for v in pool.values())
            near = [k for k, v in pool.items() if v['mean_development_iou'] >= top - .002]
            variant = max(near, key=lambda k: (pool[k]['alpha'], pool[k]['mean_development_iou']))
            chosen[arm] = dict(variant=variant, passing=bool(passing),
                               records=[r for r in records if r['variant'] == variant])
        models = [dict(name='f0', run=None)]
        # Raw endpoints are the fixed equal-compute comparison, even if they fail
        # retention. Common-alpha choices are separately marked derived models.
        raw = [r for r in records if r.get('alpha', 1.) == 1.]
        derived = [r for value in chosen.values() for r in value['records']]
        for item in raw + derived + [provisional['selected']]:
            name = Path(item['run']).name
            if name not in {m['name'] for m in models}:
                models.append(dict(name=name, run=item['run']))
        dump(plan_path, dict(created_utc=datetime.now(timezone.utc).isoformat(),
             models=models, development_choices=chosen,
             provisional_checkpoint_sha256=provisional['checkpoint_sha256'],
             development_report_sha256=sha(study / 'reports/perception_development_comparison.json'),
             bundle_manifest_sha256=sha(study / 'g1_quality/bundle/bundle_manifest.json'),
             evaluation_manifest_sha256=sha(study / 'protocol/quality_heldout_diagnostic_cached.jsonl'),
             primary='confirmation three roads,366 frames; main and condition shift reported separately',
             selection='All choices fixed on development; heldout cannot tune alpha, features, thresholds, or policy',
             source_sha256=sha(__file__)))
    plan = json.loads(plan_path.read_text())
    if plan['bundle_manifest_sha256'] != sha(study / 'g1_quality/bundle/bundle_manifest.json'):
        raise ValueError('quality policy changed after heldout freeze')
    dump(study / 'FINAL_WORK_IN_PROGRESS.json', dict(status='heldout_neural_evaluation',
         opened_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=sha(plan_path)))
    mounts = ['-v', str(args.repo) + ':/repo', '-v', '/mnt/e:/mnt/e',
              '-v', '/mnt/d:/mnt/d:ro', '-v', str(args.checkpoints) + ':/checkpoints:ro',
              '-w', '/repo', '-e', 'PYTHONPATH=/repo/src',
              '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OMP_NUM_THREADS=1']
    baseline = '/checkpoints/twinlite_medium_morai_v6_road_marking_init_b_b12_e18_20260908/best_mean.pt'
    for item in plan['models']:
        output = study / 'evaluations' / item['name'] / 'heldout_diagnostic'
        if (output / '_SUCCESS').exists():
            continue
        command = ['docker', 'run', '--name', 'ml-cycle-final-evaluation', '--rm',
                   '--gpus', 'all', '--shm-size', '1g', *mounts, 'ml-perception:rtx50-cu128',
                   'python', '-m', 'quality_estimation.evaluate_cached_perception',
                   '--manifest', str(study / 'protocol/quality_heldout_diagnostic_cached.jsonl'),
                   '--cache', str(study / 'cache'), '--output', str(output),
                   '--batch-size', '4', '--workers', '2', '--role', 'heldout_diagnostic']
        command += ['--adaptation-run', item['run']] if item['run'] else ['--checkpoint', baseline]
        with (study / 'logs' / ('heldout_' + item['name'] + '.log')).open('a') as log:
            subprocess.run(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('HELDOUT EVALUATED', item['name'], flush=True)
    for name, source in [('f0_quality', ['--checkpoint', baseline]),
                         ('g1_quality', ['--adaptation-run', provisional['selected']['run']])]:
        command = ['docker', 'run', '--name', 'ml-cycle-final-extraction', '--rm',
                   '--gpus', 'all', '--shm-size', '1g', *mounts, 'ml-perception:rtx50-cu128',
                   'python', 'scripts/experiments/roi9_gp_20261125/extract.py',
                   '--index', str(study / 'protocol/quality_index_frozen.json'), '--data-root', '/mnt',
                   *source, '--output', str(study / name / 'packs'),
                   '--keys', 'heldout_diagnostic', '--batch-size', '4']
        with (study / 'logs' / (name + '_heldout_packs.log')).open('a') as log:
            subprocess.run(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT, check=True)
    for script in ['assess_perception.py', 'assess_quality.py', 'finalize_pair.py']:
        with (study / 'logs' / script.replace('.py', '.log')).open('a') as log:
            subprocess.run([str(args.python), 'scripts/experiments/selection_cycle_20261126/' + script,
                            '--study', str(study)], cwd=args.repo,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    with (study / 'logs/verify_final_runtime.log').open('a') as log:
        subprocess.run([str(args.python), 'scripts/experiments/selection_cycle_20261126/verify_final_runtime.py',
                        '--study', str(study), '--repo', str(args.repo)], cwd=args.repo,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    with (study / 'logs/figures_results.log').open('a') as log:
        subprocess.run([str(args.python), 'scripts/experiments/selection_cycle_20261126/figures_results.py',
                        '--study', str(study), '--output', str(study / 'figures_results')], cwd=args.repo,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    case_command = ['docker', 'run', '--name', 'ml-cycle-case-export', '--rm', '--gpus', 'all',
                    *mounts, 'ml-perception:rtx50-cu128', 'python',
                    'scripts/experiments/selection_cycle_20261126/export_perception_cases.py',
                    '--study', str(study), '--checkpoint', baseline,
                    '--output', str(study / 'figures_cases')]
    with (study / 'logs/figures_cases.log').open('a') as log:
        subprocess.run(case_command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (study / 'logs/export_public_evidence.log').open('a') as log:
        subprocess.run([str(args.python), 'scripts/experiments/selection_cycle_20261126/export_public_evidence.py',
                        '--study', str(study), '--output',
                        str(args.repo / 'docs/baselines/selection_cycle_20261127')], cwd=args.repo,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    dump(study / 'FINAL_WORK_IN_PROGRESS.json', dict(status='heldout_evaluation_complete',
         completed_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=sha(plan_path),
         interpretation='Offline research result; live connection and further cycles are separate tasks'))


if __name__ == '__main__':
    main()
