"""Standalone 16:9 figures from completed, frozen offline-cycle results."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figures_initial import canvas, save, NAVY, TEAL, GOLD, GRAY

COLORS = dict(Random=GRAY, QualityRank=NAVY, ClusterBO=TEAL, ReplayOnly=GOLD)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    study, out = args.study, args.output
    out.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    development = json.loads((study / 'reports/perception_development_comparison.json').read_text())
    held = json.loads((study / 'reports/perception_heldout_assessment.json').read_text())
    quality = json.loads((study / 'reports/quality_heldout_assessment.json').read_text())
    decision = json.loads((study / 'deployment/candidate_pair.json').read_text())
    methods = list(COLORS)
    fig = canvas('New-condition improvement and original-condition retention')
    axes = [fig.add_axes([.08, .23, .39, .60]), fig.add_axes([.58, .23, .36, .60])]
    for i, arm in enumerate(methods):
        result = held['development_selected_methods'][arm]
        gains = np.array([r['confirmation_gain'] for r in result['runs']])
        delta = np.array([r['retention']['lane'] - development['baseline']['retention']['lane'] for r in result['runs']])
        for ax, values in zip(axes, [gains, delta]):
            ax.bar(i, values.mean(), color=COLORS[arm], width=.65)
            ax.scatter(i + np.array([-.08, .08]), values, color='black', s=20, zorder=5)
    for ax in axes:
        ax.axhline(0, color=GRAY, linewidth=.8)
        ax.set_xticks(range(4),[arm+'\n'+r'$\alpha$='+f"{held['development_selected_methods'][arm]['runs'][0]['alpha']:g}" for arm in methods],fontsize=10)
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Lane IoU change vs f0')
    axes[0].set_title('Reserved confirmation roads')
    axes[1].set_title('Original validation: pooled pixels')
    axes[1].axhline(-.005, color='#b64643', linestyle='--', label='Retention limit: -0.005')
    axes[1].legend(frameon=False, fontsize=9)
    save(fig, out, '06_retraining_and_retention',
         'Two optimizer seeds share admissions. Each method has a development-selected alpha; matched-alpha comparisons are separate.')

    fig = canvas('Retention is a constraint: inspect the interpolation trade-off')
    ax = fig.add_axes([.10, .21, .78, .63])
    for arm in methods:
        points = []
        for variant, value in development['methods'].items():
            if value['base_arm'] != arm or value['seeds'] != 2:
                continue
            records = [r for r in development['records'] if r['variant'] == variant]
            loss = np.mean([r['metrics']['retention']['lane'] - development['baseline']['retention']['lane'] for r in records])
            points.append((value['alpha'], loss, value['mean_development_gain']))
        points.sort()
        ax.plot([p[1] for p in points], [p[2] for p in points], '-o', label=arm, color=COLORS[arm])
        for alpha, dx, dy in points:
            ax.annotate(f'{alpha:g}', (dx, dy), xytext=(3, 5), textcoords='offset points', fontsize=9, color=COLORS[arm])
    ax.axvline(-.005, color='#b64643', linestyle='--')
    ax.axhline(0, color=GRAY, linewidth=.7)
    ax.set_xlabel('Original validation pooled lane IoU change')
    ax.set_ylabel('Development macro-road lane IoU gain')
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    save(fig, out, '07_interpolation_tradeoff',
         'Labels are alpha. Interpolation performs no new gradient steps; all raw equal-compute endpoints remain in the comparison.')

    pack = np.load(study / 'reports/g1_heldout_predictions.npz')
    index = json.loads((study / 'protocol/quality_index_frozen.json').read_text())['heldout_diagnostic']['rows']
    primary = np.array([r['source_key'].startswith('drive_confirmation_test_') for r in index])
    actual, gp, rf = pack['Y'][primary, 0], pack['estimated_iou'][primary], pack['rf_mean'][primary]
    fig = canvas('Quality estimation is refitted to the updated perception model')
    axes = [fig.add_axes([.08, .22, .37, .62]), fig.add_axes([.56, .22, .37, .62])]
    for ax, predicted, label, color in zip(axes, [gp, rf], ['Version-matched GP', 'Same-input RF'], [TEAL, NAVY]):
        ax.scatter(actual, predicted, s=15, alpha=.6, color=color, edgecolors='none')
        ax.plot([0, 1], [0, 1], color=GRAY, linestyle='--')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Actual whole-frame lane IoU')
        ax.set_ylabel('Estimated IoU')
        value = quality['point']['confirmation']['g1_gp' if label.endswith('GP') else 'g1_rf']['macro_road_mae'][0]
        ax.set_title(f'{label} | road MAE {value:.4f}')
    save(fig, out, '08_quality_gp_rf',
         'Candidate pair held: quality MAE gate failed. Same 366 frames; one whole-frame IoU target; development-only selection.')

    fig = canvas('Same-road review budget: retrieval and review cost')
    adopted='frozen BO mix' if decision['fixed_review_mix_accepted'] else 'common probability + random audit (fixed fallback)'
    fig.text(.08,.875,'Candidate review: '+adopted,fontsize=12,color=TEAL)
    axes = [fig.add_axes([.08, .22, .38, .61]), fig.add_axes([.57, .22, .37, .61])]
    styles = [('g1_policy', 'Frozen BO mix', TEAL), ('global_probability_audit', 'Common probability + audit', NAVY),
              ('rf_low_mean', 'RF low mean', GOLD), ('random_100_seeds', 'Random (100 seeds)', GRAY)]
    rows = quality['same_road_budget_review']['confirmation']
    for policy, label, color in styles:
        selected = sorted([r for r in rows if r['policy'] == policy], key=lambda r: r['fraction'])
        x = [r['fraction'] * 100 for r in selected]
        for ax, key in zip(axes, ['recall', 'precision']):
            values = [r['mean'][key] if 'mean' in r else r[key] for r in selected]
            ax.plot(x, values, '-o', label=label, color=color)
    for ax, title in zip(axes, ['Fraction of low-IoU frames retrieved', 'Low-IoU fraction among reviewed frames']):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Requested review budget (%)')
        ax.set_ylim(0, 1.03)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9, frameon=False, loc='lower right')
    save(fig, out, '09_review_budget',
         'Absolute benchmark IoU < 0.7; per-road ceil budgets. Benchmark-nonlow does not mean operationally safe.')

    risk = quality['risk']['confirmation']
    fig = canvas('Uncertainty diagnostics: ranking, intervals, and failure types')
    ax = fig.add_axes([.08, .23, .36, .58])
    auc = [risk['quality_probability_auc'], risk['sigma_only_auc']]
    ax.bar([0, 1], [v if v is not None else 0 for v in auc], color=[TEAL, GOLD])
    ax.axhline(.5, linestyle='--', color=GRAY)
    ax.set_xticks([0, 1], ['Low-IoU probability', 'Epistemic sigma alone'])
    ax.set_ylabel('AUC for actual IoU < 0.7')
    ax.set_ylim(0, 1)
    for i, value in enumerate(auc):
        if value is not None:
            ax.text(i, value + .025, f'{value:.3f}', ha='center')
        else:ax.text(i,.03,'N/A\n(single class)',ha='center',fontsize=9)
    ax = fig.add_axes([.58, .23, .35, .58])
    intervals = risk['interval_coverage']
    coverage = [r['coverage_90'] for r in intervals]
    ax.bar(range(len(intervals)), [v if v is not None else 0 for v in coverage], color=TEAL)
    ax.axhline(.9, linestyle='--', color=GRAY, label='Nominal 90%')
    labels = ['Whole IoU', 'Near recall', 'Far recall', 'Near F1'][:len(intervals)]
    ax.set_xticks(range(len(intervals)), labels, rotation=15)
    ax.set_ylabel('Empirical interval coverage')
    ax.set_ylim(0, 1.2);ax.set_yticks(np.arange(0,1.01,.2))
    for i,value in enumerate(coverage):
        if value is not None:ax.text(i,value+.03,f"w90={intervals[i]['mean_width_90']:.2f}",ha='center',fontsize=9)
        else:ax.text(i,.03,'N/A',ha='center',fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    save(fig, out, '10_uncertainty_and_coverage',
         'w90 is mean interval width in IoU/recall units. Empirical calibration has no OOD guarantee; recall does not certify precision.')

    fig=canvas('Selection comparison with a fixed adaptation coefficient')
    for j,alpha in enumerate(['1.0','0.5']):
        ax=fig.add_axes([.08+j*.49,.23,.37,.59]);values=held['matched_coefficient_comparisons'][alpha]['methods']
        for i,arm in enumerate(methods):
            gains=np.array(values[arm]['seed_gains']);ax.bar(i,gains.mean(),color=COLORS[arm],width=.65);ax.scatter(i+np.array([-.08,.08]),gains,s=20,color='black',zorder=5)
        ax.axhline(0,color=GRAY,linewidth=.8);ax.set_xticks(range(4),methods,rotation=16,fontsize=10);ax.set_ylabel('Confirmation macro-road lane IoU gain');ax.set_title(r'$\alpha$='+alpha+' for every method');ax.grid(axis='y',alpha=.15)
    save(fig,out,'11_matched_selection','Same new-image budget, replay, optimizer steps and coefficient. Raw alpha1 endpoints may fail retention; no heldout tuning.')

    fig=canvas('What does the GP task covariance contribute?')
    joint=risk['joint_band_diagnostics'];contrast=risk['near_minus_far_recall_contrast']
    ax=fig.add_axes([.09,.24,.35,.57]);brier=[joint['multiclass_brier'],joint['independent_task_brier']]
    ax.bar([0,1],[v or 0 for v in brier],color=[TEAL,GRAY]);ax.set_xticks([0,1],['Full task\ncovariance','Cross terms\nzeroed']);ax.set_ylabel('Joint-state Brier score (lower is better)')
    for i,v in enumerate(brier):
        if v is not None:ax.text(i,v+.015,f'{v:.4f}',ha='center')
        else:ax.text(i,.02,'N/A',ha='center')
    ax=fig.add_axes([.59,.24,.34,.57]);values=[contrast['full_covariance'],contrast['independent_tasks']];width=[v['mean_width_90'] or 0 for v in values]
    ax.bar([0,1],width,color=[TEAL,GRAY]);ax.set_xticks([0,1],['Full task\ncovariance','Cross terms\nzeroed']);ax.set_ylabel('Mean90%width: near–far recall contrast');ax.set_ylim(0,max(max(width)*1.3,.2))
    for i,v in enumerate(values):
        if v['coverage_90'] is not None:ax.text(i,width[i]+max(max(width)*.03,.01),f"coverage={v['coverage_90']:.1%}",ha='center',fontsize=10)
        else:ax.text(i,.02,'N/A',ha='center')
    save(fig,out,'12_task_covariance','Ablation keeps means/marginal variances fixed and zeros cross-task covariance. Image-band states are diagnostics, not operating-normality labels.')

    source_files = ['reports/perception_development_comparison.json', 'reports/perception_heldout_assessment.json',
                    'reports/quality_heldout_assessment.json', 'reports/g1_heldout_predictions.npz', 'deployment/candidate_pair.json']
    (out / 'figure_sources.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evidence_sha256={name: hashlib.sha256((study / name).read_bytes()).hexdigest() for name in source_files},
        decision_status=decision['status'], fixed_review_mix_accepted=decision['fixed_review_mix_accepted']), indent=2) + '\n')
    print('SAVED', out, flush=True)


if __name__ == '__main__':
    main()
