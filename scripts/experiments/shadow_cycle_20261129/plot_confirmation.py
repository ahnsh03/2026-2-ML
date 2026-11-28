"""Static scientific figures for the frozen534 confirmation and native ROS run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common.bridge import sha256_file


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    study = args.study
    sources = ['new_confirmation_assessment/summary.json', 'quality_input_confirmation/summary.json',
               'quality_oof_development_envelope/summary.json', 'visibility_qa/summary.json',
               'contrast_development/summary.json', 'shadow_drive/completion.json',
               'sessions/20261128T135526-1723011/review/status.json']
    assessment, quality, oof, qa, dev, drive, runtime = [read(study / p) for p in sources]
    outputs = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})

    def finish(fig, name, note):
        fig.text(.015, .01, note, fontsize=8.5)
        for extension in ['png', 'svg']:
            path = args.output / (name + '.' + extension)
            fig.savefig(path, dpi=220, bbox_inches='tight')
            outputs.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    fig.subplots_adjust(bottom=.23, wspace=.3)
    dd = {r['name']: r for r in dev['records']}
    names = ['frozen_RF', 'RF_rank_plus_z_0.25']
    for ax, source, title in [(axes[0], dd, 'Development:334 frames'),
                              (axes[1], {k: v['metrics'] for k, v in assessment['selection']['global'].items()},
                               'Frozen confirmation:534 frames')]:
        x = np.arange(2)
        ax.bar(x - .18, [source[n]['near_error_recall'] for n in names], .36,
               color='#426b9c', label='Near-error recall')
        ax.bar(x + .18, [source[n]['utility_road_macro'] for n in names], .36,
               color='#549b86', label='Road-macro utility recall')
        ax.set_xticks(x, ['RF74 reference', 'RF75% + GP z25%'])
        ax.set_ylim(0, .8)
        ax.set_title(title)
        ax.grid(axis='y', alpha=.15)
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].text(.02, .93, 'Not adopted: road-macro gain +0.0053 < +0.01 gate',
                 transform=axes[1].transAxes, fontsize=9, color='#a44040')
    fig.suptitle('Diagnostic GP contrast did not establish a selection gain')
    finish(fig, '01_contrast_confirmation',
           'Fixed global10% budget and audit10%. Different Dev/test cohorts; six physical test blocks. No retuning on534. Window100 near recall: RF .2154, blend .1951.')

    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    fig.subplots_adjust(bottom=.2)
    x, width = np.arange(3), .34
    od = {r['tag']: r for r in oof['records']}
    dev_mae = [od[n]['road_macro_mae'] for n in ['RF_raw', 'GP_raw', 'GP_z']]
    test_mae = [assessment['quality']['RF_only']['quality_road_macro_mae'],
                assessment['quality']['frozen_GP']['quality_road_macro_mae'],
                quality['quality_road_macro_mae']]
    ax.bar(x - width / 2, dev_mae, width, label='Development', color='#426b9c')
    ax.bar(x + width / 2, test_mae, width, label='New confirmation', color='#ad7653')
    ax.set_xticks(x, ['RF52', 'GP52 reference', 'GP57 + OOF error means,z,SD'])
    ax.set_ylabel('Road-macro IoU MAE (lower is better)')
    ax.set_ylim(0, .11)
    ax.legend(frameon=False)
    ax.set_title('GP-derived inputs improved Dev but worsened confirmation: excluded')
    ax.grid(axis='y', alpha=.15)
    finish(fig, '02_quality_input_confirmation',
           'Same2500 fit rows; spatial3-fold OOF auxiliary inputs, Cal835, Dev334, confirmation534. Candidate frozen before test feature/target/output access; during collection.')

    metrics = assessment['selection']['global']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1))
    fig.subplots_adjust(bottom=.22, wspace=.35)
    recall = [assessment['random_1000']['near_error_recall']] + [metrics[n]['metrics']['near_error_recall'] for n in ['frozen_RF', 'frozen_GP']]
    axes[0].bar(['Random mean', 'RF74', 'GP74'], recall, color=['#aaa', '#426b9c', '#549b86'])
    axes[0].axhline(54 / 246, color='#a44040', linestyle='--', label='Budget-only ceiling54/246')
    axes[0].set_ylim(0, .28)
    axes[0].set_ylabel('Near-error recall')
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title('54 review slots for246 near-error frames')
    axes[1].bar(['RF74', 'GP74'], [50 / 54, 51 / 54], color=['#426b9c', '#549b86'])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Near-error fraction among selected')
    axes[1].set_title('Selected near errors:50/54 and51/54')
    for ax in axes:
        ax.grid(axis='y', alpha=.15)
    finish(fig, '03_review_budget_context',
           '534 frames, global budget54, audit10%; unchanged policy. Ceiling ignores audit/other-error constraints. Lower recall alone does not establish worse ranking across cohorts.')

    groups = qa['by_hour']
    hours = list(groups)
    x = np.arange(len(hours))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
    fig.subplots_adjust(bottom=.23, wspace=.3)
    total = [groups[h]['near_errors'] for h in hours]
    chosen = [groups[h]['selected_errors']['frozen_RF'] for h in hours]
    axes[0].bar(x, total, color='#d3dbe4', label='All near errors')
    axes[0].bar(x, chosen, color='#426b9c', label='RF74 selected near errors')
    axes[0].set_xticks(x, hours)
    axes[0].set_xlabel('SDK hour')
    axes[0].set_ylabel('Frame count')
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title('Dawn/night remain the main missed-error populations')
    axes[1].bar(x, [groups[h]['very_low_contrast_cue'] for h in hours], color='#ad7653')
    axes[1].set_xticks(x, hours)
    axes[1].set_xlabel('SDK hour')
    axes[1].set_ylabel('Frames with offline low-contrast cue')
    axes[1].set_title('Low contrast is a diagnostic cue, not a normality label')
    finish(fig, '04_dawn_visibility_qa',
           'All534 displayed Semantic masks match scored GT counts. Cue: near GT-paint/ring median luma difference <=.02 in magnitude; uses GT offline only. No visibility exemption.')

    routes = drive['routes']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(bottom=.23, wspace=.35)
    axes[0].bar(np.arange(len(routes)), [r['distance_m'] for r in routes], color='#426b9c')
    axes[0].set_xticks(np.arange(len(routes)), [str(i + 1) for i in range(len(routes))])
    axes[0].set_xlabel('Short route segment')
    axes[0].set_ylabel('Measured travel (m)')
    axes[0].set_title(f'26R1 low-speed drive: {sum(r["distance_m"] for r in routes):.2f}m')
    axes[1].bar(['Archived', 'Inferred', 'Inference skipped'],
                [293, runtime['processed'], runtime['inference_skipped']], color=['#8c9aa8', '#549b86', '#ad7653'])
    axes[1].set_ylabel('Frame count')
    axes[1].set_title('Native ROS300s run,20Hz input /1Hz archive')
    finish(fig, '05_native_ros_shadow',
           '5070Ti, fixed models,100/100/93 windows =>10/10/10 slots per family. Processing mean .0808s/frame; not end-to-end latency or max-throughput test. No paired GT on drive.')

    (args.output / 'manifest.json').write_text(json.dumps(dict(
        source_sha256=sha256_file(Path(__file__)), input_sha256={p: sha256_file(study / p) for p in sources},
        artifacts={p.name: sha256_file(p) for p in outputs},
        scope='Frozen confirmation and functional runtime are separate populations; no new perception training.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
