"""Export matched RGB/GT/f0/f1 figures, including remaining losses and estimator misses."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.bridge import load_target_model
from quality_estimation.adaptation_cache import CachedAdaptationSamples
from quality_estimation.perception_identity import resolve_quality_perception
from figures_initial import NAVY, GRAY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['study', 'checkpoint', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    study, output = args.study, args.output
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    decision = json.loads((study / 'deployment/candidate_pair.json').read_text())
    checkpoint, identity, _ = resolve_quality_perception(study / decision['candidate_perception_run'])
    f0 = load_target_model(args.checkpoint, torch.device('cuda'), weights='ema').model
    f1 = load_target_model(checkpoint, torch.device('cuda'), weights='model').model
    before = np.load(study / 'f0_quality/packs/heldout_diagnostic.npz')
    after = np.load(study / 'g1_quality/packs/heldout_diagnostic.npz')
    predictions = np.load(study / 'reports/g1_heldout_predictions.npz')
    rows = [json.loads(line) for line in (study / 'protocol/quality_heldout_diagnostic_cached.jsonl').read_text().splitlines()]
    assert before['ids'].tolist() == after['ids'].tolist() == [r['sample_id'] for r in rows]
    primary = np.array([r['source_key'].startswith('drive_confirmation_test_') for r in rows])
    valid = primary & np.isfinite(before['y']) & np.isfinite(after['y'])
    gains = after['y'] - before['y']
    remaining = np.flatnonzero(valid).tolist()
    choices = []
    criteria = [('maximum_change', gains), ('median_change', -abs(gains - np.nanmedian(gains[valid]))),
                ('minimum_change', -gains), ('largest_prediction_error', abs(predictions['estimated_iou'] - after['y'])),
                ('near_far_gap', predictions['Y'][:, 1] - predictions['Y'][:, 2]),
                ('highest_uncertainty', predictions['quality_epistemic_sd'])]
    for name, score in criteria:
        i = max(remaining, key=lambda j: score[j] if np.isfinite(score[j]) else -np.inf)
        choices.append((name, i))
        remaining.remove(i)
    palette = np.array([[0, 0, 0], [45, 215, 240], [255, 215, 35], [245, 90, 180]], np.uint8)
    dataset = CachedAdaptationSamples(study / 'cache', [rows[i] for _, i in choices], verify=True)
    sources = []
    for j, (reason, index) in enumerate(choices):
        item = dataset[j]
        image = item['image'].unsqueeze(0).cuda()
        gt = item['targets']['road_marking'].numpy()[12:372]
        with torch.inference_mode():
            old = f0(image)[1].argmax(1).cpu().numpy()[0, 12:372]
            new = f1(image)[1].argmax(1).cpu().numpy()[0, 12:372]
        def iou(mask):
            p, t = np.isin(mask, [1, 2]), np.isin(gt, [1, 2])
            union = np.count_nonzero(p | t)
            return np.count_nonzero(p & t) / union if union else None
        assert abs(iou(old) - float(before['y'][index])) < 1e-6
        assert abs(iou(new) - float(after['y'][index])) < 1e-6
        rgb = cv2.cvtColor(cv2.imread(str(Path('/mnt') / rows[index]['image'])), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
        def overlay(mask):
            result = rgb.copy()
            lane = mask > 0
            result[lane] = np.rint(.35 * result[lane] + .65 * palette[mask[lane]]).astype(np.uint8)
            return result
        panels = [rgb, overlay(gt), overlay(old), overlay(new)]
        names = ['RGB', 'Verified GT', f'f0 | IoU {iou(old):.3f}', f'f1 | IoU {iou(new):.3f}']
        tag = f'case_{chr(65 + j)}_{reason}'
        for name, panel in zip(['rgb', 'gt', 'f0', 'f1'], panels):
            cv2.imwrite(str(output / (tag + '_' + name + '.png')), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        fig, axes = plt.subplots(2, 2, figsize=(12, 6.75), facecolor='white')
        fig.subplots_adjust(left=.025, right=.975, top=.84, bottom=.10, hspace=.20, wspace=.06)
        fig.suptitle(f'Case {chr(65+j)}: {reason.replace("_", " ")}', x=.04, y=.98, ha='left', color=NAVY, fontsize=18, weight='bold')
        fig.text(.04, .905, f'Actual IoU {iou(old):.3f} → {iou(new):.3f} | GP estimate {predictions["estimated_iou"][index]:.3f}', color=NAVY, fontsize=12)
        for ax, panel, title in zip(axes.ravel(), panels, names):
            ax.imshow(panel)
            ax.set_title(title, fontsize=12, color=NAVY)
            ax.axis('off')
        fig.text(.04, .035, 'Same RGB and verified GT. White/yellow union metric; stop lines shown in pink but excluded from lane IoU.', fontsize=9, color=GRAY)
        fig.savefig(output / (tag + '.png'), dpi=160)
        plt.close(fig)
        sources.append(dict(case=tag, selection_reason=reason, sample_id=rows[index]['sample_id'],
                            source_rgb_sha256=rows[index]['expected_sha256']['rgb'],
                            old_iou=iou(old), new_iou=iou(new), estimated_iou=float(predictions['estimated_iou'][index]),
                            network_grid='640x360 after removing letterbox padding; all panels 16:9',
                            perception_identity=identity))
    (output / 'figure_sources_private.json').write_text(json.dumps(sources, indent=2) + '\n')
    (output / 'README.txt').write_text('Panels are 640x360 (16:9), comparison sheets1920x1080. Preserve aspect ratio in PPT.\nGT pixels were checked against both stored IoU labels before export.\nExamples deliberately include improvements, losses and estimator overconfidence; they are illustrative, not randomly sampled.\nSource IDs stay private in figure_sources_private.json.\n')
    print('EXPORTED', len(sources), 'matched cases; all GT/metric checks passed', flush=True)


if __name__ == '__main__':
    main()
