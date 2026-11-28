"""Export real, count-verified RGB/GT/prediction cases without aspect distortion."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.bridge import load_target_model, sha256_file
from quality_estimation.build_dataset import native_marking
from quality_estimation.infer_images import load_rgb_row
from quality_estimation.perception_identity import resolve_public_perception
from quality_estimation.spatial_features import SpatialProfile


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['study', 'release', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    study = args.study
    rows = read(study / 'new_confirmation_registry/index.json')['confirmation']['rows']
    qa = read(study / 'visibility_qa/summary.json')
    by_id = {r['sample_id']: i for i, r in enumerate(rows)}
    cases = [dict(r) for r in qa['case_choices']]
    # Fixed same-pose/weather sequence supplements the first-eligible missed cases.
    for hour in [5, 6, 7, 13, 20]:
        sid = f'confirmation29_release26_site00_w0_h{hour:02d}'
        if sid not in {r['sample_id'] for r in cases}:
            cases.append(dict(sample_id=sid, reason='Fixed same-pose Sunny hour sequence'))
    indices = [by_id[c['sample_id']] for c in cases]
    with np.load(study / 'new_confirmation_packs/confirmation.npz') as pack:
        expected = pack['counts'].copy()
    roi = SpatialProfile.load().native[::2, ::2]
    checkpoint, identity, _ = resolve_public_perception(args.release)
    model = load_target_model(checkpoint, torch.device('cpu'), weights='model')
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.mkldnn.deterministic = True
    masks = {}
    # Preserve original CPU extractor batch membership to avoid numeric batching drift.
    with torch.inference_mode():
        for start in sorted({i // 16 * 16 for i in indices}):
            batch = [load_rgb_row(study, row, i)
                     for i, row in enumerate(rows[start:start + 16], start)]
            _, output = model.model(torch.from_numpy(np.stack([r['image'] for r in batch])))
            prob = output.float().softmax(1).numpy()
            for offset, item in enumerate(batch):
                i = start + offset
                if i in indices:
                    masks[i] = np.isin(prob[offset, :, 12:372].argmax(0), [1, 2])
    gallery = []
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    for number, (case, i) in enumerate(zip(cases, indices), 1):
        row = rows[i]
        image, semantic = [study / row[k] for k in ['image', 'semantic']]
        assert sha256_file(image) == row['expected_sha256']['rgb']
        assert sha256_file(semantic) == row['expected_sha256']['semantic']
        rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
        gt = cv2.resize(np.isin(native_marking(cv2.imread(str(semantic))), [1, 2]).astype(np.uint8),
                        (640, 360), interpolation=cv2.INTER_NEAREST).astype(bool)
        pred = masks[i]
        tp, fp, fn = pred & gt, pred & ~gt, ~pred & gt
        counts = np.stack([np.bincount(roi[m], minlength=9) for m in [tp, fp, fn]], 1)
        assert np.array_equal(counts, expected[i]), case['sample_id']
        union = int((pred | gt).sum())
        true_iou = float(tp.sum() / union) if union else None
        errors = np.zeros((360, 640, 3), np.uint8)
        errors[tp], errors[fp], errors[fn] = (50, 190, 100), (240, 85, 80), (55, 165, 235)
        folder = args.output / f'case_{number:02d}'
        folder.mkdir()
        for name, picture in [('rgb', rgb), ('gt', np.repeat((gt * 255).astype(np.uint8)[..., None], 3, 2)),
                              ('prediction', np.repeat((pred * 255).astype(np.uint8)[..., None], 3, 2)),
                              ('error', errors)]:
            cv2.imwrite(str(folder / (name + '.png')), cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))
        fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), layout='constrained')
        for ax, picture, title in zip(axes.flat, [rgb, gt, pred, errors],
                                     ['Original RGB', 'Semantic GT: lane union',
                                      'r2 prediction', 'Green TP / red FP / blue FN']):
            ax.imshow(picture, cmap='gray', vmin=0, vmax=1) if picture.ndim == 2 else ax.imshow(picture)
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(f'Case {number:02d} | SDK hour {row["hour"]:02d} | weather code {row["weather"]} | IoU {true_iou:.3f}')
        fig.savefig(args.output / f'case_{number:02d}.png', dpi=180)
        plt.close(fig)
        gallery.append(dict(case=number, sample_id=case['sample_id'], reason=case['reason'],
                            sdk_hour=int(row['hour']), sdk_weather=int(row['weather']),
                            true_iou=true_iou, gt_and_prediction_counts_match=True,
                            counts=counts.tolist(), rgb_sha256=sha256_file(image),
                            semantic_sha256=sha256_file(semantic)))
    report = dict(perception_identity=identity, cases=gallery,
                  input_sha256={name: sha256_file(study / name) for name in
                                ['new_confirmation_registry/index.json', 'visibility_qa/summary.json',
                                 'new_confirmation_packs/confirmation.npz']},
                  source_sha256=sha256_file(Path(__file__)),
                  scope='First-eligible unselected errors plus fixed same-pose hour series. Counts match exact scored GT/prediction. All component panels preserve16:9. Offline visibility cues are not physical visibility labels.')
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Count-verified gallery cases:', len(gallery))


if __name__ == '__main__':
    main()
