"""Build one verified feature pack per manifest using the frozen perception model."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.bridge import DEFAULT_CHECKPOINT_SHA256, load_target_model, sha256_file
from quality_estimation.build_dataset import collate, worker_init
from quality_estimation.labels import build_label
from quality_estimation.native import NativeSamples
from quality_estimation.roi10 import reduce_heads
from quality_estimation.spatial_features import ROI7_PROFILE_DIR, SpatialProfile, spatial_feature_schema, extract_spatial_features


MORPH = ['row_coverage', 'internal_gap', 'soft_outside']
ROAD = ['far_left', 'far_right', 'near_left', 'near_right']
ADDITIONAL = ['prospective_known', 'prospective_unseen', 'night_challenge',
              'nov23_roi10_run_a', 'nov23_roi10_run_b', 'nov23_roi10_run_c',
              'roi10_other_road_a', 'roi10_other_road_b']


def morphology(lane, soft, mask):
    active_rows = mask.any(axis=1)
    occupied = ((lane & mask).sum(axis=1) >= 2)[active_rows]
    positions = np.flatnonzero(occupied)
    gap = (np.diff(positions).max() - 1) / len(occupied) if len(positions) > 1 else 0.
    return [occupied.mean(), gap, (soft & mask).sum() / mask.sum()]


def extra_features(p, masks):
    cls = p.argmax(axis=0)
    lane = (cls == 1) | (cls == 2)
    soft = (~lane) & ((p[1] + p[2]) >= .20)
    values = []
    for mask in masks:
        values.extend(morphology(lane, soft, mask))
    entropy = -(p * np.log(np.maximum(p, np.finfo(np.float32).tiny))).sum(0) / np.log(4)
    flat = entropy.ravel()
    k = (flat.size + 9) // 10
    top = np.partition(flat, flat.size - k)[-k:].mean()
    h, w = entropy.shape
    y = np.unique(np.r_[np.arange(0, h - 31, 16), h - 32])[:, None]
    x = np.unique(np.r_[np.arange(0, w - 31, 16), w - 32])[None, :]
    cum = cv2.integral(entropy, sdepth=cv2.CV_64F)
    patch = (cum[y+32, x+32] - cum[y+32, x] - cum[y, x+32] + cum[y,x]).max() / 1024
    top2 = np.partition(p[:, lane], 2, axis=0)[-2:] if lane.any() else None
    margin = (top2[1] - top2[0]).mean() if top2 is not None else 0.
    return values + [top, patch, margin]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--reserve-motion', action='store_true',
                        help='Extract only the two complete archived driving cohorts to a separate directory')
    args = parser.parse_args()
    protocols = args.data / 'ml_roi10_20261123/afternoon/protocols'
    source = args.data / 'ml_quality_study_20261123'
    roster = json.loads((protocols / 'roster.json').read_text())
    keys = {r['key']: r for r in roster if r['role'] in
            ('train', 'development', 'calibration', 'test', 'geography')}
    keys.update({k: {} for k in ADDITIONAL})
    if args.reserve_motion:
        keys = {r['key']: r for r in roster if r['role'] in ('motion', 'switch')}
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    assert sha256_file(args.checkpoint) == DEFAULT_CHECKPOINT_SHA256
    model = load_target_model(args.checkpoint, torch.device('cuda'), weights='ema')
    profile = SpatialProfile.load(ROI7_PROFILE_DIR)
    native_half = profile.native[::2, ::2]
    masks = [np.ones_like(native_half, dtype=bool)]
    masks.extend(native_half == profile.region_names.index(s) for s in ROAD)
    schema = spatial_feature_schema(profile, include_global=True)
    names = schema['ordered_features'] + [r+'.'+v for r in ['global']+ROAD for v in MORPH]
    names += ['global.top_entropy', 'global.patch_entropy', 'global.lane_margin']
    args.out.mkdir(parents=True, exist_ok=True)
    seen_ids, seen_pixels = set(), set()
    receipts = {}
    for key, record in keys.items():
        path = protocols / (key + '.jsonl')
        digest = sha256_file(path)
        if record:
            assert digest == record['manifest_sha256'], key
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in rows:
            assert row['sample_id'] not in seen_ids and row['pixel_sha256'] not in seen_pixels
            assert set(row['expected_sha256']) == {'rgb','semantic'}
            seen_ids.add(row['sample_id'])
            seen_pixels.add(row['pixel_sha256'])
        dest = args.out / (key + '.npz')
        samples = NativeSamples(source, rows)
        loader = DataLoader(samples, batch_size=12, num_workers=4, collate_fn=collate,
                            worker_init_fn=worker_init, pin_memory=True)
        vectors, labels = [], []
        with torch.inference_mode():
            for images, batch in loader:
                da, marking = model.model(images.cuda(non_blocking=True))
                for d, p, item in zip(da.argmax(1).cpu().numpy(), marking.float().softmax(1).cpu().numpy(), batch):
                    assert item['index'] == len(labels)
                    assert item['pixel_sha256'] == rows[item['index']]['pixel_sha256']
                    heads = reduce_heads(d, p, item['geometry'])
                    base = extract_spatial_features(item['rgb'], heads, profile, include_global=True)
                    vectors.append(np.r_[base, extra_features(p[:, 12:372], masks)].astype(np.float32))
                    label = build_label(p.argmax(0).astype(np.uint8), item['target'], item['gt_valid'], item['gt_valid'])
                    labels.append([label['lane_iou'] if label['label_defined'] else np.nan,
                                   label['tp'], label['fp'], label['fn']])
        assert np.isfinite(vectors).all() and len(rows) == len(vectors)
        np.savez_compressed(dest, x=vectors, labels=labels, names=names,
                            **{f: np.asarray([r.get(f, -1) for r in rows]) for f in
                               ('sample_id','pixel_sha256','pose','xy','hour','weather','frame_id')})
        receipts[key] = dict(rows=len(rows), manifest_sha256=digest, pack_sha256=sha256_file(dest))
        print(key, len(rows), flush=True)
    receipt = dict(schema=schema, columns=names, checkpoint=model.checkpoint_sha256,
                   submodule=model.upstream_commit, epoch=model.epoch, torch=torch.__version__,
                   opencv=cv2.__version__, numpy=np.__version__, sources=receipts)
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
