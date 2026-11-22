"""IoU labels are separate from operational features and preserve undefined rows."""
from __future__ import annotations

import numpy as np


def build_label(prediction: np.ndarray, target: np.ndarray,
                geometric_valid: np.ndarray, annotation_valid: np.ndarray) -> dict:
    arrays = (prediction, target, geometric_valid, annotation_valid)
    if prediction.ndim != 2 or any(a.shape != prediction.shape for a in arrays):
        raise ValueError("class maps and validity masks must share an HxW shape")
    if (prediction.dtype != np.uint8 or target.dtype != np.uint8
            or not np.isin(prediction, [0, 1, 2, 3]).all()
            or not np.isin(target, [0, 1, 2, 3]).all()):
        raise ValueError("prediction and target must be uint8 class IDs 0..3")
    if geometric_valid.dtype != bool or annotation_valid.dtype != bool:
        raise ValueError("valid masks must be boolean")
    valid = geometric_valid & annotation_valid
    pred = ((prediction == 1) | (prediction == 2)) & valid
    gt = ((target == 1) | (target == 2)) & valid
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    union = tp + fp + fn
    return {"lane_iou": tp / union if union else None, "label_defined": bool(union),
            "undefined_reason": "" if union else ("no_valid_pixels" if not valid.any() else "empty_union"),
            "tp": tp, "fp": fp, "fn": fn, "union": union,
            "gt_pixels": tp + fn, "pred_pixels": tp + fp, "valid_pixels": int(valid.sum())}


def lane_iou_bands(prediction, target, geometric_valid, annotation_valid, *, split_fraction=.64):
    """Partition nonpadding image rows for offline visibility diagnostics, in image coordinates."""
    if not 0 < split_fraction < 1:
        raise ValueError("band boundary must lie inside the original image")
    full = build_label(prediction, target, geometric_valid, annotation_valid)
    image_rows = np.flatnonzero(geometric_valid.any(axis=1))
    if len(image_rows) < 2 or np.any(np.diff(image_rows) != 1):
        raise ValueError("expected a contiguous image with at least two nonpadding rows")
    boundary = image_rows[0] + int(len(image_rows) * split_fraction)
    if boundary <= image_rows[0]:
        raise ValueError("image too short for requested upper band")
    result = {"full": full}
    for name, region in (("upper", slice(None, boundary)), ("lower", slice(boundary, None))):
        result[name] = build_label(prediction[region], target[region], geometric_valid[region], annotation_valid[region])
    return result
