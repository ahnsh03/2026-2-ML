"""Explicit paired native RGB/Semantic input; no legacy target-cache dependency."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from camera_semantic_perception.geometry import letterbox_bgr
from .build_dataset import color_channels, native_marking, read_image
from .features import geometric_valid


def decoded_pixel_sha256(rgb):
    """Hash decoded RGB pixels, independent of PNG compression or alpha storage."""
    return hashlib.sha256(np.asarray(rgb.shape, dtype=np.int64).tobytes() + np.ascontiguousarray(rgb).tobytes()).hexdigest()


def perceptual_fingerprint(rgb):
    """64-bit horizontal dHash plus a small RGB thumbnail; metadata, never X."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = np.packbits(bits.reshape(-1)).tobytes().hex()
    return value


class NativeSamples(Dataset):
    def __init__(self, root, rows):
        self.root, self.rows = Path(root), rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if row.get("view") != "front":
            raise ValueError("quality model requires an explicit front camera")
        bgr, rgb_sha = read_image(self.root, row["image"], cv2.IMREAD_UNCHANGED)
        semantic, semantic_sha = read_image(self.root, row["semantic"], cv2.IMREAD_UNCHANGED)
        for field, actual in (("rgb", rgb_sha), ("semantic", semantic_sha)):
            expected = row.get("expected_sha256", {}).get(field)
            if expected is not None and actual != expected:
                raise ValueError("source " + field + " hash mismatch: " + row["sample_id"])
        bgr = color_channels(bgr, "RGB")
        semantic = color_channels(semantic, "Semantic")
        if bgr.shape != semantic.shape:
            raise ValueError("RGB/Semantic shape mismatch")
        if row.get("original_hw") and list(bgr.shape[:2]) != row["original_hw"]:
            raise ValueError("source image dimensions differ from lineage")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pixel_sha = decoded_pixel_sha256(rgb)
        if row.get("expected_pixel_sha256"):
            order = row.get("expected_pixel_color_order", "RGB")
            if order not in ("RGB", "BGR"):
                raise ValueError("unknown source pixel-hash color order")
            source_pixel_sha = pixel_sha if order == "RGB" else decoded_pixel_sha256(bgr)
            if source_pixel_sha != row["expected_pixel_sha256"]:
                raise ValueError("decoded pixel hash differs from source audit")
        model_input, geometry = letterbox_bgr(bgr, (384, 640))
        native = native_marking(semantic)
        valid = geometric_valid(geometry)
        target = np.zeros(geometry.target_hw, np.uint8)
        x, y, _, _ = geometry.pad_ltrb
        h, w = geometry.resized_hw
        target[y:y+h, x:x+w] = cv2.resize(native, (w, h), interpolation=cv2.INTER_NEAREST)
        # Native policy: visible actors are negative; every non-padding pixel is valid.
        return {"index": index, "image": model_input, "rgb": rgb, "geometry": geometry,
                "target": target, "gt_valid": valid, "native_target": native,
                "hashes": {"rgb": rgb_sha, "semantic": semantic_sha},
                "pixel_sha256": pixel_sha, "dhash64": perceptual_fingerprint(rgb)}
