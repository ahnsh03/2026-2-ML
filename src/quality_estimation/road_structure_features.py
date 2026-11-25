"""GT-free lane continuity descriptors on selected road bands of the fixed nine-region profile."""

from __future__ import annotations

import numpy as np

MORPHOLOGY = ("row_coverage", "internal_gap", "soft_outside")
ROAD_SCOPES = (
    "global",
    "far",
    "near",
    "far_left",
    "far_right",
    "near_left",
    "near_right",
)


def morphology(lane, soft_outside, mask):
    """Rows require two predicted lane pixels; the gap excludes unsupported outside rows."""
    active = mask.any(axis=1)
    if not active.any():
        raise ValueError("empty road scope")
    occupied = ((lane & mask).sum(axis=1) >= 2)[active]
    positions = np.flatnonzero(occupied)
    gap = (np.diff(positions).max() - 1) / len(occupied) if len(positions) > 1 else 0.0
    return np.asarray(
        [occupied.mean(), gap, (soft_outside & mask).sum() / mask.sum()], np.float32
    )


def extract_road_structure(probability, profile, scopes=ROAD_SCOPES):
    """Use unpadded 4x360x640 float32 softmax. Regions are image bands, not metric distance."""
    p = probability
    if (
        not scopes
        or len(set(scopes)) != len(scopes)
        or not set(scopes) <= set(ROAD_SCOPES)
    ):
        raise ValueError("unknown or duplicate road scopes")
    if (
        profile.version != "roi9-front-v1"
        or p.shape != (4, 360, 640)
        or p.dtype != np.float32
        or not np.isfinite(p).all()
        or p.min() < 0
        or p.max() > 1
        or not np.allclose(p.sum(0), 1, atol=2e-6, rtol=0)
    ):
        raise ValueError(
            "fixed nine-region profile and normalized unpadded float32 softmax required"
        )
    labels = profile.native[::2, ::2]
    lane = np.isin(p.argmax(0), (1, 2))
    soft = ~lane & ((p[1] + p[2]) >= 0.20)
    masks = {
        name: labels == profile.region_names.index(name) for name in ROAD_SCOPES[3:]
    }
    masks.update(
        global_=np.ones(labels.shape, bool),
        far=masks["far_left"] | masks["far_right"],
        near=masks["near_left"] | masks["near_right"],
    )
    masks["global"] = masks.pop("global_")
    values = np.concatenate([morphology(lane, soft, masks[name]) for name in scopes])
    return values


def ordered_features():
    return [scope + "." + name for scope in ROAD_SCOPES for name in MORPHOLOGY]
