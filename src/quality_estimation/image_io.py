"""Read root-relative images and identify their original file and decoded RGB pixels."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np


def resolve_input(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError("input manifest paths must be relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("input path escapes data root: " + relative) from exc
    return path


def read_image(root, relative, flags):
    path = resolve_input(root, relative)
    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
    if image is None:
        raise ValueError("image cannot be decoded: " + relative)
    return image, hashlib.sha256(data).hexdigest()


def color_channels(image, label):
    """Match the original training loader's BGR/BGRA policy; alpha is not an input."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(label + " must be a BGR/BGRA uint8 image")
    return image[:, :, :3]


def decoded_pixel_sha256(rgb):
    """Hash decoded RGB pixels, independent of PNG compression or alpha storage."""
    return hashlib.sha256(np.asarray(rgb.shape, dtype=np.int64).tobytes() + np.ascontiguousarray(rgb).tobytes()).hexdigest()
