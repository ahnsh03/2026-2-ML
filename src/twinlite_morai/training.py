# 이관: 인지 코드 레포 d0922bb `src/perception/twinlite_morai/training.py` (원작성: 안승현)
"""Small training-step bridge between a TwinLite model, batch and masked loss."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .losses import MaskedTwinLiteLoss, adapt_twinlite_outputs


def move_batch_to_device(
    batch: Dict[str, Any], device: torch.device, non_blocking: bool = True
) -> Dict[str, Any]:
    """Move trainable tensors while leaving provenance metadata on the host."""
    return {
        "image": batch["image"].to(device, non_blocking=non_blocking),
        "targets": {
            head: target.to(device, non_blocking=non_blocking)
            for head, target in batch["targets"].items()
        },
        "valid_mask": batch["valid_mask"].to(
            device, non_blocking=non_blocking
        ),
        "valid_masks": {
            head: mask.to(device, non_blocking=non_blocking)
            for head, mask in batch.get("valid_masks", {}).items()
        },
        "meta": batch.get("meta"),
    }


def twinlite_training_step(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    criterion: MaskedTwinLiteLoss,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Forward an already device-placed batch and compute named 2-head loss."""
    raw_outputs = model(batch["image"])
    outputs = adapt_twinlite_outputs(raw_outputs, criterion.secondary_head)
    valid = batch.get("valid_masks") or batch["valid_mask"]
    losses = criterion(outputs, batch["targets"], valid)
    return outputs, losses
