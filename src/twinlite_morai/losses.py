# 이관: 인지 코드 레포 d0922bb `src/perception/twinlite_morai/losses.py` (원작성: 안승현)
"""Padding-aware two-head segmentation loss for TwinLiteNet+."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tasks import BINARY_LANE_TASK, TwinLiteTaskSpec


TensorPair = Union[Sequence[torch.Tensor], Mapping[str, torch.Tensor]]


def adapt_twinlite_outputs(
    outputs: TensorPair, secondary_head: str = "lane"
) -> Dict[str, torch.Tensor]:
    """Normalize model output to named heads.

    The official TwinLiteNet+ forward contract is `(out_da, out_ll)`. Named
    output is also accepted so future in-repo models do not depend on tuple order.
    """
    if secondary_head not in ("lane", "road_marking"):
        raise ValueError(f"unsupported TwinLite secondary head: {secondary_head}")
    if isinstance(outputs, Mapping):
        if "drivable" not in outputs or secondary_head not in outputs:
            raise ValueError(
                f"model output mapping needs drivable and {secondary_head} logits"
            )
        result = {
            "drivable": outputs["drivable"],
            secondary_head: outputs[secondary_head],
        }
    elif isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        result = {"drivable": outputs[0], secondary_head: outputs[1]}
    else:
        raise TypeError(
            "TwinLite output must be (drivable_logits, lane_logits) or a named mapping"
        )
    for head, logits in result.items():
        if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
            raise ValueError(f"{head} logits must have shape [B,C,H,W]")
        expected_channels = 4 if head == "road_marking" else (1, 2)
        if isinstance(expected_channels, tuple):
            if logits.shape[1] not in expected_channels:
                raise ValueError(
                    f"{head} logits need 1 or 2 channels, got {logits.shape[1]}"
                )
        elif logits.shape[1] != expected_channels:
            raise ValueError(
                f"{head} logits need {expected_channels} channels, got {logits.shape[1]}"
            )
    return result


def _foreground_probabilities(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    return torch.sigmoid(logits[:, 0])


def _masked_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    alpha: float,
    gamma: float,
    eps: float,
) -> torch.Tensor:
    if logits.shape[1] == 2:
        base = F.cross_entropy(logits, target, reduction="none")
    else:
        base = F.binary_cross_entropy_with_logits(
            logits[:, 0], target.float(), reduction="none"
        )
    probability_true = torch.exp(-base)
    alpha_factor = torch.where(
        target.bool(),
        torch.as_tensor(alpha, dtype=base.dtype, device=base.device),
        torch.as_tensor(1.0 - alpha, dtype=base.dtype, device=base.device),
    )
    weighted = alpha_factor * (1.0 - probability_true).pow(gamma) * base
    valid_float = valid.to(dtype=weighted.dtype)
    return (weighted * valid_float).sum() / valid_float.sum().clamp_min(eps)


def _masked_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    alpha: float,
    gamma: float,
    eps: float,
) -> torch.Tensor:
    foreground = _foreground_probabilities(logits)
    probabilities = torch.stack((1.0 - foreground, foreground), dim=1)
    target_one_hot = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2)
    target_one_hot = target_one_hot.to(dtype=probabilities.dtype)
    valid_float = valid[:, None].to(dtype=probabilities.dtype)

    reduce_dims = (0, 2, 3)
    true_positive = (probabilities * target_one_hot * valid_float).sum(reduce_dims)
    false_positive = (
        probabilities * (1.0 - target_one_hot) * valid_float
    ).sum(reduce_dims)
    false_negative = (
        (1.0 - probabilities) * target_one_hot * valid_float
    ).sum(reduce_dims)
    beta = 1.0 - alpha
    score = (true_positive + eps) / (
        true_positive + alpha * false_positive + beta * false_negative + eps
    )
    return (1.0 - score).pow(gamma).mean()


def _masked_multiclass_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    gamma: float,
    eps: float,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    base = F.cross_entropy(logits, target, reduction="none")
    probability_true = torch.softmax(logits, dim=1).gather(
        1, target[:, None]
    )[:, 0]
    weighted = (1.0 - probability_true).pow(gamma) * base
    if class_weights is not None:
        weighted = weighted * class_weights[target]
    valid_float = valid.to(dtype=weighted.dtype)
    return (weighted * valid_float).sum() / valid_float.sum().clamp_min(eps)


def _masked_multiclass_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    alpha: float,
    gamma: float,
    eps: float,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    class_count = int(logits.shape[1])
    probabilities = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target, num_classes=class_count).permute(0, 3, 1, 2)
    target_one_hot = target_one_hot.to(dtype=probabilities.dtype)
    valid_float = valid[:, None].to(dtype=probabilities.dtype)
    reduce_dims = (0, 2, 3)
    true_positive = (probabilities * target_one_hot * valid_float).sum(reduce_dims)
    false_positive = (
        probabilities * (1.0 - target_one_hot) * valid_float
    ).sum(reduce_dims)
    false_negative = (
        (1.0 - probabilities) * target_one_hot * valid_float
    ).sum(reduce_dims)
    beta = 1.0 - alpha
    score = (true_positive + eps) / (
        true_positive + alpha * false_positive + beta * false_negative + eps
    )
    loss = (1.0 - score).pow(gamma)
    if class_weights is None:
        return loss.mean()
    normalized = class_weights / class_weights.sum().clamp_min(eps)
    return (loss * normalized).sum()


class MaskedTwinLiteLoss(nn.Module):
    """Focal + Tversky for a supported TwinLite two-head task.

    Defaults retain the official TwinLiteNet+ loss hyperparameters while
    replacing its front-only `[:, :, 12:-12]` crop with each sample's valid mask.
    """

    def __init__(
        self,
        drivable_tversky_alpha: float = 0.7,
        lane_tversky_alpha: float = 0.9,
        tversky_gamma: float = 4.0 / 3.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        drivable_weight: float = 1.0,
        lane_weight: float = 1.0,
        road_marking_weight: float = 1.0,
        road_marking_tversky_alpha: float = 0.9,
        road_marking_class_weights: Optional[Sequence[float]] = None,
        task_spec: TwinLiteTaskSpec = BINARY_LANE_TASK,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.task_spec = task_spec
        self.secondary_head = task_spec.secondary_head
        self.tversky_alpha = {
            "drivable": float(drivable_tversky_alpha),
            "lane": float(lane_tversky_alpha),
            "road_marking": float(road_marking_tversky_alpha),
        }
        self.head_weights = {
            "drivable": float(drivable_weight),
            "lane": float(lane_weight),
            "road_marking": float(road_marking_weight),
        }
        self.tversky_gamma = float(tversky_gamma)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.eps = float(eps)
        for name in self.task_spec.heads:
            value = self.tversky_alpha[name]
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} Tversky alpha must be in [0,1]")
        if not 0.0 <= self.focal_alpha <= 1.0:
            raise ValueError("focal alpha must be in [0,1]")
        if road_marking_class_weights is not None:
            if len(road_marking_class_weights) != 4 or any(
                float(value) <= 0 for value in road_marking_class_weights
            ):
                raise ValueError(
                    "road-marking class weights must contain four positive values"
                )
            weights = torch.tensor(road_marking_class_weights, dtype=torch.float32)
        else:
            weights = torch.empty(0, dtype=torch.float32)
        self.register_buffer("road_marking_class_weights", weights)

    def forward(
        self,
        outputs: TensorPair,
        targets: Mapping[str, torch.Tensor],
        valid_mask: Union[torch.Tensor, Mapping[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        logits_by_head = adapt_twinlite_outputs(outputs, self.secondary_head)
        valid_by_head = (
            dict(valid_mask)
            if isinstance(valid_mask, Mapping)
            else {head: valid_mask for head in self.task_spec.heads}
        )

        components: Dict[str, torch.Tensor] = {}
        total = None
        for head in self.task_spec.heads:
            if head not in targets:
                raise ValueError(f"missing {head} target")
            target = targets[head]
            if target.ndim == 4 and target.shape[1] == 1:
                target = target[:, 0]
            target = target.long()
            logits = logits_by_head[head]
            if head not in valid_by_head:
                raise ValueError(f"missing {head} valid mask")
            head_valid = valid_by_head[head]
            if head_valid.ndim == 4 and head_valid.shape[1] == 1:
                head_valid = head_valid[:, 0]
            if head_valid.ndim != 3:
                raise ValueError(
                    f"{head} valid mask must have shape [B,H,W] or [B,1,H,W]"
                )
            head_valid = head_valid.bool()
            if not torch.any(head_valid):
                raise ValueError(f"{head} valid mask contains no trainable pixels")
            expected = (logits.shape[0], logits.shape[2], logits.shape[3])
            if tuple(target.shape) != expected or tuple(head_valid.shape) != expected:
                raise ValueError(
                    f"{head} shape mismatch: logits={tuple(logits.shape)} "
                    f"target={tuple(target.shape)} valid={tuple(head_valid.shape)}"
                )
            class_count = self.task_spec.class_counts[head]
            if torch.any((target < 0) | (target >= class_count)):
                raise ValueError(
                    f"{head} target must contain class ids 0..{class_count - 1}"
                )

            if head == "road_marking":
                class_weights = (
                    self.road_marking_class_weights
                    if self.road_marking_class_weights.numel()
                    else None
                )
                focal = _masked_multiclass_focal_loss(
                    logits,
                    target,
                    head_valid,
                    gamma=self.focal_gamma,
                    eps=self.eps,
                    class_weights=class_weights,
                )
                tversky = _masked_multiclass_tversky_loss(
                    logits,
                    target,
                    head_valid,
                    alpha=self.tversky_alpha[head],
                    gamma=self.tversky_gamma,
                    eps=self.eps,
                    class_weights=class_weights,
                )
            else:
                focal = _masked_focal_loss(
                    logits,
                    target,
                    head_valid,
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                    eps=self.eps,
                )
                tversky = _masked_tversky_loss(
                    logits,
                    target,
                    head_valid,
                    alpha=self.tversky_alpha[head],
                    gamma=self.tversky_gamma,
                    eps=self.eps,
                )
            head_loss = focal + tversky
            components[f"{head}_focal"] = focal
            components[f"{head}_tversky"] = tversky
            components[f"{head}_loss"] = head_loss
            weighted = self.head_weights[head] * head_loss
            total = weighted if total is None else total + weighted

        components["loss"] = total
        return components
