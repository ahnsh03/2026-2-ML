# 이관: 인지 코드 레포 d0922bb `src/perception/twinlite_morai/tasks.py` (원작성: 안승현)
"""Task contracts and output-head adaptation for the pinned TwinLite model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple


ROAD_MARKING_CLASS_NAMES = (
    "background",
    "white_lane",
    "yellow_lane",
    "stopline",
)
ROAD_MARKING_CLASS_VALUES = {
    name: index for index, name in enumerate(ROAD_MARKING_CLASS_NAMES)
}


@dataclass(frozen=True)
class TwinLiteTaskSpec:
    name: str
    heads: Tuple[str, str]
    class_names: Dict[str, Tuple[str, ...]]

    @property
    def secondary_head(self) -> str:
        return self.heads[1]

    @property
    def class_counts(self) -> Dict[str, int]:
        return {head: len(names) for head, names in self.class_names.items()}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "heads": list(self.heads),
            "class_names": {
                head: list(names) for head, names in self.class_names.items()
            },
            "class_counts": self.class_counts,
        }


BINARY_LANE_TASK = TwinLiteTaskSpec(
    name="drivable_lane_binary",
    heads=("drivable", "lane"),
    class_names={
        "drivable": ("background", "drivable"),
        "lane": ("background", "lane"),
    },
)

ROAD_MARKING_TASK = TwinLiteTaskSpec(
    name="drivable_road_marking_4class",
    heads=("drivable", "road_marking"),
    class_names={
        "drivable": ("background", "drivable"),
        "road_marking": ROAD_MARKING_CLASS_NAMES,
    },
)


def task_from_training_heads(heads: Iterable[str]) -> TwinLiteTaskSpec:
    normalized = tuple(str(head) for head in heads)
    head_set = frozenset(normalized)
    if len(normalized) != len(head_set):
        raise ValueError(f"training heads contain duplicates: {normalized}")
    if head_set == frozenset(BINARY_LANE_TASK.heads):
        return BINARY_LANE_TASK
    if head_set == frozenset(ROAD_MARKING_TASK.heads):
        return ROAD_MARKING_TASK
    raise ValueError(
        "supported TwinLite head sets are {} or {}, got {}".format(
            BINARY_LANE_TASK.heads, ROAD_MARKING_TASK.heads, normalized
        )
    )


def task_from_checkpoint(checkpoint: Dict[str, Any]) -> TwinLiteTaskSpec:
    task = checkpoint.get("task_spec") or {}
    if task.get("heads"):
        return task_from_training_heads(task["heads"])
    args = checkpoint.get("args") or {}
    if args.get("training_heads"):
        return task_from_training_heads(args["training_heads"])
    return BINARY_LANE_TASK


def configure_twinlite_task(model, task: TwinLiteTaskSpec) -> Tuple[str, ...]:
    """Resize only the official model's final lane block for a task.

    The official tuple order remains `(drivable, secondary)`. For the marking
    task, the secondary tensor changes from two binary-lane channels to four
    mutually exclusive road-marking channels. The external checkout is never
    edited.
    """
    block = getattr(model, "out_ll", None)
    up_conv = getattr(block, "up_conv", None)
    deconv = getattr(up_conv, "deconv", None)
    if block is None or deconv is None:
        raise TypeError("unsupported TwinLite model: out_ll final block not found")
    current_classes = int(deconv.out_channels)
    requested_classes = task.class_counts[task.secondary_head]
    if current_classes == requested_classes:
        return ()
    if current_classes != 2 or requested_classes != 4:
        raise ValueError(
            f"unsupported secondary head conversion: {current_classes}->{requested_classes}"
        )
    in_channels = int(deconv.in_channels)
    model.out_ll = type(block)(in_channels, requested_classes, last=True)
    return ("out_ll",)


def _belongs_to_module(parameter_name: str, module_names: Iterable[str]) -> bool:
    return any(
        parameter_name == module_name
        or parameter_name.startswith(module_name + ".")
        for module_name in module_names
    )


def load_task_compatible_state_dict(
    model,
    source_state,
    source_task: TwinLiteTaskSpec,
    target_task: TwinLiteTaskSpec,
    reset_modules: Iterable[str],
) -> Dict[str, Any]:
    """Load an exact task or a controlled binary-lane to marking transfer."""
    reset_modules = tuple(reset_modules)
    if source_task == target_task:
        model.load_state_dict(source_state)
        return {
            "transfer": "exact_task",
            "loaded_parameter_tensors": len(source_state),
            "reset_modules": [],
            "reset_parameter_tensors": [],
        }
    if source_task != BINARY_LANE_TASK or target_task != ROAD_MARKING_TASK:
        raise RuntimeError(
            "unsupported initialization task transfer: "
            f"{source_task.name} -> {target_task.name}"
        )
    if reset_modules != ("out_ll",):
        raise RuntimeError(
            "binary-lane to road-marking transfer must reset only out_ll; "
            f"got {reset_modules}"
        )

    destination_state = model.state_dict()
    compatible = {}
    reset_parameter_tensors = []
    unexpected = []
    for name, value in source_state.items():
        if _belongs_to_module(name, reset_modules):
            reset_parameter_tensors.append(name)
        elif name not in destination_state:
            unexpected.append(name)
        elif tuple(value.shape) == tuple(destination_state[name].shape):
            compatible[name] = value
        else:
            reset_parameter_tensors.append(name)
    missing = [name for name in destination_state if name not in compatible]
    disallowed = [
        name
        for name in unexpected + missing + reset_parameter_tensors
        if not _belongs_to_module(name, reset_modules)
    ]
    if disallowed:
        raise RuntimeError(
            "initialization differs outside the permitted output module: "
            + ", ".join(sorted(set(disallowed)))
        )
    incompatible = model.load_state_dict(compatible, strict=False)
    invalid_reported_missing = [
        name
        for name in incompatible.missing_keys
        if not _belongs_to_module(name, reset_modules)
    ]
    if incompatible.unexpected_keys or invalid_reported_missing:
        raise RuntimeError(
            "unexpected partial initialization result: "
            f"missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return {
        "transfer": "binary_lane_to_road_marking",
        "loaded_parameter_tensors": len(compatible),
        "reset_modules": list(reset_modules),
        "reset_parameter_tensors": sorted(set(reset_parameter_tensors + missing)),
    }
