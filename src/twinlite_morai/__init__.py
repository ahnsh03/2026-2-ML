# 이관: 인지 코드 레포 d0922bb `src/perception/twinlite_morai/__init__.py` (원작성: 안승현)
"""MORAI dataset and loss adapters for TwinLiteNet+."""

from .dataset import MoraiTwinLiteDataset
from .external_model import (
    PINNED_TWINLITE_COMMIT,
    ExternalTwinLiteError,
    load_external_twinlite,
)
from .losses import MaskedTwinLiteLoss, adapt_twinlite_outputs
from .training import move_batch_to_device, twinlite_training_step
from .tasks import (
    BINARY_LANE_TASK,
    ROAD_MARKING_CLASS_NAMES,
    ROAD_MARKING_CLASS_VALUES,
    ROAD_MARKING_TASK,
    TwinLiteTaskSpec,
    configure_twinlite_task,
    load_task_compatible_state_dict,
    task_from_checkpoint,
    task_from_training_heads,
)

__all__ = [
    "MoraiTwinLiteDataset",
    "PINNED_TWINLITE_COMMIT",
    "ExternalTwinLiteError",
    "load_external_twinlite",
    "MaskedTwinLiteLoss",
    "adapt_twinlite_outputs",
    "move_batch_to_device",
    "twinlite_training_step",
    "BINARY_LANE_TASK",
    "ROAD_MARKING_CLASS_NAMES",
    "ROAD_MARKING_CLASS_VALUES",
    "ROAD_MARKING_TASK",
    "TwinLiteTaskSpec",
    "configure_twinlite_task",
    "load_task_compatible_state_dict",
    "task_from_checkpoint",
    "task_from_training_heads",
]
