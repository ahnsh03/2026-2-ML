# 이관: 인지 코드 레포 d0922bb `src/perception/twinlite_morai/external_model.py` (원작성: 안승현)
"""Pinned loader for the official TwinLiteNet+ reference checkout (submodule ``third_party/TwinLiteNetPlus``)."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import torch


DEFAULT_TWINLITE_ROOT = str(Path(__file__).resolve().parents[2] / "third_party" / "TwinLiteNetPlus")
PINNED_TWINLITE_COMMIT = "90f1b8695ae311d5123b05f8534b2e11e42499d2"
SUPPORTED_CONFIGS = ("nano", "small", "medium", "large")


class ExternalTwinLiteError(RuntimeError):
    pass


def _git_commit(root: Path) -> str:
    git_dir = root / ".git"
    if git_dir.is_file():
        marker = git_dir.read_text(encoding="utf-8").strip()
        if not marker.startswith("gitdir: "):
            raise ExternalTwinLiteError(f"invalid gitdir marker: {git_dir}")
        git_dir = (root / marker[len("gitdir: ") :]).resolve()
    head_path = git_dir / "HEAD"
    try:
        head = head_path.read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        ref_name = head[len("ref: ") :]
        loose_ref = git_dir / ref_name
        if loose_ref.is_file():
            return loose_ref.read_text(encoding="utf-8").strip()
        packed_refs = git_dir / "packed-refs"
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == ref_name:
                    return commit
    except (OSError, ValueError) as exc:
        raise ExternalTwinLiteError(
            f"cannot resolve TwinLiteNet+ git commit at {root}: {exc}"
        ) from exc
    raise ExternalTwinLiteError(
        f"cannot resolve TwinLiteNet+ ref {head} at {git_dir}"
    )


def load_external_twinlite(
    config: str = "medium",
    root: Optional[str] = None,
    expected_commit: Optional[str] = PINNED_TWINLITE_COMMIT,
) -> Tuple[torch.nn.Module, str]:
    """Load the official model from the pinned upstream checkout (MIT, not modified)."""
    config = str(config)
    if config not in SUPPORTED_CONFIGS:
        raise ExternalTwinLiteError(
            f"unsupported TwinLite config {config}; choose {SUPPORTED_CONFIGS}"
        )
    source_root = Path(
        root or os.environ.get("TWINLITE_ROOT", DEFAULT_TWINLITE_ROOT)
    ).expanduser().resolve()
    model_path = source_root / "model" / "model.py"
    license_path = source_root / "LICENSE"
    if not model_path.is_file() or not license_path.is_file():
        raise ExternalTwinLiteError(
            f"official TwinLiteNet+ checkout or LICENSE not found: {source_root}"
        )

    commit = _git_commit(source_root)
    if expected_commit and commit != expected_commit:
        raise ExternalTwinLiteError(
            f"TwinLiteNet+ commit mismatch: expected={expected_commit} actual={commit}"
        )

    existing = sys.modules.get("model")
    if existing is not None:
        existing_file = Path(getattr(existing, "__file__", "") or "/").resolve()
        try:
            existing_file.relative_to(source_root)
        except ValueError as exc:
            raise ExternalTwinLiteError(
                f"Python module name 'model' is already owned by {existing_file}"
            ) from exc
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    module = importlib.import_module("model.model")
    model_class = getattr(module, "TwinLiteNetPlus", None)
    if model_class is None:
        raise ExternalTwinLiteError(
            f"TwinLiteNetPlus class not found in {model_path}"
        )
    model = model_class(SimpleNamespace(config=config))
    return model, commit
