"""외부 인지 코드 레포(읽기 전용 클론)와 대상 인지 모델을 이 레포에서 쓰기 위한 연결부.

외부 코드는 수정하지 않고 ``sys.path``로 불러온다. 선별 근거는
``docs/05_구현계획/외부코드_선별.md``에 있다.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# 프로젝트 전 기간 동안 고정하는 대상 모델 (구현 계획 §2)
DEFAULT_CHECKPOINT_RELPATH = (
    "twinlite_medium_morai_v5_actor_negative_init_ema_b12_e18_20260906/best_mean.pt"
)
DEFAULT_DATASET_VERSION = "twinlite_morai_v5_marking_actor_negative"
VIEWS = ("front", "left", "right")

_IMPORT_ROOTS = (
    "src/perception",  # twinlite_morai
    "src/perception/drivable_bev/src",  # drivable_bev
    "src/perception/camera_semantic_perception/src",  # camera_semantic_perception
)


def resolve_perception_repo(value: Optional[str] = None) -> Path:
    root = Path(value or os.environ.get("PERCEPTION_REPO", "")).expanduser()
    if not (root / "src" / "perception" / "twinlite_morai").is_dir():
        raise RuntimeError(
            "인지 코드 레포를 찾을 수 없습니다: {!r}. --perception-repo 또는 PERCEPTION_REPO를 지정하세요".format(
                str(root)
            )
        )
    return root.resolve()


def add_perception_import_paths(repo: Path) -> Path:
    """외부 인지 패키지를 import할 수 있도록 경로를 추가한다."""
    for relative in _IMPORT_ROOTS:
        path = str(repo / relative)
        if path not in sys.path:
            sys.path.insert(0, path)
    return repo


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(repo: Path) -> Optional[str]:
    """``git`` 실행 없이 ``.git``에서 HEAD 커밋을 읽는다 (읽기 전용 마운트 대응)."""
    git_dir = Path(repo) / ".git"
    try:
        if git_dir.is_file():
            marker = git_dir.read_text(encoding="utf-8").strip()
            git_dir = (Path(repo) / marker[len("gitdir: "):]).resolve()
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        ref = head[len("ref: "):]
        loose = git_dir / ref
        if loose.is_file():
            return loose.read_text(encoding="utf-8").strip()
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith(("#", "^")):
                    commit, name = line.split(" ", 1)
                    if name == ref:
                        return commit
    except (OSError, ValueError):
        return None
    return None  # 아직 커밋이 없는 브랜치


def parse_condition(run_id: str) -> Dict[str, str]:
    """run_id 규칙(``..._<weather>_sim<hh><am|pm>_...``)에서 날씨·시각을 읽는다."""
    weather = "legacy"
    for candidate in ("sunny", "foggy", "cloudy", "rainy"):
        if "_{}_".format(candidate) in run_id:
            weather = candidate
            break
    match = re.search(r"_sim_?([^_]+)_", run_id)  # sim11am / sim_unknown
    return {"weather": weather, "sim_time": match.group(1) if match else "legacy"}


def resolve_checkpoint(value: Optional[str] = None) -> Path:
    if value:
        path = Path(value).expanduser()
    else:
        root = Path(os.environ.get("PERCEPTION_CKPT_ROOT", "")).expanduser()
        path = root / DEFAULT_CHECKPOINT_RELPATH
    if not path.is_file():
        raise RuntimeError("체크포인트가 없습니다: {}".format(path))
    return path.resolve()


@dataclass(frozen=True)
class LoadedModel:
    model: object
    checkpoint: Path
    checkpoint_sha256: str
    epoch: Optional[int]
    weights: str
    upstream_commit: str


def load_target_model(
    checkpoint: Path,
    device,
    twinlite_root: Optional[str] = None,
    config: str = "medium",
    weights: str = "auto",
) -> LoadedModel:
    """외부 평가 스크립트(``evaluate_twinlite.py``)와 같은 절차로 대상 모델을 불러온다.

    EMA 가중치를 우선 사용하고, 체크포인트와 TwinLiteNet+ 소스의 upstream 커밋이
    다르면 중단한다.
    """
    import torch
    from twinlite_morai import load_external_twinlite

    state = torch.load(str(checkpoint), map_location="cpu")
    use_ema = weights == "ema" or (weights == "auto" and state.get("ema_state_dict") is not None)
    if use_ema and state.get("ema_state_dict") is None:
        raise RuntimeError("체크포인트에 EMA 가중치가 없습니다")
    state_key = "ema_state_dict" if use_ema else "model_state_dict"

    model, upstream_commit = load_external_twinlite(config=config, root=twinlite_root)
    if state.get("upstream_commit") != upstream_commit:
        raise RuntimeError("체크포인트와 TwinLiteNet+ 소스의 upstream 커밋이 다릅니다")
    model.load_state_dict(state[state_key])
    model = model.to(device).eval()
    return LoadedModel(
        model=model,
        checkpoint=Path(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        epoch=state.get("epoch"),
        weights=state_key,
        upstream_commit=upstream_commit,
    )
