"""사진 한 장에서 안개 정도(0~1)를 잰다.

원리: 맑은 날 사진에는 어느 부분을 봐도 아주 어두운 픽셀이 조금씩 있다(그림자, 타이어, 짙은 색 물체).
안개가 끼면 뿌연 빛이 사진 전체에 덧씌워져 가장 어두운 픽셀도 밝아진다.
그래서 "작은 칸마다 가장 어두운 값"의 평균을 안개 정도로 쓴다.

  1. 픽셀마다 파랑·초록·빨강 세 값 중 가장 작은 값을 고른다.
  2. 픽셀마다 주변 patch×patch 칸 안에서 1번 값 중 가장 작은 값을 고른다.
  3. 정한 행 범위에서 2번 값을 평균하고 255로 나눈다.

하늘은 맑아도 원래 밝아서 값이 크게 나오므로, 도로 쪽 행 범위에서 재는 것이 좋다.

  python3 -m foggy.fog_estimate 1.png 2.png --region 도로:215:320 --region 위쪽:35:205
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def dark_channel(bgr: np.ndarray, patch: int = 15) -> np.ndarray:
    """픽셀마다 주변 patch×patch 칸에서 세 색 중 가장 어두운 값 (0~255)."""
    darkest_color = bgr[:, :, :3].min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(darkest_color, kernel)


def fog_score(bgr: np.ndarray, rows: Optional[Tuple[int, int]] = None, patch: int = 15) -> float:
    """rows=(위, 아래) 행 범위에서 잰 안개 정도. 0에 가까울수록 맑고, 1에 가까울수록 짙다."""
    dark = dark_channel(bgr, patch)
    if rows is not None:
        dark = dark[rows[0]:rows[1]]
    return float(dark.mean() / 255.0)


def brightness_spread(bgr: np.ndarray, rows: Optional[Tuple[int, int]] = None) -> float:
    """밝은 곳과 어두운 곳의 차이(밝기의 표준편차, 0~1). 안개가 짙을수록 작아진다."""
    gray = cv2.cvtColor(np.ascontiguousarray(bgr[:, :, :3]), cv2.COLOR_BGR2GRAY)
    if rows is not None:
        gray = gray[rows[0]:rows[1]]
    return float(gray.std() / 255.0)


def _parse_region(text: str) -> Tuple[str, Tuple[int, int]]:
    name, top, bottom = text.split(":")
    return name, (int(top), int(bottom))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="사진에서 안개 정도를 잰다")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--region", action="append", type=_parse_region, default=[],
        help="이름:위행:아래행 (여러 번 줄 수 있음). 없으면 사진 전체",
    )
    parser.add_argument("--patch", type=int, default=15, help="가장 어두운 값을 찾는 칸 크기(픽셀)")
    args = parser.parse_args(argv)

    regions = args.region or [("전체", None)]
    header = ["사진"] + [f"안개({name})" for name, _ in regions] + [f"밝기차이({name})" for name, _ in regions]
    print("\t".join(header))
    for path in args.images:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"사진을 읽을 수 없음: {path}")
        fog = [fog_score(bgr, rows, args.patch) for _, rows in regions]
        spread = [brightness_spread(bgr, rows) for _, rows in regions]
        print("\t".join([path.name] + [f"{v:.3f}" for v in fog + spread]))


if __name__ == "__main__":
    main()
