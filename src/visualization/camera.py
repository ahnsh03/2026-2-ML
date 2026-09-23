"""카메라 영상 위에 추론·GT 마스크를 그리는 함수 (OpenCV BGR 기준)."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

# 색 범례 — 발표 자료 전체에서 같은 색을 쓴다 (BGR)
DRIVABLE_BGR = (60, 200, 60)  # 주행가능영역: 초록
LANE_BGR = (0, 215, 255)  # 차선: 노랑
TP_BGR = (255, 255, 255)  # 차선 맞춤: 흰색
FN_BGR = (40, 40, 235)  # 차선 놓침(GT에만 있음): 빨강
FP_BGR = (235, 130, 30)  # 차선 오검출(예측에만 있음): 파랑


def probability_to_u8(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(probability, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


def mask_to_u8(mask: np.ndarray) -> np.ndarray:
    """{0,1}/bool 마스크를 일반 뷰어에서 보이는 {0,255}로 바꾼다."""
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def _blend(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> None:
    selected = np.asarray(mask) > 0
    if not np.any(selected):
        return
    color_array = np.asarray(color, dtype=np.float32)
    image[selected] = np.clip(
        image[selected].astype(np.float32) * (1.0 - alpha) + color_array * alpha, 0, 255
    ).astype(np.uint8)


def overlay_masks(
    bgr: np.ndarray,
    drivable: Optional[np.ndarray],
    lane: Optional[np.ndarray],
    drivable_alpha: float = 0.38,
    lane_alpha: float = 0.9,
) -> np.ndarray:
    """주행가능영역(반투명 초록)과 차선(노랑)을 원본 위에 그린다."""
    canvas = np.ascontiguousarray(bgr[:, :, :3]).copy()
    if drivable is not None:
        _blend(canvas, drivable, DRIVABLE_BGR, drivable_alpha)
    if lane is not None:
        _blend(canvas, lane, LANE_BGR, lane_alpha)
    return canvas


def lane_error_image(
    bgr: np.ndarray, predicted_lane: np.ndarray, gt_lane: np.ndarray, dim: float = 0.45
) -> np.ndarray:
    """차선 오류맵: 맞춤=흰색, 놓침(FN)=빨강, 오검출(FP)=파랑. 배경은 어둡게."""
    canvas = np.clip(bgr[:, :, :3].astype(np.float32) * dim, 0, 255).astype(np.uint8)
    predicted = np.asarray(predicted_lane) > 0
    target = np.asarray(gt_lane) > 0
    canvas[predicted & target] = TP_BGR
    canvas[~predicted & target] = FN_BGR
    canvas[predicted & ~target] = FP_BGR
    return canvas


def probability_heatmap(bgr: np.ndarray, probability: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    """확률을 TURBO 색으로 겹친다. 확률이 높을수록 진하게 보인다."""
    value = probability_to_u8(probability)
    heat = cv2.applyColorMap(value, cv2.COLORMAP_TURBO).astype(np.float32)
    weight = (np.asarray(probability, dtype=np.float32) * alpha)[:, :, None]
    base = bgr[:, :, :3].astype(np.float32)
    return np.clip(base * (1.0 - weight) + heat * weight, 0, 255).astype(np.uint8)


def rgba_layer(drivable: np.ndarray, lane: np.ndarray, drivable_alpha: float = 0.45) -> np.ndarray:
    """배경이 투명한 마스크 레이어 (슬라이드에서 원본 위에 직접 겹칠 때 사용)."""
    height, width = np.asarray(drivable).shape[:2]
    layer = np.zeros((height, width, 4), dtype=np.uint8)
    drivable_selected = np.asarray(drivable) > 0
    lane_selected = np.asarray(lane) > 0
    layer[drivable_selected, :3] = DRIVABLE_BGR
    layer[drivable_selected, 3] = int(round(255 * drivable_alpha))
    layer[lane_selected, :3] = LANE_BGR
    layer[lane_selected, 3] = 255
    return layer


def put_label(image: np.ndarray, lines: Iterable[str], origin: Tuple[int, int] = (8, 8), scale: float = 0.6) -> np.ndarray:
    """왼쪽 위에 반투명 배경 상자와 함께 글자를 쓴다 (영문·숫자만)."""
    lines = [str(line) for line in lines if line]
    if not lines:
        return image
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(scale * 2)))
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    line_height = max(size[1] for size in sizes) + int(10 * scale) + 4
    box_width = max(size[0] for size in sizes) + 12
    box_height = line_height * len(lines) + 6
    x0, y0 = origin
    x1 = min(image.shape[1], x0 + box_width)
    y1 = min(image.shape[0], y0 + box_height)
    region = image[y0:y1, x0:x1].astype(np.float32)
    image[y0:y1, x0:x1] = (region * 0.35).astype(np.uint8)
    for index, line in enumerate(lines):
        baseline = y0 + 4 + line_height * (index + 1) - int(6 * scale)
        cv2.putText(image, line, (x0 + 6, baseline), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image


def resize_to_width(image: np.ndarray, width: int, interpolation=cv2.INTER_AREA) -> np.ndarray:
    if image.shape[1] == width:
        return image
    height = int(round(image.shape[0] * width / float(image.shape[1])))
    return cv2.resize(image, (width, height), interpolation=interpolation)


def resize_to_height(image: np.ndarray, height: int, interpolation=cv2.INTER_AREA) -> np.ndarray:
    if image.shape[0] == height:
        return image
    width = int(round(image.shape[1] * height / float(image.shape[0])))
    return cv2.resize(image, (width, height), interpolation=interpolation)


def hstack_same_height(images: Sequence[np.ndarray], height: int, gap: int = 4) -> np.ndarray:
    tiles = [resize_to_height(image, height) for image in images]
    spacer = np.full((height, gap, 3), 255, dtype=np.uint8)
    parts = []
    for index, tile in enumerate(tiles):
        if index:
            parts.append(spacer)
        parts.append(tile)
    return np.concatenate(parts, axis=1)


def vstack_same_width(images: Sequence[np.ndarray], width: int, gap: int = 4) -> np.ndarray:
    tiles = [resize_to_width(image, width) for image in images]
    spacer = np.full((gap, width, 3), 255, dtype=np.uint8)
    parts = []
    for index, tile in enumerate(tiles):
        if index:
            parts.append(spacer)
        parts.append(tile)
    return np.concatenate(parts, axis=0)


def fit_width(row: np.ndarray, width: int) -> np.ndarray:
    """가로 폭을 정확히 맞춘다 (넘치면 축소, 모자라면 흰 여백)."""
    if row.shape[1] > width:
        return resize_to_width(row, width)
    if row.shape[1] < width:
        pad = np.full((row.shape[0], width - row.shape[1], 3), 255, dtype=np.uint8)
        return np.concatenate([row, pad], axis=1)
    return row
