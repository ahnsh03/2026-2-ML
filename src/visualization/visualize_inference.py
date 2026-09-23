"""대상 인지 모델의 추론 결과를 카메라별·BEV로 시각화하고 원본·마스크를 따로 저장한다.

컨테이너에서 실행한다 (docs/05_구현계획/개발환경.md):

  scripts/run_container.sh python3 -m visualization.visualize_inference \\
      --split val --name v5b_val

결과 구조와 색 범례는 출력 폴더의 README.md에 함께 기록된다.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import time
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from common.bridge import (
    DEFAULT_DATASET_VERSION,
    VIEWS,
    add_perception_import_paths,
    git_head,
    load_target_model,
    parse_condition,
    resolve_perception_repo,
    resolve_checkpoint,
    sha256_file,
)
from visualization import camera as cam
from visualization.bev import BevScene

NATIVE_TARGET_DIR = "perception_targets_native_v3_actor_negative"
LETTERBOX_KEYS = ("original_hw", "target_hw", "resized_hw", "pad_ltrb", "scale_xy")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=os.environ.get("PERCEPTION_DATA", "/data"))
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--run-id", action="append", default=[], help="이 run만 (반복 가능)")
    parser.add_argument("--every-n", type=int, default=1, help="run마다 N frame 간격으로 선택")
    parser.add_argument("--max-frames-per-run", type=int, default=0, help="run마다 고르게 최대 N frame (0 = 전부)")
    parser.add_argument("--checkpoint", help="기본: $PERCEPTION_CKPT_ROOT/<고정 대상 모델>")
    parser.add_argument("--twinlite-root", default=os.environ.get("TWINLITE_ROOT"))
    parser.add_argument("--perception-repo", default=os.environ.get("PERCEPTION_REPO"))
    parser.add_argument("--output-root", default=os.environ.get("ML_OUTPUT_ROOT", "outputs"))
    parser.add_argument("--name", help="출력 폴더 이름 (기본: <split>)")
    parser.add_argument("--grid", default="dev", choices=("dev", "pilot"), help="BEV 격자 (drivable_bev 설정)")
    parser.add_argument("--bev-scale", type=int, default=2, help="BEV 이미지 확대 배율 (0.1 m/cell 기준)")
    parser.add_argument("--lane-threshold", type=float, default=0.5)
    parser.add_argument("--drivable-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--writer-threads", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--skip-existing", action="store_true", help="mosaic이 있는 frame은 건너뜀")
    return parser.parse_args()


# ---------------------------------------------------------------------- 선택
def select_frames(rows, run_filter, every_n, max_per_run) -> List[Tuple[str, int, Dict[str, int]]]:
    groups: Dict[Tuple[str, int], Dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        if run_filter and row["run_id"] not in run_filter:
            continue
        groups[(row["run_id"], int(row["frame_id"]))][row["view"]] = index
    by_run: Dict[str, List[int]] = defaultdict(list)
    for (run_id, frame_id), views in groups.items():
        if all(view in views for view in VIEWS):
            by_run[run_id].append(frame_id)
    selected = []
    for run_id in sorted(by_run):
        frames = sorted(by_run[run_id])[:: max(1, every_n)]
        if max_per_run and len(frames) > max_per_run:
            picks = np.unique(np.linspace(0, len(frames) - 1, max_per_run).round().astype(int))
            frames = [frames[i] for i in picks]
        for frame_id in frames:
            selected.append((run_id, frame_id, groups[(run_id, frame_id)]))
    return selected


# ---------------------------------------------------------------------- 저장
class Writer:
    """이미지 저장을 스레드로 병렬화한다 (cv2.imwrite는 GIL을 놓는다)."""

    def __init__(self, threads: int, jpeg_quality: int) -> None:
        self.pool = ThreadPoolExecutor(max_workers=max(1, threads))
        self.pending = deque()
        self.jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        self.png_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        self.count = 0

    def _write(self, path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        params = self.jpeg_params if path.suffix.lower() in (".jpg", ".jpeg") else self.png_params
        if not cv2.imwrite(str(path), image, params):
            raise RuntimeError("이미지 저장 실패: {}".format(path))

    def submit(self, path: Path, image: np.ndarray) -> None:
        self.pending.append(self.pool.submit(self._write, path, np.ascontiguousarray(image)))
        self.count += 1
        while len(self.pending) > 512:
            self.pending.popleft().result()

    def close(self) -> None:
        while self.pending:
            self.pending.popleft().result()
        self.pool.shutdown()


def native_target_dir(data_root: Path, row) -> Optional[Path]:
    candidates = []
    actor = (row.get("auxiliary_targets") or {}).get("actor_occupancy_gt")
    if actor:
        candidates.append((data_root / actor).parents[2])
    candidates.append(data_root / "datasets" / row["run_id"] / "derived" / NATIVE_TARGET_DIR)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def read_native_gt(data_root: Path, row) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    base = native_target_dir(data_root, row)
    if base is None:
        return None, None
    name = "{:06d}.png".format(int(row["frame_id"]))
    masks = []
    for head in ("lane_masks", "drivable_masks"):
        path = base / head / row["view"] / name
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
        masks.append(None if mask is None else (mask > 0))
    return masks[0], masks[1]


def iou(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> Tuple[float, int, int]:
    pred = pred & valid
    target = target & valid
    union = int((pred | target).sum())
    intersection = int((pred & target).sum())
    value = float("nan") if union == 0 else intersection / union
    return value, int(target.sum()), int(pred.sum())


def fmt(value: float) -> str:
    return "nan" if value != value else "{:.3f}".format(value)


# ---------------------------------------------------------------------- frame 처리
def process_frame(ctx, run_id: str, frame_id: int, view_indices: Dict[str, int], results) -> Tuple[List[dict], dict]:
    from camera_semantic_perception.geometry import LetterboxGeometry, restore_probability

    data_root, out, writer, bev, args, split = ctx["data_root"], ctx["out"], ctx["writer"], ctx["bev"], ctx["args"], ctx["split"]
    condition = parse_condition(run_id)
    stem = "{:06d}".format(frame_id)
    run_dir = out / run_id
    sample_rows = []
    per_view = {}

    for view in VIEWS:
        index = view_indices[view]
        row = ctx["rows"][index]
        result = results[index]
        geometry = LetterboxGeometry(**{key: tuple(row["geometry"][key]) for key in LETTERBOX_KEYS})
        lane_prob = restore_probability(result["lane_prob"], geometry)
        drivable_prob = restore_probability(result["drivable_prob"], geometry)
        lane_mask = lane_prob >= args.lane_threshold
        drivable_mask = drivable_prob >= args.drivable_threshold
        image = cv2.imread(str(data_root / row["image"]), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError("원본 영상을 읽을 수 없습니다: {}".format(row["image"]))
        bgr = np.ascontiguousarray(image[:, :, :3])
        gt_lane, gt_drivable = read_native_gt(data_root, row)
        view_dir = run_dir / view

        overlay_pred = cam.overlay_masks(bgr, drivable_mask, lane_mask)
        paths = {
            "rgb": view_dir / "rgb" / (stem + ".png"),
            "pred_lane_mask": view_dir / "pred_lane_mask" / (stem + ".png"),
            "pred_drivable_mask": view_dir / "pred_drivable_mask" / (stem + ".png"),
            "pred_lane_prob": view_dir / "pred_lane_prob" / (stem + ".png"),
            "pred_drivable_prob": view_dir / "pred_drivable_prob" / (stem + ".png"),
            "pred_layer_rgba": view_dir / "pred_layer_rgba" / (stem + ".png"),
            "overlay_pred": view_dir / "overlay_pred" / (stem + ".jpg"),
            "lane_prob_heat": view_dir / "lane_prob_heat" / (stem + ".jpg"),
        }
        writer.submit(paths["rgb"], bgr)
        writer.submit(paths["pred_lane_mask"], cam.mask_to_u8(lane_mask))
        writer.submit(paths["pred_drivable_mask"], cam.mask_to_u8(drivable_mask))
        writer.submit(paths["pred_lane_prob"], cam.probability_to_u8(lane_prob))
        writer.submit(paths["pred_drivable_prob"], cam.probability_to_u8(drivable_prob))
        writer.submit(paths["pred_layer_rgba"], cam.rgba_layer(drivable_mask, lane_mask))
        writer.submit(paths["overlay_pred"], overlay_pred)
        writer.submit(paths["lane_prob_heat"], cam.probability_heatmap(bgr, lane_prob))

        label = "{} | lane IoU {} | drivable IoU {}".format(view, fmt(result["lane_iou"]), fmt(result["drivable_iou"]))
        if gt_lane is not None and gt_drivable is not None:
            overlay_gt = cam.overlay_masks(bgr, gt_drivable, gt_lane)
            error = cam.lane_error_image(bgr, lane_mask, gt_lane)
            paths.update({
                "gt_lane_mask": view_dir / "gt_lane_mask" / (stem + ".png"),
                "gt_drivable_mask": view_dir / "gt_drivable_mask" / (stem + ".png"),
                "overlay_gt": view_dir / "overlay_gt" / (stem + ".jpg"),
                "lane_error": view_dir / "lane_error" / (stem + ".jpg"),
            })
            writer.submit(paths["gt_lane_mask"], cam.mask_to_u8(gt_lane))
            writer.submit(paths["gt_drivable_mask"], cam.mask_to_u8(gt_drivable))
            writer.submit(paths["overlay_gt"], overlay_gt)
            writer.submit(paths["lane_error"], error)
            top = cam.hstack_same_height(
                [cam.put_label(bgr.copy(), ["RGB " + view]), cam.put_label(overlay_gt.copy(), ["GT"])], bgr.shape[0])
            bottom = cam.hstack_same_height(
                [cam.put_label(overlay_pred.copy(), ["Prediction", label]),
                 cam.put_label(error.copy(), ["Lane error: white=hit red=miss blue=false"])], bgr.shape[0])
            compare = cam.vstack_same_width([top, cam.fit_width(bottom, top.shape[1])], top.shape[1])
        else:
            gt_drivable = None
            compare = cam.hstack_same_height(
                [cam.put_label(bgr.copy(), ["RGB " + view]), cam.put_label(overlay_pred.copy(), ["Prediction", label])],
                bgr.shape[0])
        paths["compare"] = view_dir / "compare" / (stem + ".jpg")
        writer.submit(paths["compare"], compare)

        per_view[view] = {
            "bgr": bgr, "overlay_pred": overlay_pred, "label": label,
            "lane_mask": lane_mask, "drivable_u8": cam.probability_to_u8(drivable_prob),
            "lane_conf": cam.probability_to_u8(lane_prob),
            "gt_lane": gt_lane, "gt_drivable": gt_drivable,
        }
        sample_rows.append(OrderedDict([
            ("run_id", run_id), ("frame_id", frame_id), ("view", view), ("split", split),
            ("weather", condition["weather"]), ("sim_time", condition["sim_time"]),
            ("lane_iou", fmt(result["lane_iou"])), ("drivable_iou", fmt(result["drivable_iou"])),
            ("lane_target_px", result["lane_target_px"]), ("lane_pred_px", result["lane_pred_px"]),
            *[(key, str(path.relative_to(out))) for key, path in paths.items()],
        ]))

    # ---------------------------------------------------------------- BEV
    scale = args.bev_scale
    rgb_bev = bev.fuse_rgb({view: per_view[view]["bgr"] for view in VIEWS})
    pred_layers = bev.fuse_semantics(
        {view: per_view[view]["drivable_u8"] for view in VIEWS},
        {view: per_view[view]["lane_mask"] for view in VIEWS},
        {view: per_view[view]["lane_conf"] for view in VIEWS},
    )
    bev_dir = run_dir / "bev"
    bev_rgb_view = bev.render_rgb(rgb_bev, scale)
    bev_pred_view = bev.render_semantic(pred_layers, scale)
    bev_overlay = bev.render_overlay(rgb_bev, pred_layers, scale)
    bev_paths = {
        "rgb": bev_dir / "rgb" / (stem + ".png"),
        "pred": bev_dir / "pred" / (stem + ".png"),
        "overlay_pred": bev_dir / "overlay_pred" / (stem + ".jpg"),
        "layer_rgb": bev_dir / "layers" / "rgb" / (stem + ".png"),
        "layer_pred_drivable_prob": bev_dir / "layers" / "pred_drivable_prob" / (stem + ".png"),
        "layer_pred_lane_mask": bev_dir / "layers" / "pred_lane_mask" / (stem + ".png"),
    }
    writer.submit(bev_paths["rgb"], bev_rgb_view)
    writer.submit(bev_paths["pred"], bev_pred_view)
    writer.submit(bev_paths["overlay_pred"], bev_overlay)
    writer.submit(bev_paths["layer_rgb"], rgb_bev)
    writer.submit(bev_paths["layer_pred_drivable_prob"], pred_layers.drivable_probability)
    writer.submit(bev_paths["layer_pred_lane_mask"], pred_layers.lane_mask)
    panel_tiles = [cam.put_label(bev_rgb_view.copy(), ["BEV RGB"]), cam.put_label(bev_overlay.copy(), ["BEV RGB + prediction"])]
    has_gt = all(per_view[view]["gt_lane"] is not None and per_view[view]["gt_drivable"] is not None for view in VIEWS)
    if has_gt:
        gt_layers = bev.fuse_semantics(
            {view: cam.mask_to_u8(per_view[view]["gt_drivable"]) for view in VIEWS},
            {view: per_view[view]["gt_lane"] for view in VIEWS},
        )
        bev_gt_view = bev.render_semantic(gt_layers, scale)
        bev_paths.update({
            "gt": bev_dir / "gt" / (stem + ".png"),
            "layer_gt_drivable_mask": bev_dir / "layers" / "gt_drivable_mask" / (stem + ".png"),
            "layer_gt_lane_mask": bev_dir / "layers" / "gt_lane_mask" / (stem + ".png"),
        })
        writer.submit(bev_paths["gt"], bev_gt_view)
        writer.submit(bev_paths["layer_gt_drivable_mask"], gt_layers.drivable_probability)
        writer.submit(bev_paths["layer_gt_lane_mask"], gt_layers.lane_mask)
        panel_tiles += [cam.put_label(bev_gt_view.copy(), ["BEV GT"]), cam.put_label(bev_pred_view.copy(), ["BEV prediction"])]
    else:
        panel_tiles.append(cam.put_label(bev_pred_view.copy(), ["BEV prediction"]))
    bev_paths["panel"] = bev_dir / "panel" / (stem + ".jpg")
    writer.submit(bev_paths["panel"], cam.hstack_same_height(panel_tiles, bev_rgb_view.shape[0]))

    # ---------------------------------------------------------------- mosaic (슬라이드용 한 장)
    front = cam.put_label(per_view["front"]["overlay_pred"].copy(), [per_view["front"]["label"]])
    bev_tile = cam.put_label(bev_overlay.copy(), ["BEV (3-cam fused)"])
    row1 = cam.hstack_same_height([front, bev_tile], front.shape[0])
    side_width = (row1.shape[1] - 4) // 2
    left = cam.resize_to_width(cam.put_label(per_view["left"]["overlay_pred"].copy(), [per_view["left"]["label"]], scale=0.5), side_width)
    right = cam.resize_to_width(cam.put_label(per_view["right"]["overlay_pred"].copy(), [per_view["right"]["label"]], scale=0.5), side_width)
    row2 = cam.fit_width(cam.hstack_same_height([left, right], left.shape[0]), row1.shape[1])
    mosaic = cam.vstack_same_width([row1, row2], row1.shape[1])
    cam.put_label(mosaic, ["{}  frame {}  [{} / {}]  split={}".format(run_id, frame_id, condition["weather"], condition["sim_time"], split)],
                  origin=(8, mosaic.shape[0] - 40), scale=0.6)
    mosaic_path = run_dir / "mosaic" / (stem + ".jpg")
    writer.submit(mosaic_path, mosaic)

    lane_values = {row["view"]: float(row["lane_iou"]) for row in sample_rows}
    drivable_values = [float(row["drivable_iou"]) for row in sample_rows]
    frame_row = OrderedDict([
        ("run_id", run_id), ("frame_id", frame_id), ("split", split),
        ("weather", condition["weather"]), ("sim_time", condition["sim_time"]),
        ("lane_iou_front", fmt(lane_values["front"])), ("lane_iou_left", fmt(lane_values["left"])),
        ("lane_iou_right", fmt(lane_values["right"])),
        ("lane_iou_mean", fmt(float(np.nanmean(list(lane_values.values()))))),
        ("drivable_iou_mean", fmt(float(np.nanmean(drivable_values)))),
        ("mosaic", str(mosaic_path.relative_to(out))),
        *[("bev_" + key, str(path.relative_to(out))) for key, path in bev_paths.items()],
    ])
    return sample_rows, frame_row


# ---------------------------------------------------------------------- 문서화
README_TEMPLATE = """# 추론 시각화 — {name}

생성: {created} · 대상 모델: `{checkpoint}` (SHA-256 `{sha}`, {weights}, epoch {epoch})
데이터: `{dataset}` / split `{split}` · frame {frames}개 × 3뷰 · BEV 격자 `{grid}` ({x0}~{x1} m 전방, {y0}~{y1} m 좌우, {res} m/cell)

> 원본 데이터에서 만든 파생 이미지다. **공개 저장소에 올리지 않는다.**
> split이 `train`이면 대상 모델이 학습에 쓴 frame이라 성능이 부풀려져 보인다. 성능 주장에는 `val`/`test`만 쓴다.

## 찾아보기

- `index.html` — 모든 frame의 mosaic과 front lane IoU 하위 목록
- `frames.csv` — frame별 IoU와 mosaic·BEV 경로 · `samples.csv` — 카메라(view)별 IoU와 파일 경로
- `export_meta.json` — 체크포인트·데이터·코드 커밋·설정

## 폴더 구조

```text
<run_id>/
├── front|left|right/
│   ├── rgb/                 원본 영상 (PNG, 원본 해상도)
│   ├── pred_lane_mask/      추론 차선 마스크 {{0,255}} (원본 해상도)
│   ├── pred_drivable_mask/  추론 주행가능영역 마스크 {{0,255}}
│   ├── pred_lane_prob/      차선 확률 0~255 (회색조)
│   ├── pred_drivable_prob/  주행가능영역 확률 0~255
│   ├── pred_layer_rgba/     투명 배경 마스크 레이어 — 슬라이드에서 원본 위에 겹치기용
│   ├── overlay_pred/        원본 + 추론
│   ├── lane_prob_heat/      차선 확률 히트맵 (TURBO)
│   ├── gt_lane_mask/, gt_drivable_mask/, overlay_gt/   정답(GT, 원본 해상도)
│   ├── lane_error/          차선 오류맵
│   └── compare/             [원본 | GT] / [추론 | 오류맵] 2×2 패널
├── bev/
│   ├── rgb/                 3카메라 원본 영상을 지면에 투영·융합한 BEV
│   ├── pred/, gt/           BEV 추론 / GT (검은 배경)
│   ├── overlay_pred/        BEV 원본 + 추론
│   ├── panel/               [BEV 원본 | BEV 추론 겹침 | BEV GT | BEV 추론]
│   └── layers/              격자 해상도 원시 레이어 (분석용, 확대·장식 없음)
└── mosaic/                  front + BEV + left/right 한 장 (슬라이드용)
```

## 색 범례

| 색 | 의미 |
|---|---|
| 초록 (반투명) | 주행가능영역 |
| 노랑 | 차선 (white·yellow 통합 binary lane) |
| 흰색 / 빨강 / 파랑 (오류맵) | 차선 맞춤 / 놓침(GT에만 있음) / 오검출(추론에만 있음) |
| BEV 흰 사각형 | 차량(Ioniq 5) 외곽, 삼각형이 전방. 격자 5 m |

IoU는 팀 평가(`evaluate_twinlite.py`)와 같이 384×640 letterbox 공간·valid mask 기준이며,
union이 0이면 `nan`이다. 마스크 이미지는 확률을 원본 해상도로 복원한 뒤 임계값 {lane_th}/{drv_th}로 이진화했다.
BEV는 인지 코드 레포의 `drivable_bev` 설정 `cameras_v2.yaml`(노면 z = {ground} m)과 융합 규칙을 그대로 사용했다.
"""


def write_index_html(out: Path, frame_rows: List[dict], name: str) -> None:
    def figure(row):
        caption = "{} #{} · {}/{} · front lane IoU {}".format(
            html.escape(row["run_id"]), row["frame_id"], row["weather"], row["sim_time"], row["lane_iou_front"])
        return ('<figure><a href="{0}"><img loading="lazy" src="{0}"></a><figcaption>{1}</figcaption></figure>'
                .format(html.escape(row["mosaic"]), caption))

    def key(row):
        value = float(row["lane_iou_front"])
        return 9.0 if value != value else value

    worst = sorted(frame_rows, key=key)[:30]
    sections = ['<h2>front lane IoU 하위 30</h2><div class="grid">{}</div>'.format("".join(figure(r) for r in worst))]
    by_run = defaultdict(list)
    for row in frame_rows:
        by_run[row["run_id"]].append(row)
    for run_id in sorted(by_run):
        rows = sorted(by_run[run_id], key=lambda r: int(r["frame_id"]))
        sections.append('<h2>{} <small>({} frames)</small></h2><div class="grid">{}</div>'.format(
            html.escape(run_id), len(rows), "".join(figure(r) for r in rows)))
    page = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:sans-serif;margin:16px;background:#fafafa}}h2{{margin-top:28px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:10px}}
figure{{margin:0;background:#fff;border:1px solid #ddd;padding:4px}}img{{width:100%;display:block}}
figcaption{{font-size:12px;color:#333;padding:4px 2px}}</style></head><body>
<h1>{title}</h1><p>초록 = 주행가능영역, 노랑 = 차선. 이미지를 누르면 원본 크기로 열린다. 구조는 README.md 참고.</p>
{sections}</body></html>""".format(title=html.escape("추론 시각화 — " + name), sections="\n".join(sections))
    (out / "index.html").write_text(page, encoding="utf-8")


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------- main
def main():
    args = parse_args()
    perception_repo = add_perception_import_paths(resolve_perception_repo(args.perception_repo))

    import torch
    from torch.utils.data import DataLoader, Dataset
    from twinlite_morai import MoraiTwinLiteDataset, adapt_twinlite_outputs

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_root = Path(args.data_root).expanduser().resolve()
    name = args.name or args.split
    out = Path(args.output_root).expanduser().resolve() / "inference_vis" / name
    out.mkdir(parents=True, exist_ok=True)

    dataset = MoraiTwinLiteDataset(data_root=str(data_root), dataset_version=args.dataset_version, split=args.split)
    if "lane" not in dataset.heads:
        raise RuntimeError("binary lane head dataset이 필요합니다 (현재 heads={})".format(dataset.heads))
    selected = select_frames(dataset.rows, set(args.run_id), args.every_n, args.max_frames_per_run)
    if args.skip_existing:
        selected = [item for item in selected if not (out / item[0] / "mosaic" / "{:06d}.jpg".format(item[1])).is_file()]
    if not selected:
        print("선택된 frame이 없습니다", flush=True)
        return
    order = [index for _, _, views in selected for index in (views[v] for v in VIEWS)]
    frame_of_index = {}
    for run_id, frame_id, views in selected:
        for view in VIEWS:
            frame_of_index[views[view]] = (run_id, frame_id)
    print("frames={} samples={} -> {}".format(len(selected), len(order), out), flush=True)

    checkpoint = resolve_checkpoint(args.checkpoint)
    loaded = load_target_model(checkpoint, device, twinlite_root=args.twinlite_root)
    bev = BevScene(perception_repo, grid_name=args.grid)
    if bev.calibration_warning:
        print("[WARN] calibration: " + bev.calibration_warning, flush=True)

    class Indexed(Dataset):
        def __len__(self):
            return len(order)

        def __getitem__(self, position):
            index = order[position]
            return index, dataset[index]

    def collate(batch):
        return (
            [item[0] for item in batch],
            torch.stack([item[1]["image"] for item in batch]),
            {head: torch.stack([item[1]["targets"][head] for item in batch]) for head in ("lane", "drivable")},
            {head: torch.stack([item[1]["valid_masks"][head] for item in batch]) for head in ("lane", "drivable")},
        )

    loader = DataLoader(Indexed(), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate, pin_memory=device.type == "cuda")
    writer = Writer(args.writer_threads, args.jpeg_quality)
    ctx = {"data_root": data_root, "out": out, "writer": writer, "bev": bev, "args": args,
           "rows": dataset.rows, "split": args.split}
    results: Dict[int, dict] = {}
    remaining = {(run_id, frame_id): set(views.values()) for run_id, frame_id, views in selected}
    views_of = {(run_id, frame_id): views for run_id, frame_id, views in selected}
    sample_rows, frame_rows = [], []
    started = time.perf_counter()

    with torch.no_grad():
        for batch_index, (indices, images, targets, valid) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = adapt_twinlite_outputs(loaded.model(images))
            probabilities = {head: torch.softmax(outputs[head].float(), dim=1)[:, 1] for head in ("lane", "drivable")}
            for position, index in enumerate(indices):
                entry = {}
                for head in ("lane", "drivable"):
                    probability = probabilities[head][position]
                    prediction = (probability >= 0.5).cpu().numpy()
                    target = targets[head][position].numpy() > 0
                    mask = valid[head][position].numpy()
                    value, target_px, pred_px = iou(prediction, target, mask)
                    entry[head + "_iou"] = value
                    entry[head + "_prob"] = probability.cpu().numpy().astype(np.float32)
                    if head == "lane":
                        entry["lane_target_px"], entry["lane_pred_px"] = target_px, pred_px
                results[index] = entry
                key = frame_of_index[index]
                remaining[key].discard(index)
                if not remaining[key]:
                    rows, frame_row = process_frame(ctx, key[0], key[1], views_of[key], results)
                    sample_rows.extend(rows)
                    frame_rows.append(frame_row)
                    for view_index in views_of[key].values():
                        results.pop(view_index, None)
                    if len(frame_rows) % 25 == 0 or len(frame_rows) == len(selected):
                        print("frames {}/{}  ({:.1f}s)".format(len(frame_rows), len(selected), time.perf_counter() - started), flush=True)
    writer.close()

    # 기존 결과(--skip-existing)와 합쳐 목록을 다시 쓴다
    def merge(path: Path, new_rows: List[dict], key_fields) -> List[dict]:
        merged = OrderedDict()
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    merged[tuple(str(row[k]) for k in key_fields)] = row
        for row in new_rows:
            merged[tuple(str(row[k]) for k in key_fields)] = row
        return sorted(merged.values(), key=lambda r: tuple(str(r[k]) if k != "frame_id" else "{:08d}".format(int(r[k])) for k in key_fields))

    sample_rows = merge(out / "samples.csv", sample_rows, ("run_id", "frame_id", "view"))
    frame_rows = merge(out / "frames.csv", frame_rows, ("run_id", "frame_id"))
    write_csv(out / "samples.csv", sample_rows)
    write_csv(out / "frames.csv", frame_rows)
    write_index_html(out, frame_rows, name)

    split_manifest = data_root / "dataset_versions" / args.dataset_version / (args.split + ".jsonl")
    meta = {
        "schema": "ml2026-inference-vis-1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": vars(args),
        "frames_total": len(frame_rows),
        "images_written_this_run": writer.count,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "model": {
            "checkpoint": str(loaded.checkpoint),
            "checkpoint_sha256": loaded.checkpoint_sha256,
            "epoch": loaded.epoch,
            "weights": loaded.weights,
            "upstream_commit": loaded.upstream_commit,
        },
        "dataset": {
            "version": args.dataset_version,
            "split": args.split,
            "split_manifest_sha256": sha256_file(split_manifest),
        },
        "code": {
            "perception_repo_commit": git_head(perception_repo),
            "ml_repo_commit": git_head(Path(__file__).resolve().parents[2]),
        },
        "bev": bev.describe(),
        "device": str(device),
    }
    (out / "export_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(README_TEMPLATE.format(
        name=name, created=meta["created_at_utc"], checkpoint=Path(loaded.checkpoint).parent.name + "/" + Path(loaded.checkpoint).name,
        sha=loaded.checkpoint_sha256[:12], weights=loaded.weights, epoch=loaded.epoch,
        dataset=args.dataset_version, split=args.split, frames=len(frame_rows), grid=args.grid,
        x0=bev.grid.x_min_m, x1=bev.grid.x_max_m, y0=bev.grid.y_min_m, y1=bev.grid.y_max_m,
        res=bev.grid.resolution_m, lane_th=args.lane_threshold, drv_th=args.drivable_threshold,
        ground=bev.ground_z_m), encoding="utf-8")
    print("done: {} frames, {} images, {:.1f}s -> {}".format(len(frame_rows), writer.count, time.perf_counter() - started, out), flush=True)


if __name__ == "__main__":
    main()
