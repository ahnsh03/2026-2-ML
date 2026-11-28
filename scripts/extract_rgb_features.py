#!/usr/bin/env python3
"""Extract 28 RGB features per image; save CSVs and diagnostic plots.

Example (conditions correspond to lexicographically sorted image filenames):
  python3 scripts/extract_rgb_features.py --input ~/rgb_samples/clear_day \
    --output ~/rgb_samples/analysis --conditions clear_day dusk shadow fog objects
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


FEATURES = ("mean_brightness", "dark_ratio", "brightness_std", "edge_density")


def extract(bgr):
    """Use original uint8 BGR pixels; compute edges once before ROI slicing."""
    if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected a decoded uint8 BGR image")
    height, width = bgr.shape[:2]
    if height < 3 or width < 2:
        raise ValueError("Image too small for a 2 x 3 grid")
    rgb = bgr[:, :, ::-1].astype(np.float64) / 255.0
    luminance = rgb @ np.array([0.299, 0.587, 0.114])
    gray = np.rint(luminance * 255).clip(0, 255).astype(np.uint8)
    smooth = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.0, sigmaY=1.0,
                              borderType=cv2.BORDER_REFLECT_101)
    edges = cv2.Canny(smooth, 50, 150, apertureSize=3, L2gradient=True) > 0
    dark = luminance < 0.2
    regions = {"full": (0, 0, width, height)}
    for row in range(3):
        for col in range(2):
            regions["r{}c{}".format(row + 1, col + 1)] = (
                col * width // 2, row * height // 3,
                (col + 1) * width // 2, (row + 1) * height // 3)
    stats = {}
    for name, (x0, y0, x1, y1) in regions.items():
        y = luminance[y0:y1, x0:x1]
        stats[name] = dict(zip(FEATURES, map(float, (
            y.mean(), dark[y0:y1, x0:x1].mean(), y.std(ddof=0),
            edges[y0:y1, x0:x1].mean()))))
    return stats, regions, luminance, dark, edges


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--conditions", nargs="+",
                        help="One condition per sorted input image; overrides sidecars in outputs only")
    args = parser.parse_args()
    paths = sorted(p for p in args.input.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths:
        parser.error("No images found")
    if args.conditions and len(args.conditions) != len(paths):
        parser.error("Number of conditions must match number of images")
    if args.output.resolve() == args.input.resolve():
        parser.error("Use a separate output directory")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    args.output.mkdir(parents=True, exist_ok=True)
    long_rows, wide_rows, full_rows, inputs = [], [], [], []
    fig, axes = plt.subplots(len(paths), 4, figsize=(20, 3.3 * len(paths)), squeeze=False)
    for index, path in enumerate(paths):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        stats, regions, luminance, dark, edges = extract(bgr)
        sidecar_path = path.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
        condition = args.conditions[index] if args.conditions else sidecar.get("condition", "unknown")
        identity = {"image": path.name, "condition": condition}
        wide = dict(identity)
        for region, values in stats.items():
            long_rows.append(dict(identity, region=region, **values))
            wide.update({region + "__" + k: v for k, v in values.items()})
        wide_rows.append(wide)
        full_rows.append(dict(identity, **stats["full"]))
        inputs.append(dict(identity, path=str(path.resolve()),
                           capture_condition=sidecar.get("condition"),
                           width=bgr.shape[1], height=bgr.shape[0], roi_xyxy=regions))
        panels = (bgr[:, :, ::-1], luminance, dark, edges)
        titles = ("RGB + uniform ROI", "Brightness Y (0-1)", "Dark mask (Y < 0.2)", "Canny edges (50, 150)")
        for col, (panel, title) in enumerate(zip(panels, titles)):
            ax = axes[index, col]
            if col == 0:
                ax.imshow(panel)
                for name, (x0, y0, x1, y1) in regions.items():
                    if name == "full":
                        continue
                    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                                           fill=False, edgecolor="cyan", linewidth=0.7))
                    ax.text(x0 + 12, y0 + 32, name, color="cyan", fontsize=8,
                            bbox=dict(facecolor="black", alpha=0.5, pad=1))
            else:
                ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
            ax.set_title("{}: {} | {}".format(index + 1, condition, title), fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(args.output / "diagnostics.png"), dpi=140)
    plt.close(fig)
    write_csv(args.output / "features_28.csv", wide_rows)
    write_csv(args.output / "features_by_region.csv", long_rows)
    write_csv(args.output / "features_full.csv", full_rows)
    settings = {
        "luminance": "(0.299 R + 0.587 G + 0.114 B) / 255; no per-image normalization",
        "dark_threshold": 0.2, "standard_deviation_ddof": 0,
        "roi": "provisional full-image uniform grid: 2 columns x 3 rows; no cropping",
        "canny": {"input": "round(255 * Y), uint8", "gaussian_kernel": [5, 5],
                  "gaussian_sigma": 1.0, "border": "REFLECT_101", "low": 50,
                  "high": 150, "aperture": 3, "L2gradient": True,
                  "scope": "full image first; aggregate per ROI afterwards"},
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "condition_source": "user-supplied ordered labels" if args.conditions else "capture sidecars",
        "inputs": inputs,
    }
    (args.output / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in full_rows:
        print("{condition:12} mean={mean_brightness:.6f} dark={dark_ratio:.6%} "
              "std={brightness_std:.6f} edges={edge_density:.6%}".format(**row))
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()
