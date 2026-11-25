"""Fit release-specific Gaussian moment calibration without inspecting test scores."""

from __future__ import annotations
import argparse, copy, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.spatial_risk import calibrate_moments
from study import load, read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, required=True)
    a = ap.parse_args()
    r = a.study
    out = r / "calibration26"
    out.mkdir(exist_ok=False)
    data, s = load(r / "packs", ["release26_calibration"])
    selected = read(r / "coverage/frozen.json")
    primary = next(v for v in selected["models"] if v["name"] == selected["selected"])
    old = next(
        v for v in read(r / "finalists/frozen.json")["models"] if v["name"] == "base_GP"
    )
    independent = read(r / "models/coverage_independent.json")
    models = [
        dict(primary, root=str(r / "coverage")),
        dict(old, root=str(r / "finalists")),
        dict(
            independent,
            name="coverage_independent",
            kind="SpatialGP",
            root=str(r),
            model_file="models/coverage_independent.npz",
        ),
    ]
    calibrators = []
    for record in models:
        path = Path(record["root"]) / record["model_file"]
        assert sha(path) == record["model_sha256"]
        model = SpatialQualityGP.load(path)
        x = feature_matrix(data, s, record["ordered_features"])
        mean, cov = model.predict(x)
        cal = calibrate_moments(mean, cov, data["Y"])
        cal.update(
            model_sha256=sha(path),
            simulator_release="26.R1",
            calibration_keys=["release26_calibration"],
            calibration_ids_sha256=__import__("hashlib")
            .sha256("\n".join(data["ids"]).encode())
            .hexdigest(),
            calibration_rows=len(x),
            calibration_sites=len(set(data["site"])),
            role="Four original calibration poses across two original calibration zones, re-captured in26R1; test scores not accessed.",
        )
        target = out / (record["name"] + ".json")
        save(target, cal)
        calibrators.append(
            dict(
                name=record["name"],
                file=target.name,
                sha256=sha(target),
                model_sha256=sha(path),
            )
        )
    export = r / "export_v3_26r1"
    export.mkdir(exist_ok=False)
    for name in [
        "quality_spatial_gp.npz",
        "quality_rf.joblib",
        "calibration_legacy.json",
    ]:
        shutil.copyfile(r / "export_v3" / name, export / name)
    cal = read(out / (primary["name"] + ".json"))
    cal.update(
        parent_model_sha256=cal["model_sha256"],
        model_sha256=sha(export / "quality_spatial_gp.npz"),
    )
    save(export / "calibration_26r1.json", cal)
    recipe = read(r / "export_v3/feature_recipe.json")
    recipe.update(
        created_utc=datetime.now(timezone.utc).isoformat(),
        parent_recipe_sha256=sha(r / "export_v3/feature_recipe.json"),
        active_simulator_release="26.R1",
    )
    recipe["tasks"]["quality_spatial_gp"].update(
        calibration_filename="calibration_26r1.json",
        calibration_sha256=sha(export / "calibration_26r1.json"),
    )
    save(export / "feature_recipe.json", recipe)
    save(
        export / "receipt.json",
        dict(
            files={f.name: sha(f) for f in export.iterdir() if f.is_file()},
            source_sha256=sha(Path(__file__)),
            parent_export_receipt_sha256=sha(r / "export_v3/receipt.json"),
        ),
    )
    freeze = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        calibration_pack_sha256=sha(r / "packs/release26_calibration.npz"),
        models=[{k: v for k, v in m.items() if k != "root"} for m in models],
        calibrators=calibrators,
        selected_primary=primary["name"],
        export_receipt_sha256=sha(export / "receipt.json"),
        source_sha256=sha(Path(__file__)),
        no_test_scores_accessed=True,
        scope="Point-model weights and fixed96-frame calibration procedure frozen before26R1 test scoring.",
    )
    save(out / "frozen.json", freeze)
    print(
        "calibrated",
        len(data["ids"]),
        "frames",
        len(set(data["site"])),
        "sites; no test scores accessed",
        flush=True,
    )


if __name__ == "__main__":
    main()
