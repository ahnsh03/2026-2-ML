"""Package frozen GP weights and exact training/role provenance for teammates."""

from __future__ import annotations
import argparse, hashlib, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP
from study import read, save, sha, load


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, help="New immutable bundle directory")
    a = ap.parse_args()
    r = a.study
    out = a.output or r / "export_v3_final"
    out.mkdir(exist_ok=False)
    source = r / "export_v3_26r1"
    for name in [
        "quality_spatial_gp.npz",
        "quality_rf.joblib",
        "calibration_legacy.json",
        "calibration_26r1.json",
    ]:
        shutil.copyfile(source / name, out / name)
    recipe = read(source / "feature_recipe.json")
    columns = recipe["tasks"]["quality_spatial_gp"]["ordered_features"]
    protocol = read(r / "coverage_protocol.json")
    keys = (
        protocol["training_keys"]
        + protocol["augmentation_pool"]
        + protocol["motion_training_keys"]
    )
    data, schema = load(r / "packs", keys)
    selected = read(r / "coverage/selection.json")["ids"]
    lookup = {sid: i for i, sid in enumerate(data["ids"])}
    ix = np.array([lookup[sid] for sid in selected])
    ix = ix[np.isfinite(data["Y"][ix]).all(1)]
    x = feature_matrix(data, schema, columns)[ix]
    y = data["Y"][ix]
    model = SpatialQualityGP.load(out / "quality_spatial_gp.npz")
    transformed = (
        (x[:, model.selected_].astype(np.float64) - model.x_mean_)
        / model.x_scale_
        * model.x_weight_
    )
    np.testing.assert_allclose(transformed, model.X_, atol=1e-12, rtol=0)
    z = (y - model.y_mean_) / model.y_scale_
    weights = model.input_eigenvectors_ @ (
        (model.input_eigenvectors_.T @ z @ model.task_eigenvectors_)
        / model.denominator_
    )
    np.testing.assert_allclose(weights, model.weights_, atol=1e-11, rtol=0)
    index = read(r / "extraction_index.json")
    original = {
        row["sample_id"]: (key, row) for key, v in index.items() for row in v["rows"]
    }
    rows = []
    for i in ix:
        sid = str(data["ids"][i])
        key, row = original[sid]
        rows.append(
            dict(
                sample_id=sid,
                source_key=key,
                image=row["image"],
                semantic=row["semantic"],
                rgb_sha256=row["expected_sha256"]["rgb"],
                semantic_sha256=row["expected_sha256"]["semantic"],
                pixel_sha256=str(data["pixel_sha256"][i]),
                targets=data["Y"][i].tolist(),
            )
        )
    save(
        out / "training_membership.json",
        dict(
            rows=rows,
            checkpoint_sha256=recipe["checkpoint_sha256"],
            ordered_features=columns,
            raw_X_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
            raw_Y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
            scope="Private data provenance, not a model input. Formerly diagnosed pool/motion frames are training members here.",
        ),
    )
    roles = dict(training=[row["sample_id"] for row in rows])
    for name, source_keys in [
        ("development", protocol["development_keys"]),
        ("calibration_legacy", protocol["calibration_keys"]),
        ("calibration26", ["release26_calibration"]),
        ("test26_transfer", ["release26_transfer"]),
        ("test26_heldroads", ["release26_heldroads"]),
    ]:
        other, _ = load(r / "packs", source_keys)
        roles[name] = other["ids"].tolist()
    save(out / "split_roles.json", roles)
    paths = [p for p in (a.repo / "src").rglob("*.py")] + [
        p for p in (a.repo / "scripts/experiments/roi9_gp_20261125").glob("*.py")
    ]
    save(
        out / "code_manifest.json",
        {str(path.relative_to(a.repo)): sha(path) for path in sorted(paths)},
    )
    recipe.update(
        created_utc=datetime.now(timezone.utc).isoformat(),
        parent_recipe_sha256=sha(source / "feature_recipe.json"),
        training_membership_sha256=sha(out / "training_membership.json"),
        split_roles_sha256=sha(out / "split_roles.json"),
        status="Frozen9-region/28-input GP integration baseline for November27. Final policy, perception-retraining retention and live ROS adoption are pending.",
    )
    save(out / "feature_recipe.json", recipe)
    manifest = dict(
        version="roi9-gp-bundle-1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        artifact_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()},
        training_rows=len(rows),
        training_reconstruction="Stored standardized X and fitted target weights reconstructed from exact membership and source packs.",
        limitations="Private membership/source data stay outside Git. GP NPZ portable inference verified; RF joblib requires the training scikit-learn version.",
    )
    save(out / "bundle_manifest.json", manifest)
    print(
        "packaged",
        len(rows),
        "training rows; exact GP X/target-weight reconstruction passed",
        flush=True,
    )


if __name__ == "__main__":
    main()
