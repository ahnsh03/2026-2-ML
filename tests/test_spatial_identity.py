import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from common.bridge import DEFAULT_CHECKPOINT_SHA256, sha256_file
from quality_estimation.perception_identity import (
    baseline_identity,
    identity_from_adaptation,
    TARGET_POLICY_SHA256,
)
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.spatial_predictor import FrozenSpatialPredictor
from quality_estimation.spatial_features import SpatialProfile
from quality_estimation.spatial_identity import (
    bind_model_perception,
    recipe_perception,
    resolve_spatial_perception,
    validate_model_perception,
    validate_pack_perception,
    validate_prediction_perception,
)


class SpatialIdentityTests(unittest.TestCase):
    def adaptation(self, root):
        (root / "checkpoint.pt").write_bytes(b"test trained state, no Torch loading")
        (root / "protocol.json").write_text("{}")
        (root / "training_membership.json").write_text("[]")
        run = dict(
            status="complete",
            initial_checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,
            target_policy_sha256=TARGET_POLICY_SHA256,
            artifact_sha256={p.name: sha256_file(p) for p in root.iterdir()},
        )
        (root / "run.json").write_text(json.dumps(run))
        (root / "_SUCCESS").write_text(sha256_file(root / "run.json"))
        identity = identity_from_adaptation(root)
        recipe = dict(
            checkpoint_sha256=identity["checkpoint_sha256"],
            perception_identity=identity,
            perception_weights="model_state_dict",
        )
        return identity, recipe

    def test_existing_v3_recipe_keeps_baseline_ema(self):
        identity, state = recipe_perception(
            dict(checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256)
        )
        self.assertEqual(identity, baseline_identity())
        self.assertEqual(state, "ema_state_dict")

    def test_adapted_weights_require_verified_run_and_explicit_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity, recipe = self.adaptation(root)
            path, actual, _, state = resolve_spatial_perception(
                recipe, adaptation_run=root
            )
            self.assertEqual(
                (path, actual, state),
                (root / "checkpoint.pt", identity, "model_state_dict"),
            )
            with self.assertRaisesRegex(ValueError, "verified adaptation"):
                resolve_spatial_perception(recipe, checkpoint=root / "checkpoint.pt")
            missing = dict(recipe)
            missing.pop("perception_identity")
            with self.assertRaisesRegex(ValueError, "identity/checkpoint"):
                recipe_perception(missing)
            with self.assertRaisesRegex(ValueError, "weight state"):
                recipe_perception(dict(recipe, perception_weights="ema_state_dict"))

    def test_new_gp_roundtrip_binds_run_and_rejects_changed_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "adaptation"
            run.mkdir()
            identity, recipe = self.adaptation(run)
            x = np.arange(24).reshape(8, 3)
            model = SpatialQualityGP().fit(x, np.linspace(0.1, 0.9, 8))
            bind_model_perception(model, identity, "model_state_dict")
            path = root / "gp.npz"
            model.save(path)
            loaded = SpatialQualityGP.load(path)
            validate_model_perception(loaded, recipe)
            changed = dict(
                recipe,
                perception_identity=dict(identity, adaptation_run_sha256="f" * 64),
            )
            with self.assertRaisesRegex(ValueError, "lineage"):
                validate_model_perception(loaded, changed)
            del loaded.perception_identity_json_
            with self.assertRaisesRegex(ValueError, "embedded perception"):
                validate_model_perception(loaded, recipe)

    def test_old_quality_recipe_cannot_silently_accept_new_perception(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.adaptation(root)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                resolve_spatial_perception(
                    dict(checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256),
                    adaptation_run=root,
                )

    def test_features_and_prediction_rows_cannot_mix_model_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            identity, recipe = self.adaptation(Path(directory))
            validate_pack_perception(recipe, identity, "model_state_dict")
            with self.assertRaisesRegex(ValueError, "pack perception"):
                validate_pack_perception(
                    dict(checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256),
                    identity,
                    "model_state_dict",
                )
            run = dict(
                recipe,
                profile="roi9",
                regions_sha256="mask",
                model_sha256="gp1",
                recipe_sha256="recipe1",
            )
            validate_prediction_perception([run], run)
            for key in ["model_sha256", "recipe_sha256", "regions_sha256"]:
                with self.assertRaisesRegex(ValueError, "provenance"):
                    validate_prediction_perception([dict(run, **{key: "other"})], run)

    def test_g1_calibration_requires_matching_perception_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adaptation = root / "adaptation"
            adaptation.mkdir()
            identity, recipe = self.adaptation(adaptation)
            profile = SpatialProfile.load()
            rng = np.random.default_rng(11)
            x = rng.normal(size=(12, 3))
            y = rng.uniform(size=(12, 3))
            model = SpatialQualityGP().fit(x, y)
            bind_model_perception(model, identity, "model_state_dict")
            columns = [
                "hood.intensity",
                "near_left.lane_fraction",
                "global.top_entropy",
            ]
            for key, value in dict(
                feature_names_=columns,
                profile_version_=profile.version,
                regions_sha256_=profile.regions_sha256,
                target_names_=[
                    "whole_frame_lane_iou",
                    "near_lane_recall",
                    "far_lane_recall",
                ],
            ).items():
                setattr(model, key, np.asarray(value))
            path = root / "gp.npz"
            model.save(path)
            calibration = root / "cal.json"
            cal = dict(model_sha256=sha256_file(path), bias=[0, 0, 0], scale=[1, 1, 1])
            calibration.write_text(json.dumps(cal))
            recipe.update(
                profile=profile.version,
                regions_sha256=profile.regions_sha256,
                tasks=dict(
                    quality_spatial_gp=dict(
                        kind="SpatialGP",
                        ordered_features=columns,
                        model_sha256=sha256_file(path),
                        calibration_sha256=sha256_file(calibration),
                    )
                ),
            )
            spec = root / "recipe.json"
            spec.write_text(json.dumps(recipe))
            with self.assertRaisesRegex(ValueError, "calibration perception"):
                FrozenSpatialPredictor(
                    spec, path, profile, calibration_path=calibration
                )
            cal.update(
                perception_identity=identity, perception_weights="model_state_dict"
            )
            calibration.write_text(json.dumps(cal))
            recipe["tasks"]["quality_spatial_gp"]["calibration_sha256"] = sha256_file(
                calibration
            )
            spec.write_text(json.dumps(recipe))
            result = FrozenSpatialPredictor(
                spec, path, profile, calibration_path=calibration
            ).predict_vector(x[0])
            self.assertTrue(result["calibrated"])


if __name__ == "__main__":
    unittest.main()
