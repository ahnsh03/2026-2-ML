import hashlib, json, tempfile, unittest
from pathlib import Path
import numpy as np
from common.bridge import DEFAULT_CHECKPOINT_SHA256
from quality_estimation.spatial_features import SpatialProfile
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.spatial_predictor import FrozenSpatialPredictor


class SpatialPredictorTests(unittest.TestCase):
    def test_embedded_order_and_model_calibration_hashes_are_enforced(self):
        profile = SpatialProfile.load()
        rng = np.random.default_rng(49)
        x = rng.normal(size=(12, 3))
        y = rng.uniform(size=(12, 3))
        model = SpatialQualityGP(task_correlation=0.4).fit(x, y)
        columns = ["hood.intensity", "near_left.lane_fraction", "global.top_entropy"]
        for name, value in dict(
            feature_names_=columns,
            profile_version_=profile.version,
            regions_sha256_=profile.regions_sha256,
            checkpoint_sha256_=DEFAULT_CHECKPOINT_SHA256,
            target_names_=[
                "whole_frame_lane_iou",
                "near_lane_recall",
                "far_lane_recall",
            ],
        ).items():
            setattr(model, name, np.asarray(value))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "gp.npz"
            model.save(path)
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            cal = root / "cal.json"
            cal.write_text(
                json.dumps(
                    dict(
                        model_sha256=sha(path), bias=[0.01, 0, 0], scale=[1.2, 1.1, 1.3]
                    )
                )
            )
            recipe = dict(
                profile=profile.version,
                regions_sha256=profile.regions_sha256,
                checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,
                tasks=dict(
                    quality_spatial_gp=dict(
                        kind="SpatialGP",
                        ordered_features=columns,
                        model_sha256=sha(path),
                        calibration_sha256=sha(cal),
                    )
                ),
            )
            spec = root / "recipe.json"
            spec.write_text(json.dumps(recipe))
            prediction = FrozenSpatialPredictor(
                spec, path, profile, calibration_path=cal
            ).predict_vector(x[0])
            self.assertEqual(prediction["point_mean_source"], "uncalibrated_model")
            self.assertAlmostEqual(
                prediction["calibrated_iou"] - prediction["predicted_iou"], 0.01
            )
            self.assertEqual(len(prediction["posterior_covariance"]), 3)
            calibration = json.loads(cal.read_text())
            calibration["simulator_release"] = "26.R1"
            cal.write_text(json.dumps(calibration))
            recipe["active_simulator_release"] = "26.R1"
            recipe["tasks"]["quality_spatial_gp"]["calibration_sha256"] = sha(cal)
            spec.write_text(json.dumps(recipe))
            with self.assertRaisesRegex(ValueError, "matching simulator release"):
                FrozenSpatialPredictor(spec, path, profile, calibration_path=cal)
            FrozenSpatialPredictor(
                spec, path, profile, calibration_path=cal, simulator_release="26.R1"
            )
            recipe["tasks"]["quality_spatial_gp"]["ordered_features"] = columns[::-1]
            spec.write_text(json.dumps(recipe))
            with self.assertRaisesRegex(ValueError, "input contract"):
                FrozenSpatialPredictor(spec, path, profile)
            recipe["tasks"]["quality_spatial_gp"]["ordered_features"] = columns
            spec.write_text(json.dumps(recipe))
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                FrozenSpatialPredictor(spec, path, profile)


if __name__ == "__main__":
    unittest.main()
