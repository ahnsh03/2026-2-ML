import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from quality_estimation.portable_forest import PortableForest
from quality_estimation.region_inference import main
from quality_estimation.spatial_features import SpatialProfile


class PortableForestCLITests(unittest.TestCase):
    def fixture(self, root, quality=False):
        profile = SpatialProfile.load()
        image = root / "rgb.png"
        cv2.imwrite(str(image), np.full((720, 1280, 3), 120, np.uint8))
        x = np.asarray([[0], [0.5], [1.0]])
        cls = RandomForestRegressor if quality else RandomForestClassifier
        fitted = cls(n_estimators=3, random_state=2, bootstrap=False).fit(x, [0, 1, 1])
        columns = ["hood.intensity"]
        context = dict(
            profile=profile.version,
            regions_sha256=profile.regions_sha256,
            checkpoint_sha256="frozen-cp" if quality else None,
        )
        model = root / "forest.npz"
        PortableForest.from_sklearn(fitted, columns, context).save(model)
        recipe = root / "recipe.json"
        recipe.write_text(
            json.dumps(
                dict(
                    profile=profile.version,
                    regions_sha256=profile.regions_sha256,
                    checkpoint_sha256="frozen-cp",
                    tasks=dict(
                        task=dict(
                            kind="RF" if quality else "SceneRF",
                            format="portable-random-forest-1",
                            ordered_features=columns,
                            model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                        )
                    ),
                )
            )
        )
        return [
            "region_inference",
            "--image",
            str(image),
            "--model",
            str(model),
            "--recipe",
            str(recipe),
            "--task",
            "task",
            "--output",
            str(root / "result.json"),
        ]

    def test_rgb_classifier_runs_without_joblib_runtime_or_semantic_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = self.fixture(root)
            with patch("sys.argv", argv), patch.dict("sys.modules", {"joblib": None}):
                main()
            result = json.loads((root / "result.json").read_text())
            self.assertEqual(result["class_id"], 1)
            self.assertAlmostEqual(sum(result["class_probabilities"].values()), 1.0)
            self.assertNotIn("head_maps_sha256", result)

    def test_quality_rf_requires_matching_perception_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = self.fixture(root, quality=True)
            with patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()
            self.assertFalse((root / "result.json").exists())
            with patch(
                "sys.argv", argv + ["--perception-checkpoint-sha256", "frozen-cp"]
            ):
                main()
            self.assertIn(
                "predicted_iou", json.loads((root / "result.json").read_text())
            )


if __name__ == "__main__":
    unittest.main()
