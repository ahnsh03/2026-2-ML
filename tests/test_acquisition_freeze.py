import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from common.bridge import sha256_file
from quality_estimation.spatial_gp import SpatialQualityGP


def entrypoint():
    directory = (
        Path(__file__).resolve().parents[1] / "scripts/experiments/roi9_gp_20261125"
    )
    spec = importlib.util.spec_from_file_location(
        "test_freeze_acquisition_entrypoint", directory / "freeze_acquisition.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(directory), *sys.path]):
        spec.loader.exec_module(module)
    return module.main


class AcquisitionFreezeTests(unittest.TestCase):
    def fixture(self, root, extra_truth=False):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(12, 3))
        y = rng.uniform(size=(12, 3))
        bundle = root / "bundle"
        bundle.mkdir()
        model = bundle / "gp.npz"
        SpatialQualityGP().fit(x, y).save(model)
        cal = bundle / "cal.json"
        cal.write_text(json.dumps(dict(bias=[0, 0, 0], scale=[1, 1, 1])))
        columns = ["a", "b", "c"]
        (bundle / "feature_recipe.json").write_text(
            json.dumps(
                dict(
                    tasks=dict(
                        quality_spatial_gp=dict(
                            artifact_filename=model.name,
                            model_sha256=sha256_file(model),
                            calibration_filename=cal.name,
                            calibration_sha256=sha256_file(cal),
                            ordered_features=columns,
                        )
                    )
                )
            )
        )
        protocol = root / "protocol.json"
        protocol.write_text(
            json.dumps(
                dict(
                    budgets=[2, 4],
                    methods=[
                        "rf_mean",
                        "global_mean",
                        "global_probability",
                        "dual_probability",
                        "dual_conditional",
                        "gp_information_only",
                        "random_23411",
                    ],
                )
            )
        )
        inputs = root / "inputs.npz"
        arrays = dict(
            X=x[:8],
            ids=np.asarray([f"sample{i}" for i in range(8)]),
            rf_point_mean=np.linspace(0, 1, 8),
            ordered_features=np.asarray(columns),
        )
        if extra_truth:
            arrays["Y"] = y[:8]
        np.savez_compressed(inputs, **arrays)
        return [
            "freeze_acquisition",
            "--inputs",
            str(inputs),
            "--bundle",
            str(bundle),
            "--protocol",
            str(protocol),
            "--output",
            str(root / "selection"),
        ]

    def test_all_rank_and_covariance_policies_freeze_exact_budgets_as_json(self):
        main = entrypoint()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = self.fixture(root)
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            selection = root / "selection"
            result = json.loads((selection / "selection.json").read_text())
            self.assertEqual(len(result["selections"]), 14)
            for entry in result["selections"]:
                self.assertEqual(len(set(entry["indices"])), entry["budget"])
                self.assertEqual(entry["ids"], [f"sample{i}" for i in entry["indices"]])
            self.assertFalse(result["uses_ground_truth"])
            self.assertEqual(
                (selection / "_SUCCESS").read_text().strip(),
                sha256_file(selection / "selection.json"),
            )

    def test_pack_with_hidden_ground_truth_is_rejected_before_selection(self):
        main = entrypoint()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = self.fixture(root, extra_truth=True)
            with (
                patch("sys.argv", argv),
                self.assertRaisesRegex(ValueError, "input-only"),
            ):
                main()
            self.assertFalse((root / "selection").exists())


if __name__ == "__main__":
    unittest.main()
