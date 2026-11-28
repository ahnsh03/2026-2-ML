import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from quality_estimation.roi9_handoff import load_roi9_handoff


class ROI9HandoffTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root/"code/config").mkdir(parents=True)
        (self.root/"models").mkdir()
        quality, rgb = ["q"+str(i) for i in range(27)], ["r"+str(i) for i in range(54)]
        roles = np.array(["train"]*6+["development"]*2+["diagnostic"]*2)
        ids = np.array(["frame"+str(i) for i in range(10)])
        np.savez(self.root/"v2_frames.npz", X=np.ones((10, 27), np.float32),
                 y=np.linspace(.1, .9, 10, dtype=np.float32), RGB54=np.ones((10, 54), np.float32),
                 ids=ids, split=roles, source_cohort=roles, quality_columns=quality, rgb54_columns=rgb,
                 weather=np.zeros(10, dtype=int), hour=np.full(10, 13, dtype=int))
        self.rows = [dict(row=str(i), sample_id=ids[i], split=roles[i], source_cohort=roles[i],
                          train_cv_fold=str(i//2) if i < 6 else "", group="g"+str(i//2),
                          site="site"+str(i), pixel_sha256="pixel"+str(i),
                          weather="0", hour="13", x=str(i*100), y="0") for i in range(10)]
        tasks = {}
        for task, columns, filename in (("quality_gp", quality, "quality_gp.joblib"),
                                        ("quality_rf", quality, "quality_rf.joblib"),
                                        ("environment", rgb, "environment.npz")):
            path = self.root/"models"/filename
            path.write_bytes(task.encode())
            tasks[task] = dict(ordered_features=columns, artifact_filename=filename,
                               model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (self.root/"code/config/region_feature_recipe_v2.json").write_text(json.dumps(dict(profile="roi9-front-v1", tasks=tasks)))
        self.write_rows()

    def manifest(self):
        files = sorted(p for p in self.root.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
        (self.root/"SHA256SUMS").write_text("".join(
            hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p.relative_to(self.root))+"\n" for p in files))

    def write_rows(self):
        with (self.root/"frames.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.manifest()

    def test_preserves_provided_folds_and_feature_dtype(self):
        pack = load_roi9_handoff(self.root)
        self.assertEqual(pack.arrays["X"].dtype, np.float32)
        folds, audit = pack.training_folds()
        for k, (_, train, valid) in enumerate(folds):
            np.testing.assert_array_equal(valid, [2*k, 2*k+1])
            self.assertEqual(len(train), 4)
            self.assertGreater(audit[k]["minimum_distance_m"], 0)

    def test_modified_input_fails_checksum(self):
        with (self.root/"frames.csv").open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_roi9_handoff(self.root)

    def test_valid_hash_does_not_hide_row_misalignment(self):
        self.rows[0]["sample_id"] = "wrong"
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "row alignment"):
            load_roi9_handoff(self.root)

    def test_development_pixels_cannot_reappear_in_training(self):
        self.rows[6]["pixel_sha256"] = self.rows[0]["pixel_sha256"]
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "overlapping train/development"):
            load_roi9_handoff(self.root)

    def test_site_cannot_cross_cv_folds(self):
        self.rows[2]["site"] = self.rows[0]["site"]
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "crossing CV folds"):
            load_roi9_handoff(self.root)

    def test_missing_cv_label_fails_without_random_fallback(self):
        self.rows[0]["train_cv_fold"] = ""
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "CV membership"):
            load_roi9_handoff(self.root)

    def test_schema_and_protocol_recipe_must_match(self):
        with self.assertRaisesRegex(ValueError, "frozen search protocol"):
            load_roi9_handoff(self.root, "not-the-same-recipe")
        recipe = self.root/"code/config/region_feature_recipe_v2.json"
        spec = json.loads(recipe.read_text())
        spec["tasks"]["quality_gp"]["ordered_features"].reverse()
        recipe.write_text(json.dumps(spec))
        self.manifest()
        with self.assertRaisesRegex(ValueError, "feature schema"):
            load_roi9_handoff(self.root)


if __name__ == "__main__":
    unittest.main()
