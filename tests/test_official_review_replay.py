import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from scripts.compare_official_review_replay import compare


class OfficialReplayComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.reference=self.root/"reference";self.reference.mkdir()
        records=[dict(sample_id="a",estimated_iou=.4,environment_cluster=0,uses_ground_truth=False),
                 dict(sample_id="b",estimated_iou=.5,environment_cluster=1,uses_ground_truth=False)]
        selected=[dict(sample_id=r["sample_id"],selection_rank=i,selection_component="priority",review_population=2,budget=2)
                  for i,r in enumerate(records)]
        for name,rows in [("predictions.jsonl",records),("candidates.jsonl",selected)]:
            (self.reference/name).write_text("".join(json.dumps(r)+"\n"for r in rows))
        np.savez(self.reference/"features.npz",X=np.array([[.1],[.2]],dtype=np.float32),ids=["a","b"],ordered_features=["x"])
        self.publish(self.reference)
        self.actual=self.root/"actual";shutil.copytree(self.reference,self.actual)

    def publish(self,path):
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        run=dict(status="complete",rows=2,budget=2,perception_identity={"id":"r2"},profile="roi9",regions_sha256="regions",
                 bundle_manifest_sha256="bundle",manifest_sha256="manifest",source_sha256="source",
                 artifact_sha256={n:sha(path/n)for n in ["features.npz","predictions.jsonl","candidates.jsonl"]})
        (path/"run.json").write_text(json.dumps(run))
        (path/"_SUCCESS").write_text(sha(path/"run.json"))

    def test_identical_completed_runs_match(self):
        result=compare(self.actual,self.reference)
        self.assertEqual(result["status"],"matched")
        self.assertEqual(result["selection_overlap"],2)

    def test_changed_file_fails_hash_validation(self):
        (self.actual/"predictions.jsonl").write_text("changed")
        with self.assertRaisesRegex(ValueError,"checksum"):
            compare(self.actual,self.reference)

    def test_selection_order_is_required_not_just_overlap(self):
        p=self.actual/"candidates.jsonl"
        p.write_text("\n".join(p.read_text().splitlines()[::-1])+"\n")
        self.publish(self.actual)
        result=compare(self.actual,self.reference)
        self.assertEqual(result["selection_overlap"],2)
        self.assertFalse(result["selection_ids_order_reasons_equal"])

    def test_large_feature_difference_is_reported_even_when_predictions_match(self):
        np.savez(self.actual/"features.npz",X=np.array([[.101],[.2]],dtype=np.float32),ids=["a","b"],ordered_features=["x"])
        self.publish(self.actual)
        result=compare(self.actual,self.reference)
        self.assertIn("features.X",result["failures"])
        self.assertTrue(result["selection_ids_order_reasons_equal"])

    def test_small_fp32_prediction_difference_is_measured(self):
        p=self.actual/"predictions.jsonl"
        rows=[json.loads(s)for s in p.read_text().splitlines()];rows[0]["estimated_iou"]+=5e-7
        p.write_text("".join(json.dumps(r)+"\n"for r in rows));self.publish(self.actual)
        result=compare(self.actual,self.reference)
        self.assertEqual(result["status"],"matched")
        self.assertAlmostEqual(result["max_numeric_differences"]["estimated_iou"],5e-7)


if __name__=="__main__":
    unittest.main()
