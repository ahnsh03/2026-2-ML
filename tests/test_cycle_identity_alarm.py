import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common.bridge import DEFAULT_CHECKPOINT_SHA256, sha256_file
from quality_estimation.alarm import fit_alarm_cutoff
from quality_estimation.metrics import failure_detection_metrics
from quality_estimation.perception_identity import (baseline_identity, identity_from_adaptation,
    load_adaptation, require_matching_identity, TARGET_POLICY_SHA256, validate_identity)


class IdentityTests(unittest.TestCase):
    def adaptation(self, path):
        (path/"checkpoint.pt").write_bytes(b"toy updated weights; no deserialization in this test")
        (path/"protocol.json").write_text("{}")
        (path/"training_membership.json").write_text("[]")
        run = dict(status="complete", initial_checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,
                   target_policy_sha256=TARGET_POLICY_SHA256, checkpoint_sha256=sha256_file(path/"checkpoint.pt"),
                   artifact_sha256={p.name:sha256_file(p) for p in path.iterdir()})
        (path/"run.json").write_text(json.dumps(run))
        (path/"_SUCCESS").write_text(sha256_file(path/"run.json"))
        return identity_from_adaptation(path)

    def test_legacy_baseline_remains_default(self):
        self.assertEqual(require_matching_identity(DEFAULT_CHECKPOINT_SHA256), baseline_identity())

    def test_updated_weights_require_explicit_matching_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            identity=self.adaptation(Path(temp))
            with self.assertRaisesRegex(ValueError,"mismatch"):
                require_matching_identity(identity['checkpoint_sha256'],identity)
            self.assertEqual(require_matching_identity(identity['checkpoint_sha256'],identity,identity),identity)
            with self.assertRaisesRegex(ValueError,"mismatch"):
                require_matching_identity(DEFAULT_CHECKPOINT_SHA256,None,identity)

    def test_changed_update_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);self.adaptation(p);(p/'checkpoint.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                load_adaptation(p)

    def test_existing_updates_use_their_hash_bound_checkpoint_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);self.adaptation(p)
            run=json.loads((p/'run.json').read_text());run.pop('checkpoint_sha256')
            (p/'run.json').write_text(json.dumps(run));(p/'_SUCCESS').write_text(sha256_file(p/'run.json'))
            self.assertEqual(load_adaptation(p)[0],p/'checkpoint.pt')

    def test_identity_cannot_omit_lineage_or_change_target_policy(self):
        identity=dict(baseline_identity(),kind='adaptation',checkpoint_sha256='b'*64)
        with self.assertRaisesRegex(ValueError,'lineage'):
            validate_identity(identity)
        identity=dict(baseline_identity(),target_policy_sha256='b'*64)
        with self.assertRaisesRegex(ValueError,'policy'):
            validate_identity(identity)


class AlarmTests(unittest.TestCase):
    def test_strict_cutoff_holds_empirical_normal_budget(self):
        y=np.array([.8]*10+[.2]*3,np.float32)
        score=np.array([.2,.3,.4,.5,.6,.7,.8,.9,.95,1.,.1,.25,.7])
        policy=fit_alarm_cutoff(y,score,.1,min_normals=10)
        self.assertEqual(policy['alarm_threshold'],.3)
        self.assertEqual(policy['actual_false_positives'],1)
        m=failure_detection_metrics(y,score,.7,policy['alarm_threshold'])
        self.assertEqual((m['tp'],m['fp'],m['fn'],m['tn']),(2,1,1,9))

    def test_ties_are_not_split_to_create_false_precision(self):
        policy=fit_alarm_cutoff(np.full(20,.8),np.full(20,.5),.1)
        self.assertEqual(policy['alarm_threshold'],.5)
        self.assertEqual(policy['actual_false_positives'],0)

    def test_float32_target_boundary_and_missing_are_preserved(self):
        y=np.array([.7]*20+[np.nan],np.float32)
        score=np.array([.4]*20+[.1])
        policy=fit_alarm_cutoff(y,score,.1)
        self.assertEqual(policy['normal_rows'],20)
        self.assertEqual(policy['undefined_targets'],1)

    def test_no_normals_cannot_calibrate_a_false_alarm_budget(self):
        with self.assertRaisesRegex(ValueError,'insufficient'):
            fit_alarm_cutoff(np.full(25,.1),np.full(25,.2),.1)

    def test_label_cutoff_and_alarm_cutoff_have_distinct_meanings(self):
        y=np.array([.5,.8]);estimate=np.array([.6,.6])
        m=failure_detection_metrics(y,estimate,.7,.4)
        self.assertEqual((m['tp'],m['fp'],m['fn'],m['tn']),(0,0,1,1))
        self.assertEqual(m['auroc'],.5)
