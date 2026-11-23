import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import hypergeom

from quality_estimation.offline_review import make_queue, replay_queue, review_sample, write_receipt
from quality_estimation.offline_review_admit import join_candidates
from quality_estimation.offline_review_audit import audit_counts, total_interval
from quality_estimation.perception_identity import baseline_identity
from quality_estimation.roi10_study import _check_membership, _completed, _publish, _schema, fit


class FrozenReviewTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.array(list("abcdefgh"))
        self.pixels = np.array([format(i, "064x") for i in range(8)])
        self.scores = np.array([.1, .2, .2, .8, .8, .8, .9, 1])

    def queue(self, budget=.5, ranked=False):
        return dict(population=8, selected=review_sample(self.ids, self.pixels, self.scores, budget, 19,
                                                         ranked_only=ranked))

    def test_selection_ties_permutation_probability_and_ceiling(self):
        mixed = self.queue()["selected"]
        self.assertEqual(["a", "b"], [r["sample_id"] for r in mixed[:2]])
        self.assertEqual([1, 1, 2/6, 2/6], [r["inclusion_probability"] for r in mixed])
        self.assertEqual(mixed, review_sample(self.ids[::-1], self.pixels[::-1], self.scores[::-1], .5, 19))
        self.assertEqual(1, len(self.queue(.1)["selected"]))
        self.assertEqual(8, len({r["sample_id"] for r in self.queue(1)["selected"]}))
        with self.assertRaises(ValueError):
            review_sample(self.ids, self.ids[[0]*8], self.scores, .5, 19)
        with self.assertRaises(ValueError):
            review_sample(self.ids, self.pixels, self.scores * np.nan, .5, 19)

    def test_delayed_partial_join_and_census(self):
        counts = np.zeros((8, 10, 3), dtype=int)
        counts[:4, 4, 1] = counts[4:, 4, 0] = 10
        labels = dict(ids=self.ids, pixel_sha256=self.pixels, counts=counts)
        queue = self.queue()
        full = audit_counts(queue, labels, .7)
        indices = [list(self.ids).index(r["sample_id"]) for r in queue["selected"]]
        self.assertEqual(full, audit_counts(queue, {k:v[indices] for k,v in labels.items()}, .7))
        self.assertEqual("unobserved_complement", audit_counts(self.queue(ranked=True), labels, .7)["estimate_status"])
        self.assertEqual([4, 4], audit_counts(self.queue(1, True), labels, .7)["finite_pool_estimate"]["count_interval_95"])
        self.assertIsNone(audit_counts(queue, {k:v[1:] for k,v in labels.items()}, .7)["finite_pool_estimate"])
        counts[0] = 0
        self.assertEqual(1, audit_counts(queue, labels, .7)["selection"]["priority"]["empty_union"])
        with self.assertRaises(ValueError):
            audit_counts(queue, dict(labels, pixel_sha256=self.pixels[::-1]), .7)

    def test_admissions_need_the_same_reserved_annotation_in_frozen_order(self):
        q = dict(self.queue(), arm="QualityRankAudit", seed=19, budget=.5, model_run_sha256="a"*64,
                 inference_run_sha256="b"*64)
        rows = [dict(sample_id=r["sample_id"], parent_session_id="session", view="front",
                     preregistered_quality_role="candidate_pool", image="frame", semantic="label",
                     expected_sha256={"rgb":"a"*64, "semantic":"b"*64}, pixel_sha256=r["pixel_sha256"])
                for r in reversed(q["selected"])]
        result = join_candidates(q, rows, "c"*64)
        self.assertEqual([r["sample_id"] for r in result], [r["sample_id"] for r in q["selected"]])
        for broken in (rows[:-1], rows+[rows[0]],
                       [dict(rows[0], preregistered_quality_role="final_test")]+rows[1:],
                       [dict(rows[0], pixel_sha256="wrong")]+rows[1:],
                       [dict(rows[0], input_sha256={"rgb":"c"*64})]+rows[1:]):
            with self.assertRaises(ValueError):
                join_candidates(q, broken, "c"*64)

    def test_exact_interval_matches_enumeration(self):
        for n, draws in [(1, 1), (11, 5), (8, 8)]:
            for x in range(draws + 1):
                good = [k for k in range(n+1) if hypergeom.cdf(x, n, k, draws) >= .025
                        and hypergeom.sf(x-1, n, k, draws) >= .025]
                self.assertEqual([min(good), max(good)], total_interval(n, draws, x))

    def test_additional_quality_training_verifies_role_geometry_and_overlap(self):
        def pack(path, ids, purpose="training", pixels=None, region_pixels=None):
            path.mkdir()
            n = len(ids)
            out = path / "rows.npz"
            np.savez(out, ids=ids, pixel_sha256=pixels or ids, x56=np.zeros((n, 56)),
                     nodes=np.zeros((n, 10, 8)), heads=np.zeros((n, 10, 4)), split_group=ids,
                     predicted_pixels=np.full((n, 10), 2), counts=np.ones((n, 10, 3), dtype=int),
                     region_pixels=np.ones(10)*10 if region_pixels is None else region_pixels)
            _publish(path, dict(**_schema(), paired=True, purpose=purpose), [out])
            return path
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = pack(root / "base", ["a", "b"])
            for other, msg in [(pack(root / "duplicate", ["c"], pixels=["b"]), "duplicate"),
                               (pack(root / "test", ["d"], purpose="evaluation"), "training"),
                               (pack(root / "resized", ["e"], region_pixels=np.ones(10)*11), "geometry")]:
                args = SimpleNamespace(data=first, additional_training=[other], jobs=1, output=root / "out")
                with self.assertRaisesRegex(ValueError, msg):
                    fit(args)
            extra = pack(root / "extra", ["e", "f"])
            args = SimpleNamespace(data=first, additional_training=[extra], jobs=1, output=root / "combined")
            fit(args)
            meta = _completed(args.output, "training.npz")
            self.assertEqual(4, meta["fit_rows"])
            self.assertEqual(2, len(meta["training_pack_sha256s"]))
            with np.load(args.output / "training.npz") as training:
                self.assertEqual(list(training["ids"]), ["a", "b", "e", "f"])

    def test_queue_refuses_paired_training_or_edited_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "scores.npz"
            np.savez(artifact, ids=self.ids, pixel_sha256=self.pixels, estimated_iou=self.scores)
            base = dict(**_schema(), model_sha256="a"*64, overlapping_training_rows=0,
                        uses_gt_for_prediction=False, input_paired=False)
            for field, value in (("uses_gt_for_prediction", True), ("input_paired", True),
                                 ("overlapping_training_rows", 1)):
                _publish(root, dict(base, **{field:value}), [artifact])
                with self.assertRaises(ValueError):
                    make_queue(root, .5, 19)
            _publish(root, base, [artifact])
            queue = make_queue(root, .5, 19)
            queue["selected"][0]["score"] = 0
            path = root / "queue.json"
            write_receipt(path, queue)
            with self.assertRaisesRegex(ValueError, "decisions"):
                replay_queue(path, root)
            with artifact.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                make_queue(root, .5, 19)

    def test_identity_and_receipt_no_overwrite(self):
        f1 = dict(baseline_identity(), kind="adaptation", checkpoint_sha256="a"*64, adaptation_run_sha256="b"*64)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "payload"
            write_receipt(file, {"test": True})
            with self.assertRaises(FileExistsError):
                write_receipt(file, {})
            self.assertTrue(json.loads(file.read_text())["test"])
            _publish(root, _schema(f1), [file])
            self.assertEqual(_completed(root, file.name, f1)["perception_identity"], f1)
            with self.assertRaises(ValueError):
                _completed(root, file.name)
            with self.assertRaises(ValueError):
                _completed(root, file.name, dict(f1, adaptation_run_sha256="c"*64))

    def test_independent_quality_fit_blocks_shared_physical_session(self):
        training = dict(ids=["a"], pixel_sha256=["x"], parent_session_id=["road-A"])
        test = dict(ids=["b"], pixel_sha256=["y"], parent_session_id=["road-A"])
        with self.assertRaisesRegex(ValueError, "physical"):
            _check_membership(test, training, "evaluation")
        self.assertEqual(0, _check_membership(dict(test, parent_session_id=["road-B"]), training, "evaluation"))
