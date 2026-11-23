import unittest

import numpy as np

from camera_semantic_perception.geometry import compute_letterbox_geometry
from quality_estimation.roi10 import (FrontProfile, PAIRS, REGIONS, extract_region_rgb,
                                      native_scene_histograms, predicted_sky, reduce_heads,
                                      scene_histogram_views, summarize_head_maps)


class RegionTenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = FrontProfile.load()

    def test_profile_assigns_every_pixel_and_keeps_hood_silhouette(self):
        p = self.profile.native
        self.assertEqual(p.shape, (720, 1280))
        self.assertEqual(np.unique(p).tolist(), list(range(10)))
        self.assertEqual(np.count_nonzero(p == 9), 118199)
        self.assertEqual([int(p[y, 640]) for y in [337, 338, 418, 419, 578, 579]], [1, 4, 4, 7, 7, 9])
        for i in range(len(REGIONS)):
            self.assertTrue(np.any(self.profile.small == i))

    def test_empty_sky_and_known_node_values(self):
        rgb = np.full((720, 1280, 3), (70, 90, 110), dtype=np.uint8)
        result = extract_region_rgb(rgb, self.profile)
        self.assertEqual(result["weather100"].shape, (100,))
        self.assertEqual(result["hour187"].shape, (187,))
        self.assertEqual(result["haze30"].shape, (30,))
        np.testing.assert_allclose(result["nodes"][:, 1], 40 / 255., atol=1e-6)
        np.testing.assert_allclose(result["signed_differences"],
                                   [result["nodes"][a] - result["nodes"][b] for a, b in PAIRS])
        sky = predicted_sky(rgb[::4, ::4], self.profile)
        self.assertFalse(sky[self.profile.small >= 3].any())
        for i, support in enumerate(result["sky27"][-3:]):
            self.assertAlmostEqual(support, sky[self.profile.small == i].mean(), places=6)
            if support == 0:
                np.testing.assert_array_equal(result["sky27"][i*8:i*8+8], np.zeros(8))

    def test_native_histogram_bins_and_channel_order(self):
        rgb = np.full((720, 1280, 3), 128, dtype=np.uint8)
        h = native_scene_histograms(rgb, self.profile)
        np.testing.assert_array_equal(h.sum(2), np.ones((10, 5)))
        np.testing.assert_array_equal(h[:, :, :].argmax(2), np.tile([5, 0, 0, 0, 0], (10, 1)))
        examples = np.arange(2 * h.size).reshape(2, *h.shape)
        views = scene_histogram_views(examples)
        np.testing.assert_array_equal(views["weather150"], examples[:, :3].reshape(2, 150))
        np.testing.assert_array_equal(views["hour400"][0].reshape(10, 4, 10),
                                      examples[0][:, [0, 1, 3, 4]])
        with self.assertRaises(ValueError):
            scene_histogram_views(h.transpose(1, 0, 2))

    def test_environment_cluster_profile_checks_feature_order_and_camera(self):
        from quality_estimation.environment_clusters import (ROI10_HAZE, assign_environment_clusters,
                                                            fit_environment_clusters, rgb_extractor)
        rgb = np.full((720, 1280, 3), [60, 70, 80], dtype=np.uint8)
        vector = rgb_extractor(ROI10_HAZE)(rgb)
        np.testing.assert_allclose(vector, extract_region_rgb(rgb, self.profile)["haze30"])
        features = np.stack([vector, vector + .05, vector + .9, vector + .95])
        centers = fit_environment_clusters(features, 2, profile=ROI10_HAZE)
        labels = assign_environment_clusters(centers, features)
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[3])
        with self.assertRaisesRegex(ValueError, "camera profile"):
            assign_environment_clusters(dict(centers, camera_profile_sha256="changed"), features)
        with self.assertRaises(ValueError):
            assign_environment_clusters(centers, np.zeros((1, 73)))

    def test_native_heads_exclude_padding_and_stopline(self):
        geometry = compute_letterbox_geometry((720, 1280), (384, 640))
        q = np.zeros((4, 384, 640), np.float32)
        q[1] = 1
        q[:, 12:372] = 0
        q[3, 12:372] = 1
        drivable = np.ones((384, 640), dtype=np.uint8)
        maps = reduce_heads(drivable, q, geometry)
        np.testing.assert_array_equal(maps[0], 255)
        np.testing.assert_array_equal(maps[1:], 0)
        np.testing.assert_array_equal(summarize_head_maps(maps, self.profile),
                                      np.tile([1., 0., 0., 0.], (10, 1)))
        q[0, 14, 50] = np.nan
        with self.assertRaises(ValueError):
            reduce_heads(drivable, q, geometry)

    def test_region_major_channel_order(self):
        p = self.profile.small
        maps = np.stack([p * 20 + i for i in range(4)]).astype(np.uint8)
        expected = (np.arange(10)[:, None] * 20 + np.arange(4)) / 255.
        np.testing.assert_allclose(summarize_head_maps(maps, self.profile), expected, atol=1e-6)
        with self.assertRaises(ValueError):
            summarize_head_maps(maps.transpose(1, 2, 0), self.profile)
        with self.assertRaises(ValueError):
            extract_region_rgb(np.zeros((180, 320, 3), np.uint8), self.profile)


class RegionalQualityTests(unittest.TestCase):
    def test_whole_image_is_ratio_of_sums_and_missing_union_is_explicit(self):
        from quality_estimation.roi10_quality import RegionalQualityModel

        class Constant:
            def __init__(self, value):
                self.value = value

            def predict(self, x):
                return np.repeat(self.value, len(x))

        heads = [(Constant(1.), Constant(0.)), (Constant(.25), Constant(np.log1p(200.)))]
        heads.extend([(None, Constant(0.))] * 8)
        x = np.zeros((1, 10, 31))
        p = np.array([[10, 20, 0, 0, 0, 0, 0, 0, 0, 0]])
        out = RegionalQualityModel(heads).predict(x, p, np.full(10, 100))
        self.assertAlmostEqual(out["estimated_iou"][0], 15 / 50)
        self.assertAlmostEqual(out["per_region_iou"][0, 1], 5 / 40)
        self.assertEqual(out["per_region_union"][0, 2], 0)
        self.assertAlmostEqual(out["estimated_union"][0], 50)

    def test_delayed_labels_join_by_identity_and_do_not_change_review_budget(self):
        from quality_estimation.roi10_study import evaluate_paired_scores

        scored = dict(ids=np.array(["first", "second", "unlabeled"]), pixel_sha256=np.array(["A", "B", "C"]),
                      estimated_iou=np.array([.8, .5, .1]), per_region_iou=np.zeros((3, 10)))
        counts = np.zeros((2, 10, 3), int)
        counts[0, 7] = [50, 40, 10]
        labels = dict(ids=np.array(["second", "first"]), pixel_sha256=np.array(["B", "A"]), counts=counts)
        report = evaluate_paired_scores(labels, scored, alarm_cutoff=.6)
        self.assertEqual((report["scored"], report["paired"], report["defined"]), (3, 2, 1))
        self.assertEqual(report["mae"], 0)
        self.assertEqual(report["alarm"]["tp"], 1)
        self.assertEqual(report["region_mae_minimum_union_32"]["near_center"]["supported"], 1)
        self.assertEqual(report["pooled_lowest_ten_percent"]["unknown_truth"], 1)
        with self.assertRaisesRegex(ValueError, "pixels differ"):
            evaluate_paired_scores(dict(labels, pixel_sha256=np.array(["X", "A"])), scored)
        with self.assertRaisesRegex(ValueError, "never scored"):
            evaluate_paired_scores(dict(labels, ids=np.array(["missing", "first"])), scored)

    def test_counts_reconstruct_global_iou(self):
        from quality_estimation.roi10_quality import label_counts

        region = FrontProfile.load().native[::2, ::2]
        y, x = np.indices(region.shape)
        pred = ((x + y) % 7) < 2
        target = ((x - y) % 5) < 2
        counts = label_counts(pred, target, region)
        np.testing.assert_array_equal(counts.sum(0), [(pred & target).sum(), (pred & ~target).sum(),
                                                      (~pred & target).sum()])
        np.testing.assert_array_equal(counts[:, :2].sum(1), np.bincount(region[pred], minlength=10))
