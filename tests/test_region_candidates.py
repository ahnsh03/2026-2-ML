import unittest

import numpy as np

from quality_estimation.region_candidates import candidate_schema, extract_candidates
from quality_estimation.region_feature_views import environment_views, feature_matrix, quality_views
from quality_estimation.spatial_features import SpatialProfile, extract_spatial_features
from quality_estimation.selected_region_features import extract_selected_features


class RegionCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = SpatialProfile.load()

    def test_base_statistics_match_existing_rgb_and_head_definitions(self):
        rng = np.random.default_rng(20261124)
        rgb = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        maps = rng.integers(0, 256, (4, 180, 320), dtype=np.uint8)
        extracted = extract_candidates(rgb, maps, self.profile)
        baseline = extract_spatial_features(rgb, maps, self.profile, include_global=True).reshape(10, 12)
        np.testing.assert_array_equal(extracted["rgb"].reshape(10, 18)[:, :8], baseline[:, :8])
        np.testing.assert_array_equal(extracted["output"].reshape(10, 8)[:, :4], baseline[:, 8:])
        np.testing.assert_allclose(extracted["histograms"].reshape(10, 5, 10).sum(-1), 1., atol=1e-6)

    def test_zero_support_and_constant_scene_have_defined_values(self):
        rgb = np.full((720, 1280, 3), 120, np.uint8)
        maps = np.zeros((4, 180, 320), np.uint8)
        result = extract_candidates(rgb, maps, self.profile)
        np.testing.assert_array_equal(result["output"], 0.)
        np.testing.assert_allclose(result["relations"], 0., atol=1e-6)
        self.assertTrue(all(np.isfinite(v).all() for v in result.values()))

    def test_weighted_entropy_preserves_empty_region_support(self):
        rgb = np.zeros((720, 1280, 3), np.uint8)
        maps = np.zeros((4, 180, 320), np.uint8)
        maps[1, self.profile.small == 4] = 255
        maps[2, self.profile.small == 4] = 204
        maps[3, self.profile.small == 4] = 102
        out = extract_candidates(rgb, maps, self.profile, include_histograms=False)["output"].reshape(10, 8)
        np.testing.assert_allclose(out[4, 4:], [.4, .8, 1., 1.], atol=1e-6)
        np.testing.assert_array_equal(out[0], 0.)

    def test_view_columns_are_unique_and_environment_never_uses_perception(self):
        schema = candidate_schema(self.profile)
        views = quality_views(schema)
        views.update(environment_views(schema))
        pack = {name: np.zeros((2, len(schema[name])), np.float32)
                for name in ("rgb", "output", "relations", "histograms")}
        for columns in views.values():
            self.assertEqual(feature_matrix(pack, schema, columns).shape, (2, len(columns)))
        forbidden = set(schema["output"])
        for columns in environment_views(schema).values():
            self.assertFalse(forbidden.intersection(columns))
            self.assertFalse(any(n.split(".")[1].startswith(("lane_", "marking_")) for n in columns))

    def test_selected_extraction_matches_full_candidate_columns(self):
        rng = np.random.default_rng(29)
        rgb = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        maps = rng.integers(0, 256, (4, 180, 320), dtype=np.uint8)
        schema = candidate_schema(self.profile)
        full = {k: v[None] for k, v in extract_candidates(rgb, maps, self.profile).items()}
        columns = quality_views(schema)["role_rel_all"]
        np.testing.assert_array_equal(extract_selected_features(rgb, maps, self.profile, columns),
                                      feature_matrix(full, schema, columns)[0])
        columns = environment_views(schema)["top_hood_hist"]
        np.testing.assert_array_equal(extract_selected_features(rgb, None, self.profile, columns),
                                      feature_matrix(full, schema, columns)[0])
        with self.assertRaises(ValueError):
            extract_selected_features(rgb, None, self.profile, ["near_left.lane_fraction"])
        with self.assertRaises(ValueError):
            extract_selected_features(rgb, None, self.profile, ["near_minus_far.marking_entropy"])


if __name__ == "__main__":
    unittest.main()
