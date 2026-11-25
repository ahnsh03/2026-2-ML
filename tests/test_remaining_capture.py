import importlib.util, unittest
from pathlib import Path

path = (
    Path(__file__).resolve().parents[1]
    / "scripts/experiments/roi9_gp_20261125/remaining_capture.py"
)
spec = importlib.util.spec_from_file_location("remaining_capture", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RemainingCaptureTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(
            simulator_release="26.R1",
            weather_ids=[0, 1],
            hours=[5, 13],
            poses=[dict(id=400), dict(id=401)],
        )
        self.row = dict(
            sample_id="release26_site400_w0_h05",
            weather=0,
            hour=5,
            simulator_release="26.R1",
            files={name: {} for name in ["Intensity", "Semantic", "Depth", "Instance"]},
        )

    def test_resume_skips_accepted_rows_and_limits_a_chunk_without_mutating_parent(
        self,
    ):
        result = module.remaining_plan(self.plan, [self.row], max_frames=3)
        self.assertEqual(result["expected_frames"], 3)
        self.assertEqual(result["remaining_after_this_plan"], 4)
        self.assertEqual(result["poses"][0]["conditions"], [[0, 13], [1, 5], [1, 13]])
        self.assertNotIn("conditions", self.plan["poses"][0])

    def test_duplicate_wrong_release_and_incomplete_channels_cannot_suppress_capture(
        self,
    ):
        for rows in [
            [self.row, self.row],
            [dict(self.row, simulator_release="old")],
            [dict(self.row, files={})],
            [dict(self.row, weather=1)],
        ]:
            with self.assertRaises(ValueError):
                module.remaining_plan(self.plan, rows)


if __name__ == "__main__":
    unittest.main()
