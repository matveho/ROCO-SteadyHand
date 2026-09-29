import tempfile
import unittest
from pathlib import Path

from steadyhand.models import Pose
from tools.vega_head_fallback import (
    load_profiles,
    match_expected_parts,
    save_profiles,
)


class HeadFallbackTests(unittest.TestCase):
    def setUp(self):
        self.runtime = (
            None,
            None,
            ((0.5, 0.0), (1.0, 0.0), (0.0, 1.0), {"coefficients": (0.0, 0.0, 0.5), "anchors": []}),
            None,
        )
        parts = (
            "gear_60teeth", "gear_20teeth", "rod_16mm", "bolt_8mm", "usb_a",
            "hdmi", "pin", "battery_size1", "battery_size5",
        )
        self.targets = {
            f"task.{part}.pick": Pose((0.1 + index * 0.03, 0.0, 0.6), (1.0, 0.0, 0.0, 0.0))
            for index, part in enumerate(parts)
        }

    def test_head_detection_is_used_only_for_unique_nearby_match(self):
        scene = {
            "board": {"center_base_m_coarse": [0.5, 0.0, 0.456]},
            "parts": [{"center_base_m_coarse": [0.1, 0.0, 0.456]}],
        }
        observations = match_expected_parts(scene, self.runtime, self.targets)
        self.assertEqual(observations["gear_60teeth"]["selection"], "head_detection")
        self.assertEqual(observations["battery_size1"]["selection"], "expected_coordinate")

    def test_missing_scene_falls_back_to_expected_coordinates(self):
        observations = match_expected_parts({"board": {}, "parts": []}, self.runtime, self.targets)
        self.assertTrue(all(item["selection"] == "expected_coordinate" for item in observations.values()))

    def test_profiles_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head_profiles.json"
            value = {"schema_version": 1, "camera": "head_camera", "parts": {"battery_size1": {"offset_base_xy_m": [0.001, -0.002]}}}
            save_profiles(path, value)
            self.assertEqual(load_profiles(path), value)


if __name__ == "__main__":
    unittest.main()
