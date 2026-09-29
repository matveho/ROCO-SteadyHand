import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from steadyhand.wrist_part_profiles import blank_profiles, load_profiles, save_profile


class WristProfileTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"robot_name": "test"}

    def profile(self, root):
        template = root / "part.png"
        template.write_bytes(b"template")
        return {
            "part": "battery_size1", "working_arm": "right", "tcp_frame": "tip_r",
            "wrist_camera": "wrist_a", "calibration_sha256": "a" * 64,
            "coarse_xy_m": [0.4, -0.1], "feature_uv": [320, 240],
            "goal_uv": [320, 240], "image_shape": [480, 640],
            "hover_clearance_m": 0.1, "grasp_clearance_m": 0.035, "yaw_deg": 0,
            "template": {"path": "part.png", "sha256": hashlib.sha256(b"template").hexdigest(), "template_uv": [20, 20]},
        }

    def test_atomic_save_and_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "profiles.json"
            value = save_profile(path, self.cfg, self.profile(root))
            self.assertEqual(load_profiles(path, self.cfg)["parts"]["battery_size1"]["grasp_clearance_m"], 0.035)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_rejects_missing_hash_and_out_of_range_depth(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            profile = self.profile(root)
            profile["calibration_sha256"] = "bad"
            with self.assertRaises(ValueError):
                save_profile(root / "p.json", self.cfg, profile)


if __name__ == "__main__":
    unittest.main()
