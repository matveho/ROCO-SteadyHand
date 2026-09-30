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

    def test_rejects_missing_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            profile = self.profile(root)
            profile["calibration_sha256"] = "bad"
            with self.assertRaises(ValueError):
                save_profile(root / "p.json", self.cfg, profile)

    def test_operator_depth_and_yaw_round_trip_without_surface_clamp(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            profile = self.profile(root)
            profile.update(grasp_clearance_m=-.005, yaw_deg=90,
                           place={"offset_board_xy_m": [.12, -.08], "clearance_m": -.010, "yaw_deg": -90})
            path = root / "p.json"
            save_profile(path, self.cfg, profile)
            saved = load_profiles(path, self.cfg)["parts"]["battery_size1"]
            self.assertEqual(saved, profile)
            for invalid in (float('nan'), float('inf')):
                profile['grasp_clearance_m'] = invalid
                with self.assertRaises(ValueError):
                    save_profile(path, self.cfg, profile)

    def test_shipped_profiles_keep_template_hashes_and_board_hashes(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "calibration" / "wrist_part_profiles.json"
        if not path.is_file():
            self.skipTest("onsite calibration bundle is not present")
        value = json.loads(path.read_text(encoding="utf-8"))
        for part, profile in value.get("parts", {}).items():
            self.assertEqual(len(profile.get("calibration_sha256", "")), 64, part)
            template = root / profile["template"]["path"]
            self.assertTrue(template.is_file(), part)
            self.assertEqual(
                hashlib.sha256(template.read_bytes()).hexdigest(),
                profile["template"]["sha256"],
                part,
            )


if __name__ == "__main__":
    unittest.main()
