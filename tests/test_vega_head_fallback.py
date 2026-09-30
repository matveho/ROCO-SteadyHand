import tempfile
import unittest
from pathlib import Path

from steadyhand.models import Pose
from tools.vega_head_fallback import (
    HeadFallbackSession,
    _is_can_network_down,
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

    def test_board_frame_matching_survives_board_pose_change(self):
        scene = {
            "board": {"center_base_m_coarse": [0.8, 0.3, 0.456]},
            "parts": [{
                "center_base_m_coarse": [0.81, 0.31, 0.456],
                "center_board_m": [0.05, 0.02, 0.0],
            }],
        }
        task_data = {
            "source_board_center_xy_m": [0.193, 0.193],
            "task_coordinate_rotation_deg": 0.0,
            "parts": {
                part: {"pick": [0.193, 0.193, 0.0]}
                for part in (
                    "gear_60teeth", "gear_20teeth", "rod_16mm", "bolt_8mm",
                    "usb_a", "hdmi", "pin", "battery_size1", "battery_size5",
                )
            },
        }
        observations = match_expected_parts(scene, self.runtime, self.targets, task_data=task_data)
        self.assertEqual(observations["gear_60teeth"]["selection"], "head_detection")

    def test_profiles_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head_profiles.json"
            value = {"schema_version": 1, "camera": "head_camera", "parts": {"battery_size1": {"offset_base_xy_m": [0.001, -0.002]}}}
            save_profiles(path, value)
            self.assertEqual(load_profiles(path), value)

    def test_depth_allows_exact_board_contact_boundary(self):
        session = HeadFallbackSession.__new__(HeadFallbackSession)
        session.floor = 0.456
        session.hover_pose = Pose((0.5, 0.0, 0.6), (1.0, 0.0, 0.0, 0.0))
        session.grasp_clearance_m = None
        session.surface = lambda x, y: 0.5
        session.set_depth(100)
        self.assertEqual(session.grasp_clearance_m, 0.0)
        session.set_depth(105)
        self.assertAlmostEqual(session.grasp_clearance_m, -.005)
        for invalid in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                session.set_depth(invalid)

    def test_can_network_down_detection_is_narrow(self):
        self.assertTrue(_is_can_network_down(RuntimeError("Network is down [Error Code 100]")))
        self.assertTrue(_is_can_network_down(OSError("errno 100")))
        self.assertFalse(_is_can_network_down(RuntimeError("motor did not respond")))

    def test_grab_opens_at_hover_and_lifts_after_grip_failure(self):
        events = []

        class FakeRobot:
            def connect_gripper(self):
                events.append("connect")

            def open_gripper(self, part):
                events.append(("open", part))

            def grip(self, part):
                events.append(("grip", part))
                raise RuntimeError("CAN reply lost")

        session = HeadFallbackSession.__new__(HeadFallbackSession)
        session.robot = FakeRobot()
        session.part = "battery_size1"
        session.grasp_clearance_m = 0.0
        session.hover_pose = Pose((0.5, 0.0, 0.6), (1.0, 0.0, 0.0, 0.0))
        session.surface = lambda x, y: 0.5
        session.holding = False
        session.move = lambda target, slow=False: events.append(("move", target.position_m[2]))

        session.grab()
        self.assertEqual(events[0:3], ["connect", ("open", "battery_size1"), ("move", 0.5)])
        self.assertEqual(events[-1], ("move", 0.6))
        self.assertTrue(session.holding)


if __name__ == "__main__":
    unittest.main()
