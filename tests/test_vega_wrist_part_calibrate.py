import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from steadyhand.models import Pose
from tools.vega_wrist_part_calibrate import PartSession, _check_ready


class _FakeKinematics:
    def solve(self, pose, seed):
        return seed


class _FakeRobot:
    def __init__(self, position=(0.10, 0.20, 0.60)):
        self._kinematics = _FakeKinematics()
        self.pose = Pose(position, (1.0, 0.0, 0.0, 0.0))
        self.opened = []
        self.released = []

    def get_tcp_pose(self):
        return self.pose

    def _read_joint_positions(self):
        return [0.0] * 7

    def open_gripper(self, part):
        self.opened.append(part)

    def release_gripper(self, part):
        self.released.append(part)


class WristPartReturnTests(unittest.TestCase):
    def test_field_recalibration_does_not_invalidate_saved_profile(self):
        profile = {
            "part": "battery_size1",
            "calibration_sha256": "a" * 64,
            "grasp_clearance_m": 0.008,
            "grasp_verified": True,
        }
        cfg = {"robot_name": "test"}
        with mock.patch(
            "tools.vega_wrist_part_calibrate.load_board_calibration",
            return_value={"sha256": "b" * 64},
        ), mock.patch("builtins.print") as output:
            _check_ready(profile, cfg, "pick", no_cv=True)
        self.assertTrue(
            any("board calibration changed" in str(call).lower() for call in output.call_args_list)
        )

    def test_return_part_releases_and_clears_holding_state(self):
        session = PartSession.__new__(PartSession)
        session.holding = True
        session.part = "battery_size1"
        session.robot = _FakeRobot()
        session.floor = 0.45
        session.surface = lambda x, y: 0.50

        moves = []
        events = []
        session.move = lambda pose, slow=False: moves.append((pose, slow))
        session.event = lambda kind, fields: events.append((kind, fields))

        session.return_part(0.02)

        self.assertFalse(session.holding)
        self.assertEqual(session.robot.released, ["battery_size1"])
        self.assertEqual(session.robot.opened, [])
        self.assertEqual(len(moves), 2)
        self.assertEqual(moves[0][0].position_m, (0.10, 0.20, 0.52))
        self.assertTrue(moves[0][1])
        self.assertEqual(moves[1][0].position_m, (0.10, 0.20, 0.60))
        self.assertTrue(moves[1][1])
        self.assertEqual(events[0][0], "place_release")
        self.assertEqual(
            events[0][1]["settings"],
            {"return_to_source": True, "clearance_m": 0.02},
        )

    def test_close_writes_auditable_run_summary(self):
        with tempfile.TemporaryDirectory() as td:
            session = PartSession.__new__(PartSession)
            session.output = Path(td)
            session.cameras = mock.Mock()
            session.robot = mock.Mock()
            session.status = "completed"
            session.part = "battery_size1"
            session.action = "pick_place"
            session.holding = False
            session.last_error = None
            session.coarse = Pose((0.42, -0.12, 0.62), (1.0, 0.0, 0.0, 0.0))
            session.yaw = 3.0
            session.last_grasp_clearance = 0.04
            session.board_scene_paths = ["board_001.json"]
            session.args = type("Args", (), {"profiles": "calibration/wrist_part_profiles.json"})()

            PartSession.close(session)

            summary = json.loads((Path(td) / "run_summary.json").read_text())
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["part"], "battery_size1")
            self.assertEqual(summary["action"], "pick_place")
            self.assertFalse(summary["holding_may_be_true"])
            self.assertEqual(summary["coarse_xy_m"], [0.42, -0.12])
            self.assertEqual(summary["board_scene_paths"], ["board_001.json"])
            session.cameras.close.assert_called_once_with()
            session.robot.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
