import unittest
from unittest.mock import patch

from steadyhand.models import Pose
from steadyhand.vega_camera_clear import (
    CAMERA_CLEAR_RETAIN_HIGH_POSE_ABOVE_FLOOR_M,
    _can_retain_high_pose,
    move_camera_clear,
)


class _FakeKinematics:
    def __init__(self):
        self.config = {}


class _FakeRobot:
    def __init__(self, pose):
        self.pose = pose
        self._kinematics = _FakeKinematics()
        self.config = {"motion": {"joint_reached_tolerance_rad": 0.005}}

    def get_tcp_pose(self):
        return self.pose


class VegaCameraClearTests(unittest.TestCase):
    def test_high_pose_is_eligible_for_no_motion_fallback(self):
        floor = 0.456
        pose = Pose(
            (0.7266, -0.1776, floor + CAMERA_CLEAR_RETAIN_HIGH_POSE_ABOVE_FLOOR_M + 0.10),
            (1.0, 0.0, 0.0, 0.0),
        )
        self.assertTrue(_can_retain_high_pose(pose, floor_m=floor))

    def test_verticalization_failure_retains_existing_high_pose(self):
        floor = 0.456
        pose = Pose((0.7266, -0.1776, 1.0056), (1.0, 0.0, 0.0, 0.0))
        robot = _FakeRobot(pose)

        with patch(
            "steadyhand.vega_camera_clear._reachable_verticalize_pose",
            side_effect=RuntimeError("IK did not converge"),
        ):
            reached = move_camera_clear(robot, floor_m=floor, speed_scale=0.90)

        self.assertIs(reached, pose)
        self.assertEqual(
            robot.config["motion"]["joint_reached_tolerance_rad"],
            0.005,
        )
        self.assertEqual(robot._kinematics.config, {})

    def test_verticalization_failure_still_rejects_non_high_pose(self):
        floor = 0.456
        pose = Pose((0.60, 0.0, floor + 0.35), (1.0, 0.0, 0.0, 0.0))
        robot = _FakeRobot(pose)

        with patch(
            "steadyhand.vega_camera_clear._reachable_verticalize_pose",
            side_effect=RuntimeError("IK did not converge"),
        ):
            with self.assertRaisesRegex(RuntimeError, "IK did not converge"):
                move_camera_clear(robot, floor_m=floor, speed_scale=0.90)


if __name__ == "__main__":
    unittest.main()
