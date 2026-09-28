import unittest
from unittest.mock import patch

from steadyhand.models import Pose
from steadyhand.vega_camera_clear import (
    CAMERA_CLEAR_RETAIN_HIGH_POSE_ABOVE_FLOOR_M,
    _can_retain_high_pose,
    move_camera_clear,
    move_camera_clear_for_image,
    move_verticalize_after_image,
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
    def test_image_clear_preserves_orientation(self):
        floor = 0.456
        start = Pose((0.49, -0.06, 0.73), (0.91, 0.10, -0.20, 0.35))
        robot = _FakeRobot(start)

        planned = Pose((0.49, -0.06, 1.056), start.quaternion_wxyz)

        def fake_move(_robot, target, **kwargs):
            robot.pose = target

        with patch(
            "steadyhand.vega_camera_clear._reachable_image_clear_pose",
            return_value=planned,
        ), patch(
            "steadyhand.vega_camera_clear.move_tcp_segmented",
            side_effect=fake_move,
        ):
            reached = move_camera_clear_for_image(
                robot, floor_m=floor, speed_scale=0.90
            )

        self.assertEqual(reached.quaternion_wxyz, start.quaternion_wxyz)
        self.assertEqual(reached.position_m, planned.position_m)

    def test_post_image_verticalization_runs_restored_two_stage_sequence(self):
        floor = 0.456
        start = Pose((0.36, 0.0, 1.056), (1.0, 0.0, 0.0, 0.0))
        robot = _FakeRobot(start)
        verticalize = Pose(
            (0.50, 0.0, 0.906),
            (2 ** -0.5, 0.0, 0.0, -(2 ** -0.5)),
        )
        preset = Pose((0.35, 0.0, 1.056), verticalize.quaternion_wxyz)
        calls = []

        def fake_move(_robot, target, **kwargs):
            calls.append(target)
            robot.pose = target

        with patch(
            "steadyhand.vega_camera_clear._reachable_verticalize_pose",
            return_value=verticalize,
        ), patch(
            "steadyhand.vega_camera_clear._highest_reachable_camera_clear_pose",
            return_value=preset,
        ), patch(
            "steadyhand.vega_camera_clear.move_tcp_segmented",
            side_effect=fake_move,
        ):
            reached = move_verticalize_after_image(
                robot, floor_m=floor, speed_scale=0.90
            )

        self.assertEqual(calls, [verticalize, preset])
        self.assertIs(reached, preset)

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

    def test_escape_lift_refreshes_pose_before_verticalization_fallback(self):
        floor = 0.4560002716867571
        start = Pose((0.4933, -0.0631, 0.7273), (1.0, 0.0, 0.0, 0.0))
        robot = _FakeRobot(start)

        def fake_move(_robot, target, **kwargs):
            robot.pose = target

        with patch(
            "steadyhand.vega_camera_clear._ik_feasible",
            return_value=True,
        ), patch(
            "steadyhand.vega_camera_clear.move_tcp_segmented",
            side_effect=fake_move,
        ), patch(
            "steadyhand.vega_camera_clear._reachable_verticalize_pose",
            side_effect=RuntimeError("IK did not converge"),
        ):
            reached = move_camera_clear(robot, floor_m=floor, speed_scale=0.90)

        self.assertAlmostEqual(reached.position_m[2], start.position_m[2] + 0.08)
        self.assertGreaterEqual(
            reached.position_m[2],
            floor + CAMERA_CLEAR_RETAIN_HIGH_POSE_ABOVE_FLOOR_M,
        )

    def test_verticalization_failure_still_rejects_low_pose_if_escape_infeasible(self):
        floor = 0.456
        pose = Pose((0.60, 0.0, floor + 0.20), (1.0, 0.0, 0.0, 0.0))
        robot = _FakeRobot(pose)

        with patch(
            "steadyhand.vega_camera_clear._ik_feasible",
            return_value=False,
        ), patch(
            "steadyhand.vega_camera_clear._reachable_verticalize_pose",
            side_effect=RuntimeError("IK did not converge"),
        ):
            with self.assertRaisesRegex(RuntimeError, "IK did not converge"):
                move_camera_clear(robot, floor_m=floor, speed_scale=0.90)


if __name__ == "__main__":
    unittest.main()
