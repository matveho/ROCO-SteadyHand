import math
import unittest
from unittest import mock

from steadyhand.models import Pose
from tools.vega_board_manual_calibrate import (
    DEFAULT_FORWARD_RISE_ANGLE_DEG,
    ORIENTATION_TEACH_MIN_ABOVE_FLOOR_M,
    RIGHT_READY_MIN_ABOVE_FLOOR_M,
    _board_parallel_jog_delta,
    _capture_current_right_ready,
    _coarse_target_preserving_orientation,
    _predicted_y_reference,
    _rotate_quaternion_in_base,
)


class BoardManualCalibrationTests(unittest.TestCase):
    def test_orientation_teach_clearance_matches_camera_clear_fallback(self):
        self.assertAlmostEqual(ORIENTATION_TEACH_MIN_ABOVE_FLOOR_M, 0.30)
        self.assertAlmostEqual(RIGHT_READY_MIN_ABOVE_FLOOR_M, 0.10)

    def test_manual_right_ready_records_live_joints_and_tip_pose(self):
        class FakeRobot:
            def _read_joint_positions(self):
                return (-2.1, -0.1, 0.2, -1.4, 0.3, 0.7, -0.2)

            def get_tcp_pose(self):
                return Pose((0.40, -0.20, 0.90), (0.5, 0.5, -0.5, 0.5))

        with mock.patch("builtins.input", return_value=""):
            joints, pose = _capture_current_right_ready(FakeRobot(), floor=0.456)

        self.assertEqual(joints, (-2.1, -0.1, 0.2, -1.4, 0.3, 0.7, -0.2))
        self.assertEqual(pose.position_m, (0.40, -0.20, 0.90))

    def test_manual_right_ready_rejects_low_pose(self):
        class FakeRobot:
            def _read_joint_positions(self):
                return (0.0,) * 7

            def get_tcp_pose(self):
                return Pose((0.40, -0.20, 0.50), (1.0, 0.0, 0.0, 0.0))

        with mock.patch("builtins.input", return_value=""):
            with self.assertRaisesRegex(RuntimeError, "below"):
                _capture_current_right_ready(FakeRobot(), floor=0.456)

    def test_coarse_target_preserves_taught_orientation(self):
        current = Pose(
            (0.72, -0.18, 1.00),
            (0.61, 0.12, -0.31, 0.72),
        )
        target = _coarse_target_preserving_orientation(
            (0.435, 0.104, 0.456),
            0.550,
            current,
        )
        self.assertEqual(target.position_m, (0.435, 0.104, 0.55))
        self.assertEqual(target.quaternion_wxyz, current.quaternion_wxyz)

    def test_unvisited_y_reference_uses_corrected_center_and_head_axis(self):
        center = Pose((0.48, -0.03, 0.55), (1.0, 0.0, 0.0, 0.0))
        y = _predicted_y_reference(center, (0.0, 1.0, 0.0), 0.1)
        self.assertEqual(y.position_m, (0.48, 0.07, 0.55))
        self.assertEqual(y.quaternion_wxyz, center.quaternion_wxyz)

    def test_base_axis_rotation_is_normalized_and_changes_orientation(self):
        q = _rotate_quaternion_in_base((1.0, 0.0, 0.0, 0.0), "pitch", 10.0)
        self.assertAlmostEqual(sum(v*v for v in q), 1.0, places=12)
        self.assertGreater(abs(q[2]), 0.0)

    def test_default_plane_angle_uses_latest_0_85_over_386(self):
        expected = math.degrees(math.atan((85.0 - 0.0) / 386.0))
        self.assertAlmostEqual(DEFAULT_FORWARD_RISE_ANGLE_DEG, expected, places=9)

    def test_forward_rise_compensation_moves_tcp_down(self):
        dx, dy, dz = _board_parallel_jog_delta("forward", 50.0, 10.0)
        self.assertAlmostEqual(dx, 0.050)
        self.assertAlmostEqual(dy, 0.0)
        self.assertAlmostEqual(dz, -0.050 * math.tan(math.radians(10.0)))

    def test_back_compensation_is_exact_inverse(self):
        forward = _board_parallel_jog_delta("forward", 20.0, 10.0)
        back = _board_parallel_jog_delta("back", 20.0, 10.0)
        for a, b in zip(forward, back):
            self.assertAlmostEqual(a, -b)

    def test_left_right_have_no_unmeasured_z_compensation(self):
        self.assertEqual(_board_parallel_jog_delta("left", 10.0, 10.0), (0.0, 0.010, 0.0))
        self.assertEqual(_board_parallel_jog_delta("right", 10.0, 10.0), (0.0, -0.010, 0.0))


if __name__ == "__main__":
    unittest.main()
