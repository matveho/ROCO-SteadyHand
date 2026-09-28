import unittest
from unittest import mock

from steadyhand.config import load_bundle
from steadyhand.models import Pose
from steadyhand.vega_presets import configured_right_preset, preset_max_delta
from steadyhand.vega_camera_clear import move_camera_clear_for_image


class VegaPresetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_bundle("vega")["robot"]

    def test_measured_ready_and_camera_clear_presets_are_valid(self):
        ready_q, ready_pose = configured_right_preset(self.config, "right_ready")
        clear_q, clear_pose = configured_right_preset(self.config, "right_camera_clear")
        self.assertEqual(len(ready_q), 7)
        self.assertEqual(len(clear_q), 7)
        self.assertGreater(ready_pose.position_m[2], 0.45)
        self.assertGreater(clear_pose.position_m[2], 0.45)
        self.assertLess(preset_max_delta(ready_q, clear_q), 1.0)

    def test_bad_joint_order_is_rejected(self):
        config = dict(self.config)
        config["joint_presets"] = dict(self.config["joint_presets"])
        broken = dict(config["joint_presets"]["right_ready"])
        broken["joint_names"] = list(reversed(broken["joint_names"]))
        config["joint_presets"]["right_ready"] = broken
        with self.assertRaisesRegex(ValueError, "joint_names"):
            configured_right_preset(config, "right_ready")

    def test_camera_clear_uses_measured_joints_without_ik(self):
        ready_q, ready_pose = configured_right_preset(self.config, "right_ready")
        clear_q, clear_pose = configured_right_preset(self.config, "right_camera_clear")

        class FakeRobot:
            def __init__(self):
                self.config = self_config
                self.q = ready_q
                self.pose = ready_pose
                self._kinematics = object()

            def get_tcp_pose(self):
                return self.pose

            def _read_joint_positions(self):
                return self.q

            def move_joints(self, target, *, speed_scale):
                self.q = tuple(target)
                self.pose = clear_pose

        self_config = self.config
        robot = FakeRobot()
        with mock.patch("steadyhand.vega_camera_clear.move_tcp_segmented") as move_tcp:
            reached = move_camera_clear_for_image(robot, floor_m=0.456, speed_scale=0.45)
        self.assertEqual(robot.q, clear_q)
        self.assertEqual(reached, clear_pose)
        move_tcp.assert_not_called()


if __name__ == "__main__":
    unittest.main()
