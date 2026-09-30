import unittest
from types import SimpleNamespace
from unittest import mock

from steadyhand.models import Pose
from steadyhand.executor import move_tcp_segmented
from steadyhand.remote_motion import needs_low_clearance_confirmation
from tools.vega_wrist_part_calibrate import PartSession


def pose(z):
    return Pose((.45, -.1, z), (1., 0., 0., 0.))


class RemoteClearanceTests(unittest.TestCase):
    def test_threshold_checks_both_measured_start_and_next_target(self):
        plane = lambda x, y: .5
        for start, end, expected in ((.6, .54, False), (.54, .54, False),
                                      (.6, .539, True), (.539, .6, True)):
            self.assertEqual(needs_low_clearance_confirmation(pose(start), pose(end), plane), expected)
        self.assertTrue(needs_low_clearance_confirmation(pose(.6), pose(.6), lambda x, y: float('nan')))

    def session(self, z):
        s = PartSession.__new__(PartSession)
        s.remote_safe = True
        s.remote_checkpoint_index = 0
        s.robot = SimpleNamespace(get_tcp_pose=lambda: pose(z), _read_joint_positions=lambda: [0.] * 7)
        s.surface = lambda x, y: .5
        s.event = mock.Mock()
        return s

    def test_high_checkpoints_and_head_do_not_prompt(self):
        with mock.patch('builtins.input', side_effect=AssertionError('unexpected prompt')):
            self.session(.6).remote_checkpoint('before_coarse_hover', capture=False)
            self.session(.52).remote_checkpoint('before_head_down', capture=False)

    def test_low_target_prompts_before_motion_even_from_high_pose(self):
        session = self.session(.6)
        with mock.patch('builtins.input', return_value='abort') as prompt:
            with self.assertRaises(KeyboardInterrupt):
                session.remote_checkpoint('before_low_waypoint', capture=False, target=pose(.53))
            prompt.assert_called_once()

    def test_waypoint_guard_aborts_before_first_low_command(self):
        robot = SimpleNamespace(pose=pose(.56), moves=[])
        robot.get_tcp_pose = lambda: robot.pose
        def move(p, speed_scale):
            robot.moves.append(p)
            robot.pose = p
        robot.move_tcp = move
        def guard(p):
            if needs_low_clearance_confirmation(robot.pose, p, lambda x, y: .5):
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            move_tcp_segmented(robot, pose(.52), speed_scale=.1,
                max_translation_step_m=.008, max_orientation_step_rad=.1, waypoint_guard=guard)
        self.assertTrue(robot.moves)
        self.assertTrue(all(p.position_m[2] >= .54 for p in robot.moves))


if __name__ == '__main__':
    unittest.main()
