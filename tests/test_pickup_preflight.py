"""Pre-motion pickup checks and teaching recovery without robot hardware."""
import unittest
from types import SimpleNamespace
from unittest import mock

from steadyhand.models import Pose
from steadyhand.kinematics import IKError
from tools.vega_wrist_part_calibrate import PartSession, PickupPreflightError


class PickupPreflightTests(unittest.TestCase):
    def session(self, failure=None):
        s = PartSession.__new__(PartSession)
        s.robot = SimpleNamespace(pose=Pose((.4, -.1, .6), (1., 0., 0., 0.)))
        robot = s.robot
        robot.get_tcp_pose = lambda: robot.pose
        robot._read_joint_positions = lambda: [robot.pose.position_m[2]] + [0.] * 6
        def solve(pose, seed):
            z = pose.position_m[2]
            if abs(z - seed[0]) > .020001:
                raise IKError('oversized one-shot solve')
            if failure == 'descent' and z < .51:
                raise IKError('descent unreachable')
            if failure == 'lift' and z > seed[0]:
                raise IKError('lift unreachable')
            return [z] + [0.] * 6
        robot._kinematics = SimpleNamespace(solve=mock.Mock(side_effect=solve))
        def move(pose, speed_scale):
            robot.pose = pose
        robot.move_tcp = mock.Mock(side_effect=move)
        robot.connect_gripper = mock.Mock()
        robot.grip = mock.Mock()
        robot._gripper = SimpleNamespace(last_grip_result=lambda: {'gripped': True})
        s.surface = lambda x, y: .5
        s.part = 'gear_60teeth'
        s.goal, s.alignment_verified = (100, 100), True
        s.holding, s.remote_safe = False, False
        s.gripper_open_fraction = .25
        s.args = SimpleNamespace(speed_scale=.38)
        s.event = mock.Mock()
        s._set_gripper_fraction = mock.Mock()
        return s

    def test_grab_preflights_full_segmented_roundtrip_before_jaws_or_motion(self):
        s = self.session()
        result = s.grab(.005)
        self.assertTrue(result['gripped'])
        self.assertTrue(s.holding)
        self.assertEqual(s.robot.get_tcp_pose().position_m, (.4, -.1, .6))
        self.assertEqual(s.robot.move_tcp.call_count, 10)
        s.robot.grip.assert_called_once_with('gear_60teeth')

    def test_unreachable_descent_or_lift_never_issues_movement_or_jaw_command(self):
        for stage in ('descent', 'lift'):
            with self.subTest(stage=stage):
                s = self.session(stage)
                with self.assertRaisesRegex(PickupPreflightError, f'Pickup {stage} preflight failed: waypoint'):
                    s.grab(.005)
                self.assertFalse(s.holding)
                s.robot.move_tcp.assert_not_called()
                s.robot.connect_gripper.assert_not_called()
                s.robot.grip.assert_not_called()
                s._set_gripper_fraction.assert_not_called()
                self.assertEqual(s.event.call_args.args[1]['stage'], stage)

    def test_teaching_requires_explicit_retry_after_dry_run_rejection(self):
        s = self.session()
        s.grab = mock.Mock(side_effect=[PickupPreflightError('not converged'), {'gripped': True}])
        with mock.patch('builtins.input', side_effect=['', 'retry']) as prompt:
            self.assertTrue(s._teaching_pickup({'grasp_clearance_m': .005}))
        self.assertEqual(s.grab.call_count, 2)
        self.assertEqual(prompt.call_count, 2)

    def test_operator_can_inspect_and_abort_without_retry(self):
        s = self.session()
        s.remote_safe = True
        s.frame = mock.Mock()
        s.grab = mock.Mock(side_effect=PickupPreflightError('not converged'))
        with mock.patch('builtins.input', side_effect=['image', 'abort']):
            self.assertFalse(s._teaching_pickup({'grasp_clearance_m': .005}))
        s.grab.assert_called_once()
        self.assertEqual(s.frame.call_count, 2)

    def test_motion_ik_or_hardware_failure_is_not_hidden_as_dry_run_rejection(self):
        for error in (IKError('motion solve failed'), RuntimeError('joint timeout')):
            with self.subTest(error=error):
                s = self.session()
                s.grab = mock.Mock(side_effect=error)
                with mock.patch('builtins.input', side_effect=AssertionError('unexpected retry prompt')):
                    with self.assertRaises(type(error)):
                        s._teaching_pickup({'grasp_clearance_m': .005})


if __name__ == '__main__':
    unittest.main()
