import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from steadyhand.models import Pose
from steadyhand.executor import move_tcp_segmented
from steadyhand.remote_motion import needs_low_clearance_confirmation
from tools.vega_wrist_part_calibrate import PartSession
from tools import vega_wrist_part_calibrate as wrist
from tools import vega_competition_pipeline as pipeline


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

    def moving_session(self, *, remote=True):
        s = self.session(.6)
        s.remote_safe = remote
        s.args = SimpleNamespace(speed_scale=.38)
        s.robot.pose = pose(.6)
        s.robot.get_tcp_pose = lambda: s.robot.pose
        s.robot._kinematics = SimpleNamespace(solve=mock.Mock(side_effect=lambda p, seed: seed))
        s.moves = []
        def move(p, speed_scale):
            s.moves.append((p, speed_scale))
            s.robot.pose = p
        s.robot.move_tcp = move
        def frame(label):
            # The inspection image must precede the first descent command.
            self.assertEqual(s.moves, [])
            return None, wrist.ROOT / 'runs' / 'before_lowering.png'
        s.frame = mock.Mock(side_effect=frame)
        return s

    def test_remote_descent_keeps_normal_speed_and_path_with_one_before_image(self):
        for slow in (False, True):
            with self.subTest(slow=slow):
                normal = self.moving_session(remote=False)
                remote = self.moving_session()
                normal.args.speed_scale = remote.args.speed_scale = .60
                normal.move(pose(.508), slow=slow)
                with mock.patch('builtins.input', return_value='') as prompt:
                    remote.move(pose(.508), slow=slow)
                self.assertEqual(remote.moves, normal.moves)
                self.assertEqual(remote.robot._kinematics.solve.call_args_list,
                                 normal.robot._kinematics.solve.call_args_list)
                self.assertEqual(len(remote.moves), 5)
                self.assertTrue(all(speed == (.42 if slow else .60) for _, speed in remote.moves))
                remote.frame.assert_called_once_with('remote checkpoint before_lowering')
                normal.frame.assert_not_called()
                self.assertGreater(prompt.call_count, 0)

    def test_board_parallel_move_and_stage_checkpoints_take_no_extra_photos(self):
        s = self.moving_session()
        s.surface = lambda x, y: .5 - .2 * (x - .45)
        target = Pose((.55, -.1, .58), (1., 0., 0., 0.))
        with mock.patch('builtins.input', side_effect=AssertionError('unexpected prompt')):
            s.remote_checkpoint('before_coarse_hover')
            s.move(target)
            s.remote_checkpoint('coarse_hover')
        s.frame.assert_not_called()

    def test_abort_before_lowering_captures_once_without_motion(self):
        s = self.moving_session()
        with mock.patch('builtins.input', return_value='abort'):
            with self.assertRaises(KeyboardInterrupt):
                s.move(pose(.508), slow=True)
        self.assertEqual(s.moves, [])
        s.frame.assert_called_once()

    def test_pipeline_and_teaching_cli_do_not_cap_remote_speed(self):
        with mock.patch.object(pipeline, 'run_wrist_part_calibration', return_value=0) as run:
            self.assertEqual(pipeline.main(['--wrist-calibrate', 'battery_size1', '--remote-safe',
                                          '--speed-scale', '.38', '--confirm-physical-motion']), 0)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('--speed-scale') + 1], '0.38')
        cfg = {'working_arm': 'right', 'kinematics': {'ee_frame': 'tip_r'},
               'motion': {'joint_reached_tolerance_rad': .005}}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(wrist, 'load_bundle', return_value={'robot': cfg}), \
             mock.patch.object(wrist, 'load_profiles', return_value={'parts': {}}), \
             mock.patch.object(wrist, 'PartSession') as session:
            session.return_value.teach.return_value = 0
            self.assertEqual(wrist.main(command + ['--output', str(Path(directory) / 'out')]), 0)
            self.assertEqual(session.call_args.args[0].speed_scale, .38)

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
