import math
from types import SimpleNamespace
import unittest
from unittest import mock

from steadyhand.executor import preflight_tcp_segmented, move_tcp_segmented
from steadyhand.kinematics import IKError
from steadyhand.models import Pose
from steadyhand.operator_input import clean_choice
from steadyhand.adapters.vega import VegaAdapter
from tools import vega_competition_pipeline as pipeline


def pose(x, y=0., z=.6):
    return Pose((x, y, z), (1., 0., 0., 0.))


class PreviewMotionTests(unittest.TestCase):
    def test_long_transfer_preflights_local_deltas_not_full_seed_jump(self):
        calls = []
        def solve(target, seed):
            answer = [target.position_m[0] * 10] + [0.] * 6
            if abs(answer[0] - seed[0]) > 1.5:
                raise IKError("exceeding max_seed_delta_rad=1.500")
            calls.append(target)
            return answer
        start, end = pose(.2), pose(.6)
        kin = SimpleNamespace(solve=solve)
        seed = [2.] + [0.] * 6
        with self.assertRaises(IKError):
            kin.solve(end, seed)
        steps = dict(max_translation_step_m=.02, max_orientation_step_rad=.1)
        result = preflight_tcp_segmented(kin, seed, start, end, **steps)
        self.assertAlmostEqual(result[0], 6.)
        robot = SimpleNamespace(get_tcp_pose=lambda: start, move_tcp=mock.Mock())
        move_tcp_segmented(robot, end, speed_scale=.16, **steps)
        self.assertEqual(calls, [call.args[0] for call in robot.move_tcp.call_args_list])

    def test_menu_accepts_plain_and_bracketed_paste_on_first_entry(self):
        names = {"board.center": pose(.4), "board.bottom_left": pose(.2)}
        for text in ("2", " 2\r", "\x1b[200~2\x1b[201~", "\ufeff2\u200b"):
            with self.subTest(text=text), mock.patch('builtins.input', return_value=text) as read:
                self.assertEqual(pipeline._prompt_next_location("board.center", names), "board.bottom_left")
                read.assert_called_once()
        self.assertEqual(clean_choice("garbage 2"), "garbage 2")

    def test_preview_mode_is_explicit_with_100mm_default_and_40mm_opt_in(self):
        for choice, height, center in (("", "100", False), ("2", "100", True),
                                       ("3", "40", False), ("4", "40", True)):
            with mock.patch.object(pipeline, "_choose", return_value=["gear_60teeth"]), \
                 mock.patch('builtins.input', return_value=choice), \
                 mock.patch('tools.vega_head_target_preview.main', return_value=0) as run:
                self.assertEqual(pipeline._run_head_preview_menu(SimpleNamespace(check_only=False, remote_safe=True, speed_scale=.38)), 0)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('--hover-clearance-mm') + 1], height)
            self.assertEqual('--center' in command, center)
            self.assertEqual(command[command.index('--speed-scale') + 1], '0.38')

    def test_blocked_segment_reprompts_without_motion_or_limit_bypass(self):
        robot = SimpleNamespace(
            connect=mock.Mock(), close=mock.Mock(), move_joints=mock.Mock(),
            _read_joint_positions=lambda: [0.] * 7, get_tcp_pose=lambda: pose(.4),
            _kinematics=SimpleNamespace(solve=mock.Mock(side_effect=IKError('exceeding max_seed_delta_rad=1.500'))),
            move_tcp=mock.Mock(),
        )
        with mock.patch.object(pipeline, 'VegaAdapter', return_value=robot), \
             mock.patch.object(pipeline, '_capture_downward_head_frame'), \
             mock.patch.object(pipeline, 'configured_right_preset', return_value=([0.]*7, pose(.4))), \
             mock.patch('builtins.input', return_value='e') as read:
            result = pipeline._run_motion_targets({'far': pose(.6)}, {'robot': {}},
                confirm_physical=True, check_only=False, speed_scale=.16, interactive_next=True)
        self.assertEqual(result, 0)
        robot.move_tcp.assert_not_called()
        read.assert_called_once()

    def adapter(self, *, error=.020113, tcp_error=.002, velocity=0., stale=False, stopped=False):
        # No SDK, clocks or CAN: advancing state plus known FK error.
        adapter = VegaAdapter.__new__(VegaAdapter)
        adapter.config = {"motion": {"joint_timeout_s": .5, "joint_reached_tolerance_rad": .02}}
        adapter._read_joint_positions = lambda: (error,) + (0.,) * 6
        stamp = [10]
        def timestamp():
            if not stale:
                stamp[0] += 1
            return stamp[0]
        adapter._state_timestamp = timestamp
        adapter._arm = SimpleNamespace(get_joint_vel=lambda: (velocity,) * 7)
        adapter._read_estop_status = lambda: {"software_estop_enabled": stopped, "button_pressed": False}
        adapter._kinematics = SimpleNamespace(forward=lambda values: pose(tcp_error if values[0] else 0.))
        return adapter

    def wait(self, adapter, **kw):
        clock = iter(i*.02 for i in range(200))
        with mock.patch('steadyhand.adapters.vega.time.monotonic', side_effect=lambda: next(clock)), \
             mock.patch('steadyhand.adapters.vega.time.sleep'):
            return adapter._wait_for_joint_state(target=(0.,)*7, newer_than=1, **kw)

    def test_tiny_joint_boundary_requires_finished_stationary_fresh_cartesian_match(self):
        self.assertAlmostEqual(self.wait(self.adapter(), motion_finished=True)[0], .020113)
        for params in ({'error': .0211}, {'tcp_error': .009}, {'velocity': .01},
                       {'stale': True}, {'stopped': True}):
            with self.subTest(params=params), self.assertRaisesRegex(RuntimeError, 'timeout'):
                self.wait(self.adapter(**params), motion_finished=True)
        with self.assertRaisesRegex(RuntimeError, 'timeout'):
            self.wait(self.adapter(), motion_finished=False)

    def test_head_stall_or_nonfinite_head_never_captures_board(self):
        for measured in ([1.55, -.83, 0.], [math.nan, 0., 0.]):
            head = SimpleNamespace(
                get_joint_pos=lambda: measured,
                move_to_joint_pos=mock.Mock(return_value=SimpleNamespace(wait=lambda timeout: 'failed')),
                set_joint_pos=mock.Mock(),
            )
            robot = SimpleNamespace(_robot=SimpleNamespace(head=head))
            with mock.patch.object(pipeline, 'move_camera_clear_for_image'), \
                 mock.patch.object(pipeline, 'VegaHeadCamera') as camera:
                with self.assertRaisesRegex(RuntimeError, 'head did not follow'):
                    pipeline._capture_downward_head_frame(robot, floor_m=.4, bundle={})
                camera.assert_not_called()


if __name__ == '__main__':
    unittest.main()
