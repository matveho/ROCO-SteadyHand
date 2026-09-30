import io
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import vega_competition_pipeline as pipeline


class CompetitionMenuTests(unittest.TestCase):
    def run_menu(self, choices):
        with mock.patch('builtins.input', side_effect=choices), redirect_stdout(io.StringIO()):
            return pipeline.main(['--confirm-head-motion', '--confirm-physical-motion'])

    def test_requested_numbers_dispatch_to_correct_workflows(self):
        for choice, function, kwargs in (
            ('1', '_recalibrate', {}),
            ('2', '_run_motion_targets', None),
            ('3', '_run_head_preview_menu', {}),
            ('4', '_all_calibrated_competition_run', {'place_cv': True}),
            ('5', '_all_calibrated_competition_run', {'place_cv': False}),
            ('6', '_configured_competition_run', {}),
            ('7', '_reload_operator_settings', {}),
            ('8', '_run_rollback_menu', {}),
            ('10', '_run_task_tests_menu', {}),
        ):
            with self.subTest(choice=choice), mock.patch.object(pipeline, function, return_value=0) as run:
                self.assertEqual(self.run_menu([choice, '9']), 0)
                run.assert_called_once()
                if kwargs is not None:
                    self.assertEqual(run.call_args.kwargs, kwargs)

    def test_reload_and_rollback_are_available_without_board_calibration(self):
        with mock.patch.object(pipeline, '_load_runtime', side_effect=AssertionError('board load')), \
                mock.patch.object(pipeline, '_reload_operator_settings', return_value=0), \
                mock.patch.object(pipeline, '_run_rollback_menu', return_value=0):
            self.assertEqual(self.run_menu(['7', '8', '9']), 0)

    def test_all_part_action_aliases_map_to_pick_or_pick_place(self):
        for part in pipeline.PART_NAMES:
            for verb, action in (('grab', 'pick'), ('place', 'pick_place')):
                with self.subTest(part=part, action=action), \
                        mock.patch('builtins.input', return_value=f'{verb}_{part}'), \
                        mock.patch.object(pipeline, '_task_test_command', return_value=['test']) as command, \
                        mock.patch.object(pipeline, 'run_wrist_part_calibration', return_value=0), \
                        redirect_stdout(io.StringIO()):
                    args = SimpleNamespace(check_only=False)
                    self.assertEqual(pipeline._run_task_tests_menu(args), 0)
                    command.assert_called_once_with(args, part, action)

    def test_held_part_stops_menu_before_another_action(self):
        with mock.patch.object(pipeline, '_run_task_tests_menu', return_value=3):
            self.assertEqual(self.run_menu(['10']), 3)

    def test_rollback_forwards_motion_authorization_and_check_only_lists(self):
        with mock.patch('tools.vega_version_menu.main', return_value=0) as run:
            pipeline._run_rollback_menu(SimpleNamespace(check_only=False))
            run.assert_called_once_with(['--confirm-head-motion', '--confirm-physical-motion'])
            run.reset_mock()
            pipeline._run_rollback_menu(SimpleNamespace(check_only=True))
            run.assert_called_once_with(['--list'])


if __name__ == '__main__':
    unittest.main()
