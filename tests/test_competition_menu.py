import copy
import io
import json
from pathlib import Path
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import vega_competition_pipeline as pipeline


class CompetitionMenuTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(check_only=False, speed_scale=.38, remote_safe=False)
        self.profiles = {'parts': {
            'battery_size1': {'grasp_verified': True, 'place_verified': True,
                              'place': {'clearance_m': .04}, 'place_cv': {'enabled': True}},
            'bolt_8mm': {'grasp_verified': True, 'place_verified': False},
            'usb_a': {'grasp_verified': False, 'place_verified': True, 'place': {'clearance_m': .02}},
        }}
        self.patch = mock.patch.object(pipeline, '_menu_profiles', return_value=self.profiles)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.output = io.StringIO()
        self.redirect = redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def run_menu(self, choices, check_only=False):
        with mock.patch('builtins.input', side_effect=choices):
            return pipeline.main(['--check-only'] if check_only else
                ['--confirm-head-motion', '--confirm-physical-motion'])

    def test_every_top_level_choice_dispatches_to_existing_correct_function(self):
        for choice, function in (
            ('1', '_configured_competition_run'), ('2', '_run_task_tests_menu'),
            ('3', '_run_wrist_calibration_menu'), ('4', '_run_placement_calibration_menu'),
            ('5', '_show_menu_readiness'), ('6', '_run_advanced_tools_menu'),
        ):
            with self.subTest(choice=choice), mock.patch.object(pipeline, function, return_value=0) as run:
                self.assertEqual(self.run_menu([choice, '0']), 0)
                run.assert_called_once()
                self.assertEqual(run.call_args.kwargs, {})

    def test_every_advanced_choice_and_back(self):
        for choice, function, kwargs in (
            ('1', '_recalibrate', {}), ('2', '_run_position_testing_menu', {}),
            ('3', '_run_head_preview_menu', {}),
            ('4', '_all_calibrated_competition_run', {'place_cv': True}),
            ('5', '_all_calibrated_competition_run', {'place_cv': False}),
            ('6', '_reload_operator_settings', {}), ('7', '_run_rollback_menu', {}),
        ):
            with self.subTest(choice=choice), mock.patch.object(pipeline, function, return_value=0) as run:
                self.assertEqual(self.run_menu(['6', choice, '0', '0']), 0)
                run.assert_called_once()
                self.assertEqual(run.call_args.kwargs, kwargs)
        self.assertEqual(self.run_menu(['6', '0', '0']), 0)

    def test_position_testing_still_calls_existing_motion_workflow(self):
        with mock.patch.object(pipeline, '_run_motion_targets', return_value=0) as run:
            self.assertEqual(self.run_menu(['6', '2', '0', '0']), 0)
        run.assert_called_once()
        self.assertTrue(run.call_args.kwargs['interactive_next'])
        self.assertTrue(run.call_args.kwargs['prompt_after_capture'])

    def test_part_tests_dispatch_selected_parts_to_competition_without_action_prompt(self):
        with mock.patch.object(pipeline, '_configured_competition_run', return_value=0) as run:
            self.assertEqual(self.run_menu(['2', 'battery_size1,bolt_8mm', '0']), 0)
            run.assert_called_once()
            self.assertEqual(run.call_args.kwargs, {'selected_parts': ['battery_size1', 'bolt_8mm']})

    def test_only_verified_pickups_listed(self):
        with mock.patch.object(pipeline, '_configured_competition_run', return_value=0):
            self.assertEqual(self.run_menu(['2', 'bolt_8mm', '0']), 0)
        self.assertNotIn('usb_a', self.output.getvalue())
        self.assertNotIn('gear_60teeth', self.output.getvalue())
        with mock.patch.object(pipeline, '_configured_competition_run') as run:
            self.assertEqual(self.run_menu(['2', 'usb_a', '0']), 0)
            run.assert_not_called()

    def test_selected_competition_uses_config_and_no_prompts_or_unselected_parts(self):
        settings = pipeline._load_competition_actions()
        settings['pipeline_speed_scale'] = .31
        settings['parts']['battery_size1'].update(enabled=False, pick_enabled=False)
        settings['parts']['bolt_8mm'].update(mode='pick_place', use_wrist_pick_cv=False)
        self.profiles['parts']['bolt_8mm']['place_verified'] = True
        before = copy.deepcopy(settings)
        commands = []
        def action(command):
            commands.append(command)
            output = Path(command[command.index('--output') + 1])
            output.mkdir(parents=True)
            mode = command[command.index('--action') + 1]
            (output / 'run_summary.json').write_text(json.dumps({
                'status': 'completed' if mode == 'pick_place' else 'pick_complete_returned',
                'holding_may_be_true': False,
            }))
            return 0
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(pipeline, 'ROOT', Path(td)), \
                mock.patch.object(pipeline, 'load_profiles', return_value=self.profiles), \
                mock.patch.object(pipeline, '_load_competition_actions', return_value=settings), \
                mock.patch('builtins.input', side_effect=AssertionError('execution prompted')), \
                mock.patch.object(pipeline, 'run_wrist_part_calibration', side_effect=action):
            # Full runner, command builder, and durable progress path are real.
            self.assertEqual(pipeline._configured_competition_run(
                self.args, selected_parts=['bolt_8mm', 'battery_size1']), 0)
            self.assertEqual([(c[c.index('--part') + 1], c[c.index('--action') + 1])
                             for c in commands], [('battery_size1', 'pick_place'), ('bolt_8mm', 'pick')])
            for command in commands:
                self.assertIn('--competition', command)
                self.assertEqual(command[command.index('--speed-scale') + 1], '0.31')
                self.assertIn('--execution-offsets-json', command)
            self.assertIn('--place-cv', commands[0])
            self.assertIn('--no-cv', commands[1])
            self.assertNotIn('--place-cv', commands[1])
            progress = json.loads((Path(td) / 'runs/competition_progress.json').read_text())
            self.assertEqual(progress['status'], 'finished')
        self.assertEqual(settings, before)

    def test_selected_competition_preserves_retry_settings_and_hard_stops(self):
        settings = pipeline._load_competition_actions()
        settings['parts']['battery_size1']['max_attempts'] = 2
        settings['head_reacquire_on_failure'] = False
        settings['retry_without_wrist_cv'] = False
        for result, count in ((-1, 2), (2, 1), (3, 1)):
            with self.subTest(result=result), \
                    mock.patch.object(pipeline, 'load_profiles', return_value=self.profiles), \
                    mock.patch.object(pipeline, '_load_competition_actions', return_value=settings), \
                    mock.patch.object(pipeline, '_start_competition_progress', return_value=True), \
                    mock.patch('builtins.input', side_effect=AssertionError('execution prompted')), \
                    mock.patch.object(pipeline, '_run_competition_action', side_effect=[result, 0]) as run:
                code = pipeline._configured_competition_run(
                    self.args, selected_parts=['battery_size1', 'bolt_8mm'])
                self.assertEqual(code, 0 if result == -1 else result)
                self.assertEqual(run.call_count, count)
                self.assertEqual(run.call_args_list[0].kwargs, {
                    'retries': 1, 'no_cv': False, 'place_cv': True,
                    'head_reacquire': False, 'retry_without_cv': False,
                })

    def test_full_competition_still_respects_disabled_parts(self):
        settings = pipeline._load_competition_actions()
        for entry in settings['parts'].values():
            entry['enabled'] = False
        with mock.patch.object(pipeline, 'load_profiles', return_value=self.profiles), \
                mock.patch.object(pipeline, '_load_competition_actions', return_value=settings), \
                mock.patch.object(pipeline, '_run_competition_action') as run:
            self.assertEqual(pipeline._configured_competition_run(self.args), 2)
            run.assert_not_called()

    def test_placement_submenu_calls_separate_existing_workflows(self):
        for choice, function in (('1', '_run_drop_calibration_menu'), ('2', '_run_place_cv_menu')):
            with self.subTest(choice=choice), mock.patch.object(pipeline, function, return_value=0) as run:
                self.assertEqual(self.run_menu(['4', 'battery_size1', choice, '0', '0']), 0)
                run.assert_called_once()
                self.assertEqual(run.call_args.kwargs, {'part': 'battery_size1'})
        self.assertEqual(self.run_menu(['4', '0', '0']), 0)
        self.assertEqual(self.run_menu(['4', 'battery_size1', '0', '0']), 0)

    def test_placement_cv_unavailable_until_physical_placement_verified(self):
        with mock.patch.object(pipeline, '_run_place_cv_menu') as cv:
            self.assertEqual(self.run_menu(['4', 'bolt_8mm', '2', '0', '0']), 0)
            cv.assert_not_called()
        self.assertIn('Placement CV unavailable', self.output.getvalue())

    def test_physical_release_teaching_unlocks_cv_in_same_submenu(self):
        def teach(args, part=None):
            self.profiles['parts'][part].update(place={'clearance_m': .04}, place_verified=True)
            return 0
        with mock.patch.object(pipeline, '_run_drop_calibration_menu', side_effect=teach), \
                mock.patch.object(pipeline, '_run_place_cv_menu', return_value=0) as cv:
            self.assertEqual(self.run_menu(['4', 'bolt_8mm', '1', '2', '0', '0']), 0)
            self.assertEqual(cv.call_args.kwargs, {'part': 'bolt_8mm'})

    def test_calibration_dispatch_preserves_existing_commands_without_second_picker(self):
        for function, mode in ((pipeline._run_drop_calibration_menu, 'drop'), (pipeline._run_place_cv_menu, 'place-cv')):
            with self.subTest(mode=mode), mock.patch('builtins.input', side_effect=AssertionError('second picker')), \
                    mock.patch.object(pipeline, 'run_wrist_part_calibration', return_value=0) as run:
                function(self.args, part='battery_size1')
                run.assert_called_once_with(['--part', 'battery_size1', '--mode', mode,
                    '--confirm-head-motion', '--confirm-physical-motion', '--speed-scale', '0.38'])
        with mock.patch.object(pipeline, 'run_wrist_part_calibration', return_value=0) as run:
            self.assertEqual(self.run_menu(['3', 'bolt_8mm', '0']), 0)
            self.assertEqual(run.call_args.args[0][:4], ['--part', 'bolt_8mm', '--mode', 'calibrate'])

    def test_readiness_is_read_only_without_robot_or_board_initialization(self):
        before = copy.deepcopy(self.profiles)
        with mock.patch.object(pipeline, '_load_runtime', side_effect=AssertionError('board runtime')), \
                mock.patch.object(pipeline, 'VegaAdapter', side_effect=AssertionError('robot')), \
                mock.patch.object(pipeline, 'run_wrist_part_calibration', side_effect=AssertionError('motion')):
            self.assertEqual(self.run_menu(['5', '0'], check_only=True), 0)
        self.assertEqual(self.profiles, before)
        text = self.output.getvalue()
        for part in pipeline.PART_NAMES:
            self.assertIn(part, text)
        for label in ('Pickup verified', 'Placement verified', 'Placement CV', 'Competition enabled', 'saved/enabled', 'not taught'):
            self.assertIn(label, text)

    def test_back_no_profiles_invalid_choices_and_holding_do_not_launch_extra_actions(self):
        self.assertEqual(self.run_menu(['2', '0', '0']), 0)
        self.assertEqual(self.run_menu(['4', 'battery_size1', 'invalid', '0', '0']), 0)
        with mock.patch.object(pipeline, '_run_task_tests_menu', return_value=3):
            self.assertEqual(self.run_menu(['2']), 3)
        self.profiles['parts'] = {}
        self.assertEqual(self.run_menu(['2', '4', '0']), 0)
        self.assertIn('No parts have verified pickup', self.output.getvalue())

    def test_advanced_check_only_does_not_launch_calibration_and_rollback_only_lists(self):
        with mock.patch.object(pipeline, '_recalibrate') as run:
            self.assertEqual(self.run_menu(['6', '1', '0', '0'], check_only=True), 0)
            run.assert_not_called()
        with mock.patch('tools.vega_version_menu.main', return_value=0) as run:
            pipeline._run_rollback_menu(SimpleNamespace(check_only=False))
            run.assert_called_once_with(['--confirm-head-motion', '--confirm-physical-motion'])
            run.reset_mock()
            pipeline._run_rollback_menu(SimpleNamespace(check_only=True))
            run.assert_called_once_with(['--list'])

    def test_direct_cli_calibration_test_competition_flags_still_dispatch(self):
        for flag, mode in (('--wrist-calibrate', 'calibrate'), ('--drop-calibrate', 'drop'), ('--place-cv-calibrate', 'place-cv')):
            with self.subTest(flag=flag), mock.patch.object(pipeline, 'run_wrist_part_calibration', return_value=0) as run:
                self.assertEqual(pipeline.main([flag, 'battery_size1', '--confirm-physical-motion']), 0)
                self.assertEqual(run.call_args.args[0][:4], ['--part', 'battery_size1', '--mode', mode])
        with mock.patch.object(pipeline, '_configured_competition_run', return_value=0) as run:
            self.assertEqual(pipeline.main(['--competition-run', '--check-only']), 0)
            run.assert_called_once()
        with mock.patch.object(pipeline, '_task_test_command', return_value=['unchanged']), \
                mock.patch.object(pipeline, 'run_wrist_part_calibration', return_value=0) as run:
            self.assertEqual(pipeline.main(['--task-test', 'battery_size1.pick', '--confirm-physical-motion']), 0)
            run.assert_called_once_with(['unchanged'])


if __name__ == '__main__':
    unittest.main()
