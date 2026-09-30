import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from steadyhand.competition_progress import CompetitionProgress


class ProgressTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / 'progress.json'
        self.plan = [('battery_size1', 'pick_place'), ('gear_20teeth', 'pick')]

    def start(self, **kwargs):
        return CompetitionProgress(self.path, 'configured', self.plan, **kwargs)

    def test_stopped_run_preserves_completed_actions_and_attempt_budget(self):
        p = self.start()
        p.begin_attempt(*self.plan[0], self.path.parent / 'first')
        p.finish(*self.plan[0], 0)
        p.begin_attempt(*self.plan[1], self.path.parent / 'second')
        p.finish(*self.plan[1], 2)
        with self.assertRaisesRegex(ValueError, 'gripper is empty'):
            self.start()
        p = self.start(recovered_empty=True)
        self.assertEqual(p.entry(*self.plan[0])['status'], 'completed')
        self.assertEqual(p.entry(*self.plan[1])['status'], 'pending')
        self.assertEqual(p.entry(*self.plan[1])['attempts'], 1)

    def test_crash_after_successful_child_summary_does_not_repeat_pickup(self):
        p = self.start()
        output = self.path.parent / 'child'
        output.mkdir()
        p.begin_attempt(*self.plan[0], output)
        (output / 'run_summary.json').write_text(json.dumps({
            'status': 'completed', 'holding_may_be_true': False}))
        p = self.start()
        self.assertEqual(p.entry(*self.plan[0])['status'], 'completed')

    def test_unknown_or_held_crash_never_blindly_restarts(self):
        p = self.start()
        output = self.path.parent / 'child'
        output.mkdir()
        p.begin_attempt(*self.plan[0], output)
        for summary in ({}, {'status': 'completed', 'holding_may_be_true': True}):
            (output / 'run_summary.json').write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                self.start()
            with self.assertRaises(ValueError):
                self.start(new_run=True)

    def test_completed_run_archived_next_launch_and_safe_plan_changes_are_explicit(self):
        p = self.start()
        p.finish(*self.plan[0], 0)
        with self.assertRaisesRegex(ValueError, 'differs'):
            CompetitionProgress(self.path, 'all', self.plan)
        p.finish(*self.plan[1], -1)
        p = self.start()
        self.assertEqual(p.entry(*self.plan[0])['attempts'], 0)
        self.assertEqual(p.entry(*self.plan[0])['status'], 'pending')
        self.assertEqual(len(list(self.path.parent.glob('competition_progress_*.json'))), 1)

    def test_safe_failed_attempt_resumes_without_resetting_attempt_count(self):
        p = self.start()
        p.begin_attempt(*self.plan[0], self.path.parent / 'child')
        p.retry_pending(*self.plan[0], {'automatic_continuation_safe': True})
        p = self.start()
        self.assertEqual(p.entry(*self.plan[0])['attempts'], 1)

    def test_real_action_loop_resumes_next_part_without_repeating_completed_pick(self):
        from tools import vega_competition_pipeline as pipeline
        progress = self.start()
        args = SimpleNamespace(speed_scale=.38, competition_progress=progress)
        def run(command):
            output = Path(command[command.index('--output') + 1])
            output.mkdir(parents=True)
            part = command[command.index('--part') + 1]
            # The action was recorded BEFORE robot code was invoked.
            saved = json.loads(self.path.read_text())
            action = command[command.index('--action') + 1]
            self.assertEqual(saved['actions'][f'{part}.{action}']['status'], 'running')
            if part == 'gear_20teeth' and progress.entry(part, action)['attempts'] == 1:
                raise KeyboardInterrupt()
            status = 'completed' if action == 'pick_place' else 'pick_complete_returned'
            (output / 'run_summary.json').write_text(json.dumps({'status': status, 'holding_may_be_true': False}))
            return 0
        with mock.patch.object(pipeline, 'ROOT', self.path.parent), \
                mock.patch.object(pipeline, 'run_wrist_part_calibration', side_effect=run) as runner:
            self.assertEqual(pipeline._run_competition_action(args, *self.plan[0], retries=2), 0)
            with self.assertRaises(KeyboardInterrupt):
                pipeline._run_competition_action(args, *self.plan[1], retries=2)
            with self.assertRaises(ValueError):
                self.start()
            progress = self.start(recovered_empty=True)
            args.competition_progress = progress
            self.assertEqual(pipeline._run_competition_action(args, *self.plan[0], retries=2), 0)
            self.assertEqual(pipeline._run_competition_action(args, *self.plan[1], retries=2), 0)
            self.assertEqual(runner.call_count, 3)
            self.assertIn('--no-cv', runner.call_args.args[0])
        self.assertEqual(progress.data['status'], 'finished')

    def test_last_child_completed_before_crash_is_not_repeated_on_resume(self):
        p = self.start()
        p.finish(*self.plan[0], 0)
        output = self.path.parent / 'last_child'
        output.mkdir()
        p.begin_attempt(*self.plan[1], output)
        (output / 'run_summary.json').write_text(json.dumps({
            'status': 'pick_complete_returned', 'holding_may_be_true': False}))
        resumed = self.start()
        self.assertEqual(resumed.data['status'], 'finished')
        self.assertEqual(resumed.entry(*self.plan[1])['status'], 'completed')
        self.assertEqual(self.start().entry(*self.plan[1])['status'], 'pending')


if __name__ == '__main__':
    unittest.main()
