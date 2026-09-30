import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from tools import vega_collect_status as collector


class CollectorTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'robot'
        self.output = Path(temp.name) / 'evidence.zip'
        self.write('tools/vega_competition_pipeline.py', b'# placeholder')
        self.write('configs/competition_actions.json', {'parts': {'battery_size1': {'enabled': True}}})
        self.write('competition_offsets.json', {'pickup': {'forward_mm': 2}})
        self.write('calibration/place_templates/battery.png', b'original template bytes')
        self.write('calibration/wrist_part_profiles.json', {'parts': {'battery_size1': {
            'grasp_verified': True, 'place_verified': True,
            'place_cv': {'enabled': True, 'template': 'calibration/place_templates/battery.png',
                         'reference_image': 'runs/old/reference.png'}}}})
        self.write('runs/old/reference.png', b'old original reference')
        self.patch = mock.patch.object(collector, 'command', return_value={
            'returncode': 0, 'stdout': 'test', 'stderr': ''})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return path

    def bundle(self, **kwargs):
        report = collector.collect(self.root, self.output, **kwargs)
        with zipfile.ZipFile(self.output) as archive:
            files = {name.removeprefix(collector.PREFIX): archive.read(name) for name in archive.namelist()}
        return report, files

    def test_includes_original_images_templates_inputs_failed_and_unfinished_runs(self):
        self.write('runs/place/events.jsonl', b'{"event":"place_cv_low_confidence"}\n')
        self.write('runs/place/run_summary.json', {'part': 'battery_size1', 'action': 'place_cv',
            'status': 'failed', 'holding_may_be_true': True, 'last_error': 'lost feature'})
        self.write('runs/place/000_wrist_a.png', b'exact original pixels')
        self.write('runs/active/events.jsonl', b'{"event":"board_reference"}\n')
        before = {p: p.read_bytes() for p in collector.files_under(self.root)}
        report, files = self.bundle()
        self.assertFalse(report['hardware_accessed'])
        self.assertEqual(files['runs/place/000_wrist_a.png'], b'exact original pixels')
        self.assertEqual(files['runs/old/reference.png'], b'old original reference')
        self.assertIn('calibration/place_templates/battery.png', files)
        self.assertIn('competition_offsets.json', files)
        self.assertTrue(report['latest_by_part']['battery_size1']['place_cv']['holding_may_be_true'])
        self.assertTrue(any(r['status'] == 'no_final_summary' for r in report['recent_runs']))
        self.assertEqual(before, {p: p.read_bytes() for p in collector.files_under(self.root)})
        manifest = json.loads(files['manifest.json'])
        for name, metadata in manifest.items():
            self.assertEqual(metadata['sha256'], collector.digest(files[name]))

    def test_latest_per_part_metadata_survives_full_image_window(self):
        a = self.write('runs/older/run_summary.json', {'part': 'gear_20teeth', 'action': 'pick', 'status': 'failed'})
        self.write('runs/older/events.jsonl', b'{"reason":"unreachable"}\n')
        self.write('runs/older/old.png', b'old photo')
        b = self.write('runs/new/run_summary.json', {'part': 'battery_size1', 'action': 'pick', 'status': 'completed'})
        os.utime(a, (1, 1))
        os.utime(a.parent / 'events.jsonl', (1, 1))
        os.utime(b, (2, 2))
        report, files = self.bundle(recent_runs=1)
        self.assertIn('runs/older/run_summary.json', files)
        self.assertIn('runs/older/events.jsonl', files)
        self.assertNotIn('runs/older/old.png', files)
        self.assertIn('gear_20teeth', report['latest_by_part'])

    def test_explicit_run_outside_window_is_collected_and_omissions_are_visible(self):
        self.write('runs/specific/raw.dat', b'a' * 3000)
        self.write('runs/specific/evidence.png', b'image')
        with mock.patch.object(collector, 'FILE_LIMIT', 2048):
            report, files = self.bundle(include_runs=['runs/specific'])
        self.assertIn('runs/specific/evidence.png', files)
        self.assertNotIn('runs/specific/raw.dat', files)
        self.assertTrue(report['omitted_files'])
        self.assertFalse(report['evidence_complete_within_selection'])
        with self.assertRaisesRegex(ValueError, 'under'):
            self.bundle(include_runs=['../'])

    def test_bad_json_missing_references_and_failed_diagnostics_still_produce_bundle(self):
        self.write('calibration/broken.json', b'{broken')
        (self.root / 'runs/old/reference.png').unlink()
        with mock.patch.object(collector, 'command', return_value={
                'returncode': 2, 'stdout': '', 'stderr': 'invalid config'}):
            report, files = self.bundle()
        self.assertIn('calibration/broken.json', files)
        self.assertEqual(report['diagnostics']['competition_check']['returncode'], 2)
        self.assertIn('runs/old/reference.png', report['missing_references'])
        self.assertTrue(report['warnings'])

    def test_symlinks_and_external_profile_references_are_not_exported(self):
        secret = self.root.parent / 'unrelated.txt'
        secret.write_text('do not export')
        (self.root / 'calibration/external.txt').symlink_to(secret)
        self.write('calibration/wrist_part_profiles.json', {'parts': {}, 'reference': str(secret)})
        report, files = self.bundle()
        self.assertNotIn('calibration/external.txt', files)
        self.assertIn(str(secret), report['missing_references'])
        self.assertFalse(any(b'do not export' == value for value in files.values()))

    def test_commands_are_explicit_no_motion_and_tests_are_not_run(self):
        with mock.patch.object(collector, 'command', return_value={
                'returncode': 0, 'stdout': '', 'stderr': ''}) as run:
            self.bundle()
        commands = [c.args[1] for c in run.call_args_list]
        self.assertIn([collector.sys.executable, 'tools/vega_preflight.py', '--json'], commands)
        self.assertIn([collector.sys.executable, 'tools/vega_competition_pipeline.py', '--check-only', '--competition-run'], commands)
        readiness = next(c for c in commands if 'tools/vega_competition_readiness.py' in c)
        self.assertIn('--skip-tests', readiness)
        self.assertTrue(all('--confirm-physical-motion' not in c for c in commands))


if __name__ == '__main__':
    unittest.main()
