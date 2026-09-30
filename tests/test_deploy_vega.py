"""Exercise the actual robot deployment shell against disposable Git repositories."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/deploy_vega_remote.sh'


@unittest.skipUnless(all(shutil.which(x) for x in ('git', 'bash', 'flock')), 'requires Git, Bash, flock')
class FastDeployTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.live = self.root / 'live'
        self.source.mkdir()
        self.git(self.source, 'init', '-q', '-b', 'main')
        self.git(self.source, 'config', 'user.name', 'Deploy Test')
        self.git(self.source, 'config', 'user.email', 'deploy@example.invalid')
        self.write(self.source / 'calibration/wrist_part_profiles.json', '{"old": true}\n')
        self.write(self.source / 'code.py', 'old code\n')
        self.write(self.source / 'obsolete.py', 'old unused code\n')
        self.write(self.source / 'configs/competition_actions.json', '{}\n')
        self.commit('initial')
        self.before = self.git(self.source, 'rev-parse', 'HEAD')
        self.git(self.root, 'clone', '-q', str(self.source), str(self.live))
        self.write(self.live / 'calibration/wrist_part_profiles.json', '{"onsite": true}\n')
        self.write(self.live / 'calibration/wrist_templates/current.png', 'actual image bytes')
        self.write(self.live / 'runs/important/events.jsonl', 'irreplaceable run\n')
        self.write(self.live / 'code.py', 'onsite code to overwrite\n')
        self.write(self.source / 'code.py', 'new code\n')
        (self.source / 'obsolete.py').unlink()
        self.write(self.source / 'calibration/wrist_part_profiles.json', '{"repo": true}\n')
        self.write(self.source / 'calibration/new_default.json', '{}\n')
        self.commit('update')
        self.expected = self.git(self.source, 'rev-parse', 'HEAD')
        self.bundle = self.root / 'update.bundle'
        self.git(self.source, 'bundle', 'create', str(self.bundle), 'main', '^' + self.before)
        self.settings = self.root / 'settings'
        self.settings.mkdir()
        for name in ('competition_actions.json', 'competition_plan.json', 'competition_offsets.json'):
            self.write(self.settings / name, json.dumps({'laptop': name}) + '\n')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.pgrep = self.bin / 'pgrep'
        self.write(self.pgrep, '#!/bin/sh\nexit 1\n')
        self.pgrep.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'])

    @staticmethod
    def write(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    @staticmethod
    def git(cwd, *args):
        return subprocess.check_output(['git', '-C', str(cwd), *args], text=True,
                                       stderr=subprocess.PIPE).strip()

    def commit(self, message):
        self.git(self.source, 'add', '.')
        self.git(self.source, 'commit', '-qm', message)

    def deploy(self, bundle=None, expected=None, preflight='0'):
        return subprocess.run(['bash', str(SCRIPT), str(self.live),
                               str(self.bundle) if bundle is None else str(bundle),
                               expected or self.expected, str(self.settings), preflight],
                              env=self.env, text=True, capture_output=True, timeout=15)

    def assert_runtime_kept(self):
        self.assertEqual((self.live / 'calibration/wrist_part_profiles.json').read_text(), '{"onsite": true}\n')
        self.assertEqual((self.live / 'calibration/wrist_templates/current.png').read_text(), 'actual image bytes')
        self.assertEqual((self.live / 'runs/important/events.jsonl').read_text(), 'irreplaceable run\n')

    def test_incremental_overwrite_preserves_runtime_installs_laptop_settings_without_archives(self):
        result = self.deploy()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.git(self.live, 'rev-parse', 'HEAD'), self.expected)
        self.assertEqual((self.live / 'code.py').read_text(), 'new code\n')
        self.assertFalse((self.live / 'obsolete.py').exists())
        self.assertTrue((self.live / 'calibration/new_default.json').exists())
        self.assert_runtime_kept()
        self.assertEqual((self.live / 'competition_offsets.json').read_bytes(),
                         (self.settings / 'competition_offsets.json').read_bytes())
        for name in ('competition_actions.json', 'competition_plan.json'):
            self.assertEqual((self.live / 'configs' / name).read_bytes(), (self.settings / name).read_bytes())
        self.assertFalse(list(self.root.glob('live.calibration.*')))
        self.assertEqual({p.name for p in self.root.iterdir() if p.is_dir()},
                         {'source', 'live', 'settings', 'bin'})

    def test_same_commit_refresh_needs_no_bundle(self):
        self.assertEqual(self.deploy().returncode, 0)
        self.write(self.live / 'code.py', 'dirty again\n')
        self.bundle.unlink()
        result = self.deploy(bundle='-')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.live / 'code.py').read_text(), 'new code\n')
        self.assert_runtime_kept()

    def test_initial_install_with_full_bundle(self):
        self.live = self.root / 'fresh'
        full = self.root / 'full.bundle'
        self.git(self.source, 'bundle', 'create', str(full), 'main')
        result = self.deploy(bundle=full)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.git(self.live, 'rev-parse', 'HEAD'), self.expected)
        self.assertEqual((self.live / 'calibration/wrist_part_profiles.json').read_text(), '{"repo": true}\n')

    def test_running_robot_refuses_before_changing_checkout(self):
        self.write(self.pgrep, '#!/bin/sh\necho "robot running"\nexit 0\n')
        result = self.deploy()
        self.assertEqual(result.returncode, 8)
        self.assertEqual(self.git(self.live, 'rev-parse', 'HEAD'), self.before)
        self.assert_runtime_kept()

    def test_wrong_bundle_commit_refuses_before_overwrite(self):
        result = self.deploy(expected='f' * 40)
        self.assertEqual(result.returncode, 4)
        self.assertEqual((self.live / 'code.py').read_text(), 'onsite code to overwrite\n')
        self.assert_runtime_kept()

    def test_checkout_failure_restores_calibration(self):
        git = shutil.which('git')
        wrapper = self.bin / 'git'
        self.write(wrapper, '#!/bin/bash\n'
                   'if [[ " $* " == *" checkout "* ]]; then\n'
                   '  rm -f "$2/calibration/wrist_part_profiles.json"\n'
                   '  exit 7\nfi\n'
                   f'exec "{git}" "$@"\n')
        wrapper.chmod(0o755)
        result = self.deploy()
        self.assertEqual(result.returncode, 7)
        self.assert_runtime_kept()

    def test_preflight_is_opt_in(self):
        self.write(self.source / 'tools/vega_preflight.py', 'raise SystemExit(19)\n')
        self.commit('failing optional preflight')
        self.expected = self.git(self.source, 'rev-parse', 'HEAD')
        self.git(self.source, 'bundle', 'create', str(self.bundle), 'main', '^' + self.before)
        self.assertEqual(self.deploy().returncode, 0)
        result = self.deploy(bundle='-', preflight='1')
        self.assertEqual(result.returncode, 10)
        self.assertIn('code was updated in place', result.stderr)
        self.assert_runtime_kept()


if __name__ == '__main__':
    unittest.main()
