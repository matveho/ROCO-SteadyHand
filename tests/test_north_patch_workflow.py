"""Exercise patch-workspace boundaries without a vendor SDK or robot."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import shlex
import sys
import tempfile
import unittest
from unittest import mock


HELPERS = Path(__file__).resolve().parents[1] / "patches/sharpa-north"
sys.path.insert(0, str(HELPERS))
spec = importlib.util.spec_from_file_location("north_prepare_sdk", HELPERS / "prepare_sdk.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
sys.path.pop(0)


@unittest.skipUnless(shutil.which("patch"), "GNU patch is required")
class NorthWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sdk = self.root / "original sdk"
        self.workspace = self.root / "candidate workspace"
        self.package = self.root / "patch package"
        self.package.mkdir()
        self.relative = "src/plugins/north_controller.cpp"
        self.source = self.sdk / self.relative
        self.source.parent.mkdir(parents=True)
        self.source.write_text("original controller\n")
        (self.sdk / "include").mkdir()
        (self.sdk / "include/header.h").write_text("original header\n")
        self.original_hashes = {name: workflow.digest(self.sdk / name)
                                for name in (self.relative, "include/header.h")}
        changed = self.root / "expected.cpp"
        changed.write_text("patched controller\n")
        baseline = {"original_sha256": self.original_hashes,
                    "patched_sha256": {self.relative: workflow.digest(changed)}}
        (self.package / "baseline.json").write_text(json.dumps(baseline))
        (self.package / "0001-fix-arm-index-and-validation.patch").write_text(
            "--- a/" + self.relative + "\n+++ b/" + self.relative + "\n"
            "@@ -1 +1 @@\n-original controller\n+patched controller\n")
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(workflow, "HERE", self.package).start()

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return workflow.prepare(self.sdk, self.workspace)

    def test_copy_is_independent_and_does_not_start_or_build(self):
        with mock.patch.object(workflow, "build") as builder:
            receipt = self.prepare()
        builder.assert_not_called()
        self.assertEqual(receipt["activation"], "not_performed")
        self.assertEqual(receipt["status"], "patched_sources_only")
        copied = self.workspace / "sdk" / self.relative
        self.assertEqual(copied.read_text(), "patched controller\n")
        workflow.verify(self.sdk, self.original_hashes)
        self.assertNotEqual(copied.stat().st_ino, self.source.stat().st_ino)
        copied.write_text("further candidate edit\n")
        self.assertEqual(self.source.read_text(), "original controller\n")

    def test_wrong_baseline_does_not_create_workspace(self):
        self.source.write_text("unknown SDK revision\n")
        with self.assertRaisesRegex(ValueError, "baseline mismatch"):
            self.prepare()
        self.assertFalse(self.workspace.exists())

    def test_existing_workspace_is_preserved(self):
        self.workspace.mkdir()
        marker = self.workspace / "keep.txt"
        marker.write_text("keep me")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare()
        self.assertEqual(marker.read_text(), "keep me")

    def test_workspace_inside_original_is_rejected(self):
        self.workspace = self.sdk / "candidate"
        with self.assertRaisesRegex(ValueError, "separate trees"):
            self.prepare()
        self.assertFalse(self.workspace.exists())

    def test_workspace_inside_git_checkout_is_rejected(self):
        repo = self.root / "repo"
        repo.mkdir()
        (repo / ".git").write_text("gitdir: /elsewhere")
        self.workspace = repo / "untracked/candidate"
        with self.assertRaisesRegex(ValueError, "outside every Git"):
            self.prepare()
        self.assertFalse(self.workspace.exists())

    def test_symlink_destination_into_original_is_rejected(self):
        alias = self.root / "alias"
        alias.symlink_to(self.sdk, target_is_directory=True)
        self.workspace = alias / "candidate"
        with self.assertRaisesRegex(ValueError, "separate trees"):
            self.prepare()

    def test_external_sdk_symlink_is_rejected_before_copy(self):
        external = self.root / "external"
        external.write_text("do not touch")
        (self.sdk / "external").symlink_to(external)
        with self.assertRaisesRegex(ValueError, "External SDK symlink"):
            self.prepare()
        self.assertFalse(self.workspace.exists())
        self.assertEqual(external.read_text(), "do not touch")

    def test_internal_absolute_link_is_relocated(self):
        (self.sdk / "linked-header").symlink_to(self.sdk / "include/header.h")
        self.prepare()
        link = self.workspace / "sdk/linked-header"
        self.assertEqual(link.resolve(), self.workspace / "sdk/include/header.h")
        link.write_text("candidate header\n")
        self.assertEqual((self.sdk / "include/header.h").read_text(), "original header\n")

    def test_logs_and_backups_are_not_copied(self):
        for name in ("logs", ".backups", "recordings", ".git"):
            folder = self.sdk / name
            folder.mkdir()
            (folder / "unrelated.txt").write_text("not part of candidate")
        self.prepare()
        for name in ("logs", ".backups", "recordings", ".git"):
            self.assertFalse((self.workspace / "sdk" / name).exists())
            self.assertTrue((self.sdk / name / "unrelated.txt").exists())

    def test_bad_patch_never_changes_original_or_marks_success(self):
        (self.package / "0001-fix-arm-index-and-validation.patch").write_text(
            "--- a/" + self.relative + "\n+++ b/" + self.relative + "\n"
            "@@ -1 +1 @@\n-unexpected old value\n+replacement\n")
        with self.assertRaises(workflow.subprocess.CalledProcessError):
            self.prepare()
        workflow.verify(self.sdk, self.original_hashes)
        self.assertFalse((self.workspace / "receipt.json").exists())

    def test_build_failure_does_not_write_success_receipt(self):
        # Use the real patch calls, then fail the offline C++ test.
        real_run = workflow.subprocess.run
        with mock.patch.object(workflow.subprocess, "run", wraps=real_run) as run:
            def execute(command, **kwargs):
                if command[0] == sys.executable:
                    raise workflow.subprocess.CalledProcessError(1, command)
                return real_run(command, **kwargs)
            run.side_effect = execute
            with self.assertRaises(workflow.subprocess.CalledProcessError):
                workflow.prepare(self.sdk, self.workspace, compile_plugin=True)
        workflow.verify(self.sdk, self.original_hashes)
        self.assertFalse((self.workspace / "receipt.json").exists())


class BuildBoundaryTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("c++") and shutil.which("readelf"),
                         "C++ compiler and readelf are required")
    def test_compiler_and_linker_use_copy_even_with_original_cmake_paths(self):
        with tempfile.TemporaryDirectory(prefix="north fixture ") as root:
            original = Path(root) / "original SDK"
            copied = Path(root) / "copied SDK"
            for sdk in (original, copied):
                (sdk / "include").mkdir(parents=True)
            (original / "include/fixture.h").write_text('#error Used original SDK\n')
            (copied / "include/fixture.h").write_text('constexpr int fixture = 7;\n')
            sources = copied / "src/plugins"
            sources.mkdir(parents=True)
            (sources / "north_controller.cpp").write_text(
                '#include "fixture.h"\nint controller() { return fixture; }\n')
            (sources / "master_controller_plugin.cpp").write_text(
                'int controller();\nint plugin() { return controller(); }\n')
            settings = copied / "build/src/plugins/CMakeFiles/master_controller_plugin.dir"
            settings.mkdir(parents=True)
            (settings / "flags.make").write_text(
                'CXX_DEFINES = \nCXX_INCLUDES = ' + shlex.quote('-I' + str(original / 'include')) +
                '\nCXX_FLAGS = -fPIC\n')
            link = ['c++', '-shared', '-o', str(original / 'build/plugins/master_controller_plugin.so'),
                    'CMakeFiles/master_controller_plugin.dir/north_controller.cpp.o',
                    'CMakeFiles/master_controller_plugin.dir/master_controller_plugin.cpp.o',
                    '-Wl,-rpath,' + str(original / 'build/lib') + ':$ORIGIN']
            (settings / "link.txt").write_text(shlex.join(link))
            binary = workflow.build(copied, Path(root) / "output", original)
            dynamic = workflow.subprocess.run(['readelf', '-d', str(binary)],
                                              capture_output=True, text=True, check=True).stdout
            self.assertIn(str(copied / 'build/lib'), dynamic)
            self.assertNotIn(str(original), dynamic)
            self.assertIn('$ORIGIN', dynamic)
            self.assertFalse((original / 'build').exists())

    def test_relocation_preserves_argv_and_origin_token(self):
        original = Path("/old SDK")
        copied = Path("/new SDK")
        args = ["-I/old SDK/include", "/old SDK/build/library.so", r"-Wl,-rpath,\$ORIGIN"]
        self.assertEqual(workflow.build.__globals__["relocate"](args, original, copied),
                         ["-I/new SDK/include", "/new SDK/build/library.so", "-Wl,-rpath,$ORIGIN"])

    def test_builder_rejects_original_as_copy(self):
        with tempfile.TemporaryDirectory() as root:
            sdk = Path(root) / "sdk"
            with self.assertRaisesRegex(ValueError, "separate copy"):
                workflow.build(sdk, Path(root) / "build", sdk)

    def test_builder_rejects_output_in_original(self):
        with tempfile.TemporaryDirectory() as root:
            original = Path(root) / "original"
            with self.assertRaisesRegex(ValueError, "separate from both"):
                workflow.build(Path(root) / "copy", original / "output", original)


if __name__ == "__main__":
    unittest.main()
