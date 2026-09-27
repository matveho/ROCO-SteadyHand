#!/usr/bin/env python3
"""Copy, patch and optionally test/build North without modifying the original.

This utility never starts or stops a robot process and never publishes commands.
An existing destination is rejected. A failed workspace is retained for review.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from build_candidate import build

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
SKIP = {".git", ".backups", "__pycache__", "logs", "log", "recordings"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(root, hashes):
    for relative, expected in hashes.items():
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError("SDK baseline mismatch: " + relative)


def check_paths(sdk, workspace):
    if not sdk.is_dir():
        raise ValueError("Original SDK directory does not exist")
    if workspace.exists():
        raise ValueError("Workspace already exists; choose a new directory")
    if workspace == sdk or sdk in workspace.parents or workspace in sdk.parents:
        raise ValueError("Workspace and original SDK must be separate trees")
    # Reject vendor source/binaries anywhere inside a Git checkout, including
    # through a symlink. Only the patch tooling belongs in this repository.
    for parent in (workspace, *workspace.parents):
        if parent == REPOSITORY or (parent / ".git").exists():
            raise ValueError("Workspace must be outside every Git checkout")


def ignored(directory, names):
    return {name for name in names if skip_name(name)}


def skip_name(name):
    return (name in SKIP or name.endswith('.log') or '.log.' in name or
            (name.startswith('motor_mem_dump_') and name.endswith('.json')))


def check_links(sdk):
    """Reject links that could send writes outside the copied SDK."""
    for directory, folders, files in os.walk(sdk, followlinks=False):
        folders[:] = [name for name in folders if not skip_name(name)]
        for name in folders + [name for name in files if not skip_name(name)]:
            path = Path(directory) / name
            if path.is_symlink():
                target = path.resolve(strict=True)
                if target != sdk and sdk not in target.parents:
                    raise ValueError("External SDK symlink needs review: " + str(path.relative_to(sdk)))
                if any(skip_name(part) for part in target.relative_to(sdk).parts):
                    raise ValueError("SDK symlink points into an excluded directory")


def copy_sdk(sdk, copied):
    # copy2 creates independent file contents; do not use hardlinks to live SDKs.
    shutil.copytree(sdk, copied, symlinks=True, ignore=ignored)
    for directory, folders, files in os.walk(copied, followlinks=False):
        for name in folders + files:
            path = Path(directory) / name
            if path.is_symlink() and os.path.isabs(os.readlink(path)):
                target = Path(os.readlink(path))
                relative = target.relative_to(sdk)
                path.unlink()
                path.symlink_to(os.path.relpath(copied / relative, path.parent))


def prepare(sdk, workspace, compile_plugin=False):
    sdk, workspace = sdk.resolve(), workspace.resolve()
    check_paths(sdk, workspace)
    baseline = json.loads((HERE / "baseline.json").read_text())
    verify(sdk, baseline["original_sha256"])
    check_links(sdk)
    workspace.mkdir(parents=True, exist_ok=False)
    copied = workspace / "sdk"
    copy_sdk(sdk, copied)
    verify(copied, baseline["original_sha256"])
    patch = HERE / "0001-fix-arm-index-and-validation.patch"
    command = ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch)]
    subprocess.run(command + ["--dry-run"], cwd=copied, check=True)
    subprocess.run(command, cwd=copied, check=True)
    verify(copied, baseline["patched_sha256"])
    receipt = {"status": "patched_sources_only", "baseline": baseline,
               "patch_sha256": digest(patch), "activation": "not_performed"}
    try:
        if compile_plugin:
            test = HERE / "test_action_extraction.py"
            source = copied / "src/plugins/north_controller.cpp"
            subprocess.run([sys.executable, "-B", str(test), "--sdk", str(copied),
                            "--source", str(source)], check=True)
            binary = build(copied, workspace / "build", sdk)
            # Inspect relocations without loading a controller or its constructors.
            with (workspace / "build/plugin-dynamic.txt").open("w") as log:
                subprocess.run(["readelf", "-d", str(binary)], stdout=log, check=True)
            receipt["candidate_plugin_sha256"] = digest(binary)
            receipt["status"] = "offline_tests_and_build_passed"
            # Keep the built plugin outside the SDK copy until runtime library and
            # configuration paths have been reviewed. No hidden deployment step.
    finally:
        verify(sdk, baseline["original_sha256"])
    (workspace / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Prepared:", workspace)
    print("Original SDK verified unchanged. No controller started or plugin installed.")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True, help="Unmodified original SDK")
    parser.add_argument("--workspace", type=Path, required=True,
                        help="New directory outside SDK and Git repositories")
    parser.add_argument("--build", action="store_true",
                        help="Run offline sanitizers and compile on the SDK build machine")
    args = parser.parse_args()
    try:
        prepare(args.sdk, args.workspace, args.build)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, str(error) + "\nNothing was activated; keep any partial workspace for review.\n")


if __name__ == "__main__":
    main()
