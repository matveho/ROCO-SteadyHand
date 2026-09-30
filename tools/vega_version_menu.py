"""List or launch an archived Vega checkout without deleting the live one.

The deploy script keeps complete rollback copies in the sibling
``ROCO-SteadyHand-versions`` directory.  This helper is intentionally small
and standard-library-only so it remains usable when the robot runtime is
partly unavailable.  Launching an archive replaces this process with that
archive's competition menu; it does not copy files or modify calibration.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def versions_root(root: Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    override = os.environ.get("VEGA_VERSION_ROOT")
    return Path(override) if override else Path(__file__).resolve().parents[1].parent / "ROCO-SteadyHand-versions"


def list_versions(root: Path | None = None) -> list[Path]:
    base = versions_root(root)
    if not base.is_dir():
        return []
    return sorted(
        (path for path in base.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: path.name,
        reverse=True,
    )


def _valid_archive(path: Path) -> bool:
    return (path / "tools" / "vega_competition_pipeline.py").is_file() and (path / ".git").is_dir()


def launch(version: Path, argv: list[str] | None = None) -> None:
    if not _valid_archive(version):
        raise ValueError(f"archive is missing a competition menu or Git metadata: {version}")
    command = [sys.executable, str(version / "tools" / "vega_competition_pipeline.py")]
    command.extend(argv or [])
    print(f"LAUNCHING ARCHIVED VEGA VERSION: {version}", flush=True)
    os.execv(sys.executable, command)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print archived versions and exit")
    parser.add_argument("--launch", metavar="VERSION", help="archive directory name to launch")
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    args, remainder = parser.parse_known_args(argv)
    available = list_versions(args.root)
    if args.list:
        for path in available:
            print(path.name)
        return 0
    if args.launch:
        selected = versions_root(args.root) / args.launch
        launch(selected, remainder)
        return 0
    if not available:
        print(f"No archived Vega versions found in {versions_root(args.root)}", file=sys.stderr)
        return 2
    print("ARCHIVED VEGA VERSIONS")
    for index, path in enumerate(available, 1):
        print(f"  {index}. {path.name}")
    answer = input("Choose a version number, or 0 to cancel: ").strip()
    if answer in ("", "0", "q", "quit", "exit"):
        return 0
    try:
        selected = available[int(answer) - 1]
    except (ValueError, IndexError):
        print("Unknown archived version", file=sys.stderr)
        return 2
    launch(selected, remainder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
