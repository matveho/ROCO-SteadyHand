"""Pull wrist images from Vega without copying robot-side paths by hand.

Legacy interactive mode is preserved: run this file with no arguments and
paste any remote path. The preferred mode is ``--watch --open`` on the
Windows laptop while the robot teaching session is running. The wrist tool
publishes a stable manifest/image pair and this helper downloads each new
capture with the part and teaching stage in the local filename.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


DEFAULT_HOST = "192.168.50.20"
DEFAULT_USER = "dexmate"
DEFAULT_REMOTE_DIR = "/home/dexmate/ROCO-SteadyHand-live/runs/wrist_live"
DEFAULT_LOCAL_DIR = Path.home() / "roco_wrist_images"


def _safe_name(value, fallback):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    return text or fallback


def _scp(host, user, remote_path, destination):
    source = f"{user}@{host}:{remote_path}"
    result = subprocess.run(
        ["scp", "-o", "ConnectTimeout=8", source, str(destination)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return False, result.stderr.strip() or f"scp exited {result.returncode}"
    return True, ""


def _open_image(path):
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    opener = shutil.which("xdg-open")
    if opener:
        subprocess.Popen([opener, str(path)])


def _manifest_key(manifest):
    return (
        manifest.get("created_at_utc"),
        manifest.get("capture_index"),
        manifest.get("source_run"),
    )


def pull_latest(args, *, state=None):
    """Pull one new stable capture; return (new_state, local_image) or state."""
    args.local_dir.mkdir(parents=True, exist_ok=True)
    temp_manifest = args.local_dir / ".wrist_live_manifest.json"
    ok, error = _scp(
        args.host,
        args.user,
        f"{args.remote_dir}/latest_wrist_a.json",
        temp_manifest,
    )
    if not ok:
        if args.quiet_missing:
            return state, None
        print(f"WAITING FOR WRIST IMAGE: {error}", flush=True)
        return state, None
    try:
        manifest = json.loads(temp_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"WRIST MANIFEST INVALID: {exc}", flush=True)
        return state, None
    key = _manifest_key(manifest)
    if state == key:
        return state, None

    part = _safe_name(manifest.get("part"), "unknown_part")
    stage = _safe_name(manifest.get("stage"), "wrist")
    index = _safe_name(manifest.get("capture_index"), "0")
    destination = args.local_dir / f"{part}_{stage}_{index}.png"
    ok, error = _scp(
        args.host,
        args.user,
        f"{args.remote_dir}/latest_wrist_a.png",
        destination,
    )
    if not ok:
        print(f"WRIST IMAGE COPY FAILED: {error}", flush=True)
        return state, None

    local_manifest = destination.with_suffix(".json")
    local_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"WRIST IMAGE READY: part={part} stage={stage}\n"
        f"  local={destination}\n"
        f"  robot_source={manifest.get('source_run', 'unknown')}",
        flush=True,
    )
    if args.open:
        _open_image(destination)
    return key, destination


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="watch the stable wrist image until Ctrl+C")
    parser.add_argument("--latest", action="store_true", help="pull the current stable wrist image once")
    parser.add_argument("--open", action="store_true", help="open each downloaded image with the default image app")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--interval", type=float, default=1.0)
    return parser.parse_args(argv)


def _interactive(args):
    """Retain the original arbitrary-path transfer loop."""
    args.local_dir.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            remote_path = input("Dex path (or 'latest', 'quit'): ").strip().strip('"').strip("'")
            if remote_path.lower() in ("q", "quit", "exit"):
                return 0
            if remote_path.lower() == "latest":
                args.latest = True
                args.quiet_missing = False
                pull_latest(args)
                args.latest = False
                continue
            if not remote_path:
                continue
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination = args.local_dir / f"wrist_{timestamp}.png"
            ok, error = _scp(args.host, args.user, remote_path, destination)
            if ok:
                print(f"OK: {destination}")
                if args.open:
                    _open_image(destination)
            else:
                print(f"SCP FAILED: {error}")
        except KeyboardInterrupt:
            print("\nExiting.")
            return 0


def main(argv=None):
    args = _parse_args(argv)
    args.quiet_missing = bool(args.watch)
    if not args.watch and not args.latest:
        return _interactive(args)
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    if args.latest:
        pull_latest(args)
        return 0
    state = None
    try:
        while True:
            state, _ = pull_latest(args, state=state)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nWrist image watcher stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
