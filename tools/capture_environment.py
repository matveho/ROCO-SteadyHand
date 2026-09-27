"""Capture reproducibility information without contacting robot hardware.

Writes JSON to stdout. Redirect it into the active session folder:
    python tools/capture_environment.py > runs/.../environment.json
"""

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


PACKAGES = (
    "dexcontrol",
    "dexcomm",
    "numpy",
    "opencv-python",
    "torch",
    "lerobot",
)


def command(args):
    try:
        result = subprocess.run(
            args,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
        return {"returncode": result.returncode, "output": result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": type(exc).__name__, "detail": str(exc)}


def package_versions():
    out = {}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def main():
    safe_env = {
        key: os.environ.get(key)
        for key in ("ROBOT_NAME", "ROBOT_IP")
        if os.environ.get(key)
    }
    snapshot = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": package_versions(),
        "safe_environment": safe_env,
        "git_head": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "git_branch": command(["git", "branch", "--show-current"]),
    }
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
