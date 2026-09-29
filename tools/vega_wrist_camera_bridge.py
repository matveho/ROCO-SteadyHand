"""System-Python helper for the Conda wrist-camera bridge.

The Jetson's vendor binding imports GObject from system Python 3.10 while the
robot/IK stack runs in Conda Python 3.13.  This process owns the vendor camera
context and exchanges fresh NumPy frames through request directories.
"""

import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from wrist_cameras import WristCameras


def _write_observation(obs, root, request):
    directory = root / f"request_{request:06d}"
    directory.mkdir(parents=True, exist_ok=False)
    result = {"ok": True, "directory": str(directory)}
    for label in ("wrist_a", "wrist_b"):
        record = obs[label]
        np.save(directory / f"{label}.npy", np.asarray(record["rgb"]), allow_pickle=False)
        result[label] = {
            "frame_id": record.get("frame_id"),
            "timestamp_ns": record.get("timestamp_ns"),
            "received_monotonic_ns": record.get("received_monotonic_ns"),
        }
    return result


def main():
    root = Path(os.environ.get("VEGA_WRIST_BRIDGE_DIR", "/tmp/vega_wrist_bridge"))
    root.mkdir(parents=True, exist_ok=True)
    manager = WristCameras()
    cameras = manager.__enter__()

    def restart_manager():
        nonlocal manager, cameras
        try:
            manager.__exit__(None, None, None)
        except Exception:
            pass
        time.sleep(0.25)
        manager = WristCameras()
        cameras = manager.__enter__()

    try:
        print(json.dumps({"ready": True}), flush=True)
        for raw in sys.stdin:
            request = json.loads(raw)
            if request.get("op") == "close":
                return 0
            if request.get("op") != "read":
                raise ValueError(f"unknown wrist bridge operation: {request.get('op')!r}")
            try:
                kwargs = {
                    "timeout": float(request.get("timeout_s", 3.0)),
                    "fresh": bool(request.get("fresh", True)),
                }
                try:
                    obs = cameras.get_obs(**kwargs)
                except Exception as first_exc:
                    if "trigger worker exited" not in str(first_exc).lower():
                        raise
                    # The vendor worker can die once when the arm has just
                    # settled. Recreate the vendor context exactly once; the
                    # parent control process remains alive and no blind motion
                    # is issued by this recovery.
                    restart_manager()
                    obs = cameras.get_obs(**kwargs)
                print(json.dumps(_write_observation(obs, root, int(request["request"]))), flush=True)
            except Exception as exc:
                print(json.dumps({
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }), flush=True)
    finally:
        try:
            manager.__exit__(None, None, None)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
