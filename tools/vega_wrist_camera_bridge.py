"""System-Python helper for the Conda wrist-camera bridge.

The Jetson's vendor binding imports GObject from system Python 3.10 while the
robot/IK stack runs in Conda Python 3.13.  This process owns the vendor camera
context and exchanges fresh NumPy frames through request directories.
"""

import json
import os
from pathlib import Path
import sys
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
    with WristCameras() as cameras:
        print(json.dumps({"ready": True}), flush=True)
        for raw in sys.stdin:
            request = json.loads(raw)
            if request.get("op") == "close":
                return 0
            if request.get("op") != "read":
                raise ValueError(f"unknown wrist bridge operation: {request.get('op')!r}")
            try:
                obs = cameras.get_obs(
                    timeout=float(request.get("timeout_s", 3.0)),
                    fresh=bool(request.get("fresh", True)),
                )
                print(json.dumps(_write_observation(obs, root, int(request["request"]))), flush=True)
            except Exception as exc:
                print(json.dumps({
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
