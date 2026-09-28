"""Aim the Vega head down and capture one RGB board-development snapshot.

This intentionally moves ONLY the head. It does not connect arms or grippers.
The live head-camera publisher must already be running.

Outputs:
  head_left_rgb.npy
  head_right_rgb.npy
  metadata.json

Use tools/identify_board_snapshot.py on the saved directory for offline tuning.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.cameras.vega import VegaHeadCamera
from steadyhand.config import load_bundle


def _serializable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head-j1", type=float, default=0.55,
                   help="onsite-verified positive value looks down")
    p.add_argument("--settle-s", type=float, default=0.6)
    p.add_argument("--output")
    p.add_argument("--confirm-head-motion", action="store_true")
    args = p.parse_args(argv)

    if not args.confirm_head_motion:
        p.error("head motion requires --confirm-head-motion")

    import numpy as np
    from dexcontrol.robot import Robot

    cfg = load_bundle("vega")["robot"]
    os.environ.setdefault("ROBOT_NAME", cfg["robot_name"])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(args.output or f"board_view_{stamp}")
    output.mkdir(parents=True, exist_ok=False)

    robot = Robot()
    camera = None
    try:
        q = np.asarray(robot.head.get_joint_pos(), dtype=float)
        q[0] = float(args.head_j1)
        robot.head.set_joint_pos(
            q,
            wait_time=1.2,
            exit_on_reach=True,
            exit_on_reach_kwargs={"tolerance": 0.02},
        )
        time.sleep(float(args.settle_s))
        q_actual = np.asarray(robot.head.get_joint_pos(), dtype=float)

        camera = VegaHeadCamera()
        camera.connect()
        frame = camera.read(include_depth=False)

        np.save(output / "head_left_rgb.npy", frame.left_rgb)
        np.save(output / "head_right_rgb.npy", frame.right_rgb)
        metadata = {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "robot_name": cfg["robot_name"],
            "head_q_rad": q_actual.tolist(),
            "head_j1_downward_sign_verified_onsite": "positive",
            "camera_info": _serializable(frame.camera_info),
            "left_shape": list(frame.left_rgb.shape),
            "right_shape": list(frame.right_rgb.shape),
            "left_timestamp_ns": frame.left_timestamp_ns,
            "right_timestamp_ns": frame.right_timestamp_ns,
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        print("HEAD Q =", q_actual.tolist(), flush=True)
        print("WROTE", output.resolve(), flush=True)
        return 0
    finally:
        if camera is not None:
            camera.close()
        robot.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
