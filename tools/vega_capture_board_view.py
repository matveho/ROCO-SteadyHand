"""Aim the Vega head down and capture one RGB board-development snapshot.

This intentionally moves ONLY the head. It does not connect arms or grippers.
The live head-camera publisher must already be running.

Outputs:
  head_left_rgb.npy
  head_right_rgb.npy
  head_left_rgb.png
  head_right_rgb.png
  metadata.json

The camera subscription is opened before head motion. A baseline frame timestamp
is recorded, then capture waits for multiple strictly newer frames after the
head reaches its requested pose. This prevents saving a cached/stale Zenoh
observation as the new board view.

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
    p.add_argument("--fresh-frames", type=int, default=3,
                   help="require this many newer timestamped frames after head motion")
    p.add_argument("--fresh-timeout-s", type=float, default=8.0)
    p.add_argument("--output")
    p.add_argument("--confirm-head-motion", action="store_true")
    args = p.parse_args(argv)

    if not args.confirm_head_motion:
        p.error("head motion requires --confirm-head-motion")
    if not 1 <= int(args.fresh_frames) <= 30:
        p.error("--fresh-frames must be 1..30")
    if not 0.5 <= float(args.fresh_timeout_s) <= 30.0:
        p.error("--fresh-timeout-s must be 0.5..30")

    import numpy as np
    from dexcontrol.robot import Robot

    cfg = load_bundle("vega")["robot"]
    os.environ.setdefault("ROBOT_NAME", cfg["robot_name"])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(args.output or f"board_view_{stamp}")
    output.mkdir(parents=True, exist_ok=False)

    robot = None
    camera = VegaHeadCamera()
    try:
        # Subscribe before any head motion and remember the cached-stream
        # baseline. A post-motion capture is accepted only after timestamps
        # advance beyond this observation.
        camera.connect()
        baseline = camera.read(include_depth=False, timeout_s=15.0)
        baseline_left_ts = baseline.left_timestamp_ns
        baseline_right_ts = baseline.right_timestamp_ns
        if baseline_left_ts is None or baseline_right_ts is None:
            raise RuntimeError(
                "Head RGB timestamps are required for stale-frame rejection"
            )
        print(
            "BASELINE RGB TIMESTAMPS =",
            baseline_left_ts,
            baseline_right_ts,
            flush=True,
        )

        robot = Robot()
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

        # Require several distinct, newer frames after the head motion. Reading
        # the same cached record repeatedly does not count.
        deadline = time.monotonic() + float(args.fresh_timeout_s)
        frame = None
        accepted = 0
        last_left_ts = baseline_left_ts
        last_right_ts = baseline_right_ts
        while accepted < int(args.fresh_frames):
            candidate = camera.read(include_depth=False, timeout_s=1.0, poll_s=0.02)
            left_ts = candidate.left_timestamp_ns
            right_ts = candidate.right_timestamp_ns
            if (
                left_ts is not None
                and right_ts is not None
                and left_ts > last_left_ts
                and right_ts > last_right_ts
            ):
                frame = candidate
                last_left_ts = left_ts
                last_right_ts = right_ts
                accepted += 1
                print(
                    f"FRESH RGB FRAME {accepted}/{int(args.fresh_frames)} "
                    f"timestamps={left_ts},{right_ts}",
                    flush=True,
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Head RGB timestamps did not advance enough after head motion; "
                    "publisher/stream appears stale"
                )
            if accepted < int(args.fresh_frames):
                time.sleep(0.03)

        np.save(output / "head_left_rgb.npy", frame.left_rgb)
        np.save(output / "head_right_rgb.npy", frame.right_rgb)

        # Write viewable files directly for fast offload.
        import cv2
        cv2.imwrite(
            str(output / "head_left_rgb.png"),
            cv2.cvtColor(frame.left_rgb, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(output / "head_right_rgb.png"),
            cv2.cvtColor(frame.right_rgb, cv2.COLOR_RGB2BGR),
        )

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
            "baseline_left_timestamp_ns": baseline_left_ts,
            "baseline_right_timestamp_ns": baseline_right_ts,
            "fresh_frames_required": int(args.fresh_frames),
            "stale_frame_rejection": "timestamps_strictly_advanced_after_head_motion",
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        print("HEAD Q =", q_actual.tolist(), flush=True)
        print("WROTE", output.resolve(), flush=True)
        return 0
    finally:
        camera.close()
        if robot is not None:
            robot.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
