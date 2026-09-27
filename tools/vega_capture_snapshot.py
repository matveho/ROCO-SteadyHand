"""Capture a no-motion Vega perception snapshot for offline development.

Run onboard with /usr/bin/python3. This does not construct dexcontrol Robot().

Output:
- head_left_rgb.npy
- head_right_rgb.npy
- head_depth_m.npy
- wrist_a_rgb.npy / wrist_b_rgb.npy (unless --head-only)
- metadata.json

RGB arrays remain RGB; no OpenCV BGR conversion is applied.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.cameras.vega import VegaHeadCamera, VegaWristCameras


def serializable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return {str(k): serializable(v) for k, v in vars(value).items()}
    return repr(value)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output")
    p.add_argument("--head-only", action="store_true")
    p.add_argument("--rgb-only", action="store_true",
                   help="capture rectified head stereo RGB without requiring depth")
    args = p.parse_args(argv)

    import numpy as np

    if not os.environ.get("ROBOT_NAME"):
        raise SystemExit("ROBOT_NAME is not set")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output or f"vega_snapshot_{stamp}")
    output.mkdir(parents=True, exist_ok=False)

    head = VegaHeadCamera()
    wrists = None if args.head_only else VegaWristCameras()
    try:
        head.connect()
        if wrists:
            wrists.connect()

        h = head.read(include_depth=not args.rgb_only)
        np.save(output / "head_left_rgb.npy", h.left_rgb)
        np.save(output / "head_right_rgb.npy", h.right_rgb)
        if h.depth_m is not None:
            np.save(output / "head_depth_m.npy", h.depth_m)

        metadata = {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "robot_name": os.environ.get("ROBOT_NAME"),
            "head": {
                "left_timestamp_ns": h.left_timestamp_ns,
                "right_timestamp_ns": h.right_timestamp_ns,
                "depth_timestamp_ns": h.depth_timestamp_ns,
                "camera_info": serializable(h.camera_info),
                "left_shape": list(h.left_rgb.shape),
                "right_shape": list(h.right_rgb.shape),
                "depth_shape": None if h.depth_m is None else list(h.depth_m.shape),
            },
        }

        if wrists:
            w = wrists.read()
            np.save(output / "wrist_a_rgb.npy", w.wrist_a.rgb)
            np.save(output / "wrist_b_rgb.npy", w.wrist_b.rgb)
            metadata["wrists"] = {
                "wrist_a": serializable(asdict(w.wrist_a) | {"rgb": None}),
                "wrist_b": serializable(asdict(w.wrist_b) | {"rgb": None}),
                "wrist_a_shape": list(w.wrist_a.rgb.shape),
                "wrist_b_shape": list(w.wrist_b.rgb.shape),
            }

        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        print(output)
        return 0
    finally:
        if wrists:
            wrists.close()
        head.close()


if __name__ == "__main__":
    raise SystemExit(main())
