"""Operator-assisted battery_size1 grasp calibration.

This tool records two physical values without inventing either:
1. wrist_b pixel where the selected battery feature appears when the jaws are
   mechanically aligned over the battery;
2. measured tip_l Z at the operator-taught grasp height.

It never commands arm motion. The record-grasp-z mode constructs VegaAdapter
only to read the live TCP; Robot() initialization may move the head.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.vega import VegaAdapter
from steadyhand.battery_size1 import (
    PART_NAME,
    WRIST_CAMERA,
    blank_calibration,
    calibration_is_complete,
)
from steadyhand.cameras.vega import VegaWristCameras
from steadyhand.config import load_bundle
from steadyhand.geometry import quaternion_to_matrix
from steadyhand.skill_config import load_vega_skills

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "calibration" / "battery_size1_grasp.json"


def _path(value):
    p = Path(value) if value else DEFAULT_PATH
    return p if p.is_absolute() else ROOT / p


def _load_or_blank(path, robot_config):
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("calibration file must contain a JSON object")
        if value.get("part") not in (None, PART_NAME):
            raise ValueError("calibration file is for a different part")
        return value
    return blank_calibration(robot_config)


def _write(path, value):
    value["calibration_complete"] = bool(calibration_is_complete(value))
    value["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print("WROTE", path.resolve(), flush=True)
    print("CALIBRATION_COMPLETE =", value["calibration_complete"], flush=True)


def _capture_wrist_b(output, *, attempts=8):
    import cv2
    import numpy as np

    cameras = VegaWristCameras()
    cameras.connect()
    try:
        for attempt in range(1, attempts + 1):
            pair = cameras.read(timeout=3.0, fresh=True)
            frame = pair.wrist_b
            rgb = np.asarray(frame.rgb)
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ValueError(
                    f"wrist_b expected uint8 HxWx3 RGB, got {rgb.shape}/{rgb.dtype}"
                )
            mean, std = float(rgb.mean()), float(rgb.std())
            print(
                f"WRIST_B attempt={attempt} frame_id={frame.frame_id} "
                f"mean={mean:.2f} std={std:.2f}",
                flush=True,
            )
            if mean < 4.0 or std < 2.0:
                time.sleep(0.10)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"failed to write {output}")
            metadata = {
                "camera": WRIST_CAMERA,
                "frame_id": frame.frame_id,
                "timestamp_ns": frame.timestamp_ns,
                "received_monotonic_ns": frame.received_monotonic_ns,
                "image_size_px": [int(rgb.shape[1]), int(rgb.shape[0])],
                "mean": mean,
                "std": std,
            }
            output.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2, default=int) + "\n",
                encoding="utf-8",
            )
            print("GOAL IMAGE =", output.resolve(), flush=True)
            print("IMAGE SIZE =", metadata["image_size_px"], flush=True)
            return metadata
        raise RuntimeError("wrist_b remained black/invalid during goal-pixel capture")
    finally:
        cameras.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration", default=str(DEFAULT_PATH.relative_to(ROOT)))
    sub = p.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--force", action="store_true")

    capture = sub.add_parser("capture-goal-image")
    capture.add_argument("--output", required=True)
    capture.add_argument("--attempts", type=int, default=8)

    goal = sub.add_parser("record-goal-pixel")
    goal.add_argument("--goal-pixel", nargs=2, type=float, required=True, metavar=("U", "V"))
    goal.add_argument("--source-image", required=True)

    grasp = sub.add_parser("record-grasp-z")
    grasp.add_argument("--confirm-read-current-tcp", action="store_true")

    args = p.parse_args(argv)
    bundle = load_bundle("vega")
    cfg = bundle["robot"]
    path = _path(args.calibration)

    if args.command == "init":
        if path.exists() and not args.force:
            raise SystemExit(f"{path} exists; refusing overwrite without --force")
        _write(path, blank_calibration(cfg))
        return 0

    if args.command == "capture-goal-image":
        if not 2 <= args.attempts <= 20:
            p.error("--attempts must be 2..20")
        output = _path(args.output)
        _capture_wrist_b(output, attempts=args.attempts)
        return 0

    if args.command == "record-goal-pixel":
        import cv2

        source = _path(args.source_image)
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"could not read source image {source}")
        h, w = image.shape[:2]
        u, v = (float(x) for x in args.goal_pixel)
        if not all(math.isfinite(x) for x in (u, v)):
            raise ValueError("goal pixel must be finite")
        if not (0 <= u < w and 0 <= v < h):
            raise ValueError(f"goal pixel {(u, v)} is outside image {w}x{h}")
        value = _load_or_blank(path, cfg)
        value["jaw_alignment"] = {
            "goal_pixel_uv": [u, v],
            "image_size_px": [int(w), int(h)],
            "source_image": str(source),
            "source": "operator_taught",
        }
        _write(path, value)
        print("JAW GOAL PIXEL =", [u, v], flush=True)
        return 0

    if args.command == "record-grasp-z":
        if not args.confirm_read_current_tcp:
            p.error("record-grasp-z requires --confirm-read-current-tcp")
        floor = float(load_vega_skills()["safety"]["min_tcp_z_m"])
        cfg["allow_robot_init_head_motion"] = True
        robot = VegaAdapter(cfg)
        try:
            robot.prepare()
            robot.connect()
            pose = robot.get_tcp_pose()
            if pose.position_m[2] < floor:
                raise RuntimeError(
                    f"current TCP z={pose.position_m[2]:.6f} is below task floor "
                    f"{floor:.6f}; refusing to record"
                )
            vertical = quaternion_to_matrix(pose.quaternion_wxyz)[2][2]
            if vertical < math.cos(0.12):
                raise RuntimeError(
                    "current tip_l is not within 0.12 rad of the established vertical "
                    "claw family; physically align it before teaching grasp Z"
                )
            value = _load_or_blank(path, cfg)
            value["grasp"] = {
                "tcp_z_m": float(pose.position_m[2]),
                "taught_tip_quaternion_wxyz": list(pose.quaternion_wxyz),
                "source": "operator_taught_current_tcp",
            }
            _write(path, value)
            print("TAUGHT GRASP TCP =", pose.position_m, flush=True)
            print("TAUGHT GRASP Z =", float(pose.position_m[2]), flush=True)
            print("TAUGHT TIP QUAT =", pose.quaternion_wxyz, flush=True)
            return 0
        finally:
            robot.close()

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
