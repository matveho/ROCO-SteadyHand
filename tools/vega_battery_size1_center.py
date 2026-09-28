"""Center battery_size1 to its taught jaw-alignment pixel using wrist_a.

This is the battery-specific bridge between coarse hover and descent. It never
changes Z, never connects the gripper and never descends. For the first
bring-up, the operator explicitly supplies the battery feature pixel in the
initial wrist image; the convergence target comes only from the dedicated
battery calibration JSON.
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
    calibration_sha256,
    load_calibration,
    require_right_battery_config,
)
from steadyhand.cameras.vega import VegaWristCameras
from steadyhand.config import load_bundle
from steadyhand.geometry import quaternion_angle
from steadyhand.skill_config import load_vega_skills
from steadyhand.vision.wrist_servo import run_xy_servo
from tools.vega_wrist_fine_center import WristAOnlyCapture

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAL = ROOT / "calibration" / "battery_size1_grasp.json"


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration", default=str(DEFAULT_CAL.relative_to(ROOT)))
    p.add_argument("--coarse-xy", nargs=2, type=float, required=True, metavar=("X", "Y"))
    p.add_argument("--feature", nargs=2, type=float, required=True, metavar=("U", "V"),
                   help="battery feature pixel in the initial wrist_a hover image")
    p.add_argument("--probe-m", type=float, default=0.008)
    p.add_argument("--gain", type=float, default=0.65)
    p.add_argument("--max-step-m", type=float, default=0.010)
    p.add_argument("--max-radius-m", type=float, default=0.040)
    p.add_argument("--tolerance-px", type=float, default=5.0)
    p.add_argument("--max-iterations", type=int, default=6)
    p.add_argument("--speed-scale", type=float, default=0.45)
    p.add_argument("--start-xy-tolerance-m", type=float, default=0.015)
    p.add_argument("--output")
    p.add_argument("--confirm-physical-motion", action="store_true")
    args = p.parse_args(argv)

    if not args.confirm_physical_motion:
        p.error("--confirm-physical-motion is required")
    values = (
        *args.coarse_xy, *args.feature, args.probe_m, args.gain,
        args.max_step_m, args.max_radius_m, args.tolerance_px,
        args.speed_scale, args.start_xy_tolerance_m,
    )
    if not all(math.isfinite(float(v)) for v in values):
        p.error("all coordinates and servo settings must be finite")
    if not (
        0.006 <= args.probe_m <= 0.012
        and 0 < args.gain <= 1
        and 0 < args.max_step_m <= 0.015
        and args.probe_m <= args.max_radius_m <= 0.060
        and args.tolerance_px > 0
        and 1 <= args.max_iterations <= 12
        and 0.45 <= args.speed_scale <= 0.70
        and 0.003 <= args.start_xy_tolerance_m <= 0.030
    ):
        p.error("servo bounds invalid; see --help")

    import cv2

    cfg = load_bundle("vega")["robot"]
    require_right_battery_config(cfg)
    floor = float(load_vega_skills()["safety"]["min_tcp_z_m"])
    cal_path = _resolve(args.calibration)
    calibration = load_calibration(cal_path, cfg, floor_m=floor)
    goal_uv = tuple(float(v) for v in calibration["jaw_alignment"]["goal_pixel_uv"])

    output = (
        _resolve(args.output)
        if args.output
        else ROOT / "runs" / datetime.now(timezone.utc).strftime(
            "battery_size1_center_%Y%m%dT%H%M%S_%fZ"
        )
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "arguments.json").write_text(
        json.dumps(vars(args), indent=2) + "\n", encoding="utf-8"
    )

    cameras = VegaWristCameras()
    capture = None
    robot = None
    last_goal = goal_uv

    def event(kind, fields):
        record = {"event": kind, "time_monotonic": time.monotonic(), **fields}
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(kind.upper(), json.dumps(fields), flush=True)
        if "feature_uv" in fields and capture is not None:
            raw = output / f"{capture.index - 1:03d}_wrist_a.png"
            annotated = cv2.imread(str(raw))
            if annotated is not None:
                uv = tuple(int(round(v)) for v in fields["feature_uv"])
                goal = tuple(int(round(v)) for v in last_goal)
                cv2.circle(annotated, uv, 20, (0, 255, 0), 2)
                cv2.drawMarker(
                    annotated, goal, (0, 0, 255), cv2.MARKER_CROSS, 25, 2
                )
                cv2.imwrite(
                    str(raw.with_name(raw.stem + "_tracked.png")), annotated
                )

    try:
        cameras.connect()
        capture = WristAOnlyCapture(cameras, output, settle_s=0.20, warmup_attempts=8)
        capture()

        cfg["allow_robot_init_head_motion"] = True
        cfg["auto_clear_software_estop_on_connect"] = True
        robot = VegaAdapter(cfg)
        robot.connect()

        start = robot.get_tcp_pose()
        coarse_xy = tuple(float(v) for v in args.coarse_xy)
        xy_error = math.dist(start.position_m[:2], coarse_xy)
        if xy_error > args.start_xy_tolerance_m:
            raise RuntimeError(
                f"live TCP is {xy_error:.4f} m from --coarse-xy; "
                "establish the coarse hover first"
            )
        if not floor + 0.060 <= start.position_m[2] <= floor + 0.120:
            raise RuntimeError("battery jaw centering requires a 60-120 mm safe hover")
        taught_hover_z = float(
            calibration["jaw_alignment"]["taught_hover_tcp_z_m"]
        )
        taught_hover_quat = tuple(
            float(v)
            for v in calibration["jaw_alignment"]["taught_tip_quaternion_wxyz"]
        )
        if abs(start.position_m[2] - taught_hover_z) > 0.005:
            raise RuntimeError(
                "current hover Z differs by >5 mm from the hover where the jaw "
                "goal pixel was taught"
            )
        if quaternion_angle(start.quaternion_wxyz, taught_hover_quat) > 0.04:
            raise RuntimeError(
                "current tip orientation differs from the pose where the jaw "
                "goal pixel was taught"
            )

        robot._kinematics.config["position_tolerance_m"] = 0.0007
        robot._kinematics.config["orientation_tolerance_rad"] = 0.01

        result = run_xy_servo(
            robot,
            capture,
            floor_m=floor,
            feature_uv=args.feature,
            goal_uv=goal_uv,
            probe_m=args.probe_m,
            gain=args.gain,
            max_step_m=args.max_step_m,
            max_radius_m=args.max_radius_m,
            tolerance_px=args.tolerance_px,
            max_iterations=args.max_iterations,
            speed_scale=args.speed_scale,
            event=event,
        )
        final_pose = robot.get_tcp_pose()
        result.update(
            {
                "part": PART_NAME,
                "working_arm": "right",
                "tcp_frame": "tip_r",
                "wrist_camera": WRIST_CAMERA,
                "calibration_sha256": calibration_sha256(cal_path),
                "tcp_position_m": list(final_pose.position_m),
                "tcp_quaternion_wxyz": list(final_pose.quaternion_wxyz),
                "supplied_coarse_xy_base_m": list(coarse_xy),
                "alignment_goal": "operator_taught_jaw_pixel",
            }
        )
        (output / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print("BATTERY_SIZE1 JAW CENTER PASS", flush=True)
        print("RESULT FILE =", (output / "result.json").resolve(), flush=True)
        return 0
    except BaseException as exc:
        # Tracking/camera/calibration failures must not assert software E-stop.
        # VegaAdapter.move_tcp() handles actual motion command failures itself.
        record = {"status": "stopped", "reason": str(exc), "exception": type(exc).__name__}
        try:
            (output / "result.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
        except BaseException:
            pass
        print("BATTERY_SIZE1 JAW CENTER STOPPED", json.dumps(record), flush=True)
        raise
    finally:
        try:
            cameras.close()
        finally:
            if robot is not None:
                robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
