"""First gated autonomous battery_size1 grasp-and-lift bring-up.

Preconditions:
- battery calibration JSON contains an operator-taught jaw goal pixel and grasp Z;
- a battery_size1 wrist_a alignment run has converged to that exact goal pixel;
- the right TCP is still at the aligned safe hover.

Sequence only:
  open -> vertical approach -> slow final descent -> 1.0 A / 240 deg/s grip -> lift

There is intentionally no transfer, placement, release or insertion.
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
    VERIFIED_GRIP_CURRENT_A,
    VERIFIED_GRIP_SPEED_DPS,
    load_alignment_result,
    load_calibration,
    require_right_battery_config,
)
from steadyhand.config import load_bundle
from steadyhand.geometry import quaternion_angle
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAL = ROOT / "calibration" / "battery_size1_grasp.json"


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration", default=str(DEFAULT_CAL.relative_to(ROOT)))
    p.add_argument("--alignment-result", required=True)
    p.add_argument("--pregrasp-offset-m", type=float, default=0.025)
    p.add_argument("--free-speed", type=float, default=0.70)
    p.add_argument("--descent-speed", type=float, default=0.12)
    p.add_argument("--confirm-physical-motion", action="store_true")
    p.add_argument("--confirm-battery-ready", action="store_true")
    p.add_argument("--output")
    args = p.parse_args(argv)

    if not args.confirm_physical_motion or not args.confirm_battery_ready:
        p.error(
            "requires --confirm-physical-motion and --confirm-battery-ready"
        )
    values = (args.pregrasp_offset_m, args.free_speed, args.descent_speed)
    if not all(math.isfinite(float(v)) for v in values):
        p.error("motion settings must be finite")
    if not (
        0.015 <= args.pregrasp_offset_m <= 0.040
        and 0.45 <= args.free_speed <= 0.90
        and 0.05 <= args.descent_speed <= 0.20
    ):
        p.error(
            "pregrasp offset must be 15-40 mm, free speed 0.45-0.90, "
            "descent speed 0.05-0.20"
        )

    cfg = load_bundle("vega")["robot"]
    require_right_battery_config(cfg)
    grip_cfg = cfg["gripper"]
    if abs(float(grip_cfg["grip_current_a"]) - VERIFIED_GRIP_CURRENT_A) > 1e-12:
        raise ValueError("robot config must retain verified battery grip current 1.0 A")
    if int(round(float(grip_cfg["grip_speed_dps"]))) != VERIFIED_GRIP_SPEED_DPS:
        raise ValueError("robot config must retain verified battery grip speed 240 deg/s")

    floor = float(load_vega_skills()["safety"]["min_tcp_z_m"])
    cal_path = _resolve(args.calibration)
    alignment_path = _resolve(args.alignment_result)
    calibration = load_calibration(cal_path, cfg, floor_m=floor)

    output = (
        _resolve(args.output)
        if args.output
        else ROOT / "runs" / datetime.now(timezone.utc).strftime(
            "battery_size1_pick_%Y%m%dT%H%M%S_%fZ"
        )
    )
    output.mkdir(parents=True, exist_ok=False)

    run = {
        "part": PART_NAME,
        "working_arm": "right",
        "tcp_frame": "tip_r",
        "wrist_camera": "wrist_a",
        "gripper_scope": "right",
        "status": "initializing",
        "calibration_file": str(cal_path),
        "alignment_result_file": str(alignment_path),
        "calibration": calibration,
        "floor_m": floor,
        "gripper_command": {
            "current_a": VERIFIED_GRIP_CURRENT_A,
            "speed_dps": VERIFIED_GRIP_SPEED_DPS,
        },
        "pregrasp_offset_m": float(args.pregrasp_offset_m),
        "free_speed": float(args.free_speed),
        "descent_speed": float(args.descent_speed),
        "lift_retention_verified": None,
    }
    (output / "arguments.json").write_text(
        json.dumps(vars(args), indent=2) + "\n", encoding="utf-8"
    )

    def persist():
        (output / "result.json").write_text(
            json.dumps(run, indent=2, default=str) + "\n", encoding="utf-8"
        )

    def event(name, **fields):
        record = {"event": name, "time_monotonic": time.monotonic(), **fields}
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=str) + "\n")
        print(name.upper(), json.dumps(fields, default=str), flush=True)

    cfg["allow_robot_init_head_motion"] = True
    cfg["auto_clear_software_estop_on_connect"] = True
    robot = VegaAdapter(cfg)

    try:
        robot.prepare()
        robot.connect()
        start = robot.get_tcp_pose()
        _, aligned_pose = load_alignment_result(
            alignment_path,
            cal_path,
            calibration,
            current_pose=start,
        )

        grasp_z = float(calibration["grasp"]["tcp_z_m"])
        taught_quat = tuple(
            float(v)
            for v in calibration["grasp"]["taught_tip_quaternion_wxyz"]
        )
        if quaternion_angle(start.quaternion_wxyz, taught_quat) > 0.08:
            raise RuntimeError(
                "live aligned tip orientation differs from the operator-taught "
                "vertical grasp orientation"
            )
        if start.position_m[2] < floor:
            raise RuntimeError("live TCP starts below the measured task floor")
        if start.position_m[2] < grasp_z + args.pregrasp_offset_m + 0.005:
            raise RuntimeError(
                "aligned hover is too low for the configured pregrasp approach; "
                "raise the safe hover before retrying"
            )

        pregrasp_z = grasp_z + float(args.pregrasp_offset_m)
        if grasp_z < floor or pregrasp_z < floor:
            raise RuntimeError("grasp/pregrasp target violates the measured task floor")
        if pregrasp_z >= start.position_m[2]:
            raise RuntimeError("pregrasp target must be below the current safe hover")

        pregrasp = Pose(
            (start.position_m[0], start.position_m[1], pregrasp_z),
            start.quaternion_wxyz,
        )
        grasp = Pose(
            (start.position_m[0], start.position_m[1], grasp_z),
            start.quaternion_wxyz,
        )
        lift = Pose(
            (start.position_m[0], start.position_m[1], start.position_m[2]),
            start.quaternion_wxyz,
        )

        # Every arm segment is vertical and every endpoint is at/above floor.
        # Therefore no interpolated point can cross below the measured floor.
        for label, pose in (("start", start), ("pregrasp", pregrasp), ("grasp", grasp), ("lift", lift)):
            if pose.position_m[2] < floor:
                raise RuntimeError(f"{label} TCP target is below measured task floor")

        run["status"] = "ready"
        run["pre_tcp"] = {
            "position_m": list(start.position_m),
            "quaternion_wxyz": list(start.quaternion_wxyz),
        }
        run["aligned_tcp"] = {
            "position_m": list(aligned_pose.position_m),
            "quaternion_wxyz": list(aligned_pose.quaternion_wxyz),
        }
        run["targets"] = {
            "pregrasp": list(pregrasp.position_m),
            "grasp": list(grasp.position_m),
            "lift": list(lift.position_m),
        }
        persist()
        event(
            "gates_passed",
            pre_tcp=start.position_m,
            grasp_z_m=grasp_z,
            jaw_goal_pixel_uv=calibration["jaw_alignment"]["goal_pixel_uv"],
        )

        robot.connect_gripper()
        status_before = robot.gripper_status()
        robot.open_gripper(PART_NAME)
        status_open = robot.gripper_status()
        run["gripper_status_before"] = status_before
        run["gripper_status_open"] = status_open
        persist()
        event("gripper_open", status=status_open)

        event("approach_start", target_tcp=pregrasp.position_m, speed_scale=args.free_speed)
        robot.move_tcp(pregrasp, speed_scale=args.free_speed)
        pregrasp_actual = robot.get_tcp_pose()
        event("approach_reached", measured_tcp=pregrasp_actual.position_m)

        event("descent_start", target_tcp=grasp.position_m, speed_scale=args.descent_speed)
        robot.move_tcp(grasp, speed_scale=args.descent_speed)
        grasp_actual = robot.get_tcp_pose()
        if grasp_actual.position_m[2] < floor:
            raise RuntimeError("measured TCP crossed below task floor after descent")
        event("descent_reached", measured_tcp=grasp_actual.position_m)

        robot.grip(PART_NAME, current_a=VERIFIED_GRIP_CURRENT_A)
        grip_result = robot._gripper.last_grip_result()
        grip_status = robot.gripper_status()
        run["grip_result"] = grip_result
        run["gripper_status_after_grip"] = grip_status
        persist()
        event("grip", result=grip_result, status=grip_status)

        if not isinstance(grip_result, dict) or grip_result.get("gripped") is not True:
            run["status"] = "grip_not_verified"
            persist()
            print("GRIP NOT VERIFIED BY DRIVER; NOT LIFTING", flush=True)
            return 3

        event("lift_start", target_tcp=lift.position_m, speed_scale=args.free_speed)
        robot.move_tcp(lift, speed_scale=args.free_speed)
        lifted = robot.get_tcp_pose()
        run["post_lift_tcp"] = {
            "position_m": list(lifted.position_m),
            "quaternion_wxyz": list(lifted.quaternion_wxyz),
        }
        event("lift_reached", measured_tcp=lifted.position_m)

        answer = input(
            "battery_size1 retained securely in jaws after lift? type yes or no: "
        ).strip().lower()
        if answer == "yes":
            retention = True
        elif answer == "no":
            retention = False
        else:
            retention = None
        run["lift_retention_verified"] = retention
        run["status"] = (
            "lift_retention_verified"
            if retention is True
            else "lift_retention_failed"
            if retention is False
            else "lift_retention_unverified"
        )
        run["gripper_status_after_lift"] = robot.gripper_status()
        persist()
        event(
            "retention_check",
            lift_retention_verified=retention,
            final_status=run["status"],
        )
        print("BATTERY_SIZE1 PICK RESULT =", run["status"], flush=True)
        return 0 if retention is True else 4 if retention is False else 5
    except BaseException as exc:
        run["status"] = "stopped"
        run["error"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            if robot._robot is not None:
                pose = robot.get_tcp_pose()
                run["last_tcp"] = {
                    "position_m": list(pose.position_m),
                    "quaternion_wxyz": list(pose.quaternion_wxyz),
                }
            if robot._gripper is not None:
                run["last_gripper_status"] = robot.gripper_status()
        except BaseException:
            pass
        persist()
        # Do not blanket-assert software E-stop for calibration/gripper/operator
        # failures. VegaAdapter.move_tcp() already stops on actual arm-motion
        # failures and interruptions.
        raise
    finally:
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
