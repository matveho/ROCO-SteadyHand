"""Interactive physical calibration of Vega head->board coarse registration.

Flow:
  1. Clear the left claw from the head-board view.
  2. Capture the board exactly like vega_board_axis_benchmark.py.
  3. Move to the predicted CENTER, BOARD_X_PLUS and BOARD_Y_PLUS hover targets.
  4. At each target, let the operator jog the TCP flat in base-frame directions:
       forward N   (+base X, away from robot)
       back N      (-base X, toward robot)
       left N      (+base Y, robot-left)
       right N     (-base Y, robot-right)
     Distances are millimetres. Repeated commands are allowed.
  5. 'done' records the measured TCP for that reference.
  6. Print a paste-ready JSON calibration block containing predicted and corrected
     points, corrected board axes, TCP quaternions, and every manual jog.

This tool calibrates BOARD POSITION/AXES only. It deliberately preserves the
current TCP Z and orientation during manual XY jogs. A visibly angled claw is
therefore reported, not silently treated as a calibrated vertical orientation.
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
from steadyhand.cameras.vega import VegaHeadCamera
from steadyhand.config import load_bundle
from steadyhand.executor import move_tcp_segmented
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills
from steadyhand.vision.scene import detect_head_task_scene
from steadyhand.vega_camera_clear import move_camera_clear
from tools.vega_board_benchmark import _yaw_quat
from tools.vega_scene_perception import _ensure_publisher


ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = {
    "forward": (1.0, 0.0),
    "f": (1.0, 0.0),
    "back": (-1.0, 0.0),
    "backward": (-1.0, 0.0),
    "b": (-1.0, 0.0),
    "left": (0.0, 1.0),
    "l": (0.0, 1.0),
    "right": (0.0, -1.0),
    "r": (0.0, -1.0),
}


def _pose_record(pose):
    return {
        "position_m": [float(v) for v in pose.position_m],
        "quaternion_wxyz": [float(v) for v in pose.quaternion_wxyz],
    }


def _print_pose(prefix, pose):
    print(
        prefix,
        "xyz_m=",
        tuple(round(float(v), 6) for v in pose.position_m),
        "quat_wxyz=",
        tuple(round(float(v), 7) for v in pose.quaternion_wxyz),
        flush=True,
    )


def _interactive_adjust(robot, *, label, floor, speed_scale, max_jog_mm, events):
    print("", flush=True)
    print(f"=== MANUAL CALIBRATION: {label} ===", flush=True)
    print(
        "Commands: forward N | back N | left N | right N  (N in mm), "
        "status, done, abort",
        flush=True,
    )
    print("forward=away from robot (+base X), left=robot-left (+base Y)", flush=True)
    _print_pose("CURRENT", robot.get_tcp_pose())

    while True:
        try:
            raw = input(f"{label}> ").strip()
        except EOFError:
            raise RuntimeError("stdin closed before calibration point was accepted")
        if not raw:
            continue
        parts = raw.lower().split()
        cmd = parts[0]

        if cmd in ("done", "d"):
            pose = robot.get_tcp_pose()
            _print_pose(f"RECORDED {label}", pose)
            return pose
        if cmd in ("abort", "quit", "q"):
            raise KeyboardInterrupt()
        if cmd in ("status", "s"):
            _print_pose("CURRENT", robot.get_tcp_pose())
            continue
        if cmd in ("help", "h", "?"):
            print(
                "forward N | back N | left N | right N; N is millimetres; "
                f"single jog <= {max_jog_mm:g} mm; status; done; abort",
                flush=True,
            )
            continue
        if cmd not in DIRECTIONS or len(parts) != 2:
            print("INVALID: use e.g. 'forward 10', 'left 5', 'status', or 'done'", flush=True)
            continue
        try:
            mm = float(parts[1])
        except ValueError:
            print("INVALID DISTANCE: enter millimetres as a number", flush=True)
            continue
        if not math.isfinite(mm) or mm <= 0 or mm > max_jog_mm:
            print(f"INVALID DISTANCE: require 0 < N <= {max_jog_mm:g} mm", flush=True)
            continue

        current = robot.get_tcp_pose()
        dx_unit, dy_unit = DIRECTIONS[cmd]
        dx = dx_unit * mm / 1000.0
        dy = dy_unit * mm / 1000.0
        target = Pose(
            (
                float(current.position_m[0]) + dx,
                float(current.position_m[1]) + dy,
                float(current.position_m[2]),
            ),
            tuple(float(v) for v in current.quaternion_wxyz),
        )
        print(
            f"JOG {cmd.upper()} {mm:g} mm ->",
            tuple(round(float(v), 6) for v in target.position_m),
            flush=True,
        )
        # Pre-plan the exact flat target before commanding it.
        robot._kinematics.solve(target, robot._read_joint_positions())
        move_tcp_segmented(
            robot,
            target,
            speed_scale=float(speed_scale),
            max_translation_step_m=min(0.05, max_jog_mm / 1000.0),
            max_orientation_step_rad=0.10,
            min_tcp_z_m=floor,
        )
        actual = robot.get_tcp_pose()
        event = {
            "label": label,
            "command": cmd,
            "distance_mm": mm,
            "requested": _pose_record(target),
            "measured": _pose_record(actual),
        }
        events.append(event)
        _print_pose("MEASURED", actual)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offset-m", type=float, default=0.100,
                   help="nominal board-axis reference offset; default 100 mm")
    p.add_argument("--hover-z", type=float, default=0.550,
                   help="predicted low hover plane for initial coarse targets")
    p.add_argument("--coarse-speed-scale", type=float, default=0.90)
    p.add_argument("--jog-speed-scale", type=float, default=0.70)
    p.add_argument("--max-jog-mm", type=float, default=50.0)
    p.add_argument("--claw-yaw-deg", type=float, default=0.0)
    p.add_argument("--settle-s", type=float, default=0.5)
    p.add_argument("--publisher-log", default="~/head_camera.log")
    p.add_argument("--output", default="calibration/vega_board_manual.json")
    p.add_argument("--confirm-physical-motion", action="store_true")
    args = p.parse_args(argv)

    if not args.confirm_physical_motion:
        p.error("--confirm-physical-motion is required")
    if not 0.05 <= float(args.offset_m) <= 0.12:
        p.error("--offset-m must be 0.05..0.12 m")
    if not 0.45 <= float(args.coarse_speed_scale) <= 1.0:
        p.error("--coarse-speed-scale must be 0.45..1.0")
    if not 0.45 <= float(args.jog_speed_scale) <= 1.0:
        p.error("--jog-speed-scale must be 0.45..1.0")
    if not 1.0 <= float(args.max_jog_mm) <= 100.0:
        p.error("--max-jog-mm must be 1..100 mm")

    import numpy as np

    cfg = load_bundle("vega")["robot"]
    safety = dict(load_vega_skills().get("safety") or {})
    floor = float(safety["min_tcp_z_m"])
    if not floor + 0.03 <= float(args.hover_z) <= floor + 0.12:
        p.error(
            f"--hover-z must stay 3-12 cm above floor {floor:.6f}; "
            f"got {float(args.hover_z):.6f}"
        )

    cfg["allow_robot_init_head_motion"] = True
    cfg["auto_clear_software_estop_on_connect"] = True
    cfg["motion"]["max_step_rad"] = max(float(cfg["motion"]["max_step_rad"]), 0.45)

    _ensure_publisher(args.publisher_log, robot_name=cfg["robot_name"])

    robot = VegaAdapter(cfg)
    camera = None
    events = []
    try:
        robot.connect()
        move_camera_clear(robot, floor_m=floor, speed_scale=0.90)

        target_head = np.asarray([0.55, 0.0, 0.0], dtype=float)
        print("HEAD BEFORE =", robot._robot.head.get_joint_pos(), flush=True)
        move_head = getattr(robot._robot.head, "move_to_joint_pos", None)
        if callable(move_head):
            handle = move_head(target_head, velocity_scale=0.45)
            wait_fn = getattr(handle, "wait", None)
            if callable(wait_fn):
                wait_fn(timeout=5.0)
            else:
                time.sleep(1.5)
        else:
            robot._robot.head.set_joint_pos(
                target_head,
                wait_time=1.2,
                exit_on_reach=True,
                exit_on_reach_kwargs={"tolerance": 0.02},
            )
        time.sleep(float(args.settle_s))
        head_q = np.asarray(robot._robot.head.get_joint_pos(), dtype=float)
        print("HEAD AFTER  =", head_q.tolist(), flush=True)

        camera = VegaHeadCamera()
        camera.connect()
        print("WAITING FOR HEAD CAMERA FRAMES", flush=True)
        frame = camera.read(include_depth=False, timeout_s=15.0)
        scene = detect_head_task_scene(
            frame.left_rgb,
            frame.camera_info,
            head_q,
            plane_z_m=floor,
            lift_m=float(cfg["kinematics"]["fixed_joint_values"]["Lift"]),
            torso_flip_rad=float(cfg["kinematics"]["fixed_joint_values"]["torso_flip"]),
            layout="unlabeled",
        )

        T = np.asarray(scene["board"]["T_base_board_center"], dtype=float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            raise RuntimeError("invalid T_base_board from live perception")
        center = T[:3, 3].copy()
        bx = T[:3, 0].copy()
        by = T[:3, 1].copy()
        bx[2] = 0.0
        by[2] = 0.0
        bx /= np.linalg.norm(bx)
        by /= np.linalg.norm(by)

        predicted = {
            "CENTER": center,
            "BOARD_X_PLUS": center + float(args.offset_m) * bx,
            "BOARD_Y_PLUS": center + float(args.offset_m) * by,
        }
        print("HEAD PREDICTED CENTER =", tuple(round(float(v), 6) for v in center), flush=True)
        print("HEAD PREDICTED +X UNIT =", tuple(round(float(v), 6) for v in bx), flush=True)
        print("HEAD PREDICTED +Y UNIT =", tuple(round(float(v), 6) for v in by), flush=True)

        quat = _yaw_quat(math.radians(float(args.claw_yaw_deg)))
        corrected = {}
        for label in ("CENTER", "BOARD_X_PLUS", "BOARD_Y_PLUS"):
            point = predicted[label]
            target = Pose(
                (float(point[0]), float(point[1]), float(args.hover_z)),
                quat,
            )
            robot._kinematics.config["position_tolerance_m"] = 0.010
            robot._kinematics.config["orientation_tolerance_rad"] = 0.12
            robot._kinematics.config["max_seed_delta_rad"] = 2.4
            robot._kinematics.config["max_iterations"] = 180
            robot._kinematics.solve(target, robot._read_joint_positions())

            print(
                f"COARSE MOVE {label} ->",
                tuple(round(float(v), 6) for v in target.position_m),
                flush=True,
            )
            move_tcp_segmented(
                robot,
                target,
                speed_scale=float(args.coarse_speed_scale),
                max_translation_step_m=0.20,
                max_orientation_step_rad=0.80,
                min_tcp_z_m=floor,
            )
            reached = robot.get_tcp_pose()
            _print_pose(f"COARSE REACHED {label}", reached)
            print(
                "Visually jog until the TCP is at the intended physical reference, "
                "then type 'done'.",
                flush=True,
            )
            corrected[label] = _interactive_adjust(
                robot,
                label=label,
                floor=floor,
                speed_scale=float(args.jog_speed_scale),
                max_jog_mm=float(args.max_jog_mm),
                events=events,
            )

        c = np.asarray(corrected["CENTER"].position_m, dtype=float)
        x = np.asarray(corrected["BOARD_X_PLUS"].position_m, dtype=float)
        y = np.asarray(corrected["BOARD_Y_PLUS"].position_m, dtype=float)
        dx = x[:2] - c[:2]
        dy = y[:2] - c[:2]
        if np.linalg.norm(dx) < 0.02 or np.linalg.norm(dy) < 0.02:
            raise RuntimeError("corrected axis references are too close to center")
        ux = dx / np.linalg.norm(dx)
        uy = dy / np.linalg.norm(dy)

        record = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "robot_name": cfg["robot_name"],
            "base_frame": cfg["kinematics"]["base_frame"],
            "tcp_frame": cfg["kinematics"]["ee_frame"],
            "floor_m": floor,
            "nominal_axis_offset_m": float(args.offset_m),
            "head_prediction": {
                "center_base_m": [float(v) for v in center],
                "board_x_unit_base": [float(v) for v in bx],
                "board_y_unit_base": [float(v) for v in by],
                "center_target_base_m": [float(v) for v in predicted["CENTER"]],
                "x_plus_target_base_m": [float(v) for v in predicted["BOARD_X_PLUS"]],
                "y_plus_target_base_m": [float(v) for v in predicted["BOARD_Y_PLUS"]],
            },
            "manual_corrected": {
                label: _pose_record(corrected[label])
                for label in ("CENTER", "BOARD_X_PLUS", "BOARD_Y_PLUS")
            },
            "corrected_board_frame_xy": {
                "center_base_xy_m": [float(v) for v in c[:2]],
                "board_x_unit_base_xy": [float(v) for v in ux],
                "board_y_unit_base_xy": [float(v) for v in uy],
                "x_reference_distance_m": float(np.linalg.norm(dx)),
                "y_reference_distance_m": float(np.linalg.norm(dy)),
            },
            "orientation_status": (
                "MEASURED_ONLY_NOT_CALIBRATED_VERTICAL; physical claw was reported angled"
            ),
            "jogs": events,
        }

        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

        print("", flush=True)
        print("=== PASTE THIS CALIBRATION BLOCK BACK TO MAIN AGENT ===", flush=True)
        print(json.dumps(record, indent=2), flush=True)
        print("=== END CALIBRATION BLOCK ===", flush=True)
        print("WROTE", out.resolve(), flush=True)
        return 0
    finally:
        if camera is not None:
            try:
                camera.close()
            except BaseException:
                pass
        try:
            robot.close()
        except BaseException as exc:
            print(f"SHUTDOWN WARNING: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
