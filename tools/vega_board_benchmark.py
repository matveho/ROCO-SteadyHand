"""Register the physical task board and benchmark vertical right-claw reach.

Milestone:
  1. Aim the head camera downward.
  2. Detect the large white task-board quadrilateral.
  3. Map its four corners + center into vega_1u_base_link.
  4. Save that board registration.
  5. Optionally move the right TCP, held vertical, through:
       center -> TL -> TR -> BR -> BL -> center

This is intentionally the coarse/global layer. Fine object centering should use
the right wrist camera after this board frame exists.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.vega import VegaAdapter
from steadyhand.cameras.vega import VegaHeadCamera, intrinsics_from_camera_info
from steadyhand.config import load_bundle
from steadyhand.board_geometry import configured_board_plane_z
from steadyhand.executor import move_tcp_segmented
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills
from steadyhand.vega_camera_clear import move_camera_clear
from steadyhand.vision.board import (
    board_frame_from_corners,
    detect_white_board_corners,
    head_left_optical_transform,
    pixels_to_horizontal_plane,
)


ROOT = Path(__file__).resolve().parents[1]


def _serializable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    return value


def _set_head_and_capture(cfg, head_j1, floor):
    import numpy as np

    cfg["allow_robot_init_head_motion"] = True
    cfg["auto_clear_software_estop_on_connect"] = True
    cfg["motion"]["max_step_rad"] = max(float(cfg["motion"]["max_step_rad"]), 0.45)
    adapter = VegaAdapter(cfg)
    camera = None
    try:
        adapter.connect()
        move_camera_clear(adapter, floor_m=floor, speed_scale=0.90)

        q = np.asarray(adapter._robot.head.get_joint_pos(), dtype=float)
        if head_j1 is not None:
            q[0] = float(head_j1)
            adapter._robot.head.set_joint_pos(
                q,
                wait_time=1.5,
                exit_on_reach=True,
                exit_on_reach_kwargs={"tolerance": 0.02},
            )
            q = np.asarray(adapter._robot.head.get_joint_pos(), dtype=float)

        camera = VegaHeadCamera()
        camera.connect()
        frame = camera.read(include_depth=False)
        return q, frame
    finally:
        if camera is not None:
            camera.close()
        adapter.close()


def _yaw_quat(yaw):
    """tip_r quaternion for a top-down physical gripper with free in-plane yaw.

    Important frame detail: the organizer's q=(0,1,0,0) top-down convention
    applies to R_ee_link_gripper_link, NOT to tip_r.  Our physical IK frame is
    tip_r, whose fixed transform from that gripper link is:
        rpy = (pi, 0, pi/2)
    Therefore:
        R_base_tip = Rz(yaw) * Rx(pi) * R_ee_tip
                   = Rz(yaw - pi/2)
    So a physically vertical gripper corresponds to a tip_r frame that looks
    like a pure base-Z rotation.  The previous implementation incorrectly put
    a 180-deg X rotation directly on tip_r and made the center effectively
    unreachable.
    """
    half = (float(yaw) - math.pi / 2.0) / 2.0
    return (
        math.cos(half),
        0.0,
        0.0,
        math.sin(half),
    )


def _pose_at(point, z, quat):
    return Pose(
        (float(point[0]), float(point[1]), float(z)),
        tuple(float(x) for x in quat),
    )


def _vertical_target_for_point(
    robot,
    label,
    point,
    center,
    hover_z,
    floor,
    yaw,
):
    """Fast fixed-plane planner while keeping one constant vertical-claw yaw."""
    import numpy as np

    robot._kinematics.config["position_tolerance_m"] = 0.010
    robot._kinematics.config["orientation_tolerance_rad"] = 0.12
    robot._kinematics.config["max_seed_delta_rad"] = 2.4
    robot._kinematics.config["max_iterations"] = 140

    if not (float(floor) + 0.03 <= float(hover_z) <= float(floor) + 0.10 + 1e-9):
        raise ValueError(
            f"benchmark hover_z={float(hover_z):.4f} must stay 3-10 cm above "
            f"configured floor {float(floor):.4f}"
        )

    seed = robot._read_joint_positions()
    point = np.asarray(point, dtype=float)
    center = np.asarray(center, dtype=float)
    quat = _yaw_quat(float(yaw))

    # Exact board point first; only inset if the coarse global registration
    # lies beyond the actual arm envelope. Z and yaw stay fixed so the arm
    # traverses one flat plane without wrist reorientation.
    alphas = (1.0,) if label == "center" else (
        1.0, 0.96, 0.92, 0.88, 0.84, 0.80, 0.75, 0.70,
    )
    last_error = None
    for alpha in alphas:
        xy = center + float(alpha) * (point - center)
        target = _pose_at(xy, float(hover_z), quat)
        try:
            robot._kinematics.solve(target, seed)
        except Exception as exc:
            last_error = exc
            continue

        print(
            f"{label.upper()} PLAN: alpha={alpha:.2f}, z={float(hover_z):.3f} m, "
            f"yaw={math.degrees(float(yaw)):+.0f} deg, "
            f"xy=({target.position_m[0]:.4f},{target.position_m[1]:.4f})",
            flush=True,
        )
        return target

    raise RuntimeError(
        f"{label}: no reachable vertical-claw pose on fixed plane "
        f"z={float(hover_z):.3f}, yaw={math.degrees(float(yaw)):+.0f}; "
        f"last IK error: {last_error}"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head-j1", type=float, default=0.55,
                   help="downward-looking head_j1; onsite +0.55 looks down")
    p.add_argument("--plane-z", type=float, default=None,
                   help="board/table plane Z in base frame; defaults to measured task floor")
    p.add_argument("--hover-z", type=float, default=0.55,
                   help="flat TCP benchmark plane; must stay <=10 cm above configured floor")
    p.add_argument("--claw-yaw-deg", type=float, default=0.0,
                   help="fixed in-plane gripper yaw for the whole benchmark")
    p.add_argument("--speed-scale", type=float, default=0.90)
    p.add_argument("--output", default="calibration/vega_board_live.json")
    p.add_argument("--reuse-registration", action="store_true",
                   help="skip head capture and reuse the saved board JSON")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--center-only", action="store_true",
                   help="move only to board center; skip all corner planning")
    p.add_argument("--confirm-physical-motion", action="store_true")
    args = p.parse_args(argv)

    bundle = load_bundle("vega")
    cfg = bundle["robot"]
    robot_name = cfg["robot_name"]

    skills_cfg = load_vega_skills()
    safety = dict(skills_cfg.get("safety") or {})
    floor = float(safety["min_tcp_z_m"])
    plane_z = (
        configured_board_plane_z(cfg, floor)
        if args.plane_z is None
        else float(args.plane_z)
    )
    if not (floor + 0.03 <= float(args.hover_z) <= floor + 0.10 + 1e-9):
        raise SystemExit(
            f"--hover-z must stay 3-10 cm above hard TCP floor {floor:.6f} m; "
            f"requested {float(args.hover_z):.6f} m"
        )

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output

    if args.reuse_registration:
        if not output.is_file():
            raise SystemExit(f"--reuse-registration requested but {output} does not exist")
        record = json.loads(output.read_text(encoding="utf-8"))
        import numpy as np
        labels = ("tl", "tr", "br", "bl")
        corners_base = np.asarray(
            [record["corners_base_m"][label] for label in labels], dtype=float
        )
        center = np.asarray(record["center_base_m"], dtype=float)
        print("REUSING BOARD REGISTRATION", output)
        print("BOARD CENTER BASE =", tuple(round(float(x), 5) for x in center))
        for label, point in zip(labels, corners_base):
            print(
                f"BOARD {label.upper()} BASE =",
                tuple(round(float(x), 5) for x in point),
            )
    else:
        print("Capturing downward head view...")
        head_q, frame = _set_head_and_capture(cfg, args.head_j1, floor)
        fx, fy, cx, cy = intrinsics_from_camera_info(frame.camera_info)

        pixels = detect_white_board_corners(frame.left_rgb)
        labels = ("tl", "tr", "br", "bl")
        print("BOARD PIXELS =", dict(zip(labels, pixels)))

        T_base_camera = head_left_optical_transform(
            head_q,
            lift_m=float(cfg["kinematics"]["fixed_joint_values"]["Lift"]),
            torso_flip_rad=float(
                cfg["kinematics"]["fixed_joint_values"]["torso_flip"]
            ),
        )
        corners_base = pixels_to_horizontal_plane(
            pixels,
            fx=fx, fy=fy, cx=cx, cy=cy,
            T_base_camera=T_base_camera,
            plane_z_m=plane_z,
        )
        T_base_board, width_m, height_m = board_frame_from_corners(corners_base)
        center = T_base_board[:3, 3]

        print("BOARD CENTER BASE =", tuple(round(float(x), 5) for x in center))
        for label, point in zip(labels, corners_base):
            print(
                f"BOARD {label.upper()} BASE =",
                tuple(round(float(x), 5) for x in point),
            )
        print(f"BOARD SIZE ~= {width_m:.3f} x {height_m:.3f} m")

        output.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            "robot_id": "vega",
            "robot_name": robot_name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_frame": cfg["kinematics"]["base_frame"],
            "camera_frame": "zed_left_camera_optical",
            "method": "head white-board quadrilateral + horizontal-plane ray intersection",
            "calibration_status": "coarse_source_robot_camera_reference_normalized_by_live_head_q",
            "warning": (
                "Absolute head-camera calibration originates from organizer source robot "
                "dm/vg3eb20f25bb-1u. Use for coarse board registration; wrist visual "
                "servo should perform fine part centering."
            ),
            "head_q_rad": _serializable(head_q),
            "head_j1_downward_sign_verified_onsite": "positive",
            "plane_z_m": plane_z,
            "hover_z_m": float(args.hover_z),
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "board_pixels": {
                label: [int(u), int(v)]
                for label, (u, v) in zip(labels, pixels)
            },
            "corners_base_m": {
                label: [float(x) for x in point]
                for label, point in zip(labels, corners_base)
            },
            "center_base_m": [float(x) for x in center],
            "width_m": width_m,
            "height_m": height_m,
            "T_base_board_center": _serializable(T_base_board),
            "T_base_camera": _serializable(T_base_camera),
        }
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print("WROTE", output)

    if not args.execute:
        print("Registration complete. Re-run with --execute --confirm-physical-motion for benchmark.")
        return 0
    if not args.confirm_physical_motion:
        raise SystemExit("--execute requires --confirm-physical-motion")

    # Restore the complete pose which produced this registration. Robot() may
    # change any head joint; retaining its J2/J3 can point the camera sideways.
    saved_head = record.get("head_q_rad")
    if (not isinstance(saved_head, (list, tuple)) or len(saved_head) != 3
            or not all(math.isfinite(float(v)) for v in saved_head)):
        raise SystemExit("Board registration lacks valid head_q_rad; capture a verified board view first")

    # Benchmark path intentionally uses only the arm. Do not home the gripper.
    cfg["allow_robot_init_head_motion"] = True
    # Fewer controller stop/start segments make the free-space benchmark smoother.
    cfg["motion"]["max_step_rad"] = max(float(cfg["motion"]["max_step_rad"]), 0.30)

    robot = VegaAdapter(cfg)
    robot.prepare()
    try:
        robot.connect()
        # Robot() homes the head forward. Re-aim it down AFTER connecting so the
        # head stays on the board during the benchmark and is ready for the next
        # perception step.
        import numpy as np
        head_q = np.asarray(saved_head, dtype=float)
        print("RESTORING REGISTERED HEAD POSE =", head_q, flush=True)
        robot._robot.head.set_joint_pos(
            head_q,
            wait_time=1.2,
            exit_on_reach=True,
            exit_on_reach_kwargs={"tolerance": 0.02},
        )
        print("HEAD DOWN =", robot._robot.head.get_joint_pos(), flush=True)

        fixed_yaw = math.radians(float(args.claw_yaw_deg))
        points = [
            ("center", center),
            ("tl", corners_base[0]),
            ("tr", corners_base[1]),
            ("br", corners_base[2]),
            ("bl", corners_base[3]),
            ("center", center),
        ]
        if args.center_only:
            points = points[:1]
        # Approach the requested low working plane. Do not rise above it; the
        # user wants the arm operating like a 3D-printer nozzle within 10 cm of
        # the configured floor. The first CENTER move establishes the plane.
        for label, point in points:
            print(f"PLANNING {label.upper()}...", flush=True)
            target = _vertical_target_for_point(
                robot,
                label,
                point,
                center,
                float(args.hover_z),
                floor,
                fixed_yaw,
            )
            print(
                f"MOVE {label.upper()} ->",
                tuple(round(float(x), 4) for x in target.position_m),
                flush=True,
            )
            move_tcp_segmented(
                robot,
                target,
                speed_scale=float(args.speed_scale),
                max_translation_step_m=0.20,
                max_orientation_step_rad=0.80,
                min_tcp_z_m=floor,
            )
            actual = robot.get_tcp_pose()
            print(
                f"REACHED {label.upper()} =",
                tuple(round(float(x), 4) for x in actual.position_m),
                flush=True,
            )

        print("BOARD BENCHMARK COMPLETE")
        return 0
    except BaseException:
        try:
            robot.stop()
        except BaseException as stop_error:
            print(f"STOP FAILED: {stop_error}; use physical e-stop", file=sys.stderr)
        raise
    finally:
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
