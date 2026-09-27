"""Register the physical task board and benchmark vertical left-claw reach.

Milestone:
  1. Aim the head camera downward.
  2. Detect the large white task-board quadrilateral.
  3. Map its four corners + center into vega_1u_base_link.
  4. Save that board registration.
  5. Optionally move the left TCP, held vertical, through:
       center -> TL -> TR -> BR -> BL -> center

This is intentionally the coarse/global layer. Fine object centering should use
the left wrist camera after this board frame exists.
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
from steadyhand.executor import move_tcp_segmented
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills
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


def _set_head_and_capture(robot_name, head_j1):
    import numpy as np
    from dexcontrol.robot import Robot

    os.environ.setdefault("ROBOT_NAME", robot_name)
    robot = Robot()
    camera = None
    try:
        q = np.asarray(robot.head.get_joint_pos(), dtype=float)
        if head_j1 is not None:
            q[0] = float(head_j1)
            robot.head.set_joint_pos(
                q,
                wait_time=1.5,
                exit_on_reach=True,
                exit_on_reach_kwargs={"tolerance": 0.02},
            )
            q = np.asarray(robot.head.get_joint_pos(), dtype=float)

        camera = VegaHeadCamera()
        camera.connect()
        frame = camera.read(include_depth=False)
        return q, frame
    finally:
        if camera is not None:
            camera.close()
        robot.shutdown()


def _yaw_quat(yaw):
    return (
        math.cos(float(yaw) / 2.0),
        0.0,
        0.0,
        math.sin(float(yaw) / 2.0),
    )


def _pose_at(point, z, quat):
    return Pose(
        (float(point[0]), float(point[1]), float(z)),
        tuple(float(x) for x in quat),
    )


def _select_vertical_yaw(robot, points, hover_z):
    """Choose a vertical-claw yaw for which the entire benchmark path has IK."""
    import numpy as np

    candidates = (0.0, math.pi / 2, -math.pi / 2, math.pi)
    initial_seed = robot._read_joint_positions()
    feasible = []

    for yaw in candidates:
        quat = _yaw_quat(yaw)
        seed = initial_seed
        cost = 0.0
        solved = []
        try:
            for _, point in points:
                target = _pose_at(point, hover_z, quat)
                answer = robot._kinematics.solve(target, seed)
                cost += float(np.linalg.norm(np.asarray(answer) - np.asarray(seed)))
                solved.append(answer)
                seed = answer
        except Exception as exc:
            print(f"vertical yaw {math.degrees(yaw):+.0f} deg: no ({exc})")
            continue
        print(f"vertical yaw {math.degrees(yaw):+.0f} deg: IK OK")
        feasible.append((cost, yaw, quat, solved))

    if not feasible:
        raise RuntimeError(
            "No tested vertical-claw yaw reaches center + all four board corners "
            "at the requested hover Z"
        )
    feasible.sort(key=lambda item: item[0])
    _, yaw, quat, _ = feasible[0]
    print(f"SELECTED VERTICAL YAW = {math.degrees(yaw):+.0f} deg")
    return quat


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head-j1", type=float, default=0.55,
                   help="downward-looking head_j1; onsite +0.55 looks down")
    p.add_argument("--plane-z", type=float, default=None,
                   help="board/table plane Z in base frame; defaults to measured task floor")
    p.add_argument("--hover-z", type=float, default=0.62,
                   help="TCP benchmark height in base frame")
    p.add_argument("--speed-scale", type=float, default=0.90)
    p.add_argument("--output", default="calibration/vega_board_live.json")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm-physical-motion", action="store_true")
    args = p.parse_args(argv)

    bundle = load_bundle("vega")
    cfg = bundle["robot"]
    robot_name = cfg["robot_name"]

    skills_cfg = load_vega_skills()
    safety = dict(skills_cfg.get("safety") or {})
    floor = float(safety["min_tcp_z_m"])
    plane_z = floor if args.plane_z is None else float(args.plane_z)
    if args.hover_z <= floor:
        raise SystemExit(
            f"--hover-z must be above hard TCP floor {floor:.6f} m"
        )

    print("Capturing downward head view...")
    head_q, frame = _set_head_and_capture(robot_name, args.head_j1)
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

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
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

    # Benchmark path intentionally uses only the arm. Do not home the gripper.
    cfg["allow_robot_init_head_motion"] = True
    robot = VegaAdapter(cfg)
    robot.prepare()
    robot.connect()
    try:
        points = [
            ("center", center),
            ("tl", corners_base[0]),
            ("tr", corners_base[1]),
            ("br", corners_base[2]),
            ("bl", corners_base[3]),
            ("center", center),
        ]
        quat = _select_vertical_yaw(robot, points, float(args.hover_z))

        # First create clearance straight upward at current XY/orientation.
        current = robot.get_tcp_pose()
        clearance = Pose(
            (
                float(current.position_m[0]),
                float(current.position_m[1]),
                max(float(current.position_m[2]), float(args.hover_z)),
            ),
            current.quaternion_wxyz,
        )
        if float(clearance.position_m[2]) > float(current.position_m[2]) + 1e-4:
            print("CLEARANCE UP")
            move_tcp_segmented(
                robot,
                clearance,
                speed_scale=float(args.speed_scale),
                max_translation_step_m=0.08,
                max_orientation_step_rad=0.35,
                min_tcp_z_m=floor,
            )

        for label, point in points:
            target = _pose_at(point, float(args.hover_z), quat)
            print(
                f"MOVE {label.upper()} ->",
                tuple(round(float(x), 4) for x in target.position_m),
            )
            move_tcp_segmented(
                robot,
                target,
                speed_scale=float(args.speed_scale),
                max_translation_step_m=0.08,
                max_orientation_step_rad=0.35,
                min_tcp_z_m=floor,
            )
            actual = robot.get_tcp_pose()
            print(
                f"REACHED {label.upper()} =",
                tuple(round(float(x), 4) for x in actual.position_m),
            )

        print("BOARD BENCHMARK COMPLETE")
        return 0
    finally:
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
