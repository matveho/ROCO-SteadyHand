"""Interactive physical calibration of Vega head->board coarse registration.

Flow:
  1. Clear the right claw from the head-board view.
  2. Capture the board exactly like vega_board_axis_benchmark.py.
  3. Move to the predicted CENTER and BOARD_X_PLUS hover targets. BOARD_Y_PLUS
     is derived from the head-predicted axis by default because the far Y reach
     is not reliably available on this right-arm setup.
  4. At each target, let the operator jog the physical claw center relative to
     the board:
       forward N   (+base X, away from robot)
       back N      (-base X, toward robot)
       left N      (+base Y, robot-left)
       right N     (-base Y, robot-right)
     Distances are millimetres. Repeated commands are allowed. A measured
     forward/rise coupling can be compensated by adding signed TCP Z correction
     to forward/back jogs so the physical claw center stays board-parallel.
  5. 'done' records the measured TCP for that reference.
  6. Print a paste-ready JSON calibration block containing predicted and corrected
     points, corrected board axes, TCP quaternions, and every manual jog.

This tool calibrates BOARD POSITION/AXES only. It preserves TCP orientation.
The operator measured the physical claw-center clearance as -5 mm at the
BOTTOM_RIGHT, 45 mm at CENTER, and 84 mm at TOP_RIGHT across the 386 mm forward
span. The resulting measured forward-rise slope is used for forward/back Z
compensation. A visibly angled claw is still reported, not silently treated as
a calibrated vertical orientation.

With --use-current-right-ready, the post-image automatic verticalization is
skipped. The operator manually establishes a high, downward-facing RIGHT_READY
pose, then the live seven right-arm joints and modeled tip_r pose are recorded
as provenance for the calibration. This is the preferred route when the robot
starts in the folded near-limit configuration.
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
from steadyhand.vega_camera_clear import (
    move_camera_clear_for_image,
    move_verticalize_after_image,
)
from steadyhand.vega_presets import configured_right_preset, preset_max_delta
from tools.vega_scene_perception import _ensure_publisher


ROOT = Path(__file__).resolve().parents[1]

# Operator-measured physical claw-center clearance change across the board.
# Active right-arm board-plane correction from the latest supervised survey:
# TOP_RIGHT=85 mm, BOTTOM_RIGHT=0 mm across the 386 mm field.
MEASURED_NEAR_CLAW_HEIGHT_MM = 0.0
MEASURED_FAR_CLAW_HEIGHT_MM = 85.0
MEASURED_FORWARD_SPAN_MM = 386.0
ORIENTATION_TEACH_MIN_ABOVE_FLOOR_M = 0.30
# The measured competition RIGHT_READY is a board-working pose about 0.15 m
# above the provisional floor, rather than the earlier high image-clear pose.
RIGHT_READY_MIN_ABOVE_FLOOR_M = 0.10

DEFAULT_FORWARD_RISE_ANGLE_DEG = math.degrees(
    math.atan(
        (MEASURED_FAR_CLAW_HEIGHT_MM - MEASURED_NEAR_CLAW_HEIGHT_MM)
        / MEASURED_FORWARD_SPAN_MM
    )
)
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


def _quat_multiply(a, b):
    """Hamilton product for normalized-or-near-normalized wxyz quaternions."""
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    q = (
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    )
    n = math.sqrt(sum(v*v for v in q))
    if not math.isfinite(n) or n <= 0:
        raise ValueError("invalid quaternion product")
    return tuple(v / n for v in q)


def _base_axis_delta_quaternion(axis, degrees):
    angle = math.radians(float(degrees))
    half = angle / 2.0
    s = math.sin(half)
    if axis == "roll":
        return (math.cos(half), s, 0.0, 0.0)
    if axis == "pitch":
        return (math.cos(half), 0.0, s, 0.0)
    if axis == "yaw":
        return (math.cos(half), 0.0, 0.0, s)
    raise ValueError(f"unknown rotation axis {axis!r}")


def _rotate_quaternion_in_base(quaternion_wxyz, axis, degrees):
    """Apply a small base-frame roll/pitch/yaw rotation to a tip quaternion."""
    return _quat_multiply(
        _base_axis_delta_quaternion(axis, degrees),
        quaternion_wxyz,
    )


def _coarse_target_preserving_orientation(point, hover_z, taught_pose):
    """Build a board-calibration coarse target using operator-taught orientation."""
    return Pose(
        (float(point[0]), float(point[1]), float(hover_z)),
        tuple(float(v) for v in taught_pose.quaternion_wxyz),
    )


def _predicted_y_reference(center_pose, board_y_unit_base, offset_m):
    """Synthesize an unvisited Y reference from corrected CENTER + head axis."""
    c = tuple(float(v) for v in center_pose.position_m)
    by = tuple(float(v) for v in board_y_unit_base)
    if len(c) != 3 or len(by) != 3 or not all(math.isfinite(v) for v in (*c, *by)):
        raise ValueError("Y reference requires finite CENTER and board axis")
    if abs(float(by[2])) > 1e-6:
        raise ValueError("board Y reference must be a planar base-frame axis")
    norm = math.hypot(by[0], by[1])
    if norm <= 1e-9 or not math.isfinite(float(offset_m)) or float(offset_m) <= 0:
        raise ValueError("board Y reference requires a nonzero planar axis and offset")
    ux, uy = by[0] / norm, by[1] / norm
    return Pose(
        (c[0] + float(offset_m) * ux, c[1] + float(offset_m) * uy, c[2]),
        tuple(float(v) for v in center_pose.quaternion_wxyz),
    )


def _interactive_teach_orientation(robot, *, floor, speed_scale=0.60, max_step_deg=10.0):
    """Teach a physically acceptable board-working orientation at a high pose."""
    current = robot.get_tcp_pose()
    if current is None:
        raise RuntimeError("orientation teach requires current TCP pose")
    # Camera-clear's validated fallback leaves the TCP about 0.30-0.35 m
    # above the measured floor. That provides substantial free-space clearance
    # for the small orientation-only teach steps without forcing another lift.
    min_teach_z = float(floor) + ORIENTATION_TEACH_MIN_ABOVE_FLOOR_M
    if float(current.position_m[2]) < min_teach_z:
        raise RuntimeError(
            f"orientation teach requires TCP z >= {min_teach_z:.3f} m; "
            f"current z={float(current.position_m[2]):.3f} m"
        )

    print("", flush=True)
    print("=== TEACH BOARD-WORKING CLAW ORIENTATION ===", flush=True)
    print(
        "At this HIGH free-space pose, visually rotate the physical claw until "
        "the jaw/approach axis points downward toward the board.",
        flush=True,
    )
    print(
        "Commands: roll N | pitch N | yaw N (signed degrees, "
        f"|N| <= {float(max_step_deg):g}), status, done, abort",
        flush=True,
    )
    print(
        "Rotations are about BASE axes and hold the modeled tip_r position fixed. "
        "Use small steps and the hardware e-stop if motion is unexpected.",
        flush=True,
    )
    _print_pose("ORIENTATION START", current)

    while True:
        raw = input("ORIENTATION> ").strip()
        if not raw:
            continue
        parts = raw.lower().split()
        cmd = parts[0]
        if cmd in ("done", "d"):
            taught = robot.get_tcp_pose()
            _print_pose("TAUGHT BOARD ORIENTATION", taught)
            return taught
        if cmd in ("abort", "quit", "q"):
            raise KeyboardInterrupt()
        if cmd in ("status", "s"):
            _print_pose("CURRENT", robot.get_tcp_pose())
            continue
        if cmd not in ("roll", "pitch", "yaw") or len(parts) != 2:
            print(
                "INVALID: use e.g. 'pitch 5', 'roll -5', 'yaw 5', "
                "'status', 'done', or 'abort'",
                flush=True,
            )
            continue
        try:
            degrees = float(parts[1])
        except ValueError:
            print("INVALID ANGLE: enter signed degrees as a number", flush=True)
            continue
        if (
            not math.isfinite(degrees)
            or degrees == 0.0
            or abs(degrees) > float(max_step_deg)
        ):
            print(
                f"INVALID ANGLE: require 0 < |N| <= {float(max_step_deg):g} deg",
                flush=True,
            )
            continue

        current = robot.get_tcp_pose()
        target = Pose(
            tuple(float(v) for v in current.position_m),
            _rotate_quaternion_in_base(
                current.quaternion_wxyz, cmd, degrees
            ),
        )
        print(
            f"ORIENTATION {cmd.upper()} {degrees:+g} deg; holding xyz ->",
            tuple(round(float(v), 6) for v in target.position_m),
            flush=True,
        )
        move_tcp_segmented(
            robot,
            target,
            speed_scale=float(speed_scale),
            max_translation_step_m=0.02,
            max_orientation_step_rad=math.radians(5.0),
            min_tcp_z_m=floor,
        )
        _print_pose("MEASURED", robot.get_tcp_pose())


def _capture_current_right_ready(robot, *, floor):
    """Accept an operator-taught ready pose without commanding a reorientation.

    The robot is under manual control while this prompt is active. Reading the
    joints only after the operator confirms avoids recording a stale shoulder
    step or an intermediate pose. This helper deliberately validates height and
    finite state but does not infer physical claw orientation from the URDF.
    """
    minimum_z = float(floor) + RIGHT_READY_MIN_ABOVE_FLOOR_M
    print("", flush=True)
    print("=== MANUAL RIGHT_READY ===", flush=True)
    print(
        "Using manual robot control, place the RIGHT claw in the measured "
        "board-ready pose with the physical jaws/approach axis pointing down.",
        flush=True,
    )
    print(
        f"Keep the TCP at or above z={minimum_z:.3f} m, leave comfortable joint "
        "margins, and keep the board visible. No arm motion will be commanded "
        "by this step.",
        flush=True,
    )
    input("When RIGHT_READY is physically established, press Enter to record it: ")

    joints = tuple(float(v) for v in robot._read_joint_positions())
    if len(joints) != 7 or any(not math.isfinite(v) for v in joints):
        raise RuntimeError("RIGHT_READY requires seven finite measured right-arm joints")
    pose = robot.get_tcp_pose()
    if pose is None:
        raise RuntimeError("RIGHT_READY requires a measured tip_r pose")
    if float(pose.position_m[2]) < minimum_z:
        raise RuntimeError(
            f"RIGHT_READY TCP z={float(pose.position_m[2]):.6f} m is below "
            f"the required high-pose minimum {minimum_z:.6f} m"
        )

    print("RIGHT_READY JOINTS =", list(joints), flush=True)
    _print_pose("RIGHT_READY TIP_R", pose)
    return joints, pose


def _move_configured_right_ready(robot, *, floor, speed_scale=0.45):
    """Move to the operator-measured RIGHT_READY endpoint without IK."""
    try:
        target_q, measured_pose = configured_right_preset(robot.config, "right_ready")
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"right_ready preset is invalid: {exc}") from exc
    if float(measured_pose.position_m[2]) < float(floor):
        raise RuntimeError("configured RIGHT_READY pose is below the TCP floor")
    current_q = robot._read_joint_positions()
    delta = preset_max_delta(current_q, target_q)
    max_delta = float(robot.config["motion"]["max_total_delta_rad"])
    if delta > max_delta:
        raise RuntimeError(
            "RIGHT_READY is too far from the live joint state for a validated "
            f"joint move ({delta:.3f} rad > {max_delta:.3f} rad); manually "
            "place the arm at RIGHT_READY and rerun"
        )
    print("RIGHT_READY TARGET Q =", list(target_q), flush=True)
    print("MOVING TO RIGHT_READY", flush=True)
    robot.move_joints(target_q, speed_scale=float(speed_scale))
    joints = tuple(float(v) for v in robot._read_joint_positions())
    pose = robot.get_tcp_pose()
    if pose is None or float(pose.position_m[2]) < float(floor):
        raise RuntimeError("measured RIGHT_READY move did not produce a valid TCP pose")
    print("RIGHT_READY JOINTS =", list(joints), flush=True)
    _print_pose("RIGHT_READY TIP_R", pose)
    return joints, pose


def _board_parallel_jog_delta(command, mm, forward_rise_angle_deg):
    """Return commanded base-frame dx,dy,dz for a board-parallel claw-center jog.

    Physical observation: with dz=0, moving +base-X ("forward") raises the
    physical center of the claw. Compensate by commanding TCP downward:
        dz = -dx * tan(angle)
    Backward motion receives the exact inverse. Left/right currently have no
    measured vertical coupling and therefore use dz=0.
    """
    dx_unit, dy_unit = DIRECTIONS[command]
    distance_m = float(mm) / 1000.0
    dx = float(dx_unit) * distance_m
    dy = float(dy_unit) * distance_m
    angle_rad = math.radians(float(forward_rise_angle_deg))
    dz = -dx * math.tan(angle_rad)
    return dx, dy, dz


def _interactive_adjust(
    robot, *, label, floor, speed_scale, max_jog_mm,
    forward_rise_angle_deg, events
):
    print("", flush=True)
    print(f"=== MANUAL CALIBRATION: {label} ===", flush=True)
    print(
        "Commands: forward N | back N | left N | right N  (N in mm), "
        "status, done, abort",
        flush=True,
    )
    print("forward=away from robot (+base X), left=robot-left (+base Y)", flush=True)
    print(
        "BOARD-PARALLEL COMPENSATION: forward physical rise "
        f"{float(forward_rise_angle_deg):.2f} deg; "
        "commanded forward includes downward TCP Z",
        flush=True,
    )
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
        dx, dy, dz = _board_parallel_jog_delta(
            cmd, mm, forward_rise_angle_deg
        )
        target = Pose(
            (
                float(current.position_m[0]) + dx,
                float(current.position_m[1]) + dy,
                float(current.position_m[2]) + dz,
            ),
            tuple(float(v) for v in current.quaternion_wxyz),
        )
        if float(target.position_m[2]) < float(floor):
            print(
                f"JOG REJECTED: compensated target z={target.position_m[2]:.6f} "
                f"is below floor {float(floor):.6f}",
                flush=True,
            )
            continue
        print(
            f"JOG {cmd.upper()} {mm:g} mm "
            f"(dx={dx*1000:+.1f}, dy={dy*1000:+.1f}, "
            f"dz_comp={dz*1000:+.1f} mm) ->",
            tuple(round(float(v), 6) for v in target.position_m),
            flush=True,
        )
        # Tight IK is essential for 5-10 mm manual jogs. A 10 mm IK tolerance
        # can legally return the current seed and produce almost no motion.
        # Preflight the complete target and turn an unreachable overshoot into
        # an interactive rejection. This is common when an operator walks past
        # a board reference with repeated 50 mm jogs; no arm command is sent in
        # that case, so the operator can recover with a smaller inverse jog.
        try:
            robot._kinematics.solve(target, robot._read_joint_positions())
        except Exception as exc:
            print(
                "JOG REJECTED BEFORE MOTION: target was not IK-reachable; "
                f"try a smaller/inverse jog ({exc})",
                flush=True,
            )
            continue
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
            "commanded_delta_m": [float(dx), float(dy), float(dz)],
            "forward_rise_compensation_deg": float(forward_rise_angle_deg),
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
    p.add_argument(
        "--forward-rise-angle-deg",
        type=float,
        default=DEFAULT_FORWARD_RISE_ANGLE_DEG,
        help=(
            "physical claw-center rise angle during +base-X motion; "
            "default is derived from onsite 0 mm bottom-right / 85 mm "
            "top-right over the 386 mm span"
        ),
    )
    p.add_argument(
        "--claw-yaw-deg",
        type=float,
        default=0.0,
        help="legacy compatibility option; board calibration preserves live TCP orientation",
    )
    p.add_argument(
        "--use-current-right-ready",
        action="store_true",
        help=(
            "after the head image, pause for a manually taught high RIGHT_READY "
            "pose and skip automatic post-image verticalization"
        ),
    )
    p.add_argument(
        "--use-configured-right-ready",
        action="store_true",
        help=(
            "after the head image, move to the measured RIGHT_READY joint "
            "preset without invoking Cartesian IK"
        ),
    )
    p.add_argument(
        "--include-board-y-plus",
        action="store_true",
        help=(
            "physically visit BOARD_Y_PLUS; omitted by default because the "
            "right arm cannot reliably reach that reference"
        ),
    )
    p.add_argument("--settle-s", type=float, default=0.5)
    p.add_argument("--publisher-log", default="~/head_camera.log")
    p.add_argument("--output", default="calibration/vega_board_manual.json")
    p.add_argument("--confirm-physical-motion", action="store_true")
    args = p.parse_args(argv)

    if not args.confirm_physical_motion:
        p.error("--confirm-physical-motion is required")
    if args.use_current_right_ready and args.use_configured_right_ready:
        p.error("choose only one RIGHT_READY mode")
    if not 0.05 <= float(args.offset_m) <= 0.12:
        p.error("--offset-m must be 0.05..0.12 m")
    if not 0.45 <= float(args.coarse_speed_scale) <= 1.0:
        p.error("--coarse-speed-scale must be 0.45..1.0")
    if not 0.45 <= float(args.jog_speed_scale) <= 1.0:
        p.error("--jog-speed-scale must be 0.45..1.0")
    if not 1.0 <= float(args.max_jog_mm) <= 100.0:
        p.error("--max-jog-mm must be 1..100 mm")
    if not math.isfinite(float(args.forward_rise_angle_deg)) or not (
        -30.0 <= float(args.forward_rise_angle_deg) <= 30.0
    ):
        p.error("--forward-rise-angle-deg must be finite and within -30..30 deg")

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
    # Operator-supervised calibration showed stationary endpoint residuals up
    # to 0.013057 rad. Accept 0.015 rad here only; normal manipulation keeps its
    # configured tighter threshold.
    cfg["motion"]["joint_reached_tolerance_rad"] = max(
        float(cfg["motion"]["joint_reached_tolerance_rad"]), 0.015
    )

    _ensure_publisher(args.publisher_log, robot_name=cfg["robot_name"])

    robot = VegaAdapter(cfg)
    camera = None
    events = []
    try:
        robot.connect()
        print(
            "MEASURED BOARD-PARALLEL SLOPE: "
            f"near={MEASURED_NEAR_CLAW_HEIGHT_MM:.1f} mm, "
            f"far={MEASURED_FAR_CLAW_HEIGHT_MM:.1f} mm, "
            f"span={MEASURED_FORWARD_SPAN_MM:.1f} mm, "
            f"angle={DEFAULT_FORWARD_RISE_ANGLE_DEG:.3f} deg",
            flush=True,
        )
        move_camera_clear_for_image(robot, floor_m=floor, speed_scale=0.90)

        target_head = np.asarray([0.55, 0.0, 0.0], dtype=float)
        print("HEAD BEFORE =", robot._robot.head.get_joint_pos(), flush=True)
        move_head = getattr(robot._robot.head, "move_to_joint_pos", None)
        moved_head = False
        if callable(move_head):
            try:
                handle = move_head(target_head, velocity_scale=0.45)
                wait_fn = getattr(handle, "wait", None)
                if callable(wait_fn):
                    wait_fn(timeout=5.0)
                else:
                    time.sleep(1.5)
                moved_head = True
            except RuntimeError as exc:
                print(
                    f"HEAD MOTION HANDLE FAILED ({exc}); "
                    "falling back to verified set_joint_pos path",
                    flush=True,
                )
        if not moved_head:
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

        print("HEAD IMAGE CAPTURE COMPLETE", flush=True)
        right_ready_joints = None
        if args.use_configured_right_ready:
            right_ready_joints, coarse_orientation_pose = _move_configured_right_ready(
                robot,
                floor=floor,
                speed_scale=0.45,
            )
            orientation_status = (
                "OPERATOR_MEASURED_RIGHT_READY_PRESET; "
                "automatic post-image verticalization skipped; "
                "physical claw orientation is operator-verified"
            )
        elif args.use_current_right_ready:
            right_ready_joints, coarse_orientation_pose = _capture_current_right_ready(
                robot,
                floor=floor,
            )
            orientation_status = (
                "OPERATOR_ACCEPTED_MANUAL_RIGHT_READY; "
                "automatic post-image verticalization skipped; "
                "physical claw orientation is operator-verified"
            )
            print(
                "COARSE ORIENTATION: preserving manually accepted RIGHT_READY tip_r quaternion =",
                tuple(round(float(v), 7) for v in coarse_orientation_pose.quaternion_wxyz),
                flush=True,
            )
        else:
            print("VERTICALIZING ARM", flush=True)
            coarse_orientation_pose = move_verticalize_after_image(
                robot,
                floor_m=floor,
                speed_scale=0.90,
            )
            orientation_status = (
                "RESTORED_POST_IMAGE_VERTICALIZATION_SEQUENCE; "
                "uses the previously successful Vega verticalization routine"
            )
            print(
                "COARSE ORIENTATION: using restored post-image verticalized quaternion =",
                tuple(round(float(v), 7) for v in coarse_orientation_pose.quaternion_wxyz),
                flush=True,
            )
        if abs(float(args.claw_yaw_deg)) > 1e-12:
            print(
                "NOTE: --claw-yaw-deg is ignored during board calibration; "
                "the accepted working orientation is used",
                flush=True,
            )

        corrected = {}
        labels = ["CENTER", "BOARD_X_PLUS"]
        if args.include_board_y_plus:
            labels.append("BOARD_Y_PLUS")
        else:
            print(
                "SKIPPING BOARD_Y_PLUS physical jog; using the head-predicted "
                "planar Y axis from corrected CENTER",
                flush=True,
            )
        for label in labels:
            point = predicted[label]
            target = _coarse_target_preserving_orientation(
                point, args.hover_z, coarse_orientation_pose
            )
            robot._kinematics.config["position_tolerance_m"] = 0.0007
            robot._kinematics.config["orientation_tolerance_rad"] = 0.02
            robot._kinematics.config["max_seed_delta_rad"] = 2.4
            robot._kinematics.config["max_iterations"] = 180

            print(
                f"COARSE MOVE {label} ->",
                tuple(round(float(v), 6) for v in target.position_m),
                flush=True,
            )
            move_tcp_segmented(
                robot,
                target,
                speed_scale=float(args.coarse_speed_scale),
                max_translation_step_m=0.12,
                max_orientation_step_rad=0.10,
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
                forward_rise_angle_deg=float(args.forward_rise_angle_deg),
                events=events,
            )

        c = np.asarray(corrected["CENTER"].position_m, dtype=float)
        x = np.asarray(corrected["BOARD_X_PLUS"].position_m, dtype=float)
        if not args.include_board_y_plus:
            corrected["BOARD_Y_PLUS"] = _predicted_y_reference(
                corrected["CENTER"], by, args.offset_m
            )
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
            "physical_claw_center_jog_compensation": {
                "forward_rise_angle_deg": float(args.forward_rise_angle_deg),
                "model": "dz_commanded=-dx_base*tan(angle)",
                "applies_to": "manual forward/back jogs only",
                "status": (
                    "DISABLED_UNVERIFIED_FOR_RIGHT_ARM"
                    if abs(float(args.forward_rise_angle_deg)) <= 1e-12
                    else "OPERATOR_SELECTED_CALIBRATION_VALUE"
                ),
                "source_measurements_mm": {
                    "near_robot_edge_claw_height": MEASURED_NEAR_CLAW_HEIGHT_MM,
                    "far_edge_claw_height": MEASURED_FAR_CLAW_HEIGHT_MM,
                    "forward_span": MEASURED_FORWARD_SPAN_MM,
                },
            },
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
            "orientation_status": orientation_status,
            "board_y_reference_status": (
                "PHYSICALLY_VISITED_OPERATOR_CORRECTED"
                if args.include_board_y_plus
                else "HEAD_PREDICTED_AXIS_FROM_CORRECTED_CENTER; NOT_PHYSICALLY_VISITED"
            ),
            "coarse_preserved_quaternion_wxyz": [
                float(v) for v in coarse_orientation_pose.quaternion_wxyz
            ],
            "jogs": events,
        }
        if right_ready_joints is not None:
            record["right_ready"] = {
                "joint_names": list(cfg["kinematics"]["right_arm_joint_names"]),
                "joint_positions_rad": list(right_ready_joints),
                "tip_r": _pose_record(coarse_orientation_pose),
                "source": "operator_taught_current_pose_after_head_image",
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
