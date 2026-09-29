"""Calibrate the live Vega board from a camera read and four TCP references.

The head camera supplies only initial estimates. The operator then corrects the
right TCP at CENTER, TOP_RIGHT, BOTTOM_RIGHT, and BOTTOM_LEFT with small
forward/back/left/right jogs. At each point the operator enters the measured
TCP-to-board clearance in millimetres; the calibrated TCP pose converts that
clearance to board surface height. The output fits a planar board surface and
stores the corrected board axes for later task motion.
No gripper command or manual claw operation is used.
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
from steadyhand.board_geometry import configured_board_plane_z
from steadyhand.board_calibration import orthonormalize_xy_axes
from steadyhand.executor import move_tcp_segmented
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills
from steadyhand.vega_camera_clear import move_camera_clear_for_image
from steadyhand.vega_presets import configured_right_preset
from steadyhand.vision.scene import detect_head_task_scene
from tools.vega_board_manual_calibrate import (
    _ensure_publisher,
    _interactive_adjust,
    _move_configured_right_ready,
    _pose_record,
    _print_pose,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("CENTER", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT")


def _target(point, hover_z, quaternion, *, z_offset_m=0.0):
    return Pose(
        tuple(float(v) for v in (point[0], point[1], float(hover_z) + float(z_offset_m))),
        tuple(quaternion),
    )


def _reachable_initial_target(
    robot, point, center, hover_z, quaternion, label, *, z_offset_m=0.0
):
    """Preflight an initial camera target, insetting only when necessary."""
    import numpy as np
    point = np.asarray(point, dtype=float)
    center = np.asarray(center, dtype=float)
    seed = robot._read_joint_positions()
    last_error = None
    alphas = (1.0,) if label == "CENTER" else (1.0, 0.96, 0.92, 0.88, 0.84, 0.80)
    for alpha in alphas:
        # The board target is planar XY; do not add a 2-vector to the
        # camera-read 3-vector (which also contains the provisional plane Z).
        xy = center[:2] + float(alpha) * (point[:2] - center[:2])
        candidate = _target(xy, hover_z, quaternion, z_offset_m=z_offset_m)
        try:
            robot._kinematics.solve(candidate, seed)
        except Exception as exc:
            last_error = exc
            continue
        if alpha < 1.0:
            print(
                f"{label}: camera estimate is outside the current IK envelope; "
                f"starting at {alpha:.0%} toward CENTER for manual correction",
                flush=True,
            )
        return candidate, alpha
    raise RuntimeError(f"{label} initial target is not reachable: {last_error}")


def _calibration_height_offset_m(cfg, label):
    """Return the configured Z bias that starts each point at 100 mm clearance."""
    board_cfg = cfg.get("board_calibration") or {}
    target_mm = float(board_cfg.get("calibration_target_clearance_mm", 100.0))
    references = board_cfg.get("calibration_reference_clearance_mm") or {}
    if label not in references:
        return 0.0
    reference_mm = float(references[label])
    if not all(math.isfinite(v) for v in (target_mm, reference_mm)):
        raise ValueError("calibration clearance references must be finite")
    return (target_mm - reference_mm) / 1000.0


def _fit_surface(samples):
    import numpy as np
    def surface_z(sample):
        tip_z = float(sample["tip_r_pose"]["position_m"][2])
        if sample.get("measured_clearance_mm") is not None:
            return tip_z - float(sample["measured_clearance_mm"]) / 1000.0
        measured = float(sample["measured_surface_z_mm"])
        # Compatibility with the first run, whose prompt called clearance a
        # surface height and recorded values such as 13..88 mm.
        return tip_z - measured / 1000.0 if measured < 200.0 else measured / 1000.0
    rows = []
    values = []
    for label in LABELS:
        pose = samples[label]["tip_r_pose"]
        x, y = (float(v) for v in pose["position_m"][:2])
        z = surface_z(samples[label])
        rows.append((x, y, 1.0))
        values.append(z)
    coefficients, _, _, _ = np.linalg.lstsq(np.asarray(rows), np.asarray(values), rcond=None)
    predicted = np.asarray(rows) @ coefficients
    residuals = np.asarray(values) - predicted
    return {
        "model": "z_m = a*x_m + b*y_m + c",
        "coefficients": {"a": float(coefficients[0]), "b": float(coefficients[1]), "c": float(coefficients[2])},
        "residuals_mm": [float(v * 1000.0) for v in residuals],
        "max_abs_residual_mm": float(np.max(np.abs(residuals)) * 1000.0),
    }


def _corrected_frame(samples):
    import numpy as np
    center = np.asarray(samples["CENTER"]["tip_r_pose"]["position_m"], dtype=float)
    tr = np.asarray(samples["TOP_RIGHT"]["tip_r_pose"]["position_m"], dtype=float)
    br = np.asarray(samples["BOTTOM_RIGHT"]["tip_r_pose"]["position_m"], dtype=float)
    bl = np.asarray(samples["BOTTOM_LEFT"]["tip_r_pose"]["position_m"], dtype=float)
    # Infer the unvisited top-left corner only to average the two measured edge
    # directions; all three other corners are physically operator-corrected.
    tl = tr + bl - br
    x_vec = ((tr - tl) + (br - bl)) / 2.0
    y_vec = ((bl - tl) + (br - tr)) / 2.0
    x_vec[2] = 0.0
    y_vec[2] = 0.0
    x_len = float(np.linalg.norm(x_vec[:2]))
    y_len = float(np.linalg.norm(y_vec[:2]))
    if x_len < 0.03 or y_len < 0.03:
        raise ValueError("corrected corner references are too close to define a board")
    raw_x = [float(v) for v in x_vec[:2]]
    raw_y = [float(v) for v in y_vec[:2]]
    ux, uy, axis_dot, axis_angle_error_deg = orthonormalize_xy_axes(raw_x, raw_y)
    return {
        "center_base_xy_m": [float(v) for v in center[:2]],
        "board_x_unit_base_xy": [float(v) for v in ux],
        "board_y_unit_base_xy": [float(v) for v in uy],
        "raw_board_x_unit_base_xy": [float(v) for v in (x_vec[:2] / x_len)],
        "raw_board_y_unit_base_xy": [float(v) for v in (y_vec[:2] / y_len)],
        "axis_dot_raw": float(axis_dot),
        "axis_angle_error_deg": float(axis_angle_error_deg),
        "width_m": x_len,
        "height_m": y_len,
        "inferred_top_left_base_m": [float(v) for v in tl],
    }


def _main_once(argv=None):
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hover-z", type=float, default=None,
                   help="initial TCP hover Z in base frame; default is floor + 80 mm")
    p.add_argument("--coarse-speed-scale", type=float, default=0.65)
    p.add_argument("--jog-speed-scale", type=float, default=0.45)
    p.add_argument(
        "--max-jog-mm", type=float, default=float("inf"),
        help="maximum single jog in mm; default is unlimited (IK still preflights)",
    )
    p.add_argument(
        "--forward-rise-angle-deg", type=float, default=None,
        help="board-parallel forward compensation; defaults to the robot config",
    )
    p.add_argument("--settle-s", type=float, default=0.5)
    p.add_argument("--publisher-log", default="~/head_camera.log")
    p.add_argument("--output", default="calibration/vega_board_manual.json")
    p.add_argument("--confirm-physical-motion", action="store_true")
    args = p.parse_args(argv)
    if not args.confirm_physical_motion:
        p.error("--confirm-physical-motion is required")
    if not 0.3 <= args.coarse_speed_scale <= 0.8:
        p.error("--coarse-speed-scale must be 0.3..0.8")
    if not 0.25 <= args.jog_speed_scale <= 0.7:
        p.error("--jog-speed-scale must be 0.25..0.7")
    if args.max_jog_mm <= 0 or math.isnan(args.max_jog_mm):
        p.error("--max-jog-mm must be positive; omit it for unlimited jog distance")

    cfg = load_bundle("vega")["robot"]
    configured_angle = float(
        (cfg.get("board_calibration") or {}).get("forward_rise_angle_deg", 13.0)
    )
    forward_rise_angle_deg = configured_angle if args.forward_rise_angle_deg is None else float(args.forward_rise_angle_deg)
    if not math.isfinite(forward_rise_angle_deg) or not -30.0 <= forward_rise_angle_deg <= 30.0:
        p.error("--forward-rise-angle-deg must be finite and within -30..30 deg")
    camera_corrections = (cfg.get("board_calibration") or {}).get("camera_target_corrections_m") or {}
    suggested_clearances = (cfg.get("board_calibration") or {}).get("suggested_clearance_mm") or {}
    safety = dict(load_vega_skills().get("safety") or {})
    floor = float(safety["min_tcp_z_m"])
    hover_z = floor + 0.08 if args.hover_z is None else float(args.hover_z)
    if not floor + 0.03 <= hover_z <= floor + 0.15:
        p.error(f"--hover-z must stay 3..15 cm above floor {floor:.6f}")
    cfg["allow_robot_init_head_motion"] = True
    cfg["auto_clear_software_estop_on_connect"] = True
    cfg["motion"]["max_step_rad"] = max(float(cfg["motion"]["max_step_rad"]), 0.45)
    # The measured right-arm endpoint residual during supervised board jogs is
    # about 0.016 rad.  The competition config's 0.005 rad gate can therefore
    # report a settled physical move as a timeout and invoke the e-stop path.
    # This calibration-only tolerance still requires a fresh state sample and
    # keeps the normal competition tolerance unchanged.
    cfg["motion"]["joint_reached_tolerance_rad"] = max(
        float(cfg["motion"]["joint_reached_tolerance_rad"]), 0.020
    )
    _ensure_publisher(args.publisher_log, robot_name=cfg["robot_name"])

    robot = VegaAdapter(cfg)
    camera = None
    samples = {}
    all_jogs = []
    try:
        robot.connect()
        move_camera_clear_for_image(robot, floor_m=floor, speed_scale=0.90)
        # +0.55 rad on head_j1 is the verified downward-looking board view.
        # Use the motion-handle API when available, then read the joints back
        # so a photo is never silently captured from the old head pose.
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
                    f"HEAD MOTION HANDLE FAILED ({exc}); falling back to "
                    "set_joint_pos",
                    flush=True,
                )
        if not moved_head:
            robot._robot.head.set_joint_pos(
                target_head, wait_time=1.2, exit_on_reach=True,
                exit_on_reach_kwargs={"tolerance": 0.02},
            )
        time.sleep(float(args.settle_s))
        head_q = np.asarray(robot._robot.head.get_joint_pos(), dtype=float)
        print("HEAD AFTER  =", head_q.tolist(), flush=True)
        if not np.allclose(head_q, target_head, atol=0.03):
            raise RuntimeError(
                f"downward head view was not reached: target={target_head.tolist()} "
                f"measured={head_q.tolist()}"
            )
        camera = VegaHeadCamera()
        camera.connect()
        frame = camera.read(include_depth=False, timeout_s=15.0)
        scene = detect_head_task_scene(
            frame.left_rgb, frame.camera_info, head_q,
            plane_z_m=configured_board_plane_z(cfg, floor),
            lift_m=float(cfg["kinematics"]["fixed_joint_values"]["Lift"]),
            torso_flip_rad=float(cfg["kinematics"]["fixed_joint_values"]["torso_flip"]),
            layout="unlabeled",
        )
        coarse = scene["board"]["corners_base_m_coarse"]
        points = {
            "CENTER": np.asarray(scene["board"]["center_base_m_coarse"], dtype=float),
            "TOP_RIGHT": np.asarray(coarse["tr"], dtype=float),
            "BOTTOM_RIGHT": np.asarray(coarse["br"], dtype=float),
            "BOTTOM_LEFT": np.asarray(coarse["bl"], dtype=float),
        }
        for label, point in points.items():
            correction = camera_corrections.get(label, (0.0, 0.0))
            if not isinstance(correction, (list, tuple)) or len(correction) != 2:
                raise ValueError(f"camera correction for {label} must be [dx,dy] metres")
            point[0] += float(correction[0])
            point[1] += float(correction[1])
        print("CAMERA BOARD READ:", json.dumps(scene["board"], indent=2), flush=True)
        ready_q, ready_pose = configured_right_preset(cfg, "right_ready")
        _move_configured_right_ready(robot, floor=floor, speed_scale=0.45)
        robot._kinematics.config.update({
            "position_tolerance_m": 0.0015,
            "orientation_tolerance_rad": 0.05,
            "max_seed_delta_rad": 2.4,
            "max_iterations": 240,
        })
        for label in LABELS:
            height_offset_m = _calibration_height_offset_m(cfg, label)
            target, alpha = _reachable_initial_target(
                robot, points[label], points["CENTER"], hover_z,
                ready_pose.quaternion_wxyz, label, z_offset_m=height_offset_m,
            )
            print(
                label,
                "CAMERA TARGET =",
                tuple(round(v, 6) for v in target.position_m),
                "HEIGHT OFFSET MM =",
                round(height_offset_m * 1000.0, 1),
                flush=True,
            )
            input(f"Press Enter to move to {label}; type anything to cancel: ")
            move_tcp_segmented(
                robot, target, speed_scale=float(args.coarse_speed_scale),
                max_translation_step_m=0.05, max_orientation_step_rad=0.15,
                min_tcp_z_m=floor,
            )
            reached = robot.get_tcp_pose()
            _print_pose(f"REACHED {label}", reached)
            point_jogs = []
            _interactive_adjust(
                robot, label=label, floor=floor,
                speed_scale=float(args.jog_speed_scale),
                max_jog_mm=float(args.max_jog_mm),
                forward_rise_angle_deg=forward_rise_angle_deg, events=point_jogs,
            )
            all_jogs.extend(point_jogs)
            corrected = robot.get_tcp_pose()
            while True:
                suggested = suggested_clearances.get(label)
                suffix = f" [default {float(suggested):g}]" if suggested is not None else ""
                raw = input(
                    f"Enter measured TCP-to-board CLEARANCE at {label} in mm "
                    f"(positive means TCP is above board){suffix}: "
                ).strip()
                try:
                    surface_mm = float(suggested) if not raw and suggested is not None else float(raw)
                    if not math.isfinite(surface_mm): raise ValueError
                    break
                except ValueError:
                    print("Enter one finite height in millimetres.", flush=True)
            samples[label] = {
                "camera_target_pose": _pose_record(target),
                "calibration_height_offset_m": float(height_offset_m),
                "camera_inset_fraction": float(alpha),
                "joint_names": list(robot._joint_names),
                "joint_positions_rad": list(robot._read_joint_positions()),
                "tip_r_pose": _pose_record(corrected),
                "measured_clearance_mm": surface_mm,
                "measured_surface_z_mm": float(corrected.position_m[2]) - surface_mm / 1000.0,
                "jogs": point_jogs,
            }
            _print_pose(f"RECORDED {label}", corrected)

        frame_xy = _corrected_frame(samples)
        plane = _fit_surface(samples)
        record = {
            "schema_version": 2,
            "calibration_kind": "vega_board_five_point_surface",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "robot_name": cfg["robot_name"],
            "base_frame": cfg["kinematics"]["base_frame"],
            "tcp_frame": cfg["kinematics"]["ee_frame"],
            "camera_frame": "zed_left_camera_optical",
            "board_width_m_declared": 0.386,
            "floor_m": floor,
            "initial_hover_z_m": hover_z,
            "forward_rise_angle_deg": forward_rise_angle_deg,
            "camera_target_corrections_m": camera_corrections,
            "head_q_rad": [float(v) for v in head_q],
            "camera_board_read": scene["board"],
            "manual_corrected": {label: samples[label]["tip_r_pose"] for label in LABELS},
            "samples": samples,
            "jogs": all_jogs,
            "corrected_board_frame_xy": frame_xy,
            "board_surface_plane_base": plane,
            "height_reference": "tcp_to_board_clearance_mm; surface_z=tip_z-clearance",
            "right_ready": {
                "joint_names": list(cfg["kinematics"]["right_arm_joint_names"]),
                "joint_positions_rad": list(ready_q),
                "tip_r": _pose_record(ready_pose),
                "source": "configured_operator_measured_RIGHT_READY",
            },
        }
        out = Path(args.output)
        if not out.is_absolute(): out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print("=== FIVE-POINT CALIBRATION ===", flush=True)
        print(json.dumps(record, indent=2), flush=True)
        print("WROTE", out.resolve(), flush=True)
        return 0
    finally:
        if camera is not None:
            try: camera.close()
            except BaseException: pass
        try: robot.close()
        except BaseException as exc: print(f"SHUTDOWN WARNING: {exc}", file=sys.stderr)


def _retryable_ik_error(exc):
    text = str(exc)
    return any(marker in text for marker in (
        "IK did not converge",
        "initial target is not reachable",
        "no reachable supervised inset",
        "downward head view was not reached",
    ))


def main(argv=None):
    """Retry transient IK failures by restarting the camera/arm sequence.

    A retry deliberately re-enters the complete startup path: the arm is
    brought through the validated camera-clear preset, a new head-camera frame
    is captured, and RIGHT_READY plus the board targets are planned again.
    Ctrl-C is the operator escape for a persistent physical problem.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return _main_once(argv)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if not _retryable_ik_error(exc):
                raise
            print(f"CALIBRATION IK RETRY {attempt}: {exc}", flush=True)
            print(
                "Re-entering camera-clear recovery, taking a fresh head-camera "
                "frame, and retrying calibration.",
                flush=True,
            )
            answer = input("Retry calibration? Type yes to continue, or no to stop: ").strip().lower()
            if answer not in ("y", "yes"):
                print("Calibration retry stopped by operator.", flush=True)
                return 2
            time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
