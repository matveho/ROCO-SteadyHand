"""Supervised software-motion survey of two Vega board corners.

This reuses the completed manual board calibration frame. The arm is moved by
the normal segmented TCP planner to TOP_RIGHT and BOTTOM_LEFT; the operator
only enters an external height measurement at each reached point. No manual
claw control is required.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.vega import VegaAdapter
from steadyhand.board_calibration import load_board_calibration
from steadyhand.config import load_bundle
from steadyhand.executor import move_tcp_segmented
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills
from steadyhand.vega_presets import configured_right_preset


ROOT = Path(__file__).resolve().parents[1]


def _finite_vector(value, size, name):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must contain {size} numbers")
    result = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _load_frame(path, config):
    manual = load_board_calibration(path, config)
    raw = manual["raw"]
    center = manual["center_base_xy_m"]
    ux = manual["board_x_unit_base_xy"]
    uy = manual["board_y_unit_base_xy"]
    center_pose = (raw.get("manual_corrected") or {}).get("CENTER")
    if not isinstance(center_pose, dict):
        raise ValueError("manual calibration lacks corrected CENTER pose")
    pose = Pose.from_mapping(center_pose)
    return raw, center, ux, uy, pose


def _corner_pose(center_xy, ux, uy, center_pose, x_offset, y_offset, quaternion=None):
    return Pose(
        (
            center_xy[0] + ux[0] * x_offset + uy[0] * y_offset,
            center_xy[1] + ux[1] * x_offset + uy[1] * y_offset,
            center_pose.position_m[2],
        ),
        center_pose.quaternion_wxyz if quaternion is None else tuple(quaternion),
    )


def _reachable_corner(robot, *, label, center_xy, ux, uy, center_pose,
                      x_offset, y_offset, quaternion):
    """Find the largest reachable inset toward a requested board corner.

    The calibrated board rectangle can extend beyond the arm envelope.  Probe
    from the requested corner toward CENTER before any motion, so the tool
    never starts a physical move with an IK target known to fail.
    """
    seed = robot._read_joint_positions()
    last_error = None
    # Keep the exact corner as the first choice; progressively inset by 2 cm.
    for alpha in (1.00, 0.94, 0.88, 0.82, 0.76, 0.70, 0.64, 0.58, 0.52):
        target = _corner_pose(
            center_xy, ux, uy, center_pose,
            float(x_offset) * alpha, float(y_offset) * alpha,
            quaternion,
        )
        try:
            robot._kinematics.solve(target, seed)
        except Exception as exc:
            last_error = exc
            continue
        if alpha < 1.0:
            print(
                f"{label}: full corner was outside the IK envelope; using "
                f"{alpha:.0%} inset toward CENTER", flush=True,
            )
        return target, alpha
    raise RuntimeError(
        f"{label} has no reachable supervised inset on the calibrated plane; "
        f"last IK error: {last_error}"
    )


def _measurement_summary(samples, kind, floor_m):
    values = [float(samples[label]["measured_value_mm"]) for label in ("TOP_RIGHT", "BOTTOM_LEFT")]
    result = {
        "measurement_kind": kind,
        "corner_difference_mm": values[0] - values[1],
        "modeled_tip_z_difference_m": (
            samples["TOP_RIGHT"]["tip_r_pose"]["position_m"][2]
            - samples["BOTTOM_LEFT"]["tip_r_pose"]["position_m"][2]
        ),
    }
    if kind == "board_surface_z_mm":
        result["candidate_global_board_z_offset_m"] = sum(values) / 2000.0 - float(floor_m)
        result["board_surface_z_range_m"] = (max(values) - min(values)) / 1000.0
    else:
        result["candidate_global_board_z_offset_m"] = None
        result["note"] = "Clearance difference diagnoses slope; it does not determine global board Z."
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration", default="calibration/vega_board_manual.json")
    p.add_argument("--x-offset-m", type=float)
    p.add_argument("--y-offset-m", type=float)
    p.add_argument("--measurement-kind", choices=("board_surface_z_mm", "claw_clearance_mm"), default="board_surface_z_mm")
    p.add_argument("--speed-scale", type=float, default=0.45)
    p.add_argument("--output", default="calibration/vega_board_corner_heights.json")
    p.add_argument("--confirm-physical-motion", action="store_true")
    args = p.parse_args(argv)
    if not args.confirm_physical_motion:
        p.error("--confirm-physical-motion is required")
    if not 0.2 <= args.speed_scale <= 0.8:
        p.error("--speed-scale must be 0.2..0.8")

    cfg = load_bundle("vega")["robot"]
    raw, center, ux, uy, center_pose = _load_frame(args.calibration, cfg)
    try:
        _, ready_pose = configured_right_preset(cfg, "right_ready")
        corner_quaternion = ready_pose.quaternion_wxyz
    except (KeyError, ValueError):
        corner_quaternion = center_pose.quaternion_wxyz
    frame = raw.get("corrected_board_frame_xy") or {}
    x_default = frame.get("x_reference_distance_m", float(frame.get("width_m", 0.2)) / 2.0)
    y_default = frame.get("y_reference_distance_m", float(frame.get("height_m", 0.2)) / 2.0)
    x_offset = float(args.x_offset_m if args.x_offset_m is not None else x_default)
    y_offset = float(args.y_offset_m if args.y_offset_m is not None else y_default)
    if not (0.03 <= x_offset <= 0.30 and 0.03 <= y_offset <= 0.30):
        p.error("corner offsets must each be within 0.03..0.30 m")

    safety = load_vega_skills()["safety"]
    floor = float(safety["min_tcp_z_m"])
    cfg["allow_robot_init_head_motion"] = True
    cfg["auto_clear_software_estop_on_connect"] = True
    cfg["motion"]["max_step_rad"] = max(float(cfg["motion"]["max_step_rad"]), 0.45)
    robot = VegaAdapter(cfg)
    samples = {}
    try:
        robot.connect()
        targets = {
            "TOP_RIGHT": (+x_offset, +y_offset),
            "BOTTOM_LEFT": (-x_offset, -y_offset),
        }
        kin_cfg = robot._kinematics.config
        kin_cfg.update({"position_tolerance_m": 0.002, "orientation_tolerance_rad": 0.05,
                        "max_seed_delta_rad": 2.4, "max_iterations": 300})
        for label, (x_corner, y_corner) in targets.items():
            target, alpha = _reachable_corner(
                robot, label=label, center_xy=center, ux=ux, uy=uy,
                center_pose=center_pose, x_offset=x_corner, y_offset=y_corner,
                quaternion=corner_quaternion,
            )
            print(label, "TARGET =", tuple(round(v, 6) for v in target.position_m), flush=True)
            input(f"Press Enter to move to {label}; type anything to cancel: ")
            move_tcp_segmented(robot, target, speed_scale=args.speed_scale,
                               max_translation_step_m=0.06,
                               max_orientation_step_rad=0.20,
                               min_tcp_z_m=floor)
            pose = robot.get_tcp_pose()
            measured = float(input(f"Enter measured {args.measurement_kind} at {label} (mm): "))
            if not math.isfinite(measured):
                raise ValueError("measured height must be finite")
            samples[label] = {
                "requested_tip_r_pose": {"position_m": list(target.position_m), "quaternion_wxyz": list(target.quaternion_wxyz)},
                "requested_corner_offsets_m": {"x": abs(x_corner), "y": abs(y_corner)},
                "reachable_inset_fraction": alpha,
                "joint_names": list(robot._joint_names),
                "joint_positions_rad": list(robot._read_joint_positions()),
                "tip_r_pose": {"position_m": list(pose.position_m), "quaternion_wxyz": list(pose.quaternion_wxyz)},
                "measured_value_mm": measured,
            }
            print(label, "REACHED =", tuple(round(v, 6) for v in pose.position_m), flush=True)

        record = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "robot_name": cfg["robot_name"],
            "base_frame": cfg["kinematics"]["base_frame"],
            "tcp_frame": cfg["kinematics"]["ee_frame"],
            "source_manual_calibration": str(Path(args.calibration)),
            "corner_offsets_m": {"x": x_offset, "y": y_offset},
            "configured_floor_m": floor,
            "samples": samples,
            "summary": _measurement_summary(samples, args.measurement_kind, floor),
        }
        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2), flush=True)
        print("WROTE", out.resolve(), flush=True)
        return 0
    finally:
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
