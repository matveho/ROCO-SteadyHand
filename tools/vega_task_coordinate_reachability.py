"""Supervised reachability test for organizer task coordinates.

This visits selected task XY locations in the live calibrated board frame. It
does not operate the gripper or attempt a pick/place. Organizer Z values are
metadata only; every target uses the accepted calibrated board hover height.
"""

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.vega import VegaAdapter
from steadyhand.board_calibration import load_board_calibration
from steadyhand.board_geometry import validate_task_board_geometry
from steadyhand.config import load_bundle
from steadyhand.executor import move_tcp_segmented
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills
from steadyhand.vega_presets import configured_right_preset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POINTS = ("battery_size1.pick", "usb_a.pick", "rod_16mm.place", "gear_20teeth.place")
DEFAULT_HOVER_CLEARANCE_MM = 100.0


def _finite_vector(value, size, name):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must contain {size} numbers")
    result = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _load_manual(path, cfg):
    manual = load_board_calibration(path, cfg)
    if manual.get("schema_version") != 2:
        raise ValueError(
            "task motion requires a completed five-point surface calibration; "
            "run tools/vega_board_five_point_calibrate.py first"
        )
    plane = dict(manual["surface_plane"])
    plane["camera_geometry_signature"] = manual.get("calibration_camera_geometry_signature")
    plane["raw_board_x_unit_base_xy"] = manual["raw_board_x_unit_base_xy"]
    plane["raw_board_y_unit_base_xy"] = manual["raw_board_y_unit_base_xy"]
    plane["axis_dot_raw"] = manual["axis_dot_raw"]
    plane["axis_angle_error_deg"] = manual["axis_angle_error_deg"]
    return (
        manual["center_base_xy_m"],
        manual["board_x_unit_base_xy"],
        manual["board_y_unit_base_xy"],
        plane,
    )


def calibrated_surface_z(x, y, surface_model):
    """Evaluate the fitted board plane plus measured local residual correction."""
    if isinstance(surface_model, dict):
        a, b, c = surface_model["coefficients"]
        anchors = surface_model.get("anchors") or []
    else:
        a, b, c = surface_model
        anchors = []
    base = float(a) * float(x) + float(b) * float(y) + float(c)
    if not anchors:
        return base
    distances = [
        (float(anchor["x_m"]) - float(x)) ** 2
        + (float(anchor["y_m"]) - float(y)) ** 2
        for anchor in anchors
    ]
    nearest = min(distances)
    if nearest < 1e-12:
        return base + float(anchors[distances.index(nearest)]["residual_m"])
    weights = [1.0 / distance for distance in distances]
    correction = sum(
        weight * float(anchor["residual_m"])
        for weight, anchor in zip(weights, anchors)
    ) / sum(weights)
    return base + correction


def _resolve_point(name, task_data):
    if "." not in name:
        raise ValueError(f"point {name!r} must be part.pick or part.place")
    part, kind = name.rsplit(".", 1)
    if part not in task_data["parts"] or kind not in task_data["parts"][part]:
        raise ValueError(f"unknown task point {name!r}")
    return part, kind, _finite_vector(task_data["parts"][part][kind], 3, name)


def _live_pose(source_xyz, *, source_center, live_center, ux, uy, surface_plane, clearance_m, quat, rotation_deg=0.0):
    dx = float(source_xyz[0]) - float(source_center[0])
    dy = float(source_xyz[1]) - float(source_center[1])
    angle = math.radians(float(rotation_deg))
    dx, dy = (
        dx * math.cos(angle) - dy * math.sin(angle),
        dx * math.sin(angle) + dy * math.cos(angle),
    )
    live_x = live_center[0] + ux[0] * dx + uy[0] * dy
    live_y = live_center[1] + ux[1] * dx + uy[1] * dy
    surface_z = calibrated_surface_z(live_x, live_y, surface_plane)
    return Pose(
        (live_x, live_y,
         float(surface_z + clearance_m)),
        tuple(quat),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("points", nargs="*", default=list(DEFAULT_POINTS),
                   help="point names such as battery_size1.pick or rod_16mm.connect")
    p.add_argument("--coordinates", default="configs/task_coordinates.json")
    p.add_argument("--calibration", default="calibration/vega_board_manual.json")
    p.add_argument("--speed-scale", type=float, default=0.35)
    p.add_argument(
        "--hover-clearance-mm", type=float, default=DEFAULT_HOVER_CLEARANCE_MM,
        help="TCP clearance above the fitted board surface (default: 100 mm)",
    )
    p.add_argument("--check-only", action="store_true", help="connect and preflight IK, but do not move")
    p.add_argument("--confirm-physical-motion", action="store_true")
    args = p.parse_args(argv)
    if not args.check_only and not args.confirm_physical_motion:
        p.error("--confirm-physical-motion is required unless --check-only is used")
    if not 0.15 <= args.speed_scale <= 0.6:
        p.error("--speed-scale must be 0.15..0.6")
    if not 20.0 <= args.hover_clearance_mm <= 100.0:
        p.error("--hover-clearance-mm must be 20..100")

    cfg = load_bundle("vega")["robot"]
    task_path = Path(args.coordinates)
    if not task_path.is_absolute(): task_path = ROOT / task_path
    task_data = json.loads(task_path.read_text(encoding="utf-8"))
    if task_data.get("source_pose_frame") != "roco_organizer_sim_stage":
        raise ValueError("task coordinates must retain the organizer source frame")
    validate_task_board_geometry(task_data)
    rotation_deg = float(task_data.get("task_coordinate_rotation_deg", 0.0))
    source_center = _finite_vector(task_data.get("source_board_center_xy_m"), 2, "source board center")
    live_center, ux, uy, surface_plane = _load_manual(args.calibration, cfg)
    _, ready_pose = configured_right_preset(cfg, "right_ready")
    points = []
    for name in args.points:
        part, kind, source_xyz = _resolve_point(name, task_data)
        points.append((name, part, kind, source_xyz,
                       _live_pose(source_xyz, source_center=source_center,
                                  live_center=live_center, ux=ux, uy=uy,
                                  surface_plane=surface_plane,
                                  clearance_m=float(args.hover_clearance_mm) / 1000.0,
                                  quat=ready_pose.quaternion_wxyz,
                                  rotation_deg=rotation_deg)))

    safety = load_vega_skills()["safety"]
    floor = float(safety["min_tcp_z_m"])
    cfg["allow_robot_init_head_motion"] = True
    cfg["auto_clear_software_estop_on_connect"] = True
    robot = VegaAdapter(cfg)
    try:
        robot.connect()
        ready_q, _ = configured_right_preset(cfg, "right_ready")
        current_q = robot._read_joint_positions()
        # All IK checks use the taught RIGHT_READY seed; no physical motion occurs
        # until every requested target has passed.
        for name, _, _, source_xyz, target in points:
            if target.position_m[2] < floor:
                raise ValueError(f"{name} target z={target.position_m[2]:.6f} is below safety floor {floor:.6f}")
            robot._kinematics.solve(target, ready_q)
            print(name, "SOURCE =", tuple(round(v, 6) for v in source_xyz),
                  "LIVE =", tuple(round(v, 6) for v in target.position_m), flush=True)
        print("PREFLIGHT OK:", len(points), "task points", flush=True)
        if args.check_only:
            return 0
        print("MOVING TO RIGHT_READY", flush=True)
        robot.move_joints(ready_q, speed_scale=args.speed_scale)
        for name, _, _, _, target in points:
            input(f"Press Enter to move to {name}; type anything to cancel: ")
            move_tcp_segmented(robot, target, speed_scale=args.speed_scale,
                               max_translation_step_m=0.04,
                               max_orientation_step_rad=0.15,
                               min_tcp_z_m=floor)
            measured = robot.get_tcp_pose()
            print(name, "MEASURED TIP_R =", tuple(round(v, 6) for v in measured.position_m), flush=True)
        return 0
    finally:
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
