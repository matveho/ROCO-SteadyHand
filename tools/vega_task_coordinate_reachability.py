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
from steadyhand.config import load_bundle
from steadyhand.executor import move_tcp_segmented
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills
from steadyhand.vega_presets import configured_right_preset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POINTS = ("battery_size1.pick", "usb_a.pick", "rod_16mm.place", "gear_20teeth.place")
FALLBACK_CALIBRATION = ROOT / "calibration" / "vega_board_manual_fallback.json"


def _finite_vector(value, size, name):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must contain {size} numbers")
    result = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _fit_plane_from_samples(samples):
    rows = []
    values = []
    for label in ("CENTER", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT"):
        sample = samples.get(label) or {}
        position = (sample.get("tip_r_pose") or {}).get("position_m") or []
        if len(position) != 3:
            continue
        if sample.get("measured_clearance_mm") is not None:
            surface_z = float(position[2]) - float(sample["measured_clearance_mm"]) / 1000.0
        elif sample.get("measured_surface_z_mm") is not None:
            value = float(sample["measured_surface_z_mm"])
            surface_z = float(position[2]) - value / 1000.0 if value < 200.0 else value / 1000.0
        else:
            continue
        rows.append((float(position[0]), float(position[1]), 1.0))
        values.append(surface_z)
    if len(rows) < 3:
        return None
    matrix = [[sum(row[i] * row[j] for row in rows) for j in range(3)] for i in range(3)]
    vector = [sum(row[i] * value for row, value in zip(rows, values)) for i in range(3)]
    for i in range(3):
        pivot = max(range(i, 3), key=lambda index: abs(matrix[index][i]))
        if abs(matrix[pivot][i]) < 1e-12:
            return None
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        vector[i], vector[pivot] = vector[pivot], vector[i]
        scale = matrix[i][i]
        matrix[i] = [value / scale for value in matrix[i]]
        vector[i] /= scale
        for row_index in range(3):
            if row_index == i:
                continue
            scale = matrix[row_index][i]
            matrix[row_index] = [a - scale * b for a, b in zip(matrix[row_index], matrix[i])]
            vector[row_index] -= scale * vector[i]
    return tuple(vector)


def _load_manual(path, cfg):
    path = Path(path)
    if not path.is_file() and path.name == "vega_board_manual.json" and FALLBACK_CALIBRATION.is_file():
        print(f"USING PERMANENT BOARD CALIBRATION FALLBACK: {FALLBACK_CALIBRATION}", flush=True)
        path = FALLBACK_CALIBRATION
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("robot_name") != cfg.get("robot_name"):
        raise ValueError("manual calibration belongs to a different robot")
    if raw.get("base_frame") != cfg["kinematics"]["base_frame"]:
        raise ValueError("manual calibration has the wrong base frame")
    if raw.get("calibration_kind") != "vega_board_five_point_surface":
        raise ValueError(
            "task motion requires a completed five-point surface calibration; "
            "run tools/vega_board_five_point_calibrate.py first"
        )
    frame = raw.get("corrected_board_frame_xy") or {}
    center = _finite_vector(frame.get("center_base_xy_m"), 2, "live board center")
    ux = _finite_vector(frame.get("board_x_unit_base_xy"), 2, "live board X axis")
    uy = _finite_vector(frame.get("board_y_unit_base_xy"), 2, "live board Y axis")
    if abs(math.hypot(*ux) - 1.0) > 0.03 or abs(math.hypot(*uy) - 1.0) > 0.03:
        raise ValueError("live board axes must be unit vectors")
    center_pose = (raw.get("manual_corrected") or {}).get("CENTER")
    if not isinstance(center_pose, dict):
        raise ValueError("manual calibration lacks corrected CENTER pose")
    plane = raw.get("board_surface_plane_base") or {}
    coefficients = plane.get("coefficients") or {}
    a, b, c = (float(coefficients.get(key)) for key in ("a", "b", "c"))
    if not all(math.isfinite(v) for v in (a, b, c)):
        raise ValueError("five-point calibration lacks a finite board surface plane")
    anchors = []
    samples = raw.get("samples") or {}
    reconstructed = _fit_plane_from_samples(samples)
    if reconstructed is not None:
        # The first completed run recorded clearances under the old field name
        # and therefore contains an invalid low-Z plane. Reconstruct from the
        # measured TCP Z and the entered TCP-to-board clearance.
        a, b, c = reconstructed
    for label in ("CENTER", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT"):
        sample = samples.get(label) or {}
        pose = sample.get("tip_r_pose") or {}
        position = pose.get("position_m") or []
        measured_mm = sample.get("measured_clearance_mm")
        measured_is_clearance = measured_mm is not None
        if measured_mm is None:
            measured_mm = sample.get("measured_surface_z_mm")
        if len(position) == 3 and measured_mm is not None:
            x, y = float(position[0]), float(position[1])
            measured_z = float(measured_mm) / 1000.0
            if measured_is_clearance or measured_z < 0.2:
                measured_z = float(position[2]) - measured_z
            plane_z = a * x + b * y + c
            if all(math.isfinite(v) for v in (x, y, measured_z, plane_z)):
                anchors.append({"label": label, "x_m": x, "y_m": y,
                                "residual_m": measured_z - plane_z})
    return center, ux, uy, {"coefficients": (a, b, c), "anchors": anchors}


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


def _live_pose(source_xyz, *, source_center, live_center, ux, uy, surface_plane, clearance_m, quat):
    dx = float(source_xyz[0]) - float(source_center[0])
    dy = float(source_xyz[1]) - float(source_center[1])
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
    p.add_argument("--hover-clearance-mm", type=float, default=50.0,
                   help="TCP clearance above the fitted board surface")
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
    if abs(float(task_data.get("board_width_m")) - 0.386) > 1e-9:
        raise ValueError("task coordinate registration must declare board_width_m=0.386")
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
                                  quat=ready_pose.quaternion_wxyz)))

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
        input("Press Enter to move to measured RIGHT_READY; type anything to cancel: ")
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
