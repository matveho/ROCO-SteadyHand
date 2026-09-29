"""Generate a hardware-free audit of every organizer task target.

The report applies the same board rotation, normalized calibrated axes and
surface model used by the live pipeline.  It is intentionally read-only: it
does not construct a robot connection or claim that IK/grasping was tested.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.config import load_bundle
from steadyhand.board_geometry import BOARD_SIZE_M, validate_task_board_geometry
from steadyhand.skill_config import load_vega_skills
from steadyhand.models import Pose
from steadyhand.vega_presets import configured_right_preset
from tools.vega_task_coordinate_reachability import (
    _finite_vector,
    _live_pose,
    _load_manual,
    _resolve_point,
    calibrated_surface_z,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = ROOT / "calibration" / "vega_board_manual.json"
DEFAULT_COORDINATES = ROOT / "configs" / "task_coordinates.json"


def build_report(task_data, calibration, *, clearance_m=0.100, floor_m=0.456, calibration_path=None):
    """Return a deterministic target audit dictionary with no hardware access."""
    if not math.isfinite(float(clearance_m)) or clearance_m <= 0:
        raise ValueError("clearance_m must be finite and positive")
    floor = float(floor_m)
    source_center = _finite_vector(task_data.get("source_board_center_xy_m"), 2, "source board center")
    rotation_deg = float(task_data.get("task_coordinate_rotation_deg", 0.0))
    live_center, ux, uy, plane = calibration
    rows = []
    for part in task_data["official_order"]:
        for kind in task_data["parts"][part]:
            if kind not in ("pick", "place", "connect", "grade"):
                continue
            name = f"{part}.{kind}"
            source_xyz = _resolve_point(name, task_data)[2]
            dx = float(source_xyz[0]) - source_center[0]
            dy = float(source_xyz[1]) - source_center[1]
            angle = math.radians(rotation_deg)
            board_x = dx * math.cos(angle) - dy * math.sin(angle)
            board_y = dx * math.sin(angle) + dy * math.cos(angle)
            pose = _live_pose(
                source_xyz, source_center=source_center,
                live_center=live_center, ux=ux, uy=uy,
                surface_plane=plane, clearance_m=clearance_m,
                quat=(1.0, 0.0, 0.0, 0.0), rotation_deg=rotation_deg,
            )
            surface_z = calibrated_surface_z(pose.position_m[0], pose.position_m[1], plane)
            rows.append({
                "name": name,
                "part": part,
                "action": kind,
                "source_xyz_m": list(source_xyz),
                "board_relative_xy_m": [board_x, board_y],
                "base_xy_m": list(pose.position_m[:2]),
                "surface_z_m": surface_z,
                "hover_z_m": float(pose.position_m[2]),
                "hover_clearance_m": float(clearance_m),
                "hover_distance_above_floor_m": float(pose.position_m[2]) - floor,
                "surface_distance_from_floor_m": surface_z - floor,
                "distance_from_board_center_m": math.hypot(board_x, board_y),
                "inside_calibrated_board_extent": (
                    abs(board_x) <= BOARD_SIZE_M / 2.0
                    and abs(board_y) <= BOARD_SIZE_M / 2.0
                ),
                "surface_below_hard_floor": surface_z < floor,
                "hover_below_hard_floor": float(pose.position_m[2]) < floor,
            })
    return {
        "schema_version": 1,
        "report_kind": "vega_task_geometry_audit",
        "board_size_m": BOARD_SIZE_M,
        "board_motion_model": task_data.get("board_motion_model"),
        "task_coordinate_rotation_deg": rotation_deg,
        "clearance_m": float(clearance_m),
        "hard_floor_m": floor,
        "calibration_path": calibration_path,
        "targets": rows,
        "summary": {
            "target_count": len(rows),
            "outside_board_count": sum(not r["inside_calibrated_board_extent"] for r in rows),
            "hover_below_floor_count": sum(r["hover_below_hard_floor"] for r in rows),
            "surface_below_floor_count": sum(r["surface_below_hard_floor"] for r in rows),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", default=str(DEFAULT_COORDINATES.relative_to(ROOT)))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION.relative_to(ROOT)))
    parser.add_argument("--clearance-mm", type=float, default=100.0)
    parser.add_argument("--output", help="JSON report path; default is calibration/task_geometry_audit.json")
    parser.add_argument("--check-ik", action="store_true", help="attempt local right-arm IK when Pinocchio is installed")
    args = parser.parse_args(argv)
    coords_path = Path(args.coordinates)
    if not coords_path.is_absolute():
        coords_path = ROOT / coords_path
    calibration_path = Path(args.calibration)
    if not calibration_path.is_absolute():
        calibration_path = ROOT / calibration_path
    task_data = json.loads(coords_path.read_text(encoding="utf-8"))
    validate_task_board_geometry(task_data)
    cfg = load_bundle("vega")["robot"]
    calibration = _load_manual(calibration_path, cfg)
    floor = float((load_vega_skills().get("safety") or {}).get("min_tcp_z_m", 0.456))
    report = build_report(task_data, calibration, clearance_m=args.clearance_mm / 1000.0, floor_m=floor, calibration_path=str(calibration_path))
    if args.check_ik:
        report["ik_audit"] = _audit_ik(report, cfg)
    output = Path(args.output) if args.output else ROOT / "calibration" / "task_geometry_audit.json"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print("WROTE", output.resolve())
    return 0


def _audit_ik(report, cfg):
    """Best-effort local IK audit; unavailable dependencies never touch hardware."""
    try:
        from steadyhand.adapters.vega import VegaAdapter
        robot = VegaAdapter(cfg)
        robot.prepare()
        seed, ready_pose = configured_right_preset(cfg, "right_ready")
        solved = []
        for row in report["targets"]:
            target = Pose(tuple(row["base_xy_m"] + [row["hover_z_m"]]), ready_pose.quaternion_wxyz)
            try:
                solution = robot._kinematics.solve(target, seed)
                solved.append({"name": row["name"], "status": "success", "joint_solution_rad": list(solution), "max_seed_delta_rad": max(abs(float(a) - float(b)) for a, b in zip(solution, seed))})
            except Exception as exc:
                solved.append({"name": row["name"], "status": "failed", "reason": str(exc)})
        return {"status": "available", "targets": solved, "success_count": sum(x["status"] == "success" for x in solved), "failure_count": sum(x["status"] == "failed" for x in solved)}
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    raise SystemExit(main())
