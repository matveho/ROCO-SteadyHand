"""Manually survey two board corners for height/plane calibration.

This tool intentionally sends no arm motion and performs no Cartesian IK. The
operator uses the physical manual controller to place the right claw at
TOP_RIGHT and BOTTOM_LEFT, then enters the measured board surface height or
claw clearance. It records the live right joints and modeled tip_r pose at
each sample so a later global offset/plane update has physical provenance.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.vega import VegaAdapter
from steadyhand.config import load_bundle
from steadyhand.skill_config import load_vega_skills


ROOT = Path(__file__).resolve().parents[1]
CORNER_LABELS = ("TOP_RIGHT", "BOTTOM_LEFT")


def summarize_height_samples(samples, *, measurement_kind, configured_floor_m):
    """Return differences and a candidate global offset from two raw samples."""
    if set(samples) != set(CORNER_LABELS):
        raise ValueError("height survey requires TOP_RIGHT and BOTTOM_LEFT samples")
    floor = float(configured_floor_m)
    values = {}
    for label in CORNER_LABELS:
        row = samples[label]
        value = float(row["measured_value"])
        modeled_z = float(row["tip_r_pose"]["position_m"][2])
        if not math.isfinite(value) or not math.isfinite(modeled_z):
            raise ValueError("height survey values must be finite")
        values[label] = value

    result = {
        "measurement_kind": measurement_kind,
        "corner_difference": values["TOP_RIGHT"] - values["BOTTOM_LEFT"],
        "modeled_tip_z_difference_m": (
            float(samples["TOP_RIGHT"]["tip_r_pose"]["position_m"][2])
            - float(samples["BOTTOM_LEFT"]["tip_r_pose"]["position_m"][2])
        ),
    }
    if measurement_kind == "board_surface_z_mm":
        result["candidate_global_board_z_offset_m"] = (
            sum(values[label] / 1000.0 for label in CORNER_LABELS) / 2.0 - floor
        )
        result["board_surface_z_range_m"] = (
            max(values.values()) - min(values.values())
        ) / 1000.0
    else:
        result["candidate_global_board_z_offset_m"] = None
        result["note"] = (
            "Clearance samples diagnose corner-to-corner variation; a global "
            "board Z offset requires an independently measured board surface Z."
        )
    return result


def _pose_record(pose):
    return {
        "position_m": [float(v) for v in pose.position_m],
        "quaternion_wxyz": [float(v) for v in pose.quaternion_wxyz],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurement-kind",
        choices=("board_surface_z_mm", "claw_clearance_mm"),
        default="board_surface_z_mm",
        help="what the entered millimetre values represent",
    )
    parser.add_argument("--output", default="calibration/vega_board_corner_heights.json")
    parser.add_argument("--confirm-physical-motion", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_physical_motion:
        parser.error("--confirm-physical-motion is required; the operator will manually move the arm")

    cfg = load_bundle("vega")["robot"]
    cfg["allow_robot_init_head_motion"] = True
    cfg["auto_clear_software_estop_on_connect"] = True
    floor = float(load_vega_skills()["safety"]["min_tcp_z_m"])
    robot = VegaAdapter(cfg)
    samples = {}
    try:
        robot.connect()
        print("NO AUTOMATIC ARM MOTION OR CARTESIAN IK WILL BE USED", flush=True)
        print("Configured provisional TCP floor =", floor, "m", flush=True)
        for label in CORNER_LABELS:
            input(
                f"Manually place the physical right claw at {label}; "
                "press Enter only when stable: "
            )
            joints = [float(v) for v in robot._read_joint_positions()]
            pose = robot.get_tcp_pose()
            if pose is None:
                raise RuntimeError(f"no tip_r pose available at {label}")
            raw = input(
                f"Enter measured {'board surface Z' if args.measurement_kind == 'board_surface_z_mm' else 'claw clearance'} "
                "in millimetres: "
            ).strip()
            try:
                measured = float(raw)
            except ValueError as exc:
                raise ValueError("measured height must be numeric millimetres") from exc
            if not math.isfinite(measured):
                raise ValueError("measured height must be finite")
            samples[label] = {
                "joint_names": list(robot._joint_names),
                "joint_positions_rad": joints,
                "tip_r_pose": _pose_record(pose),
                "measured_value": measured,
                "measured_unit": "mm",
            }
            print(label, "TIP_R =", samples[label]["tip_r_pose"], flush=True)

        summary = summarize_height_samples(
            samples,
            measurement_kind=args.measurement_kind,
            configured_floor_m=floor,
        )
        record = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "robot_name": cfg["robot_name"],
            "base_frame": cfg["kinematics"]["base_frame"],
            "tcp_frame": cfg["kinematics"]["ee_frame"],
            "configured_floor_m": floor,
            "motion_method": "operator_manual_controller_only; no_ik; no_arm_command",
            "samples": samples,
            "summary": summary,
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
