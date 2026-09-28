"""Move battery_size1 to source-localizer XY at the current safe hover only.

Input is the battery source-localizer localization.json, not raw X/Y typed from
memory. The tool revalidates the completed manual board calibration and
requires its SHA/timestamp to match the localization provenance.

Arm behavior is deliberately narrow:
- current TCP must already be at the accepted low-hover Z;
- preserve the exact live hover Z and orientation;
- move only in base-frame X/Y to the localized battery source;
- no vertical move, gripper, wrist camera/servo, grasp descent, placement, or
  insertion.

Robot() on Vega homes the head as an SDK side effect, so head-motion and
physical-motion confirmations are both explicit.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.vega import VegaAdapter
from steadyhand.battery_size1 import require_right_battery_config
from steadyhand.battery_size1_hover import (
    accepted_hover_from_manual,
    load_source_localization,
    plan_hover_stages,
)
from steadyhand.battery_size1_source import load_manual_board_calibration
from steadyhand.config import load_bundle
from steadyhand.executor import move_tcp_segmented
from steadyhand.geometry import quaternion_angle
from steadyhand.skill_config import load_vega_skills


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUAL = ROOT / "calibration" / "vega_board_manual.json"


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _pose_record(pose):
    return {
        "position_m": [float(v) for v in pose.position_m],
        "quaternion_wxyz": [float(v) for v in pose.quaternion_wxyz],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--localization",
        required=True,
        help="battery source-localizer localization.json",
    )
    p.add_argument(
        "--manual-calibration",
        default=str(DEFAULT_MANUAL.relative_to(ROOT)),
        help="completed manual board calibration used by localization",
    )
    p.add_argument("--max-calibration-age-min", type=float, default=720.0)
    p.add_argument("--max-localization-age-min", type=float, default=120.0)
    p.add_argument("--orientation-tolerance-rad", type=float, default=0.05)
    p.add_argument("--hover-z-tolerance-m", type=float, default=0.008)
    p.add_argument("--final-position-tolerance-m", type=float, default=0.008)
    p.add_argument("--speed-scale", type=float, default=0.70)
    p.add_argument("--output")
    p.add_argument(
        "--confirm-board-unchanged-since-localization",
        action="store_true",
    )
    p.add_argument("--confirm-head-motion", action="store_true")
    p.add_argument("--confirm-physical-motion", action="store_true")
    args = p.parse_args(argv)

    if not args.confirm_board_unchanged_since_localization:
        p.error(
            "--confirm-board-unchanged-since-localization is required"
        )
    if not args.confirm_head_motion or not args.confirm_physical_motion:
        p.error(
            "--confirm-head-motion and --confirm-physical-motion are required"
        )

    numeric = (
        args.max_calibration_age_min,
        args.max_localization_age_min,
        args.orientation_tolerance_rad,
        args.hover_z_tolerance_m,
        args.final_position_tolerance_m,
        args.speed_scale,
    )
    if not all(math.isfinite(float(v)) for v in numeric):
        p.error("motion/provenance settings must be finite")
    if not 15 <= args.max_calibration_age_min <= 1440:
        p.error("--max-calibration-age-min must be 15..1440")
    if not 5 <= args.max_localization_age_min <= 720:
        p.error("--max-localization-age-min must be 5..720")
    if not 0.01 <= args.orientation_tolerance_rad <= 0.15:
        p.error("--orientation-tolerance-rad must be 0.01..0.15")
    if not 0.003 <= args.hover_z_tolerance_m <= 0.015:
        p.error("--hover-z-tolerance-m must be 0.003..0.015")
    if not 0.002 <= args.final_position_tolerance_m <= 0.015:
        p.error("--final-position-tolerance-m must be 0.002..0.015")
    if not 0.45 <= args.speed_scale <= 0.90:
        p.error("--speed-scale must be 0.45..0.90")

    cfg = load_bundle("vega")["robot"]
    require_right_battery_config(cfg)
    floor = float(load_vega_skills()["safety"]["min_tcp_z_m"])

    manual_path = _resolve(args.manual_calibration)
    localization_path = _resolve(args.localization)
    manual = load_manual_board_calibration(
        manual_path,
        cfg,
        max_age_minutes=float(args.max_calibration_age_min),
    )
    localization = load_source_localization(
        localization_path,
        cfg,
        manual,
        max_age_minutes=float(args.max_localization_age_min),
    )
    hover = accepted_hover_from_manual(manual, floor_m=floor)

    output = (
        _resolve(args.output)
        if args.output
        else ROOT
        / "runs"
        / datetime.now(timezone.utc).strftime(
            "battery_size1_coarse_hover_%Y%m%dT%H%M%S_%fZ"
        )
    )
    output.mkdir(parents=True, exist_ok=False)

    audit = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "part": "battery_size1",
        "status": "starting",
        "robot_name": cfg["robot_name"],
        "base_frame": cfg["kinematics"]["base_frame"],
        "working_arm": "right",
        "tcp_frame": cfg["kinematics"]["ee_frame"],
        "wrist_camera": "wrist_a",
        "gripper_scope": "right",
        "floor_m": floor,
        "localization": {
            "path": str(localization_path),
            "sha256": localization["sha256"],
            "generated_at_utc": localization["generated_at_utc"],
            "age_minutes": localization["age_minutes"],
            "method": localization["method"],
            "coarse_base_xy_m": list(
                localization["coarse_base_xy_m"]
            ),
            "source_image_sha256": localization[
                "source_image_sha256"
            ],
        },
        "manual_board_calibration": {
            "path": str(manual_path),
            "sha256": manual["sha256"],
            "generated_at_utc": manual["generated_at_utc"],
            "age_minutes": manual["age_minutes"],
        },
        "operator_confirmed_board_unchanged_since_localization": True,
        "accepted_hover": {
            "source": "manual_corrected.CENTER",
            "reference_z_m": hover["z_m"],
            "reference_quaternion_wxyz": list(
                hover["quaternion_wxyz"]
            ),
            "orientation_status": hover["orientation_status"],
        },
        "motion_contract": {
            "xy_only": True,
            "preserve_live_z": True,
            "preserve_live_orientation": True,
            "configured_floor_m": floor,
            "gripper": False,
            "wrist_servo": False,
            "placement": False,
            "grasp_descent": False,
        },
    }
    (output / "request.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    robot = None
    try:
        cfg["allow_robot_init_head_motion"] = True
        cfg["auto_clear_software_estop_on_connect"] = True
        robot = VegaAdapter(cfg)
        robot.connect()

        start = robot.get_tcp_pose()
        if float(start.position_m[2]) < floor:
            raise RuntimeError(
                f"live TCP z={float(start.position_m[2]):.6f} is below "
                f"configured floor {floor:.6f}"
            )

        orientation_error = quaternion_angle(
            start.quaternion_wxyz,
            hover["quaternion_wxyz"],
        )
        if orientation_error > float(args.orientation_tolerance_rad):
            raise RuntimeError(
                "live TCP orientation does not match the manual-calibration "
                "accepted hover orientation: "
                f"error={orientation_error:.4f} rad"
            )

        target_xy = localization["coarse_base_xy_m"]
        stages = plan_hover_stages(
            start,
            target_xy,
            hover_z_m=hover["z_m"],
            floor_m=floor,
            hover_z_tolerance_m=float(args.hover_z_tolerance_m),
        )

        old_kin = dict(robot._kinematics.config)
        try:
            robot._kinematics.config["position_tolerance_m"] = 0.0007
            robot._kinematics.config["orientation_tolerance_rad"] = 0.01
            robot._kinematics.config["max_seed_delta_rad"] = max(
                float(
                    robot._kinematics.config.get(
                        "max_seed_delta_rad", 0.0
                    )
                ),
                2.4,
            )
            robot._kinematics.config["max_iterations"] = max(
                int(
                    robot._kinematics.config.get(
                        "max_iterations", 0
                    )
                ),
                180,
            )

            measured_stages = []
            for label, target in stages:
                robot._kinematics.solve(
                    target,
                    robot._read_joint_positions(),
                )
                print(
                    label,
                    "TARGET xyz_m=",
                    tuple(
                        round(float(v), 6)
                        for v in target.position_m
                    ),
                    flush=True,
                )
                move_tcp_segmented(
                    robot,
                    target,
                    speed_scale=float(args.speed_scale),
                    max_translation_step_m=0.08,
                    max_orientation_step_rad=0.10,
                    min_tcp_z_m=floor,
                )
                actual = robot.get_tcp_pose()
                if (
                    quaternion_angle(
                        actual.quaternion_wxyz,
                        start.quaternion_wxyz,
                    )
                    > float(args.orientation_tolerance_rad)
                ):
                    raise RuntimeError(
                        "measured TCP orientation changed during "
                        "coarse-hover move"
                    )
                if (
                    abs(
                        float(actual.position_m[2])
                        - float(start.position_m[2])
                    )
                    > float(args.final_position_tolerance_m)
                ):
                    raise RuntimeError(
                        "measured TCP Z changed during XY-only "
                        "coarse-hover move"
                    )
                measured_stages.append(
                    {
                        "label": label,
                        "requested": _pose_record(target),
                        "measured": _pose_record(actual),
                    }
                )
        finally:
            robot._kinematics.config.clear()
            robot._kinematics.config.update(old_kin)

        final_pose = robot.get_tcp_pose()
        xy_error = math.dist(
            final_pose.position_m[:2],
            localization["coarse_base_xy_m"],
        )
        z_change = abs(
            float(final_pose.position_m[2])
            - float(start.position_m[2])
        )
        hover_reference_error = abs(
            float(final_pose.position_m[2])
            - float(hover["z_m"])
        )
        final_orientation_error = quaternion_angle(
            final_pose.quaternion_wxyz,
            start.quaternion_wxyz,
        )
        tol = float(args.final_position_tolerance_m)

        if xy_error > tol or z_change > tol:
            raise RuntimeError(
                "coarse-hover endpoint outside measured tolerance: "
                f"xy_error={xy_error:.4f} m, "
                f"z_change={z_change:.4f} m, "
                f"limit={tol:.4f} m"
            )
        if (
            hover_reference_error
            > float(args.hover_z_tolerance_m) + tol
        ):
            raise RuntimeError(
                "measured final Z no longer matches accepted hover reference"
            )
        if float(final_pose.position_m[2]) < floor:
            raise RuntimeError(
                "measured final TCP is below configured floor"
            )

        result = dict(audit)
        result.update(
            {
                "status": "reached",
                "start_tcp": _pose_record(start),
                "stages": measured_stages,
                "measured_final_tcp": _pose_record(final_pose),
                "measured_final_xy_m": [
                    float(final_pose.position_m[0]),
                    float(final_pose.position_m[1]),
                ],
                "measured_final_z_m": float(
                    final_pose.position_m[2]
                ),
                "endpoint_error_m": {
                    "xy": xy_error,
                    "z_change_from_start": z_change,
                    "z_from_hover_reference": hover_reference_error,
                },
                "orientation_change_rad": (
                    final_orientation_error
                ),
            }
        )
        result_path = output / "result.json"
        result_path.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )

        x, y, z = (float(v) for v in final_pose.position_m)
        print("", flush=True)
        print("BATTERY_SIZE1 COARSE HOVER PASS", flush=True)
        print(
            "MEASURED FINAL XYZ =",
            f"{x:.6f}",
            f"{y:.6f}",
            f"{z:.6f}",
            flush=True,
        )
        print(
            "CENTER ARGUMENT =",
            "--coarse-xy",
            f"{x:.6f}",
            f"{y:.6f}",
            flush=True,
        )
        print(
            "RESULT FILE =",
            result_path.resolve(),
            flush=True,
        )
        return 0

    except BaseException as exc:
        stopped = dict(audit)
        stopped.update(
            {
                "status": "stopped",
                "reason": str(exc),
                "exception": type(exc).__name__,
            }
        )
        try:
            (output / "result.json").write_text(
                json.dumps(stopped, indent=2) + "\n",
                encoding="utf-8",
            )
        except BaseException:
            pass
        print(
            "BATTERY_SIZE1 COARSE HOVER STOPPED",
            json.dumps(
                {
                    "reason": str(exc),
                    "exception": type(exc).__name__,
                }
            ),
            flush=True,
        )
        # Do not blanket E-stop provenance/validation failures. Actual
        # motion command failures are already stopped by VegaAdapter.move_tcp().
        raise
    finally:
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
