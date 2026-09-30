"""Diagnose the physical-right Vega CAN gripper without starting the robot.

This intentionally does not construct ``VegaAdapter`` or ``Robot()``. It
isolates the right motor and local ``can1`` from Zenoh/network failures. The
default operation is a no-motion telemetry probe. Homing and motion tests are
explicit, and only the configured right motor is ever addressed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.config import load_bundle
from steadyhand.grippers.vega import VegaCanGripper


ROOT = Path(__file__).resolve().parents[1]


def _finite(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{name} returned non-finite value {value!r}")
    return number


def _status(gripper):
    status = gripper.gripper_status()
    print("RIGHT STATUS =", json.dumps(status, default=str), flush=True)
    return status


def _raw_status(gripper):
    """Read right-motor telemetry without requiring a homing calibration."""
    motor = gripper._driver.right
    alive = bool(motor.alive())
    result = {
        "alive": alive,
        "angle_deg": motor.angle(),
        "current_a": motor.current(),
        "enabled": motor.enabled(),
        "can_interface": "can1",
        "scope": "right",
    }
    print("RIGHT RAW TELEMETRY =", json.dumps(result, default=str), flush=True)
    return result


def _move_fraction(gripper, fraction, *, speed, label, report):
    print(f"RIGHT {label}: fraction={fraction:.3f} speed={speed}", flush=True)
    result = gripper.move_fraction(fraction, speed=speed)
    position = _finite(gripper.gripper_position(), "right gripper position")
    record = {
        "label": label,
        "requested_fraction": fraction,
        "result": result,
        "measured_fraction": position,
    }
    report.append(record)
    print("RIGHT MOTION RESULT =", json.dumps(record, default=str), flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="read right-motor CAN telemetry without homing or motion",
    )
    parser.add_argument(
        "--exercise",
        action="store_true",
        help="home, move right jaw through safe fractions, and report readback",
    )
    parser.add_argument(
        "--grip",
        action="store_true",
        help="after exercise, run one current-limited right-jaw grip test",
    )
    parser.add_argument('--confirm-physical-motion', action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--low-fraction", type=float, default=0.20)
    parser.add_argument("--high-fraction", type=float, default=0.80)
    parser.add_argument("--open-speed", type=int, default=None)
    parser.add_argument("--grip-current", type=float, default=None)
    parser.add_argument("--grip-speed", type=int, default=None)
    parser.add_argument("--output", help="optional JSON report path")
    args = parser.parse_args(argv)

    if not args.status_only and not args.exercise and not args.grip:
        # Safe default: telemetry only. Homing is a physical command and must
        # be requested explicitly with --exercise.
        args.status_only = True
    if args.status_only and (args.exercise or args.grip):
        parser.error("--status-only cannot be combined with --exercise or --grip")
    if args.grip and not args.exercise:
        parser.error("--grip requires --exercise")
    if not 0.02 <= args.low_fraction <= 0.98:
        parser.error("--low-fraction must be 0.02..0.98")
    if not 0.02 <= args.high_fraction <= 0.98:
        parser.error("--high-fraction must be 0.02..0.98")
    if args.low_fraction >= args.high_fraction:
        parser.error("--low-fraction must be below --high-fraction")

    bundle = load_bundle("vega")
    cfg = dict(bundle["robot"])
    gripper_cfg = dict(cfg.get("gripper") or {})
    if gripper_cfg.get("scope") != "right":
        raise RuntimeError(f"Expected right-only gripper scope, got {gripper_cfg.get('scope')!r}")
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "robot_name": cfg.get("robot_name"),
        "can_interface": "can1",
        "scope": "right",
        "can1_present": Path("/sys/class/net/can1").exists(),
        "driver_path": str(gripper_cfg.get("driver_path") or cfg.get("gripper_driver_path")),
        "status_only": bool(args.status_only),
        "motion": [],
        "errors": [],
    }
    print("VEGA RIGHT GRIPPER DIAGNOSTIC", flush=True)
    print(json.dumps({k: report[k] for k in (
        "robot_name", "can_interface", "scope", "can1_present", "driver_path"
    )}, indent=2), flush=True)
    gripper = None
    try:
        if args.status_only:
            # The production gate requires an explicit verified override before
            # skipping home. This is a read-only diagnostic mode; no position
            # command is issued and raw angle/current telemetry is used.
            gripper_cfg["home_on_connect"] = False
            gripper_cfg["skip_home_verified"] = True
        gripper = VegaCanGripper(gripper_cfg)
        gripper.connect()
        report["connected"] = True
        if args.status_only:
            report["raw_status"] = _raw_status(gripper)
            print("STATUS-ONLY COMPLETE: no right-jaw motion was commanded.", flush=True)
        else:
            report["home_status"] = _status(gripper)
            if not args.exercise:
                print("HOME/STATUS COMPLETE: use --exercise for motion readback.", flush=True)
            else:
                speed = args.open_speed or int(round(float(gripper_cfg.get("open_speed_dps", 500))))
                _move_fraction(
                    gripper, args.low_fraction, speed=speed, label="LOW TEST", report=report["motion"]
                )
                _move_fraction(
                    gripper, args.high_fraction, speed=speed, label="HIGH TEST", report=report["motion"]
                )
                _move_fraction(
                    gripper, 0.98, speed=speed, label="OPEN TEST", report=report["motion"]
                )
                if args.grip:
                    current = args.grip_current or float(gripper_cfg.get("grip_current_a", 1.0))
                    grip_speed = args.grip_speed or int(round(float(gripper_cfg.get("grip_speed_dps", 60))))
                    print(
                        f"RIGHT GRIP TEST: current={current:.3f} A speed={grip_speed}",
                        flush=True,
                    )
                    grip_result = gripper.grip(current_a=current)
                    report["grip_result"] = grip_result
                    print("RIGHT GRIP RESULT =", json.dumps(grip_result, default=str), flush=True)
                else:
                    _move_fraction(
                        gripper, 0.20, speed=speed, label="RETURN TEST", report=report["motion"]
                    )
                print("EXERCISE COMPLETE: inspect the right jaw and report the JSON/results.", flush=True)
        report["ok"] = True
    except BaseException as exc:
        report["ok"] = False
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        print(f"DIAGNOSTIC FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        if gripper is not None:
            try:
                gripper.close()
            except BaseException as exc:
                report["errors"].append(f"shutdown {type(exc).__name__}: {exc}")
                print(f"GRIPPER SHUTDOWN ERROR: {exc}", file=sys.stderr, flush=True)
        if args.output:
            output = Path(args.output)
            if not output.is_absolute():
                output = ROOT / output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
            print(f"DIAGNOSTIC REPORT = {output}", flush=True)
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
