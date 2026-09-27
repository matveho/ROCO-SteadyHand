"""No-motion Vega software/environment preflight.

This checks installed modules/files/configuration only. It does not construct
Robot(), does not open the CAN grippers, and does not contact the robot.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.config import WORKSPACE, load_bundle, read_json
from steadyhand.skill_config import load_vega_skills


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def module_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--robot-config", help="onsite Vega robot JSON")
    p.add_argument("--urdf")
    p.add_argument("--ee-frame")
    p.add_argument("--base-frame")
    p.add_argument("--working-arm", choices=("left", "right"))
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    cfg = read_json(args.robot_config) if args.robot_config else load_bundle("vega")["robot"]
    if cfg.get("robot_id") != "vega":
        raise ValueError("--robot-config must describe Vega")
    urdf = Path(args.urdf or cfg.get("urdf_path") or "").expanduser()
    if str(urdf) not in ("", ".") and not urdf.is_absolute():
        urdf = WORKSPACE / urdf
    driver = Path((cfg.get("gripper") or {}).get("driver_path")
                  or cfg.get("gripper_driver_path") or "/nonexistent").expanduser()
    if not driver.is_absolute():
        driver = WORKSPACE / driver
    skills = load_vega_skills()
    safety = dict(skills.get("safety") or {})

    checks = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "machine": platform.machine(),
        "robot_name_env_set": bool(os.environ.get("ROBOT_NAME")),
        "dexcontrol_version": package_version("dexcontrol"),
        "dexcontrol_module": module_available("dexcontrol"),
        "pinocchio_module": module_available("pinocchio"),
        "numpy_module": module_available("numpy"),
        "python_can_module": module_available("can"),
        "wrist_cameras_module": module_available("wrist_cameras"),
        "can1_present": Path("/sys/class/net/can1").exists(),
        "gripper_driver_path": str(driver),
        "gripper_driver_exists": driver.is_file(),
        "urdf_path": str(urdf) if str(urdf) not in ("", ".") else None,
        "urdf_exists": urdf.is_file() if str(urdf) not in ("", ".") else False,
        "working_arm": args.working_arm or cfg.get("working_arm"),
        "ee_frame": args.ee_frame or (cfg.get("kinematics") or {}).get("ee_frame"),
        "base_frame": args.base_frame or (cfg.get("kinematics") or {}).get("base_frame"),
        "step_wait_time_s": (cfg.get("motion") or {}).get("step_wait_time_s"),
        "control_hz": (cfg.get("motion") or {}).get("control_hz"),
        "joint_reached_tolerance_rad": (cfg.get("motion") or {}).get("joint_reached_tolerance_rad"),
        "joint_timeout_s": (cfg.get("motion") or {}).get("joint_timeout_s"),
        "max_joint_speed_rad_s": cfg.get("max_joint_speed_rad_s"),
        "grip_current_a": (cfg.get("gripper") or {}).get("grip_current_a"),
        "min_tcp_z_m": safety.get("min_tcp_z_m"),
        "table_contact_tip_z_m": safety.get("table_contact_tip_z_m"),
    }

    expected = cfg.get("sdk_version")
    checks["dexcontrol_matches_expected"] = (
        expected is None
        or checks["dexcontrol_version"] is None
        or checks["dexcontrol_version"] == expected
    )

    if args.json:
        print(json.dumps(checks, indent=2))
    else:
        for key, value in checks.items():
            print(f"{key}: {value}")

    required = (
        checks["dexcontrol_module"],
        checks["pinocchio_module"],
        checks["numpy_module"],
        checks["python_can_module"],
        checks["urdf_exists"],
        checks["gripper_driver_exists"],
        bool(checks["working_arm"]),
        bool(checks["ee_frame"]),
        bool(checks["base_frame"]),
        all(
            isinstance(checks[key], (int, float)) and checks[key] > 0
            for key in ("joint_reached_tolerance_rad", "joint_timeout_s",
                        "grip_current_a")
        ),
        isinstance(checks["min_tcp_z_m"], (int, float)),
    )
    return 0 if all(required) else 2


if __name__ == "__main__":
    raise SystemExit(main())
