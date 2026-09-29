"""Persistent provenance and validation for per-part wrist teaching."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


PART_NAMES = (
    "gear_60teeth", "gear_20teeth", "rod_16mm", "bolt_8mm", "usb_a",
    "hdmi", "pin", "battery_size1", "battery_size5",
)
WRIST_CAMERA = "wrist_a"
WORKING_ARM = "right"
TCP_FRAME = "tip_r"
SCHEMA_VERSION = 1


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_profile_path(root):
    return Path(root) / "calibration" / "wrist_part_profiles.json"


def default_template_dir(root):
    return Path(root) / "calibration" / "wrist_templates"


def blank_profiles(robot_config):
    return {
        "schema_version": SCHEMA_VERSION,
        "profiles_kind": "vega_right_wrist_part_grasp_profiles",
        "robot_name": robot_config.get("robot_name"),
        "working_arm": WORKING_ARM,
        "tcp_frame": TCP_FRAME,
        "wrist_camera": WRIST_CAMERA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parts": {},
    }


def load_profiles(path, robot_config):
    path = Path(path)
    if not path.is_file():
        return blank_profiles(robot_config)
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_profiles(value, robot_config)
    return value


def save_profile(path, robot_config, profile):
    path = Path(path)
    value = load_profiles(path, robot_config)
    value["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    value.setdefault("parts", {})[profile["part"]] = dict(profile)
    validate_profiles(value, robot_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)
    return value


def validate_profiles(value, robot_config):
    if not isinstance(value, dict):
        raise ValueError("wrist profiles must be a JSON object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported wrist profile schema_version")
    if value.get("profiles_kind") != "vega_right_wrist_part_grasp_profiles":
        raise ValueError("wrist profiles have the wrong kind")
    if value.get("robot_name") != robot_config.get("robot_name"):
        raise ValueError("wrist profiles belong to a different robot")
    if value.get("working_arm") != WORKING_ARM or value.get("tcp_frame") != TCP_FRAME:
        raise ValueError("wrist profiles require right arm / tip_r")
    if value.get("wrist_camera") != WRIST_CAMERA:
        raise ValueError("wrist profiles require physical-right wrist_a")
    parts = value.get("parts")
    if not isinstance(parts, dict):
        raise ValueError("wrist profiles require a parts mapping")
    for name, profile in parts.items():
        validate_profile(profile, robot_config, expected_part=name)
    return value


def validate_profile(profile, robot_config, *, expected_part=None):
    if not isinstance(profile, dict):
        raise ValueError("wrist part profile must be an object")
    part = profile.get("part")
    if expected_part is not None and part != expected_part:
        raise ValueError("wrist profile part key does not match profile part")
    if part not in PART_NAMES:
        raise ValueError(f"unknown wrist profile part {part!r}")
    if profile.get("working_arm") != WORKING_ARM or profile.get("tcp_frame") != TCP_FRAME:
        raise ValueError(f"{part}: wrong arm/TCP provenance")
    if profile.get("wrist_camera") != WRIST_CAMERA:
        raise ValueError(f"{part}: wrong wrist camera provenance")
    for name, size in (("feature_uv", 2), ("goal_uv", 2), ("coarse_xy_m", 2)):
        _finite_vector(profile.get(name), size, f"{part}.{name}")
    for name in ("hover_clearance_m", "yaw_deg"):
        _finite(profile.get(name), f"{part}.{name}")
    opening = profile.get("gripper_open_fraction")
    if opening is not None and not 0.0 <= _finite(opening, "gripper opening") <= 1.0:
        raise ValueError(f"{part}: gripper opening must be 0..1 (0 closed, 1 open)")
    if not 0.060 <= profile["hover_clearance_m"] <= 0.150:
        raise ValueError(f"{part}: hover clearance must be 60..150 mm")
    if abs(profile["yaw_deg"]) > 45:
        raise ValueError(f"{part}: yaw must be within 45 degrees of ready")
    grasp = profile.get("grasp_clearance_m")
    if grasp is not None and not 0 <= _finite(grasp, "grasp clearance") < profile["hover_clearance_m"]:
        raise ValueError(f"{part}: grasp clearance must be below hover and above surface")
    shape = _finite_vector(profile.get("image_shape"), 2, "image shape")
    for field in ("feature_uv", "goal_uv"):
        u, v = profile[field]
        if not 0 <= u < shape[1] or not 0 <= v < shape[0]:
            raise ValueError(f"{part}: {field} is outside image")
    if len(str(profile.get("calibration_sha256", ""))) != 64:
        raise ValueError(f"{part}: board calibration hash is required")
    place = profile.get("place")
    if place is not None:
        _finite_vector(place.get("offset_board_xy_m"), 2, "place offset")
        if not 0 <= _finite(place.get("clearance_m"), "place clearance") < profile["hover_clearance_m"]:
            raise ValueError("place clearance must be below hover and above surface")
        if abs(_finite(place.get("yaw_deg"), "place yaw")) > 45:
            raise ValueError("place yaw must be within 45 degrees of ready")
    template = profile.get("template")
    if not isinstance(template, dict) or not isinstance(template.get("path"), str):
        raise ValueError(f"{part}: template provenance is required")
    if len(str(template.get("sha256", ""))) != 64:
        raise ValueError(f"{part}: template SHA256 is malformed")
    return profile


def _finite_vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} finite numbers")
    result = [float(v) for v in value]
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _finite(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
