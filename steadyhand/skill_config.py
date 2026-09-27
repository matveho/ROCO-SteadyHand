"""Robot-specific execution parameters layered over task semantics."""

import math
from pathlib import Path

from .config import (
    WORKSPACE,
    check_pose,
    check_transform,
    numeric_vector,
    read_json,
)


def load_vega_skills(path=None):
    path = Path(path) if path else WORKSPACE / "configs" / "skills" / "vega.json"
    value = read_json(path)
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or value.get("robot_id") != "vega"):
        raise ValueError("Invalid Vega skill configuration")
    if not isinstance(value.get("defaults"), dict):
        raise ValueError("Vega skills require defaults")
    if not isinstance(value.get("parts"), dict):
        raise ValueError("Vega skills require parts mapping")
    if any(not isinstance(part, dict) for part in value["parts"].values()):
        raise ValueError("Vega skill part overrides must be mappings")
    return value


def skill_for_part(config, part_name):
    if part_name not in config["parts"]:
        raise ValueError(f"No Vega skill configured for {part_name!r}")
    merged = dict(config["defaults"])
    merged.update(config["parts"].get(part_name, {}))
    return merged


def validate_skill(skill, *, require_calibrated=True):
    """Validate effective settings before constructing Robot() or homing CAN.

    Simulation offsets/orientations are retained for reference, but are not a
    calibrated transform to any particular physical URDF end-effector frame.
    """
    if not isinstance(skill, dict):
        raise ValueError("Vega skill must be a mapping")

    def number(key, *, positive=False, default=None):
        value = skill.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"skill.{key}: configure a finite measured/validated number")
        if value < 0 or (positive and value == 0):
            relation = "> 0" if positive else ">= 0"
            raise ValueError(f"skill.{key}: must be {relation}")
        return value

    for key in ("hover_pick_m", "hover_place_m", "retract_m",
                "max_cartesian_step_m", "max_orientation_step_rad",
                "preinsert_m", "insertion_reached_tolerance_m",
                "insertion_reached_tolerance_rad"):
        number(key, positive=True)
    for key in ("grasp_settle_s", "release_settle_s"):
        number(key, default=0.0)
    if skill.get("grip_current_a") is not None or require_calibrated:
        number("grip_current_a", positive=True)
    if "gripper_open_fraction" in skill:
        fraction = number("gripper_open_fraction")
        if fraction != 1:
            raise ValueError(
                "skill.gripper_open_fraction: the current CAN wrapper supports "
                "only full both_open(), so this must be exactly 1"
            )
    for key in ("legacy_geometry_verified", "allow_snap_without_force_guard"):
        if key in skill and type(skill[key]) is not bool:
            raise ValueError(f"skill.{key}: must be a boolean")

    runtime_pose_type = skill.get("runtime_pose_type", "object")
    if runtime_pose_type not in ("object", "tcp"):
        raise ValueError("skill.runtime_pose_type must be 'object' or 'tcp'")

    transform = skill.get("T_part_tcp")
    if runtime_pose_type == "tcp":
        # Explicitly measured TCP targets need no object->TCP transform.
        pass
    elif transform is not None:
        check_transform(transform, "skill.T_part_tcp")
    else:
        check_pose(
            {"position_m": skill.get("legacy_ee_offset_m"),
             "quaternion_wxyz": skill.get("legacy_ee_orientation_wxyz")},
            "skill.legacy_geometry",
        )
        if require_calibrated and skill.get("legacy_geometry_verified") is not True:
            raise ValueError(
                "skill: configure calibrated T_part_tcp or set "
                "legacy_geometry_verified=true only after measuring the legacy "
                "offset/orientation for the selected physical URDF EE frame. "
                "The simulation orientation includes a USD/Lula frame correction."
            )

    search = skill.get("search")
    if search is not None:
        if not isinstance(search, dict) or search.get("type") != "grid":
            raise ValueError("skill.search: only a grid mapping or null is supported")
        if type(search.get("n")) is not int or search["n"] < 1:
            raise ValueError("skill.search.n: must be a positive integer")
        numeric_vector(search.get("extent_xy_m"), 2, "skill.search.extent_xy_m")
        if any(x < 0 for x in search["extent_xy_m"]):
            raise ValueError("skill.search.extent_xy_m: must be non-negative")
    return skill
