"""Load per-attempt real-world target poses.

The static task file describes semantics/order. This module consumes a fresh
runtime target file whose poses are expressed in the robot base frame.
"""

import json
from pathlib import Path

from .config import check_pose
from .models import PartGoal, Pose


def _unique_mapping(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"runtime targets: duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_runtime_targets(path, task_config, *, expected_base_frame=None):
    value = json.loads(
        Path(path).read_text(encoding="utf-8-sig"),
        object_pairs_hook=_unique_mapping,
    )
    if not isinstance(value, dict):
        raise ValueError("runtime targets: expected a JSON object")
    if value.get("schema_version") != 1:
        raise ValueError("runtime targets: unsupported schema_version")
    if value.get("robot_id") != "vega":
        raise ValueError("runtime targets: expected robot_id 'vega'")
    if value.get("pose_frame") != "robot_base":
        raise ValueError("runtime targets: pose_frame must be robot_base")
    pose_type = value.get("pose_type", "object")
    if pose_type not in ("object", "tcp"):
        raise ValueError("runtime targets: pose_type must be 'object' or 'tcp'")
    if expected_base_frame is not None:
        if not isinstance(expected_base_frame, str) or not expected_base_frame.strip():
            raise ValueError("runtime targets: expected_base_frame must name a verified URDF frame")
        if value.get("base_frame") != expected_base_frame:
            raise ValueError(
                "runtime targets: base_frame must match the verified IK base frame "
                f"{expected_base_frame!r}; got {value.get('base_frame')!r}"
            )
    if value.get("position_units", "m") != "m":
        raise ValueError("runtime targets: position_units must be m")
    if value.get("quaternion_order", "wxyz") != "wxyz":
        raise ValueError("runtime targets: quaternion_order must be wxyz")

    parts = value.get("parts")
    if not isinstance(parts, dict):
        raise ValueError("runtime targets: parts must be a mapping")

    expected = task_config["part_order"]
    unknown = set(parts) - set(expected)
    if unknown:
        raise ValueError("runtime targets: unknown parts: " + ", ".join(sorted(unknown)))
    goals = {}
    for name in expected:
        if name not in parts:
            continue
        rec = parts[name]
        if not isinstance(rec, dict):
            raise ValueError(f"runtime_targets.{name}: expected a mapping")
        unknown_fields = set(rec) - {"pick_pose", "place_pose"}
        if unknown_fields:
            raise ValueError(
                f"runtime_targets.{name}: unknown fields: "
                + ", ".join(sorted(unknown_fields))
            )
        pick = rec.get("pick_pose")
        place = rec.get("place_pose")
        if pick is not None:
            check_pose(pick, f"runtime_targets.{name}.pick_pose")
        if place is not None:
            check_pose(place, f"runtime_targets.{name}.place_pose")
        if pick is None and place is None:
            continue
        task_spec = task_config["parts"][name]
        goals[name] = PartGoal(
            name=name,
            release_mode=task_spec["simulation_release_mode"],
            pick_pose=None if pick is None else Pose.from_mapping(pick),
            place_pose=None if place is None else Pose.from_mapping(place),
            verification_method=task_spec.get("verification_method"),
        )
    return value, goals


def require_goal(goals, part_name):
    try:
        goal = goals[part_name]
    except KeyError as exc:
        raise ValueError(
            f"No runtime pose for {part_name!r}; fill its pick/place poses first"
        ) from exc
    if goal.pick_pose is None:
        raise ValueError(f"{part_name}: pick_pose is missing")
    if goal.place_pose is None:
        raise ValueError(f"{part_name}: place_pose is missing")
    return goal
