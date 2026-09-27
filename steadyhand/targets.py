"""Load per-attempt real-world target poses.

The static task file describes semantics/order. This module consumes a fresh
runtime target file whose poses are expressed in the robot base frame.
"""

import json
from pathlib import Path

from .config import check_pose
from .models import PartGoal, Pose


def load_runtime_targets(path, task_config):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("runtime targets: unsupported schema_version")
    if value.get("robot_id") != "vega":
        raise ValueError("runtime targets: expected robot_id 'vega'")
    if value.get("pose_frame") != "robot_base":
        raise ValueError("runtime targets: pose_frame must be robot_base")

    parts = value.get("parts")
    if not isinstance(parts, dict):
        raise ValueError("runtime targets: parts must be a mapping")

    expected = task_config["part_order"]
    goals = {}
    for name in expected:
        rec = parts.get(name)
        if not isinstance(rec, dict):
            continue
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
