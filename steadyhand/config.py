"""Load explicit robot/task configuration without importing a robot SDK."""

import json
import math
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
ROBOTS = ("vega", "sharpa")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def numeric_vector(value, length, label):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label}: expected {length} numbers")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError(f"{label}: values must be finite numbers")


def check_pose(pose, label):
    if not isinstance(pose, dict):
        raise ValueError(f"{label}: expected position_m and quaternion_wxyz")
    numeric_vector(pose.get("position_m"), 3, label + ".position_m")
    quat = pose.get("quaternion_wxyz")
    numeric_vector(quat, 4, label + ".quaternion_wxyz")
    if abs(sum(x * x for x in quat) - 1.0) > 0.001:
        raise ValueError(f"{label}: quaternion must have unit length")


def check_transform(matrix, label):
    if not isinstance(matrix, list) or len(matrix) != 4:
        raise ValueError(f"{label}: expected a 4x4 transform")
    for row in matrix:
        numeric_vector(row, 4, label)
    if any(abs(a - b) > 1e-6 for a, b in zip(matrix[3], [0, 0, 0, 1])):
        raise ValueError(f"{label}: last row must be [0, 0, 0, 1]")
    r = [row[:3] for row in matrix[:3]]
    for i in range(3):
        for j in range(3):
            dot = sum(r[i][k] * r[j][k] for k in range(3))
            if abs(dot - int(i == j)) > 0.001:
                raise ValueError(f"{label}: rotation must be orthonormal")
    determinant = (
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    )
    if abs(determinant - 1.0) > 0.001:
        raise ValueError(f"{label}: rotation must be right-handed")


def load_bundle(robot, root=WORKSPACE):
    if robot not in ROBOTS:
        raise ValueError(f"Unknown robot: {robot}")
    root = Path(root)
    bundle = {
        "robot": read_json(root / "configs" / "robots" / f"{robot}.json"),
        "tasks": read_json(root / "configs" / "task_board.json"),
        "calibration": read_json(root / "calibration" / f"{robot}.json"),
    }
    validate_bundle(bundle)
    if bundle["robot"]["robot_id"] != robot:
        raise ValueError("Robot filename does not match robot_id")
    return bundle


def validate_bundle(bundle):
    robot, tasks, calibration = (bundle[k] for k in ("robot", "tasks", "calibration"))
    for label, config in bundle.items():
        if config.get("schema_version") != 1:
            raise ValueError(f"{label}: unsupported schema_version")
    if robot.get("robot_id") not in ROBOTS:
        raise ValueError("Unsupported robot_id")
    if calibration.get("robot_id") != robot["robot_id"]:
        raise ValueError("Calibration belongs to a different robot")
    if tasks.get("position_units") != "m" or tasks.get("quaternion_order") != "wxyz":
        raise ValueError("Task poses must use metres and wxyz quaternions")
    order, parts = tasks.get("part_order"), tasks.get("parts")
    if not isinstance(order, list) or not order or not isinstance(parts, dict):
        raise ValueError("Tasks require a nonempty part_order and parts mapping")
    if any(not isinstance(part, str) for part in order):
        raise ValueError("Part names must be strings")
    if len(order) != len(set(order)) or set(order) != set(parts):
        raise ValueError("part_order must list each configured part exactly once")
    for name, spec in parts.items():
        if spec.get("simulation_release_mode") not in ("open", "snap"):
            raise ValueError(f"{name}: invalid simulation release mode")
        for key in ("pick_pose", "place_pose"):
            if spec.get(key) is not None:
                check_pose(spec[key], f"{name}.{key}")
    for key in ("T_base_camera", "T_base_board", "T_wrist_tcp"):
        if calibration.get(key) is not None:
            check_transform(calibration[key], key)
    return bundle


def missing_setup(bundle):
    missing = []
    for key in ("robot_name", "working_arm", "joint_names", "joint_limits_rad"):
        if not bundle["robot"].get(key):
            missing.append(f"robot.{key}")
    for key in ("base_frame", "camera_frame", "wrist_frame", "tcp_frame",
                "T_base_camera", "T_base_board", "T_wrist_tcp", "measured_at"):
        if not bundle["calibration"].get(key):
            missing.append(f"calibration.{key}")
    if not bundle["tasks"].get("pose_frame"):
        missing.append("tasks.pose_frame")
    for name, spec in bundle["tasks"]["parts"].items():
        for key in ("pick_pose", "place_pose", "verification_method"):
            if spec.get(key) is None:
                missing.append(f"tasks.{name}.{key}")
    return missing
