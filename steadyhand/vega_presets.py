"""Validated measured joint presets for the competition Vega right arm."""

import math

from .models import Pose


def configured_right_preset(robot_config, name):
    """Return ``(joint_positions, tip_r_pose)`` for a measured Vega preset.

    Presets are operator-measured endpoints. This helper validates their shape,
    joint-name order, limits, and pose before any caller can command one. It
    does not claim that the joint-space path to an endpoint is collision-free.
    """
    if robot_config.get("working_arm") != "right":
        raise ValueError("Vega measured presets require working_arm='right'")
    kin = robot_config.get("kinematics") or {}
    if kin.get("ee_frame") != "tip_r":
        raise ValueError("Vega measured presets require kinematics.ee_frame='tip_r'")
    presets = robot_config.get("joint_presets") or {}
    value = presets.get(name)
    if not isinstance(value, dict):
        raise KeyError(f"No configured Vega joint preset named {name!r}")

    expected_names = tuple(kin.get("right_arm_joint_names") or ())
    names = tuple(value.get("joint_names") or ())
    if names != expected_names:
        raise ValueError(f"Preset {name!r} joint_names do not match right-arm order")

    raw_q = value.get("joint_positions_rad")
    if not isinstance(raw_q, list) or len(raw_q) != 7:
        raise ValueError(f"Preset {name!r} requires seven joint positions")
    q = tuple(float(v) for v in raw_q)
    if not all(math.isfinite(v) for v in q):
        raise ValueError(f"Preset {name!r} joint positions must be finite")

    limits = (robot_config.get("arm_joint_limits_rad") or {}).get("right")
    if not isinstance(limits, list) or len(limits) != 7:
        raise ValueError("right arm joint limits are required for Vega presets")
    for index, (v, bound) in enumerate(zip(q, limits), 1):
        lo, hi = (float(x) for x in bound)
        if not lo <= v <= hi:
            raise ValueError(
                f"Preset {name!r} joint {index}={v:.6f} is outside [{lo:.6f}, {hi:.6f}]"
            )

    pose_value = value.get("tip_r_pose")
    if not isinstance(pose_value, dict):
        raise ValueError(f"Preset {name!r} requires tip_r_pose provenance")
    pose = Pose.from_mapping(pose_value)
    return q, pose


def preset_max_delta(current_q, target_q):
    current = tuple(float(v) for v in current_q)
    target = tuple(float(v) for v in target_q)
    if len(current) != 7 or len(target) != 7:
        raise ValueError("Vega preset delta requires seven joint positions")
    return max(abs(a - b) for a, b in zip(current, target))
