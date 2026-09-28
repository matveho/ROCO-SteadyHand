"""Battery-size1 bring-up calibration and result contracts.

This module is deliberately part-specific. It contains no motion code and
never supplies guessed calibration values. Right-arm provenance is explicit:
a calibration is valid only for working_arm=right, tcp_frame=tip_r,
wrist_camera=wrist_a, and the right CAN gripper scope.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .geometry import quaternion_angle
from .models import Pose


PART_NAME = "battery_size1"
WORKING_ARM = "right"
TCP_FRAME = "tip_r"
WRIST_CAMERA = "wrist_a"
WRIST_PHYSICAL_MOUNT = "right_wrist"
GRIPPER_SCOPE = "right"
SCHEMA_VERSION = 2
CALIBRATION_GENERATION = "right_arm_tip_r_wrist_a_clean_v1"
VERIFIED_GRIP_CURRENT_A = 1.0
VERIFIED_GRIP_SPEED_DPS = 240


def calibration_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_right_battery_config(robot_config):
    """Fail closed unless the runtime robot config is the right battery stack."""
    if robot_config.get("working_arm") != WORKING_ARM:
        raise ValueError("battery_size1 requires working_arm=right")

    kin = robot_config.get("kinematics") or {}
    if kin.get("ee_frame") != TCP_FRAME:
        raise ValueError("battery_size1 requires ee_frame=tip_r")

    cameras = robot_config.get("cameras") or {}
    wrists = cameras.get("wrists") or {}
    mapping = wrists.get("api_label_to_physical_mount") or {}
    if mapping.get(WRIST_CAMERA) != WRIST_PHYSICAL_MOUNT:
        raise ValueError(
            "battery_size1 requires wrist_a to map to physical right_wrist"
        )

    gripper = robot_config.get("gripper") or {}
    if gripper.get("scope") != GRIPPER_SCOPE:
        raise ValueError("battery_size1 requires gripper.scope=right")

    return {
        "working_arm": WORKING_ARM,
        "tcp_frame": TCP_FRAME,
        "wrist_camera": WRIST_CAMERA,
        "wrist_physical_mount": WRIST_PHYSICAL_MOUNT,
        "gripper_scope": GRIPPER_SCOPE,
    }


def blank_calibration(robot_config):
    require_right_battery_config(robot_config)
    return {
        "schema_version": SCHEMA_VERSION,
        "calibration_generation": CALIBRATION_GENERATION,
        "part": PART_NAME,
        "robot_name": robot_config["robot_name"],
        "base_frame": robot_config["kinematics"]["base_frame"],
        "working_arm": WORKING_ARM,
        "tcp_frame": TCP_FRAME,
        "wrist_camera": WRIST_CAMERA,
        "jaw_alignment": {
            "goal_pixel_uv": None,
            "image_size_px": None,
            "source_image": None,
            "taught_hover_tcp_z_m": None,
            "taught_tip_quaternion_wxyz": None,
            "source": None,
        },
        "grasp": {
            "tcp_z_m": None,
            "taught_tip_quaternion_wxyz": None,
            "source": None,
        },
        "gripper": {
            "scope": GRIPPER_SCOPE,
            "current_a": VERIFIED_GRIP_CURRENT_A,
            "speed_dps": VERIFIED_GRIP_SPEED_DPS,
        },
        "calibration_complete": False,
    }


def validate_calibration_provenance(value, robot_config):
    """Validate right-arm identity/provenance even for incomplete calibration."""
    require_right_battery_config(robot_config)

    if not isinstance(value, dict):
        raise ValueError("battery calibration must be a JSON object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "battery calibration is stale or provenance-missing; "
            "run clean right-arm init --force"
        )
    if value.get("calibration_generation") != CALIBRATION_GENERATION:
        raise ValueError(
            "battery calibration lacks clean right-arm initialization provenance; "
            "run init --force"
        )
    if value.get("part") != PART_NAME:
        raise ValueError("battery calibration must target battery_size1")
    if value.get("robot_name") != robot_config["robot_name"]:
        raise ValueError("battery calibration is for a different robot")
    if value.get("base_frame") != robot_config["kinematics"]["base_frame"]:
        raise ValueError("battery calibration has the wrong base frame")
    if value.get("working_arm") != WORKING_ARM:
        raise ValueError(
            "battery calibration working_arm is not right; "
            "left/pre-switch calibration is stale"
        )
    if value.get("tcp_frame") != TCP_FRAME:
        raise ValueError(
            "battery calibration tcp_frame is not tip_r; "
            "left/pre-switch calibration is stale"
        )
    if value.get("wrist_camera") != WRIST_CAMERA:
        raise ValueError(
            "battery calibration wrist_camera is not wrist_a; "
            "left/pre-switch calibration is stale"
        )

    gripper = value.get("gripper")
    if not isinstance(gripper, dict):
        raise ValueError("battery calibration is missing gripper provenance")
    if gripper.get("scope") != GRIPPER_SCOPE:
        raise ValueError(
            "battery calibration gripper scope is not right; "
            "left/pre-switch calibration is stale"
        )

    current = _finite_scalar(gripper.get("current_a"), "gripper current")
    speed = _finite_scalar(gripper.get("speed_dps"), "gripper speed")
    if abs(current - VERIFIED_GRIP_CURRENT_A) > 1e-12:
        raise ValueError("battery gripper current must be verified 1.0 A")
    if abs(speed - VERIFIED_GRIP_SPEED_DPS) > 1e-12:
        raise ValueError("battery gripper speed must be verified 240 deg/s")

    return value


def load_calibration(path, robot_config, *, floor_m):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_calibration_provenance(value, robot_config)

    jaw = value.get("jaw_alignment")
    grasp = value.get("grasp")
    if not isinstance(jaw, dict) or not isinstance(grasp, dict):
        raise ValueError("battery calibration is missing jaw/grasp sections")

    goal = _finite_vector(jaw.get("goal_pixel_uv"), 2, "jaw goal pixel")
    image_size = jaw.get("image_size_px")
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError("jaw image_size_px must contain [width,height]")
    width, height = (int(v) for v in image_size)
    if width <= 0 or height <= 0:
        raise ValueError("jaw image_size_px must be positive")
    if not (0 <= goal[0] < width and 0 <= goal[1] < height):
        raise ValueError("jaw goal pixel lies outside the taught image")

    floor = _finite_scalar(floor_m, "task floor")
    hover_z = _finite_scalar(
        jaw.get("taught_hover_tcp_z_m"),
        "jaw taught_hover_tcp_z_m",
    )
    hover_quat = _unit_quaternion(
        jaw.get("taught_tip_quaternion_wxyz"),
        "jaw taught tip quaternion",
    )
    if not floor + 0.060 <= hover_z <= floor + 0.120:
        raise ValueError(
            "jaw goal pixel must be taught at a 60-120 mm safe hover"
        )
    if jaw.get("source") != "operator_taught_current_tcp":
        raise ValueError(
            "jaw goal pixel must be operator-taught with a measured current TCP"
        )

    grasp_z = _finite_scalar(grasp.get("tcp_z_m"), "grasp tcp_z_m")
    if grasp_z < floor:
        raise ValueError(
            f"taught grasp TCP z={grasp_z:.6f} m is below task floor "
            f"{floor:.6f} m"
        )
    quat = _unit_quaternion(
        grasp.get("taught_tip_quaternion_wxyz"),
        "taught tip quaternion",
    )
    if grasp.get("source") != "operator_taught_current_tcp":
        raise ValueError(
            "grasp Z must come from an operator-taught measured current TCP"
        )

    if value.get("calibration_complete") is not True:
        raise ValueError("battery calibration is incomplete")
    if not calibration_is_complete(value):
        raise ValueError(
            "battery calibration completeness/provenance contract is inconsistent"
        )

    out = dict(value)
    out["jaw_alignment"] = dict(jaw)
    out["jaw_alignment"]["goal_pixel_uv"] = list(goal)
    out["jaw_alignment"]["image_size_px"] = [width, height]
    out["jaw_alignment"]["taught_hover_tcp_z_m"] = hover_z
    out["jaw_alignment"]["taught_tip_quaternion_wxyz"] = list(hover_quat)
    out["grasp"] = dict(grasp)
    out["grasp"]["tcp_z_m"] = grasp_z
    out["grasp"]["taught_tip_quaternion_wxyz"] = list(quat)
    return out


def load_alignment_result(
    path,
    calibration_path,
    calibration,
    *,
    current_pose=None,
):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("battery alignment result must be a JSON object")
    if value.get("status") != "converged":
        raise ValueError("battery alignment result is not converged")
    if (
        value.get("part") != PART_NAME
        or value.get("wrist_camera") != WRIST_CAMERA
    ):
        raise ValueError(
            "alignment result is not a battery_size1 wrist_a result"
        )
    if value.get("working_arm") != WORKING_ARM:
        raise ValueError(
            "alignment result lacks explicit working_arm=right provenance"
        )
    if value.get("tcp_frame") != TCP_FRAME:
        raise ValueError(
            "alignment result lacks explicit tcp_frame=tip_r provenance"
        )

    expected_digest = calibration_sha256(calibration_path)
    if value.get("calibration_sha256") != expected_digest:
        raise ValueError(
            "alignment result was produced with a different calibration"
        )
    goal = tuple(
        float(v)
        for v in calibration["jaw_alignment"]["goal_pixel_uv"]
    )
    actual_goal = _finite_vector(
        value.get("goal_uv"), 2, "alignment goal"
    )
    if math.dist(goal, actual_goal) > 0.5:
        raise ValueError(
            "alignment result did not center to the taught jaw goal pixel"
        )
    tcp = Pose(
        tuple(
            _finite_vector(
                value.get("tcp_position_m"),
                3,
                "alignment TCP",
            )
        ),
        tuple(
            _unit_quaternion(
                value.get("tcp_quaternion_wxyz"),
                "alignment TCP quaternion",
            )
        ),
    )
    if current_pose is not None:
        if math.dist(current_pose.position_m, tcp.position_m) > 0.003:
            raise ValueError(
                "live TCP no longer matches the successful wrist alignment"
            )
        if (
            quaternion_angle(
                current_pose.quaternion_wxyz,
                tcp.quaternion_wxyz,
            )
            > 0.025
        ):
            raise ValueError(
                "live TCP orientation changed after wrist alignment"
            )
    return value, tcp


def calibration_is_complete(value):
    try:
        jaw = value["jaw_alignment"]
        grasp = value["grasp"]
        gripper = value["gripper"]
        return (
            value.get("schema_version") == SCHEMA_VERSION
            and value.get("calibration_generation")
            == CALIBRATION_GENERATION
            and value.get("part") == PART_NAME
            and value.get("working_arm") == WORKING_ARM
            and value.get("tcp_frame") == TCP_FRAME
            and value.get("wrist_camera") == WRIST_CAMERA
            and gripper.get("scope") == GRIPPER_SCOPE
            and float(gripper.get("current_a"))
            == VERIFIED_GRIP_CURRENT_A
            and float(gripper.get("speed_dps"))
            == VERIFIED_GRIP_SPEED_DPS
            and jaw.get("goal_pixel_uv") is not None
            and jaw.get("image_size_px") is not None
            and jaw.get("taught_hover_tcp_z_m") is not None
            and jaw.get("taught_tip_quaternion_wxyz") is not None
            and jaw.get("source") == "operator_taught_current_tcp"
            and grasp.get("tcp_z_m") is not None
            and grasp.get("taught_tip_quaternion_wxyz") is not None
            and grasp.get("source") == "operator_taught_current_tcp"
        )
    except (KeyError, TypeError, ValueError):
        return False


def _finite_scalar(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _finite_vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(
            f"{name} must contain {size} finite numbers"
        )
    return tuple(_finite_scalar(v, name) for v in value)


def _unit_quaternion(value, name):
    quat = _finite_vector(value, 4, name)
    norm2 = sum(v * v for v in quat)
    if abs(norm2 - 1.0) > 0.001:
        raise ValueError(f"{name} must be unit length")
    return quat
