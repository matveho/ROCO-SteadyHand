"""Shared camera-clear arm preset for Vega board imaging.

The competition head camera can see the left claw when the arm is working low
over the board. Before board-level perception, move tip_l to a high centered
free-space pose so the claw cannot be mistaken for a dark task part.

This preset is operator-requested onsite 2026-09-28:
- about 0.50 m in front of the robot,
- centered laterally,
- about 0.80 m above the measured task floor,
- physical gripper axis vertical,
- fast free-space motion.
"""

import math

from .executor import move_tcp_segmented
from .models import Pose


CAMERA_CLEAR_X_M = 0.50
CAMERA_CLEAR_Y_M = 0.00
CAMERA_CLEAR_Z_ABOVE_FLOOR_M = 0.80
CAMERA_CLEAR_SPEED_SCALE = 0.90
CAMERA_CLEAR_MIN_LIFT_ABOVE_FLOOR_M = 0.35


def vertical_claw_tip_quaternion(yaw_rad: float = 0.0):
    """tip_l wxyz quaternion for the physical gripper axis vertical/downward.

    For the tracked competition URDF, a top-down physical gripper maps to a
    pure base-Z rotation of tip_l: R_base_tip = Rz(yaw - pi/2).
    """
    half = (float(yaw_rad) - math.pi / 2.0) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def camera_clear_pose(floor_m: float) -> Pose:
    floor_m = float(floor_m)
    return Pose(
        (
            CAMERA_CLEAR_X_M,
            CAMERA_CLEAR_Y_M,
            floor_m + CAMERA_CLEAR_Z_ABOVE_FLOOR_M,
        ),
        vertical_claw_tip_quaternion(0.0),
    )


def move_camera_clear(robot, *, floor_m: float, speed_scale: float = CAMERA_CLEAR_SPEED_SCALE):
    """Raise first, then finish at the centered high vertical-claw preset."""
    floor_m = float(floor_m)
    speed_scale = float(speed_scale)
    target = camera_clear_pose(floor_m)
    current = robot.get_tcp_pose()

    if current is None:
        raise RuntimeError("camera-clear move requires current TCP pose")
    if float(current.position_m[2]) < floor_m:
        raise RuntimeError(
            f"Current TCP z={float(current.position_m[2]):.4f} is below "
            f"configured floor {floor_m:.4f}; recover upward before camera-clear move"
        )

    # Give the high preset enough IK room without changing normal task config.
    kin_cfg = robot._kinematics.config
    kin_cfg["position_tolerance_m"] = 0.010
    kin_cfg["orientation_tolerance_rad"] = 0.12
    kin_cfg["max_seed_delta_rad"] = 2.5
    kin_cfg["max_iterations"] = 180

    lift_z = min(
        float(target.position_m[2]),
        max(
            float(current.position_m[2]),
            floor_m + CAMERA_CLEAR_MIN_LIFT_ABOVE_FLOOR_M,
        ),
    )
    if lift_z > float(current.position_m[2]) + 0.01:
        lift = Pose(
            (
                float(current.position_m[0]),
                float(current.position_m[1]),
                lift_z,
            ),
            tuple(float(v) for v in current.quaternion_wxyz),
        )
        print(
            "CAMERA CLEAR: LIFT ->",
            tuple(round(float(v), 4) for v in lift.position_m),
            flush=True,
        )
        move_tcp_segmented(
            robot,
            lift,
            speed_scale=speed_scale,
            max_translation_step_m=0.35,
            max_orientation_step_rad=1.0,
            min_tcp_z_m=floor_m,
        )

    print(
        "CAMERA CLEAR: PRESET ->",
        tuple(round(float(v), 4) for v in target.position_m),
        "vertical claw",
        flush=True,
    )
    move_tcp_segmented(
        robot,
        target,
        speed_scale=speed_scale,
        max_translation_step_m=0.35,
        max_orientation_step_rad=0.80,
        min_tcp_z_m=floor_m,
    )
    actual = robot.get_tcp_pose()
    print(
        "CAMERA CLEAR: REACHED ->",
        tuple(round(float(v), 4) for v in actual.position_m),
        flush=True,
    )
    return actual
