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
CAMERA_CLEAR_ESCAPE_LIFT_M = 0.10
CAMERA_CLEAR_VERTICALIZE_ABOVE_FLOOR_M = 0.30
CAMERA_CLEAR_JOINT_ENDPOINT_TOLERANCE_RAD = 0.020


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
    """Recover from a low/arbitrary wrist pose, then move clear of board view.

    A previous run can be interrupted mid-orientation. Do not preserve that
    arbitrary wrist attitude through a large vertical lift. Instead:
      1. make one short straight-up escape while preserving current attitude;
      2. at clear height, establish the known vertical-claw orientation;
      3. move high and centered while holding that orientation.

    This is deliberately coarse free-space repositioning, so it uses a relaxed
    measured joint endpoint tolerance locally and restores normal task settings
    before returning.
    """
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

    print(
        "CAMERA CLEAR: START ->",
        tuple(round(float(v), 4) for v in current.position_m),
        flush=True,
    )

    kin_cfg = robot._kinematics.config
    old_kin = dict(kin_cfg)
    old_joint_tol = float(robot.config["motion"]["joint_reached_tolerance_rad"])
    try:
        # Coarse free-space preset only. Normal manipulation keeps its original
        # tighter endpoint requirement after this helper returns.
        robot.config["motion"]["joint_reached_tolerance_rad"] = max(
            old_joint_tol,
            CAMERA_CLEAR_JOINT_ENDPOINT_TOLERANCE_RAD,
        )
        kin_cfg["position_tolerance_m"] = 0.010
        kin_cfg["orientation_tolerance_rad"] = 0.12
        kin_cfg["max_seed_delta_rad"] = 2.5
        kin_cfg["max_iterations"] = 180

        escape_z = min(
            float(target.position_m[2]),
            float(current.position_m[2]) + CAMERA_CLEAR_ESCAPE_LIFT_M,
        )
        if escape_z > float(current.position_m[2]) + 0.01:
            escape = Pose(
                (
                    float(current.position_m[0]),
                    float(current.position_m[1]),
                    escape_z,
                ),
                tuple(float(v) for v in current.quaternion_wxyz),
            )
            print(
                "CAMERA CLEAR: ESCAPE LIFT ->",
                tuple(round(float(v), 4) for v in escape.position_m),
                flush=True,
            )
            move_tcp_segmented(
                robot,
                escape,
                speed_scale=speed_scale,
                max_translation_step_m=0.10,
                max_orientation_step_rad=0.80,
                min_tcp_z_m=floor_m,
            )

        verticalize_z = min(
            float(target.position_m[2]),
            max(
                float(robot.get_tcp_pose().position_m[2]),
                floor_m + CAMERA_CLEAR_VERTICALIZE_ABOVE_FLOOR_M,
            ),
        )
        verticalize = Pose(
            (
                CAMERA_CLEAR_X_M,
                CAMERA_CLEAR_Y_M,
                verticalize_z,
            ),
            vertical_claw_tip_quaternion(0.0),
        )
        print(
            "CAMERA CLEAR: VERTICALIZE ->",
            tuple(round(float(v), 4) for v in verticalize.position_m),
            flush=True,
        )
        move_tcp_segmented(
            robot,
            verticalize,
            speed_scale=speed_scale,
            max_translation_step_m=0.15,
            max_orientation_step_rad=0.55,
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
            max_translation_step_m=0.20,
            max_orientation_step_rad=0.55,
            min_tcp_z_m=floor_m,
        )
        actual = robot.get_tcp_pose()
        print(
            "CAMERA CLEAR: REACHED ->",
            tuple(round(float(v), 4) for v in actual.position_m),
            flush=True,
        )
        return actual
    finally:
        robot.config["motion"]["joint_reached_tolerance_rad"] = old_joint_tol
        kin_cfg.clear()
        kin_cfg.update(old_kin)

