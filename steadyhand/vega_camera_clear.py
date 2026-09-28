"""Shared camera-clear arm preset for Vega board imaging.

The competition head camera can see the right claw when the arm is working low
over the board. Before board-level perception, move tip_r to a high centered
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
from .vega_presets import configured_right_preset, preset_max_delta


CAMERA_CLEAR_X_M = 0.50
CAMERA_CLEAR_Y_M = 0.00
CAMERA_CLEAR_Z_ABOVE_FLOOR_M = 0.80
CAMERA_CLEAR_MIN_Z_ABOVE_FLOOR_M = 0.40
CAMERA_CLEAR_Z_STEP_M = 0.05
CAMERA_CLEAR_X_CANDIDATES_M = (0.50, 0.45, 0.40, 0.35)
CAMERA_CLEAR_SPEED_SCALE = 0.90
CAMERA_CLEAR_ESCAPE_LIFT_M = 0.08
CAMERA_CLEAR_ESCAPE_ONLY_BELOW_FLOOR_PLUS_M = 0.28
CAMERA_CLEAR_VERTICALIZE_Z_CANDIDATES_ABOVE_FLOOR_M = (0.45, 0.40, 0.35, 0.30)
CAMERA_CLEAR_JOINT_ENDPOINT_TOLERANCE_RAD = 0.020
CAMERA_CLEAR_RETAIN_HIGH_POSE_ABOVE_FLOOR_M = 0.30
CAMERA_CLEAR_ESCAPE_POSITION_TOLERANCE_M = 0.002
CAMERA_IMAGE_CLEAR_Z_ABOVE_FLOOR_M = 0.60
CAMERA_IMAGE_CLEAR_MIN_Z_ABOVE_FLOOR_M = 0.45
CAMERA_IMAGE_CLEAR_X_CANDIDATES_M = (0.35, 0.40, 0.45, 0.50)


def vertical_claw_tip_quaternion(yaw_rad: float = 0.0):
    """tip_r wxyz quaternion for the physical gripper axis vertical/downward.

    For the tracked competition URDF, a top-down physical gripper maps to a
    pure base-Z rotation of tip_r: R_base_tip = Rz(yaw - pi/2).
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


def _highest_reachable_camera_clear_pose(robot, *, floor_m: float) -> Pose:
    """Pick the highest IK-feasible vertical clear pose near the requested x.

    Search height first, then prefer x closest to the operator-requested 0.50 m.
    This is a pure planning step: no robot command is sent here.
    """
    floor_m = float(floor_m)
    seed = robot._read_joint_positions()
    quat = vertical_claw_tip_quaternion(0.0)

    z = floor_m + CAMERA_CLEAR_Z_ABOVE_FLOOR_M
    z_min = floor_m + CAMERA_CLEAR_MIN_Z_ABOVE_FLOOR_M
    last_error = None
    while z >= z_min - 1e-9:
        for x in CAMERA_CLEAR_X_CANDIDATES_M:
            candidate = Pose(
                (float(x), CAMERA_CLEAR_Y_M, float(z)),
                quat,
            )
            try:
                robot._kinematics.solve(candidate, seed)
            except Exception as exc:
                last_error = exc
                continue
            print(
                "CAMERA CLEAR: PLANNED REACHABLE PRESET ->",
                tuple(round(float(v), 4) for v in candidate.position_m),
                flush=True,
            )
            return candidate
        z -= CAMERA_CLEAR_Z_STEP_M

    raise RuntimeError(
        "No reachable high vertical camera-clear pose found in search "
        f"x={CAMERA_CLEAR_X_CANDIDATES_M}, "
        f"z_above_floor={CAMERA_CLEAR_MIN_Z_ABOVE_FLOOR_M:.2f}.."
        f"{CAMERA_CLEAR_Z_ABOVE_FLOOR_M:.2f} m; last IK error: {last_error}"
    )


def _reachable_verticalize_pose(robot, *, floor_m: float) -> Pose:
    """Choose a conservative vertical-claw waypoint before the high search."""
    seed = robot._read_joint_positions()
    quat = vertical_claw_tip_quaternion(0.0)
    last_error = None
    for dz in CAMERA_CLEAR_VERTICALIZE_Z_CANDIDATES_ABOVE_FLOOR_M:
        for x in CAMERA_CLEAR_X_CANDIDATES_M:
            candidate = Pose(
                (float(x), CAMERA_CLEAR_Y_M, float(floor_m) + float(dz)),
                quat,
            )
            try:
                robot._kinematics.solve(candidate, seed)
            except Exception as exc:
                last_error = exc
                continue
            print(
                "CAMERA CLEAR: PLANNED VERTICALIZE ->",
                tuple(round(float(v), 4) for v in candidate.position_m),
                flush=True,
            )
            return candidate
    raise RuntimeError(
        "No reachable verticalization waypoint found; "
        f"last IK error: {last_error}"
    )


def _ik_feasible(robot, pose):
    """Return True only when the current live seed can solve the pose."""
    try:
        robot._kinematics.solve(pose, robot._read_joint_positions())
    except Exception:
        return False
    return True


def _can_retain_high_pose(current, *, floor_m: float) -> bool:
    """True when the current TCP is already safely high for head imaging fallback.

    This fallback is intentionally translation-free: if the model-derived
    vertical-claw orientation is not reachable from the live seed, retaining an
    already-high free-space pose is safer than forcing an unvalidated wrist
    reorientation merely to capture the board image.
    """
    return float(current.position_m[2]) >= (
        float(floor_m) + CAMERA_CLEAR_RETAIN_HIGH_POSE_ABOVE_FLOOR_M
    )




def _reachable_image_clear_pose(robot, *, floor_m: float) -> Pose:
    """Find a high camera-clear pose while preserving the live TCP orientation.

    This phase intentionally does not verticalize the claw. It only gets the
    arm high/out of the head image so perception happens before any orientation
    change. The preferred height matches the ~1.056 m pose from which the
    original verticalization sequence was physically successful onsite.
    """
    current = robot.get_tcp_pose()
    if current is None:
        raise RuntimeError("camera image-clear planning requires current TCP pose")
    seed = robot._read_joint_positions()
    quat = tuple(float(v) for v in current.quaternion_wxyz)
    last_error = None

    xy_candidates = [
        (float(current.position_m[0]), float(current.position_m[1])),
        *[(float(x), CAMERA_CLEAR_Y_M) for x in CAMERA_IMAGE_CLEAR_X_CANDIDATES_M],
    ]
    seen = set()
    unique_xy = []
    for xy in xy_candidates:
        key = (round(xy[0], 6), round(xy[1], 6))
        if key not in seen:
            seen.add(key)
            unique_xy.append(xy)

    z = float(floor_m) + CAMERA_IMAGE_CLEAR_Z_ABOVE_FLOOR_M
    z_min = float(floor_m) + CAMERA_IMAGE_CLEAR_MIN_Z_ABOVE_FLOOR_M
    while z >= z_min - 1e-9:
        for x, y in unique_xy:
            candidate = Pose((x, y, z), quat)
            try:
                robot._kinematics.solve(candidate, seed)
            except Exception as exc:
                last_error = exc
                continue
            print(
                "CAMERA IMAGE CLEAR: PLANNED ->",
                tuple(round(float(v), 4) for v in candidate.position_m),
                flush=True,
            )
            return candidate
        z -= CAMERA_CLEAR_Z_STEP_M

    raise RuntimeError(
        "No reachable orientation-preserving camera image-clear pose found; "
        f"last IK error: {last_error}"
    )


def move_camera_clear_for_image(
    robot, *, floor_m: float, speed_scale: float = CAMERA_CLEAR_SPEED_SCALE
):
    """Clear the head-camera view without changing the live TCP orientation."""
    floor_m = float(floor_m)
    current = robot.get_tcp_pose()
    if current is None:
        raise RuntimeError("camera image-clear move requires current TCP pose")
    if float(current.position_m[2]) < floor_m:
        raise RuntimeError(
            f"Current TCP z={float(current.position_m[2]):.4f} is below "
            f"configured floor {floor_m:.4f}"
        )

    print(
        "CAMERA IMAGE CLEAR: START ->",
        tuple(round(float(v), 4) for v in current.position_m),
        flush=True,
    )

    # Onsite validation: head-board capture succeeded from z≈0.806 m with the
    # arm in its existing pose. Do not invent a harder orientation-preserving
    # relocation when the TCP is already at least 0.30 m above the measured
    # floor; capture first, then run the restored verticalization sequence.
    if float(current.position_m[2]) >= float(floor_m) + 0.30:
        print(
            "CAMERA IMAGE CLEAR: CURRENT POSE ALREADY HIGH; NO ARM MOVE BEFORE IMAGE",
            flush=True,
        )
        return current

    # The competition unit now has an operator-measured camera-clear joint
    # endpoint. Use it directly so this startup phase cannot fail in local IK.
    # The adapter still enforces joint limits, endpoint convergence, and the
    # configured maximum joint delta. If the live state is too far away, stop
    # and ask for manual recovery rather than inventing an unvalidated path.
    try:
        target_q, measured_pose = configured_right_preset(
            robot.config, "right_camera_clear"
        )
    except KeyError:
        # Keep the generic path for test adapters and older non-competition
        # bundles. The competition config contains the measured preset above.
        kin_cfg = robot._kinematics.config
        old_kin = dict(kin_cfg)
        old_joint_tol = float(robot.config["motion"]["joint_reached_tolerance_rad"])
        try:
            robot.config["motion"]["joint_reached_tolerance_rad"] = max(
                old_joint_tol, CAMERA_CLEAR_JOINT_ENDPOINT_TOLERANCE_RAD
            )
            kin_cfg["position_tolerance_m"] = CAMERA_CLEAR_ESCAPE_POSITION_TOLERANCE_M
            kin_cfg["orientation_tolerance_rad"] = 0.04
            kin_cfg["max_seed_delta_rad"] = 2.5
            kin_cfg["max_iterations"] = 180
            target = _reachable_image_clear_pose(robot, floor_m=floor_m)
            move_tcp_segmented(
                robot,
                target,
                speed_scale=float(speed_scale),
                max_translation_step_m=0.08,
                max_orientation_step_rad=0.10,
                min_tcp_z_m=floor_m,
            )
            actual = robot.get_tcp_pose()
            print(
                "CAMERA IMAGE CLEAR: REACHED ->",
                tuple(round(float(v), 4) for v in actual.position_m),
                flush=True,
            )
            return actual
        finally:
            robot.config["motion"]["joint_reached_tolerance_rad"] = old_joint_tol
            kin_cfg.clear()
            kin_cfg.update(old_kin)
    except ValueError as exc:
        if "joint_presets" not in robot.config:
            # Minimal fake/legacy adapters have no measured-preset section.
            # They retain the generic IK behavior for compatibility.
            kin_cfg = robot._kinematics.config
            old_kin = dict(kin_cfg)
            old_joint_tol = float(robot.config["motion"]["joint_reached_tolerance_rad"])
            try:
                robot.config["motion"]["joint_reached_tolerance_rad"] = max(
                    old_joint_tol, CAMERA_CLEAR_JOINT_ENDPOINT_TOLERANCE_RAD
                )
                kin_cfg["position_tolerance_m"] = CAMERA_CLEAR_ESCAPE_POSITION_TOLERANCE_M
                kin_cfg["orientation_tolerance_rad"] = 0.04
                kin_cfg["max_seed_delta_rad"] = 2.5
                kin_cfg["max_iterations"] = 180
                target = _reachable_image_clear_pose(robot, floor_m=floor_m)
                move_tcp_segmented(
                    robot, target, speed_scale=float(speed_scale),
                    max_translation_step_m=0.08,
                    max_orientation_step_rad=0.10,
                    min_tcp_z_m=floor_m,
                )
                return robot.get_tcp_pose()
            finally:
                robot.config["motion"]["joint_reached_tolerance_rad"] = old_joint_tol
                kin_cfg.clear()
                kin_cfg.update(old_kin)
        raise RuntimeError(f"right_camera_clear preset is invalid: {exc}") from exc
    current_q = robot._read_joint_positions()
    delta = preset_max_delta(current_q, target_q)
    max_delta = float(robot.config["motion"]["max_total_delta_rad"])
    if delta > max_delta:
        raise RuntimeError(
            "RIGHT_CAMERA_CLEAR is too far from the live joint state for a "
            f"validated joint move ({delta:.3f} rad > {max_delta:.3f} rad); "
            "manually place the arm at RIGHT_READY or RIGHT_CAMERA_CLEAR and rerun"
        )
    if float(measured_pose.position_m[2]) < floor_m:
        raise RuntimeError("configured RIGHT_CAMERA_CLEAR pose is below the TCP floor")
    print(
        "CAMERA IMAGE CLEAR: MEASURED JOINT PRESET ->",
        tuple(round(float(v), 6) for v in target_q),
        flush=True,
    )
    robot.move_joints(target_q, speed_scale=float(speed_scale))
    actual = robot.get_tcp_pose()
    print(
        "CAMERA IMAGE CLEAR: REACHED ->",
        tuple(round(float(v), 4) for v in actual.position_m),
        flush=True,
    )
    return actual


def move_verticalize_after_image(
    robot, *, floor_m: float, speed_scale: float = CAMERA_CLEAR_SPEED_SCALE
):
    """Run the original physically successful verticalization sequence.

    This intentionally mirrors the pre-fallback onsite sequence:
      reachable verticalize waypoint -> verticalize -> highest reachable
      vertical preset.
    It is called only after the head image has been captured.
    """
    floor_m = float(floor_m)
    current = robot.get_tcp_pose()
    if current is None:
        raise RuntimeError("post-image verticalization requires current TCP pose")
    print(
        "POST-IMAGE VERTICALIZE: START ->",
        tuple(round(float(v), 4) for v in current.position_m),
        flush=True,
    )

    kin_cfg = robot._kinematics.config
    old_kin = dict(kin_cfg)
    old_joint_tol = float(robot.config["motion"]["joint_reached_tolerance_rad"])
    try:
        robot.config["motion"]["joint_reached_tolerance_rad"] = max(
            old_joint_tol, CAMERA_CLEAR_JOINT_ENDPOINT_TOLERANCE_RAD
        )
        kin_cfg["position_tolerance_m"] = 0.010
        kin_cfg["orientation_tolerance_rad"] = 0.12
        kin_cfg["max_seed_delta_rad"] = 2.5
        kin_cfg["max_iterations"] = 180

        verticalize = _reachable_verticalize_pose(robot, floor_m=floor_m)
        print(
            "POST-IMAGE VERTICALIZE: WAYPOINT ->",
            tuple(round(float(v), 4) for v in verticalize.position_m),
            flush=True,
        )

        # The endpoint was already proven IK-feasible above. Do not Cartesian-
        # interpolate the large wrist reorientation: intermediate SLERP poses
        # can be unreachable even though the endpoint is reachable. Execute the
        # pre-solved endpoint as one server-smoothed joint trajectory.
        verticalize_q = robot._kinematics.solve(
            verticalize, robot._read_joint_positions()
        )
        print("POST-IMAGE VERTICALIZE: DIRECT PRE-SOLVED JOINT MOVE", flush=True)
        robot.move_joints(verticalize_q, speed_scale=float(speed_scale))

        target = _highest_reachable_camera_clear_pose(robot, floor_m=floor_m)
        print(
            "POST-IMAGE VERTICALIZE: PRESET ->",
            tuple(round(float(v), 4) for v in target.position_m),
            flush=True,
        )
        move_tcp_segmented(
            robot,
            target,
            speed_scale=float(speed_scale),
            max_translation_step_m=0.15,
            max_orientation_step_rad=0.45,
            min_tcp_z_m=floor_m,
        )

        actual = robot.get_tcp_pose()
        print(
            "POST-IMAGE VERTICALIZE: REACHED ->",
            tuple(round(float(v), 4) for v in actual.position_m),
            flush=True,
        )
        return actual
    finally:
        robot.config["motion"]["joint_reached_tolerance_rad"] = old_joint_tol
        kin_cfg.clear()
        kin_cfg.update(old_kin)


def move_camera_clear(robot, *, floor_m: float, speed_scale: float = CAMERA_CLEAR_SPEED_SCALE):
    """Move the right claw out of the head-board view using preplanned stages.

    Every coarse stage is IK-checked before it is commanded. A short
    orientation-preserving escape lift is attempted only when the TCP is truly
    low; otherwise we go directly to a conservative vertical-claw waypoint.
    """
    floor_m = float(floor_m)
    speed_scale = float(speed_scale)
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

    if "joint_presets" in robot.config:
        try:
            target_q, measured_pose = configured_right_preset(
                robot.config, "right_camera_clear"
            )
        except (KeyError, ValueError) as exc:
            raise RuntimeError(f"right_camera_clear preset is invalid: {exc}") from exc
        if float(measured_pose.position_m[2]) < floor_m:
            raise RuntimeError("configured RIGHT_CAMERA_CLEAR pose is below the TCP floor")
        delta = preset_max_delta(robot._read_joint_positions(), target_q)
        max_delta = float(robot.config["motion"]["max_total_delta_rad"])
        if delta > max_delta:
            raise RuntimeError(
                "RIGHT_CAMERA_CLEAR is too far from the live joint state for a "
                f"validated joint move ({delta:.3f} rad > {max_delta:.3f} rad); "
                "manually place the arm at RIGHT_READY or RIGHT_CAMERA_CLEAR and rerun"
            )
        print("CAMERA CLEAR: MEASURED JOINT PRESET ->", list(target_q), flush=True)
        robot.move_joints(target_q, speed_scale=speed_scale)
        actual = robot.get_tcp_pose()
        print(
            "CAMERA CLEAR: REACHED ->",
            tuple(round(float(v), 4) for v in actual.position_m),
            flush=True,
        )
        return actual

    kin_cfg = robot._kinematics.config
    old_kin = dict(kin_cfg)
    old_joint_tol = float(robot.config["motion"]["joint_reached_tolerance_rad"])
    try:
        robot.config["motion"]["joint_reached_tolerance_rad"] = max(
            old_joint_tol,
            CAMERA_CLEAR_JOINT_ENDPOINT_TOLERANCE_RAD,
        )
        kin_cfg["position_tolerance_m"] = 0.010
        kin_cfg["orientation_tolerance_rad"] = 0.12
        kin_cfg["max_seed_delta_rad"] = 2.5
        kin_cfg["max_iterations"] = 180

        # Only try to preserve the current arbitrary wrist orientation when the
        # TCP is genuinely low. If that straight-up target itself is not
        # solvable, skip it rather than asserting software E-stop on an IK-only
        # failure.
        low_threshold = floor_m + CAMERA_CLEAR_ESCAPE_ONLY_BELOW_FLOOR_PLUS_M
        if float(current.position_m[2]) < low_threshold:
            escape = Pose(
                (
                    float(current.position_m[0]),
                    float(current.position_m[1]),
                    float(current.position_m[2]) + CAMERA_CLEAR_ESCAPE_LIFT_M,
                ),
                tuple(float(v) for v in current.quaternion_wxyz),
            )

            old_position_tol = float(kin_cfg["position_tolerance_m"])
            kin_cfg["position_tolerance_m"] = min(
                old_position_tol,
                CAMERA_CLEAR_ESCAPE_POSITION_TOLERANCE_M,
            )
            try:
                feasible = _ik_feasible(robot, escape)
            finally:
                kin_cfg["position_tolerance_m"] = old_position_tol

            if feasible:
                print(
                    "CAMERA CLEAR: ESCAPE LIFT ->",
                    tuple(round(float(v), 4) for v in escape.position_m),
                    flush=True,
                )
                old_position_tol = float(kin_cfg["position_tolerance_m"])
                kin_cfg["position_tolerance_m"] = min(
                    old_position_tol,
                    CAMERA_CLEAR_ESCAPE_POSITION_TOLERANCE_M,
                )
                try:
                    move_tcp_segmented(
                        robot,
                        escape,
                        speed_scale=speed_scale,
                        max_translation_step_m=0.04,
                        max_orientation_step_rad=0.80,
                        min_tcp_z_m=floor_m,
                    )
                finally:
                    kin_cfg["position_tolerance_m"] = old_position_tol
                current = robot.get_tcp_pose()
                if current is None:
                    raise RuntimeError(
                        "camera-clear escape completed but current TCP pose is unavailable"
                    )
                print(
                    "CAMERA CLEAR: AFTER ESCAPE ->",
                    tuple(round(float(v), 4) for v in current.position_m),
                    flush=True,
                )
            else:
                print(
                    "CAMERA CLEAR: ESCAPE LIFT skipped (IK infeasible)",
                    flush=True,
                )

        try:
            verticalize = _reachable_verticalize_pose(robot, floor_m=floor_m)
        except RuntimeError as exc:
            current = robot.get_tcp_pose()
            if current is None:
                raise RuntimeError(
                    "verticalization failed and current TCP pose is unavailable"
                ) from exc
            if _can_retain_high_pose(current, floor_m=floor_m):
                print(
                    "CAMERA CLEAR: VERTICALIZE unavailable from current seed; "
                    "retaining existing high free-space pose ->",
                    tuple(round(float(v), 4) for v in current.position_m),
                    f"({exc})",
                    flush=True,
                )
                return current
            raise
        print(
            "CAMERA CLEAR: VERTICALIZE ->",
            tuple(round(float(v), 4) for v in verticalize.position_m),
            flush=True,
        )
        move_tcp_segmented(
            robot,
            verticalize,
            speed_scale=speed_scale,
            max_translation_step_m=0.12,
            max_orientation_step_rad=0.45,
            min_tcp_z_m=floor_m,
        )

        target = _highest_reachable_camera_clear_pose(robot, floor_m=floor_m)
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
            max_translation_step_m=0.15,
            max_orientation_step_rad=0.45,
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
