"""Robot-independent physical pick/place executor.

This mirrors the submitted policy's high-level phases while replacing simulator
conveniences with:
- runtime object/target poses;
- robot-specific object->TCP grasp configuration;
- local IK through RobotAdapter.move_tcp();
- segmented Cartesian waypoint motion;
- optional force-delta-gated insertion search.

It intentionally makes no claim that a completed motion means the part was
successfully assembled.
"""

from dataclasses import dataclass
import math
import time

from .geometry import (
    compose,
    interpolate_pose,
    matrix_to_pose,
    offset_z,
    pose_distance,
    pose_to_matrix,
)
from .models import Pose
from .search import centered_grid
from .skill_config import validate_skill


class ExecutionError(RuntimeError):
    pass


class ContactLimitExceeded(ExecutionError):
    pass


@dataclass(frozen=True)
class PartExecutionResult:
    part: str
    completed: bool
    grasp_verified: bool | None
    placement_verified: bool | None
    insertion_candidate: tuple[float, float] | None = None
    note: str | None = None


def object_pose_to_tcp(object_pose: Pose, skill: dict) -> Pose:
    """Convert a runtime pose into a desired TCP pose.

    For rapid onsite bring-up, runtime_pose_type='tcp' means the runtime file
    already contains measured TCP poses in the configured base frame. Normal
    competition perception should use object poses plus calibrated T_part_tcp.
    """
    if skill.get("runtime_pose_type") == "tcp":
        return object_pose

    t_part_tcp = skill.get("T_part_tcp")
    if t_part_tcp is not None:
        return matrix_to_pose(compose(pose_to_matrix(object_pose), t_part_tcp))

    offset = skill.get("legacy_ee_offset_m")
    orientation = skill.get("legacy_ee_orientation_wxyz")
    if offset is None or orientation is None:
        raise ValueError(
            "Skill needs calibrated T_part_tcp or legacy EE offset/orientation"
        )
    return Pose(
        position_m=tuple(
            float(p) + float(d) for p, d in zip(object_pose.position_m, offset)
        ),
        quaternion_wxyz=tuple(float(x) for x in orientation),
    )


def move_tcp_segmented(
    robot,
    target: Pose,
    *,
    speed_scale: float,
    max_translation_step_m: float,
    max_orientation_step_rad: float,
    after_waypoint=None,
    before_waypoint=None,
    event=None,
    min_tcp_z_m=None,
):
    """Move through short Cartesian targets, reseeding IK from live state."""
    for label, value in (("max_translation_step_m", max_translation_step_m),
                         ("max_orientation_step_rad", max_orientation_step_rad)):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{label} must be finite and positive")
    if not math.isfinite(float(speed_scale)) or not 0 < float(speed_scale) <= 1:
        raise ValueError("speed_scale must be in (0, 1]")
    if min_tcp_z_m is not None:
        min_tcp_z_m = float(min_tcp_z_m)
        if not math.isfinite(min_tcp_z_m):
            raise ValueError("min_tcp_z_m must be finite when configured")
    current = robot.get_tcp_pose()
    if current is None:
        raise ExecutionError("Robot adapter cannot provide current TCP pose")
    if min_tcp_z_m is not None and float(current.position_m[2]) < min_tcp_z_m:
        raise ExecutionError(
            f"Current TCP z={float(current.position_m[2]):.6f} m is below "
            f"configured floor {min_tcp_z_m:.6f} m; recover upward manually "
            "before task execution"
        )
    if min_tcp_z_m is not None and float(target.position_m[2]) < min_tcp_z_m:
        raise ExecutionError(
            f"Refusing TCP target z={float(target.position_m[2]):.6f} m below "
            f"configured floor {min_tcp_z_m:.6f} m"
        )

    dist, angle = pose_distance(current, target)
    n = max(
        1,
        int(math.ceil(dist / float(max_translation_step_m))),
        int(math.ceil(angle / float(max_orientation_step_rad))),
    )
    for index in range(1, n + 1):
        if before_waypoint is not None:
            before_waypoint()
        waypoint = interpolate_pose(current, target, index / n)
        if min_tcp_z_m is not None and float(waypoint.position_m[2]) < min_tcp_z_m:
            raise ExecutionError(
                f"Refusing TCP waypoint z={float(waypoint.position_m[2]):.6f} m below "
                f"configured floor {min_tcp_z_m:.6f} m"
            )
        robot.move_tcp(waypoint, speed_scale=speed_scale)
        # Check contact before logging: slow/failing storage must not delay it.
        if after_waypoint is not None:
            after_waypoint()
        if event:
            event(
                "tcp_waypoint",
                "completed",
                {
                    "index": index,
                    "count": n,
                    "target_position_m": waypoint.position_m,
                },
            )


def tcp_reached(robot, target, *, position_tolerance_m, orientation_tolerance_rad):
    actual = robot.get_tcp_pose()
    if actual is None:
        return False
    dp, da = pose_distance(actual, target)
    return (
        dp <= float(position_tolerance_m)
        and da <= float(orientation_tolerance_rad)
    )


class ForceDeltaGuard:
    """Relative force threshold, disabled until onsite units/frame are verified."""

    def __init__(self, robot, threshold, axes=(0, 1, 2)):
        self.robot = robot
        self.threshold = None if threshold is None else float(threshold)
        self.axes = tuple(int(x) for x in axes)
        self.baseline = None
        if self.enabled and (not math.isfinite(self.threshold) or self.threshold <= 0):
            raise ValueError("force_delta_limit must be finite and positive")
        if not self.axes or len(set(self.axes)) != len(self.axes) or any(i not in (0, 1, 2) for i in self.axes):
            raise ValueError("force_axes must select distinct force channels 0, 1, 2")

    def _read(self):
        value = self.robot.read_wrench()
        if value is None or len(value) != 6 or any(not math.isfinite(float(x)) for x in value):
            raise ExecutionError("Force guard needs six finite wrench channels")
        return tuple(float(x) for x in value)

    @property
    def enabled(self):
        return self.threshold is not None

    def capture(self):
        if not self.enabled:
            return
        self.baseline = self._read()

    def check(self):
        if not self.enabled:
            return
        if self.baseline is None:
            raise RuntimeError("Force guard baseline has not been captured")
        value = self._read()
        delta = math.sqrt(
            sum(
                (float(value[i]) - self.baseline[i]) ** 2
                for i in self.axes
            )
        )
        if delta > self.threshold:
            raise ContactLimitExceeded(
                f"Force delta {delta:.4f} exceeded threshold {self.threshold:.4f}"
            )


def validate_execution(goal, skill, safety, speed_scale=1.0):
    """All static execution checks, callable before Robot() or CAN homing."""
    validate_skill(skill)
    if goal.pick_pose is None or goal.place_pose is None:
        raise ValueError(f"{goal.name}: pick_pose and place_pose are required")
    if goal.release_mode not in ("open", "snap"):
        raise ValueError(f"{goal.name}: unknown release mode {goal.release_mode!r}")
    if not math.isfinite(float(speed_scale)) or not 0 < float(speed_scale) <= 1:
        raise ValueError("speed_scale must be in (0, 1]")
    min_tcp_z_m = safety.get("min_tcp_z_m")
    if min_tcp_z_m is not None and not math.isfinite(float(min_tcp_z_m)):
        raise ValueError("safety.min_tcp_z_m must be finite when configured")
    if goal.release_mode == "snap":
        if safety.get("force_delta_limit") is None:
            raise ExecutionError(f"{goal.name}: insertion requires verified force_delta_limit")
        ForceDeltaGuard(None, safety["force_delta_limit"], safety.get("force_axes", (0, 1, 2)))
        if safety.get("wrench_units") != "N,Nm" or not safety.get("wrench_frame"):
            raise ExecutionError("Insertion requires measured safety.wrench_units='N,Nm' and wrench_frame")
        step = skill.get("max_contact_step_m")
        if step is None or not math.isfinite(float(step)) or float(step) <= 0:
            raise ExecutionError("Insertion requires a validated max_contact_step_m")
    pick_tcp = object_pose_to_tcp(goal.pick_pose, skill)
    place_tcp = object_pose_to_tcp(goal.place_pose, skill)
    min_tcp_z_m = safety.get("min_tcp_z_m")
    if min_tcp_z_m is not None:
        floor = float(min_tcp_z_m)
        for label, pose in (("pick", pick_tcp), ("place", place_tcp)):
            if float(pose.position_m[2]) < floor:
                raise ExecutionError(
                    f"{goal.name}: {label} TCP z={float(pose.position_m[2]):.6f} m "
                    f"is below configured floor {floor:.6f} m"
                )


def execute_part(robot, goal, skill, *, safety=None, speed_scale=1.0, event=None,
                 verify=None):
    """Stop on every failure, including Ctrl-C/SystemExit and verifier errors."""
    try:
        return _execute_part(robot, goal, skill, safety=safety,
                             speed_scale=speed_scale, event=event, verify=verify)
    except BaseException:
        try:
            robot.stop()
        except BaseException:
            # Preserve the original failure; runners report stop errors too.
            pass
        raise


def _verification(robot, stage, name, verify):
    method = robot.verify_grasp if stage in ("grasp", "lift") else robot.verify_place
    # No existing adapter has an insertion-completion sensor. Never infer it
    # from joint/TCP arrival or from verify_place after release.
    result = None if stage == "insertion" else method(name)
    # A driver's initial current/position result cannot prove the part stayed
    # in the jaws during lift. Require the live verifier after lifting even if
    # the cached gripper result was true.
    if stage == "lift" and result is not False:
        result = None
    if result is None and verify is not None:
        result = verify(stage, name)
    if result is not True:
        raise ExecutionError(f"{name}: {stage} verification failed or is unavailable")
    return True


def _execute_part(
    robot,
    goal,
    skill,
    *,
    safety=None,
    speed_scale=1.0,
    event=None,
    verify=None,
):
    """Execute one physical part using the submitted policy's phase structure."""
    safety = dict(safety or {})
    validate_execution(goal, skill, safety, speed_scale)

    def emit(phase, state, details=None):
        if event:
            event(phase, state, details or {})

    pick_tcp = object_pose_to_tcp(goal.pick_pose, skill)
    place_tcp = object_pose_to_tcp(goal.place_pose, skill)
    hover_pick = offset_z(pick_tcp, float(skill["hover_pick_m"]))
    hover_place = offset_z(place_tcp, float(skill["hover_place_m"]))
    retract = offset_z(place_tcp, float(skill["retract_m"]))

    move_kwargs = dict(
        speed_scale=float(speed_scale),
        max_translation_step_m=float(skill["max_cartesian_step_m"]),
        max_orientation_step_rad=float(skill["max_orientation_step_rad"]),
        event=emit,
        min_tcp_z_m=safety.get("min_tcp_z_m"),
    )

    emit("open_gripper", "started")
    robot.open_gripper(goal.name)
    emit("open_gripper", "completed")

    emit("approach_pick", "started")
    move_tcp_segmented(robot, hover_pick, **move_kwargs)
    emit("approach_pick", "completed")

    emit("descend_pick", "started")
    move_tcp_segmented(robot, pick_tcp, **move_kwargs)
    emit("descend_pick", "completed")

    emit("grasp", "started")
    robot.grip(goal.name, current_a=skill.get("grip_current_a"))
    time.sleep(float(skill.get("grasp_settle_s", 0.0)))
    grasp_verified = _verification(robot, "grasp", goal.name, verify)
    emit("grasp", "completed", {"verified": grasp_verified})
    if grasp_verified is False:
        raise ExecutionError(f"{goal.name}: grasp verification failed")

    emit("lift", "started")
    move_tcp_segmented(robot, hover_pick, **move_kwargs)
    emit("lift", "completed")
    _verification(robot, "lift", goal.name, verify)

    emit("transfer", "started")
    move_tcp_segmented(robot, hover_place, **move_kwargs)
    emit("transfer", "completed")

    insertion_candidate = None
    if goal.release_mode == "snap":
        insertion_candidate = _execute_insertion(
            robot,
            goal,
            skill,
            safety=safety,
            speed_scale=speed_scale,
            event=emit,
            verify=verify,
        )
        # _execute_insertion leaves the TCP at the successful candidate.
        successful_place = _shift_xy(
            place_tcp, insertion_candidate[0], insertion_candidate[1]
        )
        retract = offset_z(successful_place, float(skill["retract_m"]))
    else:
        emit("place", "started")
        move_tcp_segmented(robot, place_tcp, **move_kwargs)
        emit("place", "completed")

    emit("release", "started")
    robot.open_gripper(goal.name)
    time.sleep(float(skill.get("release_settle_s", 0.0)))
    emit("release", "completed")

    emit("retreat", "started")
    move_tcp_segmented(robot, retract, **move_kwargs)
    emit("retreat", "completed")

    placement_verified = _verification(robot, "placement", goal.name, verify)
    emit("verify_place", "completed", {"verified": placement_verified})

    return PartExecutionResult(
        part=goal.name,
        completed=True,
        grasp_verified=grasp_verified,
        placement_verified=placement_verified,
        insertion_candidate=insertion_candidate,
        note=(
            "Motion sequence completed; placement remains unverified"
            if placement_verified is None
            else None
        ),
    )


def _execute_insertion(robot, goal, skill, *, safety, speed_scale, event, verify=None):
    search = skill.get("search")
    if not search:
        offsets = [(0.0, 0.0)]
    elif search.get("type") == "grid":
        offsets = centered_grid(
            int(search["n"]),
            tuple(float(x) for x in search["extent_xy_m"]),
        )
    else:
        raise ValueError(f"{goal.name}: unsupported search config {search!r}")

    force_guard = ForceDeltaGuard(
        robot,
        safety.get("force_delta_limit"),
        safety.get("force_axes", (0, 1, 2)),
    )
    if not force_guard.enabled:
        raise ExecutionError(
            f"{goal.name}: snap insertion requires a verified force_delta_limit "
            "with measured units/frame"
        )

    place_tcp = object_pose_to_tcp(goal.place_pose, skill)
    preinsert = offset_z(place_tcp, float(skill["preinsert_m"]))

    move_kwargs = dict(
        speed_scale=float(speed_scale),
        max_translation_step_m=min(float(skill["max_cartesian_step_m"]), float(skill["max_contact_step_m"])),
        max_orientation_step_rad=float(skill["max_orientation_step_rad"]),
        event=event,
        min_tcp_z_m=safety.get("min_tcp_z_m"),
    )

    event("approach_place", "started", {"mode": "preinsert"})
    move_tcp_segmented(robot, preinsert, **move_kwargs)
    event("approach_place", "completed", {"mode": "preinsert"})

    force_guard.capture()

    # The simulator descended at nominal before starting the grid, including
    # for even-sized grids which have no zero sample.
    offsets = [(0.0, 0.0)] + [xy for xy in offsets if xy != (0.0, 0.0)]
    for dx, dy in offsets:
        candidate_pre = _shift_xy(preinsert, dx, dy)
        candidate_final = _shift_xy(place_tcp, dx, dy)
        event(
            "insertion_search",
            "candidate_started",
            {"dx_m": dx, "dy_m": dy},
        )

        move_tcp_segmented(robot, candidate_pre, before_waypoint=force_guard.check,
                           after_waypoint=force_guard.check, **move_kwargs)
        try:
            move_tcp_segmented(
                robot,
                candidate_final,
                after_waypoint=force_guard.check,
                before_waypoint=force_guard.check,
                **move_kwargs,
            )
            reached = tcp_reached(
                robot,
                candidate_final,
                position_tolerance_m=skill["insertion_reached_tolerance_m"],
                orientation_tolerance_rad=skill[
                    "insertion_reached_tolerance_rad"
                ],
            )
            if reached:
                event(
                    "insertion_search",
                    "candidate_reached",
                    {"dx_m": dx, "dy_m": dy},
                )
                # Reaching a pose proves neither engagement nor retention.
                if verify is None:
                    raise ExecutionError("Insertion reached TCP target but has no physical verifier; refusing release")
                verification = verify("insertion", goal.name)
                if verification is True:
                    force_guard.check()
                    return (float(dx), float(dy))
                if verification is None:
                    raise ExecutionError(
                        "Insertion reached TCP target but physical verifier is "
                        "unavailable; refusing release"
                    )
            event(
                "insertion_search",
                "candidate_not_reached",
                {"dx_m": dx, "dy_m": dy},
            )
        except ContactLimitExceeded as exc:
            # An overload is a fault, not permission to sweep another cell.
            # Stop before logging; never issue an automatic blind retract.
            robot.stop()
            event(
                "insertion_search",
                "contact_limit",
                {"dx_m": dx, "dy_m": dy, "error": str(exc)},
            )
            raise

        # Retract before trying another lateral location.
        move_tcp_segmented(robot, candidate_pre, before_waypoint=force_guard.check,
                           after_waypoint=force_guard.check, **move_kwargs)

    raise ExecutionError(
        f"{goal.name}: no insertion-search candidate reached the target"
    )


def _shift_xy(pose, dx, dy):
    x, y, z = pose.position_m
    return Pose(
        (x + float(dx), y + float(dy), z),
        pose.quaternion_wxyz,
    )
