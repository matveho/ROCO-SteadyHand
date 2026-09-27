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
    """Convert object/target pose into desired TCP pose.

    Preferred representation is T_part_tcp, which rotates naturally with the
    part/board. Until it is calibrated, legacy_ee_offset_m +
    legacy_ee_orientation_wxyz reproduces the submitted simulation policy's
    world-axis offset behavior as a bring-up fallback.
    """
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
    event=None,
):
    """Move through short Cartesian targets, reseeding IK from live state."""
    current = robot.get_tcp_pose()
    if current is None:
        raise ExecutionError("Robot adapter cannot provide current TCP pose")

    dist, angle = pose_distance(current, target)
    n = max(
        1,
        int(math.ceil(dist / float(max_translation_step_m))),
        int(math.ceil(angle / float(max_orientation_step_rad))),
    )
    for index in range(1, n + 1):
        waypoint = interpolate_pose(current, target, index / n)
        robot.move_tcp(waypoint, speed_scale=speed_scale)
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
        if after_waypoint is not None:
            after_waypoint()


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

    @property
    def enabled(self):
        return self.threshold is not None

    def capture(self):
        if not self.enabled:
            return
        value = self.robot.read_wrench()
        if value is None:
            raise ExecutionError("Force guard configured but wrench is unavailable")
        self.baseline = tuple(float(x) for x in value)

    def check(self):
        if not self.enabled:
            return
        if self.baseline is None:
            raise RuntimeError("Force guard baseline has not been captured")
        value = self.robot.read_wrench()
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


def execute_part(
    robot,
    goal,
    skill,
    *,
    safety=None,
    speed_scale=1.0,
    event=None,
):
    """Execute one physical part using the submitted policy's phase structure."""
    safety = dict(safety or {})
    if goal.pick_pose is None or goal.place_pose is None:
        raise ValueError(f"{goal.name}: pick_pose and place_pose are required")

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
    grasp_verified = robot.verify_grasp(goal.name)
    emit("grasp", "completed", {"verified": grasp_verified})
    if grasp_verified is False:
        raise ExecutionError(f"{goal.name}: grasp verification failed")

    emit("lift", "started")
    move_tcp_segmented(robot, hover_pick, **move_kwargs)
    emit("lift", "completed")

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

    placement_verified = robot.verify_place(goal.name)
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


def _execute_insertion(robot, goal, skill, *, safety, speed_scale, event):
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
    if not force_guard.enabled and not skill.get(
        "allow_snap_without_force_guard", False
    ):
        raise ExecutionError(
            f"{goal.name}: snap insertion requires a verified force_delta_limit "
            "or explicit allow_snap_without_force_guard=true"
        )

    place_tcp = object_pose_to_tcp(goal.place_pose, skill)
    preinsert = offset_z(place_tcp, float(skill["preinsert_m"]))

    move_kwargs = dict(
        speed_scale=float(speed_scale),
        max_translation_step_m=float(skill["max_cartesian_step_m"]),
        max_orientation_step_rad=float(skill["max_orientation_step_rad"]),
        event=event,
    )

    event("approach_place", "started", {"mode": "preinsert"})
    move_tcp_segmented(robot, preinsert, **move_kwargs)
    event("approach_place", "completed", {"mode": "preinsert"})

    force_guard.capture()

    for dx, dy in offsets:
        candidate_pre = _shift_xy(preinsert, dx, dy)
        candidate_final = _shift_xy(place_tcp, dx, dy)
        event(
            "insertion_search",
            "candidate_started",
            {"dx_m": dx, "dy_m": dy},
        )

        move_tcp_segmented(robot, candidate_pre, **move_kwargs)
        try:
            move_tcp_segmented(
                robot,
                candidate_final,
                after_waypoint=force_guard.check,
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
                return (float(dx), float(dy))
            event(
                "insertion_search",
                "candidate_not_reached",
                {"dx_m": dx, "dy_m": dy},
            )
        except ContactLimitExceeded as exc:
            event(
                "insertion_search",
                "contact_limit",
                {"dx_m": dx, "dy_m": dy, "error": str(exc)},
            )

        # Retract before trying another lateral location.
        move_tcp_segmented(robot, candidate_pre, **move_kwargs)

    raise ExecutionError(
        f"{goal.name}: no insertion-search candidate reached the target"
    )


def _shift_xy(pose, dx, dy):
    x, y, z = pose.position_m
    return Pose(
        (x + float(dx), y + float(dy), z),
        pose.quaternion_wxyz,
    )
