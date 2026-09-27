"""Shared assembly vocabulary above the robot-specific adapter layer."""

from .models import PartGoal, Pose


PHASES = (
    "open_gripper",
    "approach_pick",
    "descend_pick",
    "grasp",
    "lift",
    "verify_grasp",
    "transfer",
    "approach_place",
    "place",
    "release",
    "retreat",
    "verify_place",
)


def goal_from_spec(name, spec):
    """Convert task JSON into a robot-independent goal.

    Unknown onsite poses remain None. This function must not fabricate poses
    from simulation constants or silently assume a coordinate frame.
    """
    pick = spec.get("pick_pose")
    place = spec.get("place_pose")
    return PartGoal(
        name=name,
        release_mode=spec["simulation_release_mode"],
        pick_pose=None if pick is None else Pose.from_mapping(pick),
        place_pose=None if place is None else Pose.from_mapping(place),
        verification_method=spec.get("verification_method"),
    )
