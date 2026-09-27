"""Small dependency-free data models shared across robot implementations."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Pose:
    """Rigid pose expressed in metres and a wxyz unit quaternion."""

    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Pose":
        return cls(
            position_m=tuple(float(x) for x in value["position_m"]),
            quaternion_wxyz=tuple(float(x) for x in value["quaternion_wxyz"]),
        )


@dataclass(frozen=True)
class PartGoal:
    """Task-level goal. It intentionally contains no robot-specific joints."""

    name: str
    release_mode: str
    pick_pose: Pose | None
    place_pose: Pose | None
    verification_method: str | None
