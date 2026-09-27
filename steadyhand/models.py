"""Small dependency-free data models shared across robot implementations."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Mapping


@dataclass(frozen=True)
class Pose:
    """Rigid pose expressed in metres and a wxyz unit quaternion."""

    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self):
        # Poses also originate in FK and direct API calls, outside JSON parsing.
        # Never let NaN or truncated zip() operations become motion targets.
        for name, size in (("position_m", 3), ("quaternion_wxyz", 4)):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or len(values) != size:
                raise ValueError(f"Pose.{name}: expected {size} numbers")
            if any(isinstance(v, bool) or not isinstance(v, Real)
                   or not math.isfinite(v) for v in values):
                raise ValueError(f"Pose.{name}: values must be finite numbers")
            object.__setattr__(self, name, tuple(float(v) for v in values))
        if abs(sum(v*v for v in self.quaternion_wxyz) - 1.0) > 0.001:
            raise ValueError("Pose.quaternion_wxyz: quaternion must have unit length")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Pose":
        return cls(
            position_m=tuple(value["position_m"]),
            quaternion_wxyz=tuple(value["quaternion_wxyz"]),
        )


@dataclass(frozen=True)
class PartGoal:
    """Task-level goal. It intentionally contains no robot-specific joints."""

    name: str
    release_mode: str
    pick_pose: Pose | None
    place_pose: Pose | None
    verification_method: str | None
