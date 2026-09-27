"""Robot boundary shared by Vega and Sharpa implementations.

Keep SDK imports out of this module. The rest of SteadyHand should depend on
this contract, while each robot adapter translates the contract into the
installed vendor API.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..models import Pose


class HardwareUnavailableError(RuntimeError):
    """Raised when a hardware adapter is intentionally not ready to connect."""


@dataclass(frozen=True)
class AdapterCapabilities:
    joint_motion: bool = False
    tcp_motion: bool = False
    cameras: bool = False
    force_torque: bool = False
    tactile: bool = False


@dataclass(frozen=True)
class RobotObservation:
    """Minimal common observation; robot-specific data belongs in extras."""

    timestamp_s: float | None = None
    joint_positions: tuple[float, ...] = ()
    cameras: Mapping[str, Any] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)


class RobotAdapter(ABC):
    """Capability-level interface between shared task code and vendor SDKs."""

    robot_id: str

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    @property
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities:
        raise NotImplementedError

    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stop commanded motion as safely and quickly as the SDK permits."""
        raise NotImplementedError

    @abstractmethod
    def observe(self) -> RobotObservation:
        raise NotImplementedError

    @abstractmethod
    def move_joints(
        self, joint_positions: Sequence[float], *, speed_scale: float = 0.2
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def move_tcp(self, pose: Pose, *, speed_scale: float = 0.2) -> None:
        raise NotImplementedError

    @abstractmethod
    def open_gripper(self, part_name: str | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def close_gripper(self, part_name: str | None = None) -> None:
        raise NotImplementedError

    def grip(self, part_name: str | None = None, *, current_a: float | None = None) -> None:
        """Object-aware grasp; default falls back to close_gripper()."""
        self.close_gripper(part_name)

    def get_tcp_pose(self) -> Pose | None:
        return None

    def read_wrench(self):
        return None

    def verify_grasp(self, part_name: str) -> bool | None:
        return None

    def verify_place(self, part_name: str) -> bool | None:
        return None
