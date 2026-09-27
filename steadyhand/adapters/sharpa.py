"""Sharpa North hardware adapter boundary.

The public Wave-hand resources and RoCo demonstration dataset are useful
references, but they do not establish the complete North arm/body command
interface. Keep this adapter disabled until the onsite interface is verified.
"""

from .base import (
    AdapterCapabilities,
    HardwareUnavailableError,
    RobotAdapter,
    RobotObservation,
)


class SharpaAdapter(RobotAdapter):
    robot_id = "sharpa"

    @property
    def capabilities(self) -> AdapterCapabilities:
        # North arm/body capabilities remain unknown until the onsite
        # interface is inspected.
        return AdapterCapabilities(
            joint_motion=False,
            tcp_motion=False,
            cameras=False,
            force_torque=False,
            tactile=True,
        )

    def _unavailable(self):
        raise HardwareUnavailableError(
            "Sharpa North adapter is not enabled yet. Obtain and validate the "
            "onsite North control interface before sending hardware commands."
        )

    def connect(self) -> None:
        self._unavailable()

    def close(self) -> None:
        return None

    def stop(self) -> None:
        self._unavailable()

    def observe(self) -> RobotObservation:
        self._unavailable()

    def move_joints(self, joint_positions, *, speed_scale: float = 0.2) -> None:
        self._unavailable()

    def move_tcp(self, pose, *, speed_scale: float = 0.2) -> None:
        self._unavailable()

    def open_gripper(self, part_name: str | None = None) -> None:
        self._unavailable()

    def close_gripper(self, part_name: str | None = None) -> None:
        self._unavailable()


def connect(config):
    adapter = SharpaAdapter(config)
    adapter.connect()
    return adapter
