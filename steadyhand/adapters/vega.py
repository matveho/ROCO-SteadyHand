"""DexMate Vega U hardware adapter boundary.

Do not import dexcontrol at module import time. Before enabling this adapter,
verify the installed SDK, physical gripper condition, stop behavior, frames,
and the gripper-equipped collision model.
"""

from .base import (
    AdapterCapabilities,
    HardwareUnavailableError,
    RobotAdapter,
    RobotObservation,
)


class VegaAdapter(RobotAdapter):
    robot_id = "vega"

    @property
    def capabilities(self) -> AdapterCapabilities:
        # Expected from provided documentation, not proof that this stub has
        # exercised the actual competition robot.
        return AdapterCapabilities(
            joint_motion=True,
            tcp_motion=False,
            cameras=True,
            force_torque=True,
            tactile=False,
        )

    def _unavailable(self):
        raise HardwareUnavailableError(
            "Vega adapter is not enabled yet. Implement and validate against "
            "the competition robot's installed dexcontrol/gripper stack first."
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
    adapter = VegaAdapter(config)
    adapter.connect()
    return adapter
