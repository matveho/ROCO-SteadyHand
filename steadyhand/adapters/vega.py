"""DexMate Vega U hardware adapter boundary.

Arm/gripper motion remains disabled until validated on the competition robot.
Camera acquisition is implemented separately and is safe to initialize without
constructing dexcontrol Robot(), which would move the head.
"""

from ..cameras.vega import VegaHeadCamera, VegaWristCameras
from .base import (
    AdapterCapabilities,
    HardwareUnavailableError,
    RobotAdapter,
    RobotObservation,
)


class VegaAdapter(RobotAdapter):
    robot_id = "vega"

    def __init__(self, config):
        super().__init__(config)
        self._head_camera = None
        self._wrist_cameras = None

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            joint_motion=True,
            tcp_motion=False,
            cameras=True,
            force_torque=True,
            tactile=False,
        )

    def _motion_unavailable(self):
        raise HardwareUnavailableError(
            "Vega arm/gripper control is not enabled yet. Validate the actual "
            "competition robot's dexcontrol, stop behavior, gripper driver, "
            "joint mapping, and gripper-equipped collision model first."
        )

    # -------- No-motion camera path --------

    def connect_cameras(self, *, head=True, wrists=True) -> None:
        """Connect requested cameras without constructing dexcontrol Robot()."""
        opened_head = False
        try:
            if head and self._head_camera is None:
                self._head_camera = VegaHeadCamera()
                self._head_camera.connect()
                opened_head = True

            if wrists and self._wrist_cameras is None:
                self._wrist_cameras = VegaWristCameras()
                self._wrist_cameras.connect()
        except Exception:
            # Roll back only resources opened by this call.
            if opened_head and self._head_camera is not None:
                self._head_camera.close()
                self._head_camera = None
            if wrists and self._wrist_cameras is not None:
                self._wrist_cameras.close()
                self._wrist_cameras = None
            raise

    def read_cameras(self, *, head=True, wrists=True):
        """Return available camera frames.

        Head and wrist timestamps use different time bases. The returned
        records must not be treated as exposure-synchronized.
        """
        out = {}
        if head:
            if self._head_camera is None:
                raise RuntimeError("Head camera is not connected")
            out["head"] = self._head_camera.read()
        if wrists:
            if self._wrist_cameras is None:
                raise RuntimeError("Wrist cameras are not connected")
            out["wrists"] = self._wrist_cameras.read()
        return out

    def close_cameras(self) -> None:
        if self._wrist_cameras is not None:
            self._wrist_cameras.close()
            self._wrist_cameras = None
        if self._head_camera is not None:
            self._head_camera.close()
            self._head_camera = None

    # -------- Motion/control path: intentionally disabled --------

    def connect(self) -> None:
        self._motion_unavailable()

    def close(self) -> None:
        self.close_cameras()

    def stop(self) -> None:
        self._motion_unavailable()

    def observe(self) -> RobotObservation:
        self._motion_unavailable()

    def move_joints(self, joint_positions, *, speed_scale: float = 0.2) -> None:
        self._motion_unavailable()

    def move_tcp(self, pose, *, speed_scale: float = 0.2) -> None:
        self._motion_unavailable()

    def open_gripper(self, part_name: str | None = None) -> None:
        self._motion_unavailable()

    def close_gripper(self, part_name: str | None = None) -> None:
        self._motion_unavailable()


def connect(config):
    """Full hardware connection; intentionally unavailable until validated."""
    adapter = VegaAdapter(config)
    adapter.connect()
    return adapter


def connect_cameras(config=None, *, head=True, wrists=True):
    """Convenience entry point for sensor-only Vega access."""
    adapter = VegaAdapter(config or {})
    adapter.connect_cameras(head=head, wrists=wrists)
    return adapter
