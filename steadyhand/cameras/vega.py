"""Vega U camera acquisition.

This module is deliberately independent of Robot(): constructing dexcontrol
Robot() moves the head. Camera acquisition therefore uses the sensor-only head
API and the local wrist-camera API.

Head:
- ZED X Mini stereo on the head
- rectified left/right RGB + depth in metres
- configured 1920x1200 @ 30 fps; competition guidance reports ~24 Hz observed
- no native point-cloud stream; point clouds are reconstructed from depth + K

Wrists:
- two Sony ISX031 cameras
- local WristCameras API returns wrist_a / wrist_b
- physical left/right mapping must be verified onsite
- wrist timestamps are not synchronized to head timestamps
"""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class HeadCameraFrame:
    left_rgb: Any
    right_rgb: Any
    depth_m: Any
    left_timestamp_ns: int | None
    right_timestamp_ns: int | None
    depth_timestamp_ns: int | None
    camera_info: Any


@dataclass(frozen=True)
class WristCameraFrame:
    rgb: Any
    frame_id: int | None
    timestamp_ns: int | None
    received_monotonic_ns: int | None


@dataclass(frozen=True)
class WristPair:
    wrist_a: WristCameraFrame
    wrist_b: WristCameraFrame


class VegaHeadCamera:
    """Sensor-only ZED subscriber. Does not construct dexcontrol Robot()."""

    def __init__(self) -> None:
        self._sensors = None
        self._head = None

    def connect(self) -> None:
        if self._sensors is not None:
            return
        from dexcontrol.core.config import get_robot_config
        from dexcontrol.sensors.manager import Sensors

        configs = get_robot_config()
        if "head_camera" not in configs.sensors:
            raise RuntimeError("dexcontrol config has no head_camera")
        configs.sensors["head_camera"].enabled = True

        sensors = Sensors({"head_camera": configs.sensors["head_camera"]})
        try:
            sensors.wait_for_all_active(timeout=10)
        except Exception:
            sensors.shutdown()
            raise

        self._sensors = sensors
        self._head = sensors.head_camera

    def close(self) -> None:
        if self._sensors is not None:
            self._sensors.shutdown()
        self._sensors = None
        self._head = None

    def camera_info(self):
        self._require_connected()
        return self._head.get_camera_info()

    def read(self) -> HeadCameraFrame:
        self._require_connected()
        obs = self._head.get_obs(
            obs_keys=["left_rgb", "right_rgb", "depth"],
            include_timestamp=True,
        )
        required = ("left_rgb", "right_rgb", "depth")
        if any(obs.get(key) is None for key in required):
            raise RuntimeError("One or more Vega head streams are not ready")

        return HeadCameraFrame(
            left_rgb=obs["left_rgb"]["data"],
            right_rgb=obs["right_rgb"]["data"],
            depth_m=obs["depth"]["data"],
            left_timestamp_ns=obs["left_rgb"].get("timestamp_ns"),
            right_timestamp_ns=obs["right_rgb"].get("timestamp_ns"),
            depth_timestamp_ns=obs["depth"].get("timestamp_ns"),
            camera_info=self._head.get_camera_info(),
        )

    def _require_connected(self) -> None:
        if self._head is None:
            raise RuntimeError("VegaHeadCamera is not connected")


class VegaWristCameras:
    """Context-managed wrapper around the local dual-ISX031 API."""

    def __init__(self) -> None:
        self._manager = None
        self._cameras = None

    def connect(self) -> None:
        if self._cameras is not None:
            return
        from wrist_cameras import WristCameras

        manager = WristCameras()
        cameras = manager.__enter__()
        self._manager = manager
        self._cameras = cameras

    def close(self) -> None:
        if self._manager is not None:
            self._manager.__exit__(None, None, None)
        self._manager = None
        self._cameras = None

    def read(self, *, timeout: float = 3.0, fresh: bool = True) -> WristPair:
        if self._cameras is None:
            raise RuntimeError("VegaWristCameras is not connected")
        obs = self._cameras.get_obs(timeout=timeout, fresh=fresh)
        return WristPair(
            wrist_a=_wrist_record(obs["wrist_a"]),
            wrist_b=_wrist_record(obs["wrist_b"]),
        )


def _wrist_record(record: Mapping[str, Any]) -> WristCameraFrame:
    return WristCameraFrame(
        rgb=record["rgb"],
        frame_id=record.get("frame_id"),
        timestamp_ns=record.get("timestamp_ns"),
        received_monotonic_ns=record.get("received_monotonic_ns"),
    )


def intrinsics_from_camera_info(camera_info) -> tuple[float, float, float, float]:
    """Extract fx, fy, cx, cy from common camera-info representations.

    Runtime camera info is authoritative. This intentionally avoids embedding
    the example intrinsics from the field manual.
    """
    if camera_info is None:
        raise ValueError("camera_info is required")

    if isinstance(camera_info, Mapping):
        k = camera_info.get("K") or camera_info.get("k")
        if k is not None:
            if len(k) != 9:
                raise ValueError("camera_info K must contain 9 values")
            return float(k[0]), float(k[4]), float(k[2]), float(k[5])

        for keyset in (
            ("fx", "fy", "cx", "cy"),
            ("f_x", "f_y", "c_x", "c_y"),
        ):
            if all(key in camera_info for key in keyset):
                return tuple(float(camera_info[key]) for key in keyset)

    k = getattr(camera_info, "K", None)
    if k is None:
        k = getattr(camera_info, "k", None)
    if k is not None:
        if len(k) != 9:
            raise ValueError("camera_info K must contain 9 values")
        return float(k[0]), float(k[4]), float(k[2]), float(k[5])

    attrs = ("fx", "fy", "cx", "cy")
    if all(hasattr(camera_info, key) for key in attrs):
        return tuple(float(getattr(camera_info, key)) for key in attrs)

    raise ValueError("Could not extract fx, fy, cx, cy from camera_info")


def depth_to_point_cloud(
    depth_m,
    camera_info,
    *,
    stride: int = 1,
    min_depth_m: float = 0.05,
    max_depth_m: float | None = None,
):
    """Reconstruct XYZ points in the rectified left-camera optical frame.

    Returns an (N, 3) float array. Invalid/non-finite depth and values outside
    the requested range are removed. This is geometry only; it does not apply
    T_base_camera or any board transform.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")

    import numpy as np

    depth = np.asarray(depth_m)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2-D depth image")

    fx, fy, cx, cy = intrinsics_from_camera_info(camera_info)

    sampled = depth[::stride, ::stride].astype(np.float32, copy=False)
    vv, uu = np.indices(sampled.shape, dtype=np.float32)
    uu *= stride
    vv *= stride

    valid = np.isfinite(sampled) & (sampled >= float(min_depth_m))
    if max_depth_m is not None:
        valid &= sampled <= float(max_depth_m)

    z = sampled[valid]
    u = uu[valid]
    v = vv[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.column_stack((x, y, z))
