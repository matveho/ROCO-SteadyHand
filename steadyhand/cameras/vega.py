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
- VERIFIED_ON_COMP_ROBOT 2026-09-28: wrist_a=RIGHT, wrist_b=LEFT
- wrist timestamps are not synchronized to head timestamps
"""

from dataclasses import dataclass
import math
import time
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

    def read(
        self,
        *,
        include_depth: bool = True,
        timeout_s: float = 15.0,
        poll_s: float = 0.05,
    ) -> HeadCameraFrame:
        """Wait for the requested ZED streams and usable camera info.

        The competition ZED can report initialized before its first Zenoh
        frames are available, especially immediately after Robot() has moved
        the head. A single-shot read therefore creates a false failure during
        normal startup. Keep acquisition bounded, but tolerate that warm-up.
        """
        self._require_connected()
        timeout_s = float(timeout_s)
        poll_s = float(poll_s)
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and > 0")
        if not math.isfinite(poll_s) or poll_s < 0:
            raise ValueError("poll_s must be finite and >= 0")

        keys = ["left_rgb", "right_rgb"]
        if include_depth:
            keys.append("depth")

        deadline = time.monotonic() + timeout_s
        last_missing = list(keys)
        last_info_error = None
        while True:
            obs = self._head.get_obs(obs_keys=keys, include_timestamp=True)
            last_missing = [key for key in keys if obs.get(key) is None]
            if not last_missing:
                try:
                    camera_info = self._head.get_camera_info()
                except Exception as exc:
                    # The info service has also been observed to come up late.
                    # Retry it within the same bounded warm-up window.
                    last_info_error = exc
                else:
                    depth = obs.get("depth")
                    return HeadCameraFrame(
                        left_rgb=obs["left_rgb"]["data"],
                        right_rgb=obs["right_rgb"]["data"],
                        depth_m=None if depth is None else depth["data"],
                        left_timestamp_ns=obs["left_rgb"].get("timestamp_ns"),
                        right_timestamp_ns=obs["right_rgb"].get("timestamp_ns"),
                        depth_timestamp_ns=None if depth is None else depth.get("timestamp_ns"),
                        camera_info=camera_info,
                    )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if last_info_error is not None and not last_missing:
                    raise RuntimeError(
                        "Vega head streams are ready but camera info is unavailable "
                        f"after {timeout_s:.1f}s"
                    ) from last_info_error
                raise RuntimeError(
                    "Vega head streams are not ready after "
                    f"{timeout_s:.1f}s; missing: {', '.join(last_missing)}"
                )
            time.sleep(min(poll_s, remaining))

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
        # Live DexSensor ZED info nests rectified intrinsics under
        # camera_info["intrinsics"]["left"].
        nested = camera_info.get("intrinsics")
        if isinstance(nested, Mapping):
            left = nested.get("left")
            if isinstance(left, Mapping) and all(
                key in left for key in ("fx", "fy", "cx", "cy")
            ):
                return tuple(float(left[key]) for key in ("fx", "fy", "cx", "cy"))

        # NumPy arrays have no scalar truth value. Camera-info commonly uses
        # one, so never select K with boolean `or`.
        k = camera_info.get("K")
        if k is None:
            k = camera_info.get("k")
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
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise ValueError("Camera intrinsics must be finite with positive fx/fy")

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
