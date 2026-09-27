"""Observation-only North adapter using the organizer's protobuf definitions.

This creates one Zenoh subscriber and no publishers or service queries. The
vendor NorthClient is deliberately not constructed: it also owns action output.
Camera values are encoded CameraFrame objects, not policy-ready RGB arrays.
"""

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import threading
import time

from .base import (
    AdapterCapabilities,
    HardwareUnavailableError,
    RobotAdapter,
    RobotObservation,
)


JOINT_GROUPS = (("left_arm", 7), ("left_hand", 22),
                ("right_arm", 7), ("right_hand", 22), ("motor", 7))
CAMERA_FIELDS = {"head_left": "image_left", "head_right": "image_right",
                 "wrist_left": "fish_left", "wrist_right": "fish_right"}


@dataclass(frozen=True)
class CameraFrame:
    data: bytes
    format: str
    source_stamp: tuple


def _source_stamp(message):
    header = message.header
    stamp = (header.stamp.sec, header.stamp.nanosec,
             header.frame_id, header.seq)
    if not any(stamp):
        raise ValueError("Sensor message has no source timestamp or sequence")
    return stamp


def _finite(values, size, label):
    values = tuple(float(v) for v in values)
    if len(values) != size or not all(math.isfinite(v) for v in values):
        raise ValueError(f"{label} must contain {size} finite values")
    return values


def _parse(message):
    """Preserve source clocks; never compare them to the workstation's clock."""
    for field in ("robot_state", "vision", "mode"):
        if not message.HasField(field):
            raise ValueError(f"Missing North observation field: {field}")
    timestamp = float(message.timestamp)
    if not math.isfinite(timestamp) or timestamp <= 0:
        raise ValueError("Missing or invalid North observation timestamp")
    tokens = {"observation": (timestamp,), "mode": _source_stamp(message.mode)}
    positions, names, groups = [], {}, {}
    state = message.robot_state
    for name, size in JOINT_GROUPS:
        if not state.HasField(name):
            raise ValueError(f"Missing joint group: {name}")
        group = getattr(state, name)
        if name == "motor":
            if tuple(m.id for m in group.motors) != tuple(range(1, 8)):
                raise ValueError("Body/neck motor order must be IDs 1 through 7")
            values = _finite((m.position for m in group.motors), size, name)
            names[name] = tuple(m.name for m in group.motors)
            token = (group.timestamp, *(m.timestamp for m in group.motors))
            if not any(token):
                raise ValueError("Motor feedback has no source timestamps")
        else:
            values = _finite(group.joint.position, size, name)
            names[name] = tuple(group.joint.name)
            if names[name] and len(names[name]) != size:
                raise ValueError(f"Unexpected joint-name count for {name}")
            token = _source_stamp(group)
        groups[name] = values
        positions.extend(values)
        tokens[name] = token
    cameras = {}
    for name, field in CAMERA_FIELDS.items():
        if not message.vision.HasField(field):
            raise ValueError(f"Missing camera: {name}")
        frame = getattr(message.vision, field)
        if not frame.data or frame.format.lower() not in ("jpeg", "jpg", "png"):
            raise ValueError(f"Missing or unsupported encoded image: {name}")
        stamp = _source_stamp(frame)
        cameras[name] = CameraFrame(bytes(frame.data), frame.format, stamp)
        tokens[f"camera/{name}"] = stamp
    tactile = {}
    for sensor in state.tactile:
        if not sensor.HasField("force6d"):
            continue  # Absence is not a zero force measurement.
        key = sensor.header.key
        if not key or key in tactile:
            raise ValueError("Tactile sensor keys must be nonempty and unique")
        wrench = sensor.force6d
        tactile[key] = _finite((wrench.force.x, wrench.force.y, wrench.force.z,
                                wrench.torque.x, wrench.torque.y, wrench.torque.z),
                               6, f"tactile/{key}")
        tokens[f"tactile/{key}"] = _source_stamp(sensor)
    observation = RobotObservation(
        timestamp_s=timestamp,
        joint_positions=tuple(positions),
        cameras=cameras,
        extras={"observation_only": True, "joint_groups": groups,
                "reported_joint_names": names, "tactile_force6d": tactile,
                "faults": {"left_arm": state.left_arm.error_code,
                           "right_arm": state.right_arm.error_code,
                           "motor": tuple((bool(m.is_error), int(m.faults))
                                          for m in state.motor.motors)},
                "on_sleep": message.on_sleep,
                "mode": {"operation_mode": message.mode.operation_mode,
                         "state": message.mode.state,
                         "sub_state": message.mode.sub_state}},
    )
    return observation, tokens


def _load_dependencies(sdk_root):
    path = Path(sdk_root) / "sharpa_north_ces_lite/proto/north_pb2.py"
    if not path.is_file():
        raise HardwareUnavailableError(f"Organizer SDK protobuf file not found: {path}")
    try:
        import zenoh
        spec = importlib.util.spec_from_file_location("steadyhand_north_pb2", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise HardwareUnavailableError("North observations require eclipse-zenoh and protobuf") from exc
    return zenoh, module


class SharpaAdapter(RobotAdapter):
    robot_id = "sharpa"

    def __init__(self, config):
        super().__init__(config)
        self._session = self._subscriber = self._proto = None
        self._lock = threading.Lock()
        self._observation = None
        self._error = None
        self._progress = {}
        self._received = None
        self._max_age = 0.5

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            joint_motion=False,
            tcp_motion=False,
            cameras=True,
            force_torque=False,
            tactile=True,
        )

    def _unavailable(self):
        raise HardwareUnavailableError(
            "Sharpa adapter supports observations only; motion and hardware "
            "stop commands are not implemented. Use the physical e-stop when needed."
        )

    def connect(self) -> None:
        if self._session is not None:
            return
        if self.config.get("observation_only") is not True:
            raise HardwareUnavailableError("Set observation_only=true to enable North feedback")
        endpoint = self.config.get("zenoh_endpoint")
        sdk_root = self.config.get("sdk_root")
        if not isinstance(endpoint, str) or not endpoint.startswith("tcp/") or not sdk_root:
            raise ValueError("Provide sdk_root and an explicit tcp/ Zenoh endpoint")
        self._max_age = float(self.config.get("max_age_s", 0.5))
        timeout = float(self.config.get("connect_timeout_s", 5.0))
        if not all(math.isfinite(v) and v > 0 for v in (self._max_age, timeout)):
            raise ValueError("Observation timeouts must be positive and finite")
        zenoh, self._proto = _load_dependencies(sdk_root)
        import json
        config = zenoh.Config()
        config.insert_json5("mode", '"client"')
        config.insert_json5("connect/endpoints", json.dumps([endpoint]))
        config.insert_json5("scouting/multicast/enabled", "false")
        config.insert_json5("connect/timeout_ms", str(max(1, int(timeout * 1000))))
        try:
            self._session = zenoh.open(config)
            self._subscriber = self._session.declare_subscriber("north_observation", self._receive)
            deadline = time.monotonic() + timeout
            while True:
                try:
                    self.observe()
                    return
                except RuntimeError as exc:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"North feedback did not become ready: {exc}") from exc
                    time.sleep(0.02)
        except BaseException:
            self.close()
            raise

    def _receive(self, sample):
        try:
            message = self._proto.NorthObservation()
            message.ParseFromString(sample.payload.to_bytes())
            self._accept(message, time.monotonic())
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
                self._observation = None

    def _accept(self, message, received):
        observation, tokens = _parse(message)
        with self._lock:
            progress = {}
            for key, token in tokens.items():
                old = self._progress.get(key)
                if old is None:
                    progress[key] = (token, received, False)
                elif old[0] != token:
                    progress[key] = (token, received, True)
                else:
                    progress[key] = old
            self._progress = progress
            self._observation = observation
            self._received = received
            self._error = None

    def close(self) -> None:
        session, self._session = self._session, None
        try:
            if session is not None:
                session.close()
        finally:
            self._subscriber = None
            with self._lock:
                self._observation = None
                self._received = None
                self._progress = {}
                self._error = None

    def stop(self) -> None:
        self._unavailable()

    def observe(self) -> RobotObservation:
        with self._lock:
            if self._session is None:
                raise RuntimeError("North observation subscriber is not connected")
            if self._error:
                raise RuntimeError(f"Invalid North observation: {self._error}")
            if self._observation is None:
                raise RuntimeError("Waiting for North feedback")
            now = time.monotonic()
            if now - self._received > self._max_age:
                raise RuntimeError("North feedback stream is stale")
            for key, (_, changed, advanced) in self._progress.items():
                if not advanced:
                    raise RuntimeError(f"Waiting for advancing source data: {key}")
                if now - changed > self._max_age:
                    raise RuntimeError(f"North source data is frozen: {key}")
            # Give consumers their own containers; frame bytes and tuples are immutable.
            import copy
            return copy.deepcopy(self._observation)

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
