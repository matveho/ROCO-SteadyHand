"""Fault checks for the North subscriber; no robot or vendor SDK required."""

import copy
import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from steadyhand.adapters.base import HardwareUnavailableError
from steadyhand.adapters.sharpa import CAMERA_FIELDS, JOINT_GROUPS, SharpaAdapter, _parse


class Message(NS):
    def HasField(self, name):
        return getattr(self, name, None) is not None


def header(seq):
    return NS(stamp=NS(sec=1000, nanosec=seq), frame_id=str(seq), seq=seq, key="finger")


def observation(seq=1):
    groups = {}
    for name, count in JOINT_GROUPS[:-1]:
        groups[name] = Message(header=header(seq), error_code=0, joint=NS(
            position=[float(i) for i in range(count)], name=[str(i) for i in range(count)]))
    groups["motor"] = NS(timestamp=seq, motors=[
        NS(id=i, name=f"motor_{i}", position=float(i), timestamp=seq, is_error=False, faults=0) for i in range(1, 8)])
    vector = NS(x=0.0, y=0.0, z=0.0)
    groups["tactile"] = [Message(header=header(seq), force6d=NS(force=vector, torque=vector))]
    cameras = {field: Message(header=header(seq), format="jpeg", data=b"encoded image")
               for field in CAMERA_FIELDS.values()}
    # Camera clocks are intentionally different from the robot and host clocks.
    for camera in cameras.values():
        camera.header.stamp.sec = 1
    return Message(timestamp=1000.0 + seq, robot_state=Message(**groups),
                   vision=Message(**cameras), on_sleep=False,
                   mode=Message(header=header(seq), operation_mode=0, state=0, sub_state=0))


class SharpaObservationTests(unittest.TestCase):
    def ready_adapter(self):
        adapter = SharpaAdapter({})
        adapter._session = NS(close=lambda: None)
        adapter._accept(observation(1), 100.0)
        adapter._accept(observation(2), 100.1)
        return adapter

    def test_protocol_group_order_and_camera_mapping(self):
        result, _ = _parse(observation())
        self.assertEqual(len(result.joint_positions), 65)
        self.assertEqual(result.joint_positions[7:29], tuple(float(i) for i in range(22)))
        self.assertEqual(result.joint_positions[58:], tuple(float(i) for i in range(1, 8)))
        self.assertEqual(set(result.cameras), set(CAMERA_FIELDS))

    def test_faults_are_preserved_for_the_motion_gate(self):
        message = observation()
        message.robot_state.left_arm.error_code = 7
        message.robot_state.motor.motors[2].is_error = True
        message.robot_state.motor.motors[2].faults = 4
        result, _ = _parse(message)
        self.assertEqual(result.extras["faults"]["left_arm"], 7)
        self.assertEqual(result.extras["faults"]["motor"][2], (True, 4))

    def test_missing_and_invalid_components_are_rejected(self):
        cases = []
        message = observation(); message.robot_state.left_arm = None; cases.append(message)
        message = observation(); message.robot_state.left_arm.joint.position.pop(); cases.append(message)
        message = observation(); message.robot_state.left_hand.joint.position[0] = math.nan; cases.append(message)
        message = observation(); message.robot_state.motor.motors.reverse(); cases.append(message)
        message = observation(); message.vision.fish_right.data = b""; cases.append(message)
        message = observation(); message.mode = None; cases.append(message)
        for message in cases:
            with self.subTest(message=cases.index(message)), self.assertRaises(ValueError):
                _parse(message)

    def test_missing_tactile_is_not_reported_as_zero_force(self):
        message = observation(); message.robot_state.tactile[0].force6d = None
        result, _ = _parse(message)
        self.assertEqual(result.extras["tactile_force6d"], {})

    def test_readiness_requires_source_progress_and_not_clock_agreement(self):
        adapter = SharpaAdapter({}); adapter._session = object()
        adapter._accept(observation(1), 100.0)
        with patch("steadyhand.adapters.sharpa.time.monotonic", return_value=100.01):
            with self.assertRaisesRegex(RuntimeError, "advancing"):
                adapter.observe()
        adapter._accept(observation(2), 100.1)
        with patch("steadyhand.adapters.sharpa.time.monotonic", return_value=100.2):
            self.assertEqual(len(adapter.observe().joint_positions), 65)

    def test_cached_camera_in_new_aggregate_messages_is_rejected(self):
        adapter = self.ready_adapter()
        message = observation(3)
        message.vision.image_left = observation(2).vision.image_left
        adapter._accept(message, 100.8)
        with patch("steadyhand.adapters.sharpa.time.monotonic", return_value=100.81):
            with self.assertRaisesRegex(RuntimeError, "frozen: camera/head_left"):
                adapter.observe()

    def test_stale_joint_feedback_and_stream_are_rejected(self):
        adapter = self.ready_adapter()
        message = observation(3)
        message.robot_state.right_arm = observation(2).robot_state.right_arm
        adapter._accept(message, 100.8)
        with patch("steadyhand.adapters.sharpa.time.monotonic", return_value=100.81):
            with self.assertRaisesRegex(RuntimeError, "frozen: right_arm"):
                adapter.observe()
        with patch("steadyhand.adapters.sharpa.time.monotonic", return_value=102):
            with self.assertRaisesRegex(RuntimeError, "stream is stale"):
                adapter.observe()

    def test_consumers_cannot_mutate_cached_containers(self):
        adapter = self.ready_adapter()
        with patch("steadyhand.adapters.sharpa.time.monotonic", return_value=100.2):
            result = adapter.observe(); result.cameras.clear(); result.extras["mode"].clear()
            self.assertEqual(len(adapter.observe().cameras), 4)
            self.assertEqual(adapter.observe().extras["mode"]["state"], 0)

    def test_connect_only_subscribes_and_close_does_not_control_robot(self):
        class WireMessage:
            def ParseFromString(self, message):
                self.__dict__.update(message.__dict__)
            HasField = Message.HasField
        class Session:
            def __init__(self): self.closed = False; self.topics = []
            def declare_subscriber(self, topic, callback):
                self.topics.append(topic)
                for seq in (1, 2):
                    callback(NS(payload=NS(to_bytes=lambda seq=seq: observation(seq))))
                return object()
            def close(self): self.closed = True
            def declare_publisher(self, *args): raise AssertionError("Command publisher created")
            def put(self, *args): raise AssertionError("Command sent")
            def get(self, *args): raise AssertionError("Service queried")
        session = Session()
        zenoh = NS(Config=lambda: NS(insert_json5=lambda *args: None), open=lambda config: session)
        adapter = SharpaAdapter({"observation_only": True, "sdk_root": "vendor-sdk",
                                "zenoh_endpoint": "tcp/robot:7449"})
        with patch("steadyhand.adapters.sharpa._load_dependencies", return_value=(zenoh, NS(NorthObservation=WireMessage))):
            adapter.connect()
        self.assertEqual(session.topics, ["north_observation"])
        self.assertFalse(adapter.capabilities.joint_motion)
        self.assertFalse(adapter.capabilities.tcp_motion)
        for call in [lambda: adapter.move_joints([0] * 65), lambda: adapter.move_tcp(None),
                     adapter.open_gripper, adapter.close_gripper, adapter.stop]:
            with self.assertRaises(HardwareUnavailableError): call()
        adapter.close(); adapter.close()
        self.assertTrue(session.closed)
        with self.assertRaisesRegex(RuntimeError, "not connected"): adapter.observe()


if __name__ == "__main__":
    unittest.main()
