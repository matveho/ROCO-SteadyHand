"""SDK boundary tests: no dexcontrol, Pinocchio or CAN installation required.

The arm fake uses the public dexcontrol 0.5.0 move_to_joint_pos() contract and
MotionHandle-style completion. It proves our calls/guards, not hardware
tracking, force units, collision clearance, CAN timing or grasp success.
"""

import copy
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from steadyhand.adapters.vega import VegaAdapter
from steadyhand.cameras.vega import VegaHeadCamera, intrinsics_from_camera_info
from steadyhand.grippers.vega import VegaCanGripper, _load_gripper_module
from steadyhand.models import Pose


def config():
    return {
        "robot_name": "test-vega",
        "working_arm": "right",
        "sdk_version": "0.5.0",
        "allow_robot_init_head_motion": True,
        "urdf_path": "fake.urdf",
        "max_joint_speed_rad_s": 0.2,
        "arm_joint_limits_rad": {
            "left": [[-1.0, 1.0], [-0.4, 1.5], [-1.0, 1.0], [-1.0, 0.24], [-1.0, 1.0], [-1.0, 1.0], [-1.3, 1.1]],
            "right": [[-1.0, 1.0], [-1.5, 0.4], [-1.0, 1.0], [-1.0, 0.24], [-1.0, 1.0], [-1.0, 1.0], [-1.1, 1.3]],
        },
        "motion": {
            "max_step_rad": 0.12,
            "max_total_delta_rad": 0.6,
            "step_wait_time_s": 0.1,
            "control_hz": 100,
            "joint_reached_tolerance_rad": 0.005,
            "joint_timeout_s": 0.001,
        },
        "kinematics": {
            "backend": "pinocchio", "base_frame": "base", "ee_frame": "R_ee",
            "left_arm_joint_names": [f"L_arm_j{i}" for i in range(1, 8)],
            "right_arm_joint_names": [f"R_arm_j{i}" for i in range(1, 8)],
        },
    }


class FakeArm:
    def __init__(self, side, events):
        self.names = [f"{side}_arm_j{i}" for i in range(1, 8)]
        self.q = [0.0] * 7
        self.events = events
        self.stamp = 1
        self.frozen = False
        self.track = True
        self.error = None
        self.wrench_sensor = types.SimpleNamespace(
            get_wrench_state=lambda: [1, 2, 3, 4, 5, 6],
            get_button_state=lambda: 0,
        )

    def get_joint_name(self):
        return self.names[:]

    def get_joint_pos(self):
        return self.q[:]

    def get_joint_vel(self):
        return [0.0] * 7

    def get_joint_current(self):
        return [0.0] * 7

    def get_timestamp_ns(self):
        if not self.frozen:
            self.stamp += 1
        return self.stamp

    class _Handle:
        def __init__(self, arm, target):
            self.arm = arm
            self.target = list(target)
            self.state = "accepted"
            self.message = ""
            self.is_done = False
            self.cancelled = False

        def wait(self, timeout=None):
            if self.arm.error is not None:
                raise self.arm.error
            if self.arm.track:
                self.arm.q = self.target[:]
            self.state = "finished"
            self.is_done = True
            return self.state

        def cancel(self):
            self.cancelled = True
            self.state = "cancelled"
            self.is_done = True
            self.arm.events.append(("cancel",))

    # Exact shape used by dexcontrol 0.5.0 ManagedJointComponent.
    def move_to_joint_pos(self, joint_pos, *, relative=False, velocity_scale=None):
        self.events.append(
            ("move_to_joint_pos", tuple(joint_pos), relative, velocity_scale)
        )
        return self._Handle(self, joint_pos)


class FakeRobot:
    def __init__(self, events):
        self.events = events
        self.events.append(("Robot",))
        self.left_arm = FakeArm("L", events)
        self.right_arm = FakeArm("R", events)
        self.stopped = False
        self.stop_error = None
        self.estop = types.SimpleNamespace(activate=self.activate,
                                          deactivate=self.deactivate,
                                           get_status=lambda: {"software_estop_enabled": self.stopped},
                                           get_state=lambda: {
                                               "software_estop_enabled": self.stopped,
                                               "left_base_estop_enabled": False,
                                               "right_base_estop_enabled": False,
                                               "torso_estop_enabled": False,
                                               "remote_estop_enabled": False,
                                           })

    def activate(self):
        self.events.append(("estop",))
        if self.stop_error is not None:
            raise self.stop_error
        self.stopped = True

    def deactivate(self):
        self.stopped = False

    def shutdown(self):
        self.events.append(("shutdown",))


class FakeKinematics:
    def __init__(self, path, frame, names, cfg):
        self.cfg = cfg

    def forward(self, q):
        return Pose((q[0], q[1], q[2]), (1, 0, 0, 0))

    def solve(self, pose, seed):
        return (*pose.position_m, *seed[3:])


class AdapterContractTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.robot = FakeRobot(self.events)
        self.events.clear()
        robot_module = types.ModuleType("dexcontrol.robot")
        def make_robot():
            self.events.append(("Robot",))
            return self.robot
        robot_module.Robot = make_robot
        # Adapter only needs ndarray conversion; keep this test in stdlib CI.
        numpy = types.ModuleType("numpy")
        numpy.asarray = lambda value, dtype=None: list(value)
        self.patches = [
            patch.dict(sys.modules, {"dexcontrol": types.ModuleType("dexcontrol"),
                                     "dexcontrol.robot": robot_module, "numpy": numpy}),
            patch("steadyhand.adapters.vega.PinocchioArmKinematics", FakeKinematics),
            patch("steadyhand.adapters.vega.importlib.metadata.version", return_value="0.5.0"),
            patch.dict("os.environ", {"ROBOT_NAME": "test-vega"}),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.adapter = VegaAdapter(config())

    def connect(self):
        self.adapter.connect()
        return self.adapter

    def test_prepare_is_no_robot_and_shares_limits_with_solver(self):
        self.adapter.prepare()
        self.assertEqual(self.events, [])
        self.assertEqual(self.adapter._kinematics.cfg["joint_limits_rad"][1], (-1.5, 0.4))

    def test_start_state_motion_and_shutdown(self):
        adapter = self.connect()
        self.assertEqual(adapter.observe().joint_positions, (0.0,) * 7)
        self.assertEqual(adapter.read_wrench(), (1, 2, 3, 4, 5, 6))
        adapter.move_tcp(Pose((0.25, 0.0, 0.0), (1, 0, 0, 0)), speed_scale=0.5)
        calls = [event for event in self.events if event[0] == "move_to_joint_pos"]
        self.assertEqual(len(calls), 1)
        _, q, relative, velocity_scale = calls[0]
        self.assertFalse(relative)
        self.assertEqual(q[0], 0.25)
        self.assertEqual(velocity_scale, 0.5)
        self.assertEqual(adapter.get_tcp_pose().position_m, (0.25, 0, 0))
        adapter.close()
        self.assertEqual(self.events[-1], ("shutdown",))
        self.assertIsNone(adapter._robot)

    def test_malformed_local_config_fails_before_robot(self):
        for key, value in (("max_step_rad", math.nan),
                           ("max_total_delta_rad", 0),
                           ("joint_timeout_s", math.inf),
                           ("joint_reached_tolerance_rad", None)):
            with self.subTest(key=key, value=value):
                cfg = config()
                cfg["motion"][key] = value
                with self.assertRaises(ValueError):
                    VegaAdapter(cfg).connect()
                self.assertEqual(self.events, [])

    def test_ik_load_failure_precedes_head_motion(self):
        with patch("steadyhand.adapters.vega.PinocchioArmKinematics", side_effect=ValueError("invalid frame")):
            with self.assertRaisesRegex(ValueError, "invalid frame"):
                self.adapter.connect()
        self.assertEqual(self.events, [])

    def test_robot_name_mismatch_does_not_connect(self):
        self.adapter.config["robot_name"] = "different-vega"
        with self.assertRaisesRegex(ValueError, "ROBOT_NAME"):
            self.adapter.connect()
        self.assertEqual(self.events, [])

    def test_bad_sdk_order_stops_and_shutdowns(self):
        self.robot.right_arm.names.reverse()
        with self.assertRaisesRegex(ValueError, "joint order"):
            self.adapter.connect()
        self.assertEqual(self.events, [("Robot",), ("estop",), ("shutdown",)])

    def test_startup_rejects_nonfinite_short_and_stale_state(self):
        for q in ([0.0] * 6, [math.nan] * 7, [2.0] * 7):
            with self.subTest(q=q):
                self.robot.right_arm.q = q
                with self.assertRaises(ValueError):
                    VegaAdapter(config()).connect()
        self.robot.right_arm.q = [0.0] * 7
        self.robot.right_arm.frozen = True
        with self.assertRaisesRegex(RuntimeError, "timestamp"):
            VegaAdapter(config()).connect()

    def test_startup_requires_real_estop_state(self):
        self.robot.estop.get_state = lambda: {}
        with self.assertRaisesRegex(RuntimeError, "E-stop state"):
            self.adapter.connect()
        self.assertIn(("estop",), self.events)
        self.assertIn(("shutdown",), self.events)

    def test_startup_can_explicitly_clear_software_estop(self):
        self.robot.stopped = True
        self.adapter.config["auto_clear_software_estop_on_connect"] = True
        self.adapter.connect()
        self.assertFalse(self.robot.stopped)
        self.assertIsNotNone(self.adapter._robot)

    def test_startup_never_clears_pressed_hardware_estop(self):
        self.robot.stopped = True
        original = self.robot.estop.get_state
        def pressed_state():
            state = original()
            state["left_base_estop_enabled"] = True
            return state
        self.robot.estop.get_state = pressed_state
        self.adapter.config["auto_clear_software_estop_on_connect"] = True
        with self.assertRaisesRegex(RuntimeError, "Physical e-stop"):
            self.adapter.connect()
        self.assertTrue(self.robot.stopped)
        self.assertIn(("shutdown",), self.events)

    def test_joint_limits_are_asymmetric(self):
        adapter = self.connect()
        target = [0.0] * 7
        target[1] = -0.41
        with self.assertRaisesRegex(ValueError, "Joint 2"):
            adapter.move_joints(target)
        self.assertFalse(any(item[0] == "move_to_joint_pos" for item in self.events))

    def test_invalid_state_after_connect_never_commands(self):
        adapter = self.connect()
        self.robot.right_arm.q = [math.nan] * 7
        with self.assertRaisesRegex(ValueError, "finite"):
            adapter.move_joints([0.1] * 7)
        self.assertTrue(self.robot.stopped)
        self.assertFalse(any(item[0] == "move_to_joint_pos" for item in self.events))

    def test_motion_failure_and_systemexit_activate_estop(self):
        for error in (RuntimeError("vendor failed"), SystemExit(0), KeyboardInterrupt()):
            with self.subTest(error=error):
                self.robot.stopped = False
                adapter = self.connect()
                self.robot.right_arm.error = error
                with self.assertRaises(type(error)):
                    adapter.move_joints([0.01] * 7)
                self.assertTrue(self.robot.stopped)

    def test_unreached_target_stops_before_second_segment(self):
        adapter = self.connect()
        self.robot.right_arm.track = False
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            adapter.move_joints([0.3, 0, 0, 0, 0, 0, 0])
        self.assertEqual(sum(event[0] == "move_to_joint_pos" for event in self.events), 1)
        self.assertTrue(self.robot.stopped)

    def test_frozen_state_stops_before_second_segment(self):
        adapter = self.connect()
        self.robot.right_arm.frozen = True
        with self.assertRaisesRegex(RuntimeError, "timestamp"):
            adapter.move_joints([0.3, 0, 0, 0, 0, 0, 0])
        self.assertEqual(sum(event[0] == "move_to_joint_pos" for event in self.events), 1)

    def test_gripper_halt_failure_does_not_suppress_estop(self):
        adapter = self.connect()
        adapter._gripper = types.SimpleNamespace(halt=lambda: (_ for _ in ()).throw(RuntimeError("CAN failed")))
        with self.assertRaisesRegex(RuntimeError, "CAN failed"):
            adapter.stop()
        self.assertTrue(self.robot.stopped)

    def test_estop_failure_does_not_suppress_gripper_halt(self):
        adapter = self.connect()
        self.robot.stop_error = RuntimeError("estop failed")
        adapter._gripper = types.SimpleNamespace(halt=lambda: self.events.append(("halt",)))
        with self.assertRaisesRegex(RuntimeError, "estop failed"):
            adapter.stop()
        self.assertIn(("halt",), self.events)

    def test_camera_and_can_cleanup_failures_do_not_suppress_shutdown(self):
        adapter = self.connect()
        def bad_close():
            raise RuntimeError("close failed")
        adapter._head_camera = types.SimpleNamespace(close=bad_close)
        adapter._wrist_cameras = types.SimpleNamespace(close=bad_close)
        adapter._gripper = types.SimpleNamespace(close=bad_close)
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            adapter.close()
        self.assertEqual(self.events[-1], ("shutdown",))
        self.assertIsNone(adapter._robot)


DRIVER = '''from __future__ import annotations
from dataclasses import dataclass
@dataclass
class Status:
    ok: bool = True
class Motor:
    def __init__(self, side, calls): self.side, self.calls = side, calls
    def home(self): self.calls.append((self.side, "home"))
    def open(self, speed=150): self.calls.append((self.side, "open", speed))
    def close(self): self.calls.append((self.side, "close"))
    def grip(self, *, current, speed):
        self.calls.append((self.side, "grip", current, speed)); return {"gripped": True}
    def move_to(self, fraction, *, speed): self.calls.append((self.side, "move_to", fraction, speed))
    def position(self): return 0.5
    def angle(self): return 0.1
    def current(self): return 0.2
    def temperature(self): return 30.0
    def voltage(self): return 24.0
    def enabled(self): return True
    def halt(self): self.calls.append((self.side, "halt"))
    def release(self): self.calls.append((self.side, "release"))
class Grippers:
    def __init__(self):
        self.calls = []; self.left = Motor("left", self.calls); self.right = Motor("right", self.calls)
    def home(self, require_all=False): self.calls.append(("home", require_all))
    def both_open(self, speed=500): self.calls.append(("both_open", speed))
    def both_close(self): self.calls.append(("both_close",))
    def both_move_to(self, fraction, *, speed): self.calls.append(("both_move_to", fraction, speed))
    def status(self): return Status()
    def halt(self): self.calls.append(("halt",))
    def release(self): self.calls.append(("release",))
    def close_bus(self): self.calls.append(("close_bus",))
'''


class GripperContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "gripper.py"
        self.path.write_text(DRIVER, encoding="utf-8")
        self.cfg = {"driver_path": str(self.path), "scope": "right", "grip_current_a": 0.3, "home_on_connect": True}

    def test_driver_registration_and_exact_calls(self):
        gripper = VegaCanGripper(self.cfg)
        gripper.connect()
        driver = gripper._driver
        gripper.open()
        gripper.grip()
        gripper.close_empty()
        gripper.move_fraction(0.2, speed=10)
        self.assertEqual(gripper.position(), 0.5)
        self.assertTrue(gripper.status()["enabled"])
        gripper.close()
        self.assertEqual(driver.calls, [
            ("right", "home"), ("right", "open", 500.0), ("right", "grip", 0.3, 60),
            ("right", "close"), ("right", "move_to", 0.2, 10.0),
            ("right", "halt"), ("close_bus",),
        ])
        self.assertNotIn(("right", "release"), driver.calls)

    def test_config_validation_never_imports_driver(self):
        self.path.write_text("raise RuntimeError('must not import')", encoding="utf-8")
        VegaCanGripper(self.cfg).validate_config()

    def test_unknown_current_and_skip_home_fail_before_constructor(self):
        for updates in ({"grip_current_a": None}, {"grip_current_a": math.nan},
                        {"scope": "middle"}, {"home_on_connect": False}):
            with self.subTest(updates=updates):
                cfg = {**self.cfg, **updates}
                with patch("steadyhand.grippers.vega._load_gripper_module") as load:
                    with self.assertRaises(ValueError):
                        VegaCanGripper(cfg).connect()
                    load.assert_not_called()

    def test_home_failure_closes_can_bus(self):
        module = _load_gripper_module(self.path)
        driver = module.Grippers()
        def fail_home():
            raise RuntimeError("homing failed")
        driver.right.home = fail_home
        module.Grippers = lambda: driver
        with patch("steadyhand.grippers.vega._load_gripper_module", return_value=module):
            gripper = VegaCanGripper(self.cfg)
            with self.assertRaisesRegex(RuntimeError, "homing failed"):
                gripper.connect()
        self.assertEqual(driver.calls, [("right", "halt"), ("close_bus",)])
        self.assertIsNone(gripper._driver)

    def test_invalid_current_never_reaches_driver(self):
        gripper = VegaCanGripper(self.cfg)
        gripper.connect()
        for current in (-1, 0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                gripper.grip(current)
        self.assertEqual(gripper._driver.calls, [("right", "home")])
        gripper.close()

    def test_close_bus_attempted_even_when_halt_fails(self):
        gripper = VegaCanGripper(self.cfg)
        gripper.connect()
        driver = gripper._driver
        def fail_halt():
            raise RuntimeError("halt failed")
        driver.right.halt = fail_halt
        with self.assertRaisesRegex(RuntimeError, "halt failed"):
            gripper.close()
        self.assertIn(("close_bus",), driver.calls)
        self.assertIsNone(gripper._driver)

    def test_camera_intrinsics_does_not_boolean_test_array(self):
        class Array(list):
            def __bool__(self):
                raise ValueError("ambiguous array truth value")
        self.assertEqual(intrinsics_from_camera_info({"K": Array([700, 0, 960, 0, 701, 600, 0, 0, 1])}), (700, 701, 960, 600))

    def test_head_camera_read_waits_for_first_frames(self):
        class FakeHead:
            def __init__(self):
                self.calls = 0

            def get_obs(self, *, obs_keys, include_timestamp):
                self.calls += 1
                if self.calls < 3:
                    return {key: None for key in obs_keys}
                return {
                    key: {"data": key, "timestamp_ns": self.calls}
                    for key in obs_keys
                }

            def get_camera_info(self):
                return {"K": [700, 0, 960, 0, 701, 600, 0, 0, 1]}

        camera = VegaHeadCamera()
        camera._head = FakeHead()
        frame = camera.read(include_depth=False, timeout_s=0.1, poll_s=0.0)
        self.assertEqual(camera._head.calls, 3)
        self.assertEqual(frame.left_rgb, "left_rgb")
        self.assertEqual(frame.right_rgb, "right_rgb")

    def test_head_camera_read_retries_late_camera_info(self):
        class FakeHead:
            def __init__(self):
                self.info_calls = 0

            def get_obs(self, *, obs_keys, include_timestamp):
                return {
                    key: {"data": key, "timestamp_ns": 1}
                    for key in obs_keys
                }

            def get_camera_info(self):
                self.info_calls += 1
                if self.info_calls == 1:
                    raise RuntimeError("info not ready")
                return {"K": [700, 0, 960, 0, 701, 600, 0, 0, 1]}

        camera = VegaHeadCamera()
        camera._head = FakeHead()
        frame = camera.read(include_depth=False, timeout_s=0.1, poll_s=0.0)
        self.assertEqual(camera._head.info_calls, 2)
        self.assertIsNotNone(frame.camera_info)


if __name__ == "__main__":
    unittest.main()
