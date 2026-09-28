"""Offline contracts for battery_size1 calibration, jaw-centering and pick bring-up."""

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from steadyhand.battery_size1 import (
    CALIBRATION_GENERATION,
    SCHEMA_VERSION,
    VERIFIED_GRIP_CURRENT_A,
    VERIFIED_GRIP_SPEED_DPS,
    blank_calibration,
    calibration_is_complete,
    load_alignment_result,
    load_calibration,
    require_right_battery_config,
)
from steadyhand.config import load_bundle
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills
from tools.vega_battery_size1_calibrate import _load_initialized
from tools.vega_battery_size1_center import main as center_main
from tools.vega_battery_size1_pick import main as pick_main


VERTICAL = (math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5))


def complete_calibration(path, *, grasp_z=0.480, goal=(800.0, 700.0)):
    cfg = copy.deepcopy(load_bundle("vega")["robot"])
    value = blank_calibration(cfg)
    value["jaw_alignment"] = {
        "goal_pixel_uv": list(goal),
        "image_size_px": [1920, 1536],
        "source_image": "operator_goal.png",
        "taught_hover_tcp_z_m": 0.550,
        "taught_tip_quaternion_wxyz": list(VERTICAL),
        "source": "operator_taught_current_tcp",
    }
    value["grasp"] = {
        "tcp_z_m": grasp_z,
        "taught_tip_quaternion_wxyz": list(VERTICAL),
        "source": "operator_taught_current_tcp",
    }
    value["calibration_complete"] = True
    Path(path).write_text(json.dumps(value) + "\n")
    return value


def alignment_result(path, calibration_path, *, pose=None, goal=(800.0, 700.0)):
    pose = pose or Pose((0.56, 0.0, 0.55), VERTICAL)
    value = {
        "status": "converged",
        "part": "battery_size1",
        "working_arm": "right",
        "tcp_frame": "tip_r",
        "wrist_camera": "wrist_a",
        "calibration_sha256": hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest(),
        "goal_uv": list(goal),
        "tcp_position_m": list(pose.position_m),
        "tcp_quaternion_wxyz": list(pose.quaternion_wxyz),
    }
    Path(path).write_text(json.dumps(value) + "\n")
    return value


class CalibrationContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg = copy.deepcopy(load_bundle("vega")["robot"])
        self.floor = float(load_vega_skills()["safety"]["min_tcp_z_m"])

    def test_blank_calibration_is_intentionally_incomplete(self):
        value = blank_calibration(self.cfg)
        self.assertFalse(calibration_is_complete(value))
        self.assertIsNone(value["jaw_alignment"]["goal_pixel_uv"])
        self.assertIsNone(value["grasp"]["tcp_z_m"])
        self.assertEqual(value["gripper"]["current_a"], VERIFIED_GRIP_CURRENT_A)
        self.assertEqual(value["gripper"]["speed_dps"], VERIFIED_GRIP_SPEED_DPS)

    def test_blank_calibration_records_explicit_right_side_provenance(self):
        value = blank_calibration(self.cfg)
        self.assertEqual(value["schema_version"], SCHEMA_VERSION)
        self.assertEqual(
            value["calibration_generation"],
            CALIBRATION_GENERATION,
        )
        self.assertEqual(value["working_arm"], "right")
        self.assertEqual(value["tcp_frame"], "tip_r")
        self.assertEqual(value["wrist_camera"], "wrist_a")
        self.assertEqual(value["gripper"]["scope"], "right")
        # The shared floor is still only a provisional runtime guard; a fresh
        # right-arm battery artifact must not claim it as a measured tip_r
        # calibration value.
        self.assertNotIn("floor_m", value)
        self.assertNotIn("table_contact_tip_z_m", value)

    def test_stale_missing_or_left_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cal.json"
            base = complete_calibration(path)
            cases = {}

            missing = copy.deepcopy(base)
            missing.pop("working_arm")
            cases["missing working_arm"] = missing

            old_schema = copy.deepcopy(base)
            old_schema["schema_version"] = 1
            old_schema.pop("calibration_generation", None)
            cases["pre-switch schema"] = old_schema

            left_arm = copy.deepcopy(base)
            left_arm["working_arm"] = "left"
            cases["left arm"] = left_arm

            left_tcp = copy.deepcopy(base)
            left_tcp["tcp_frame"] = "tip_l"
            cases["tip_l"] = left_tcp

            left_wrist = copy.deepcopy(base)
            left_wrist["wrist_camera"] = "wrist_b"
            cases["wrist_b"] = left_wrist

            left_gripper = copy.deepcopy(base)
            left_gripper["gripper"]["scope"] = "left"
            cases["left gripper"] = left_gripper

            for label, value in cases.items():
                with self.subTest(label=label):
                    path.write_text(json.dumps(value))
                    self.assertFalse(calibration_is_complete(value))
                    with self.assertRaises(ValueError):
                        load_calibration(
                            path,
                            self.cfg,
                            floor_m=self.floor,
                        )

    def test_incremental_teaching_refuses_pre_switch_file_until_clean_init(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cal.json"
            legacy = complete_calibration(path)
            legacy["schema_version"] = 1
            legacy.pop("calibration_generation", None)
            legacy.pop("working_arm", None)
            legacy.pop("tcp_frame", None)
            legacy["wrist_camera"] = "wrist_b"
            legacy["gripper"].pop("scope", None)
            path.write_text(json.dumps(legacy))

            with self.assertRaisesRegex(ValueError, "init --force"):
                _load_initialized(path, self.cfg)

            fresh = blank_calibration(self.cfg)
            path.write_text(json.dumps(fresh))
            loaded = _load_initialized(path, self.cfg)
            self.assertFalse(calibration_is_complete(loaded))
            self.assertIsNone(loaded["jaw_alignment"]["goal_pixel_uv"])
            self.assertIsNone(loaded["grasp"]["tcp_z_m"])

    def test_runtime_config_gate_rejects_left_or_mismatched_stack(self):
        mutations = []

        cfg = copy.deepcopy(self.cfg)
        cfg["working_arm"] = "left"
        mutations.append(("working_arm", cfg))

        cfg = copy.deepcopy(self.cfg)
        cfg["kinematics"]["ee_frame"] = "tip_l"
        mutations.append(("tcp_frame", cfg))

        cfg = copy.deepcopy(self.cfg)
        cfg["cameras"]["wrists"][
            "api_label_to_physical_mount"
        ]["wrist_a"] = "left_wrist"
        mutations.append(("wrist", cfg))

        cfg = copy.deepcopy(self.cfg)
        cfg["gripper"]["scope"] = "left"
        mutations.append(("gripper", cfg))

        for label, cfg in mutations:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    require_right_battery_config(cfg)

    def test_complete_calibration_requires_operator_values_and_verified_gripper(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cal.json"
            complete_calibration(path)
            value = load_calibration(path, self.cfg, floor_m=self.floor)
            self.assertEqual(value["jaw_alignment"]["goal_pixel_uv"], [800.0, 700.0])
            self.assertAlmostEqual(value["grasp"]["tcp_z_m"], 0.480)

            bad = json.loads(path.read_text())
            bad["gripper"]["current_a"] = 0.6
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "1.0 A"):
                load_calibration(path, self.cfg, floor_m=self.floor)

    def test_missing_goal_or_grasp_and_below_floor_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cal.json"
            value = blank_calibration(self.cfg)
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_calibration(path, self.cfg, floor_m=self.floor)

            complete_calibration(path, grasp_z=self.floor - 0.001)
            with self.assertRaisesRegex(ValueError, "below task floor"):
                load_calibration(path, self.cfg, floor_m=self.floor)

    def test_jaw_goal_pixel_is_bound_to_safe_taught_hover(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cal.json"
            complete_calibration(path)
            bad = json.loads(path.read_text())
            bad["jaw_alignment"]["taught_hover_tcp_z_m"] = self.floor + 0.040
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "60-120 mm"):
                load_calibration(path, self.cfg, floor_m=self.floor)

    def test_alignment_result_must_match_calibration_goal_digest_and_live_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            cal = Path(directory) / "cal.json"
            result = Path(directory) / "alignment.json"
            value = complete_calibration(cal)
            alignment_result(result, cal)
            current = Pose((0.56, 0.0, 0.55), VERTICAL)
            loaded = load_calibration(cal, self.cfg, floor_m=self.floor)
            record, pose = load_alignment_result(
                result, cal, loaded, current_pose=current
            )
            self.assertEqual(record["status"], "converged")
            self.assertEqual(pose, current)

            record = json.loads(result.read_text())
            record["goal_uv"] = [900, 700]
            result.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "jaw goal"):
                load_alignment_result(result, cal, loaded, current_pose=current)

            alignment_result(result, cal)
            record = json.loads(result.read_text())
            record.pop("working_arm")
            result.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "working_arm=right"):
                load_alignment_result(result, cal, loaded, current_pose=current)


class FakeGripper:
    def __init__(self, result):
        self.result = result

    def last_grip_result(self):
        return self.result


class FakePickRobot:
    def __init__(self, config, *, gripped=True):
        self.config = config
        self.pose = Pose((0.56, 0.0, 0.55), VERTICAL)
        self.moves = []
        self.opened = False
        self.grip_calls = []
        self.closed = False
        self._robot = object()
        self._gripper = FakeGripper(
            {
                "gripped": gripped,
                "peak_current": 1.0,
                "stopped_by": "current" if gripped else "reached",
                "position": 0.20 if gripped else 0.02,
            }
        )

    def prepare(self):
        pass

    def connect(self):
        pass

    def close(self):
        self.closed = True

    def get_tcp_pose(self):
        return self.pose

    def connect_gripper(self):
        pass

    def gripper_status(self):
        return {"position": self._gripper.result["position"], "enabled": True}

    def open_gripper(self, part):
        self.opened = True

    def grip(self, part, *, current_a=None):
        self.grip_calls.append((part, current_a))

    def move_tcp(self, pose, *, speed_scale):
        self.moves.append((pose, speed_scale))
        self.pose = pose


class PickToolTests(unittest.TestCase):
    def run_pick(self, *, gripped=True, answer="yes"):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cal = directory / "cal.json"
            align = directory / "alignment.json"
            out = directory / "run"
            complete_calibration(cal)
            alignment_result(align, cal)
            holder = {}

            def make_robot(cfg):
                robot = FakePickRobot(cfg, gripped=gripped)
                holder["robot"] = robot
                return robot

            with patch("tools.vega_battery_size1_pick.VegaAdapter", side_effect=make_robot), \
                    patch("builtins.input", return_value=answer):
                code = pick_main(
                    [
                        "--calibration", str(cal),
                        "--alignment-result", str(align),
                        "--confirm-physical-motion",
                        "--confirm-battery-ready",
                        "--output", str(out),
                    ]
                )
            return code, holder["robot"], json.loads((out / "result.json").read_text())

    def test_verified_grip_runs_approach_slow_descent_and_lift(self):
        code, robot, result = self.run_pick(gripped=True, answer="yes")
        self.assertEqual(code, 0)
        self.assertTrue(robot.opened)
        self.assertEqual(robot.grip_calls, [("battery_size1", 1.0)])
        self.assertEqual(len(robot.moves), 3)
        self.assertAlmostEqual(robot.moves[0][0].position_m[2], 0.505)
        self.assertAlmostEqual(robot.moves[0][1], 0.70)
        self.assertAlmostEqual(robot.moves[1][0].position_m[2], 0.480)
        self.assertAlmostEqual(robot.moves[1][1], 0.12)
        self.assertAlmostEqual(robot.moves[2][0].position_m[2], 0.550)
        self.assertAlmostEqual(robot.moves[2][1], 0.70)
        self.assertEqual(result["status"], "lift_retention_verified")
        self.assertTrue(result["lift_retention_verified"])
        self.assertTrue(robot.closed)

    def test_driver_failed_grip_never_lifts(self):
        code, robot, result = self.run_pick(gripped=False)
        self.assertEqual(code, 3)
        self.assertEqual(len(robot.moves), 2)
        self.assertEqual(result["status"], "grip_not_verified")
        self.assertIsNone(result["lift_retention_verified"])

    def test_cli_requires_both_motion_gates(self):
        with self.assertRaises(SystemExit) as cm:
            pick_main(["--alignment-result", "missing.json"])
        self.assertEqual(cm.exception.code, 2)


class CenterToolTests(unittest.TestCase):
    def test_center_uses_taught_goal_and_writes_pick_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cal = directory / "cal.json"
            out = directory / "center"
            complete_calibration(cal, goal=(810.0, 710.0))

            class Cameras:
                def connect(self):
                    pass
                def close(self):
                    pass

            class Capture:
                index = 0
                def __init__(self, cameras, output, **kwargs):
                    self.index = 0
                def __call__(self):
                    self.index += 1
                    import numpy as np
                    return np.full((1536, 1920, 3), 80, dtype=np.uint8)

            class Robot:
                def __init__(self, cfg):
                    self.config = cfg
                    self.pose = Pose((0.56, 0.0, 0.55), VERTICAL)
                    self._kinematics = types.SimpleNamespace(config={})
                def connect(self):
                    pass
                def close(self):
                    pass
                def get_tcp_pose(self):
                    return self.pose

            seen = {}

            def servo(robot, capture, **kwargs):
                seen["goal"] = kwargs["goal_uv"]
                return {
                    "status": "converged",
                    "iterations": 1,
                    "error_px": 2.0,
                    "feature_uv": (810.0, 710.0),
                    "goal_uv": tuple(kwargs["goal_uv"]),
                    "jacobian_px_per_m": ((1000.0, 0.0), (0.0, 1000.0)),
                }

            with patch("tools.vega_battery_size1_center.VegaWristCameras", return_value=Cameras()), \
                    patch("tools.vega_battery_size1_center.WristAOnlyCapture", Capture), \
                    patch("tools.vega_battery_size1_center.VegaAdapter", Robot), \
                    patch("tools.vega_battery_size1_center.run_xy_servo", side_effect=servo):
                code = center_main(
                    [
                        "--calibration", str(cal),
                        "--coarse-xy", ".56", "0",
                        "--feature", "900", "700",
                        "--confirm-physical-motion",
                        "--output", str(out),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(seen["goal"], (810.0, 710.0))
            result = json.loads((out / "result.json").read_text())
            self.assertEqual(result["part"], "battery_size1")
            self.assertEqual(result["wrist_camera"], "wrist_a")
            self.assertEqual(result["alignment_goal"], "operator_taught_jaw_pixel")
            self.assertEqual(
                result["calibration_sha256"],
                hashlib.sha256(cal.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
