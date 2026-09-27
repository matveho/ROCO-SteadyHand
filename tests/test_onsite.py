"""Offline checks for configuration errors and honest dry-run outcomes."""

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters import sharpa, vega
from steadyhand.adapters.base import HardwareUnavailableError, RobotAdapter
from steadyhand.adapters.mock import MockAdapter
from steadyhand.cameras.vega import intrinsics_from_camera_info
from steadyhand.config import load_bundle, missing_setup, validate_bundle
from steadyhand.geometry import (
    compose,
    invert_rigid,
    pose_to_matrix,
    transform_point,
    transform_pose,
)
from steadyhand.models import Pose
from steadyhand.runner import dry_run
from steadyhand.search import centered_grid, square_spiral
from steadyhand.sessions import create_session
from steadyhand.skills import PHASES, goal_from_spec


EXPECTED_PART_ORDER = [
    "gear_60teeth", "gear_20teeth", "rod_16mm", "bolt_8mm", "usb_a",
    "hdmi", "pin", "battery_size1", "battery_size5",
]
ORGANIZER_COMMIT = "45dd6ad6e0792faf3450bdd2f81bb143b11bc43f"


class OnsiteTests(unittest.TestCase):
    def test_templates_are_valid_but_incomplete(self):
        for robot in ("vega", "sharpa"):
            self.assertTrue(missing_setup(load_bundle(robot)))

    def test_task_config_is_pinned_to_organizer_code(self):
        tasks = load_bundle("vega")["tasks"]
        self.assertEqual(tasks["part_order"], EXPECTED_PART_ORDER)
        self.assertEqual(tasks["source"]["commit"], ORGANIZER_COMMIT)
        self.assertEqual(tasks["source"]["source_file"], "task/param_config.py")

    def test_vega_camera_contract_is_recorded(self):
        robot = load_bundle("vega")["robot"]
        head = robot["cameras"]["head"]
        wrists = robot["cameras"]["wrists"]
        self.assertEqual(head["model"], "ZED X Mini")
        self.assertEqual((head["width"], head["height"]), (1920, 1200))
        self.assertEqual(head["configured_fps"], 30)
        self.assertEqual(head["observed_publish_hz_approx"], 24)
        self.assertFalse(head["native_point_cloud"])
        self.assertEqual(wrists["model"], "Sony ISX031")
        self.assertEqual(wrists["api_labels"], ["wrist_a", "wrist_b"])
        self.assertIsNone(wrists["api_label_to_physical_mount"])

    def test_runtime_camera_intrinsics_parser(self):
        self.assertEqual(
            intrinsics_from_camera_info(
                {"K": [700, 0, 960, 0, 701, 600, 0, 0, 1]}
            ),
            (700.0, 701.0, 960.0, 600.0),
        )

    def test_goal_conversion_does_not_invent_unknown_poses(self):
        spec = load_bundle("vega")["tasks"]["parts"]["usb_a"]
        goal = goal_from_spec("usb_a", spec)
        self.assertEqual(goal.name, "usb_a")
        self.assertEqual(goal.release_mode, "snap")
        self.assertIsNone(goal.pick_pose)
        self.assertIsNone(goal.place_pose)

    def test_shared_phase_vocabulary_has_verification_before_transfer(self):
        self.assertLess(PHASES.index("verify_grasp"), PHASES.index("transfer"))
        self.assertEqual(PHASES[-1], "verify_place")

    def test_rigid_transform_round_trip(self):
        pose = Pose(
            position_m=(0.12, -0.08, 0.44),
            quaternion_wxyz=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
        )
        t = pose_to_matrix(pose)
        identity = compose(t, invert_rigid(t))
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(identity[i][j], float(i == j), places=9)

        p = (0.01, 0.02, 0.03)
        transformed = transform_point(t, p)
        recovered = transform_point(invert_rigid(t), transformed)
        for a, b in zip(p, recovered):
            self.assertAlmostEqual(a, b, places=9)

    def test_transform_pose_respects_destination_source_convention(self):
        t_base_camera = [
            [1, 0, 0, 0.5],
            [0, 1, 0, -0.2],
            [0, 0, 1, 1.0],
            [0, 0, 0, 1],
        ]
        camera_pose = Pose((0.1, 0.2, 0.3), (1, 0, 0, 0))
        base_pose = transform_pose(t_base_camera, camera_pose)
        self.assertEqual(base_pose.position_m, (0.6, 0.0, 1.3))

    def test_search_patterns_start_at_nominal_pose(self):
        grid = centered_grid(5, (0.002, 0.003))
        self.assertEqual(grid[0], (0.0, 0.0))
        self.assertEqual(len(grid), 25)
        self.assertLessEqual(max(abs(x) for x, _ in grid), 0.002)
        self.assertLessEqual(max(abs(y) for _, y in grid), 0.003)

        spiral = square_spiral(step_m=0.001, rings=2)
        self.assertEqual(spiral[0], (0.0, 0.0))
        self.assertEqual(len(spiral), 25)
        self.assertEqual(len(set(spiral)), len(spiral))

    def test_mock_adapter_exercises_common_contract(self):
        robot = MockAdapter()
        self.assertIsInstance(robot, RobotAdapter)
        robot.connect()
        robot.open_gripper("pin")
        robot.move_joints([0.1, 0.2], speed_scale=0.1)
        robot.move_tcp(Pose((0, 0, 0.2), (1, 0, 0, 0)))
        self.assertTrue(robot.verify_grasp("pin"))
        robot.stop()
        robot.close()
        names = [command[0] for command in robot.commands]
        self.assertEqual(
            names,
            [
                "connect", "open_gripper", "move_joints", "move_tcp",
                "verify_grasp", "stop", "close",
            ],
        )

    def test_rejects_wrong_robot_calibration(self):
        bundle = load_bundle("vega")
        bundle["calibration"]["robot_id"] = "sharpa"
        with self.assertRaisesRegex(ValueError, "different robot"):
            validate_bundle(bundle)

    def test_rejects_invalid_pose_and_reflected_frame(self):
        bundle = load_bundle("vega")
        bad = copy.deepcopy(bundle)
        bad["tasks"]["parts"]["pin"]["pick_pose"] = {
            "position_m": [0, 0, float("nan")],
            "quaternion_wxyz": [1, 0, 0, 0],
        }
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_bundle(bad)
        bundle["calibration"]["T_base_board"] = [
            [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        ]
        with self.assertRaisesRegex(ValueError, "right-handed"):
            validate_bundle(bundle)

    def test_failure_stops_before_transfer_and_never_claims_physical_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = load_bundle("vega")
            folder = create_session("vega", "test", "dry_run", bundle, tmp)
            result = dry_run(folder, "vega", "pin", "verify_grasp")
            events = [
                json.loads(line)
                for line in (folder / "events.jsonl").read_text().splitlines()
            ]
            self.assertFalse(result["mock_sequence_completed"])
            self.assertIsNone(result["physical_success"])
            self.assertNotIn("transfer", [event["phase"] for event in events])
            self.assertEqual(events[-1]["event"], "mock_stop")
            self.assertEqual(len((folder / "trials.csv").read_text().splitlines()), 1)

    def test_sessions_are_unique_and_snapshot_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = load_bundle("sharpa")
            a = create_session("sharpa", "test", "dry_run", bundle, tmp)
            b = create_session("sharpa", "test", "dry_run", bundle, tmp)
            self.assertNotEqual(a, b)
            self.assertEqual(json.loads((a / "config_snapshot.json").read_text()), bundle)
            submission = Path(__file__).resolve().parents[1] / "policy.py"
            metadata = json.loads((a / "session.json").read_text())
            self.assertEqual(
                metadata["submission_sha256"],
                hashlib.sha256(submission.read_bytes()).hexdigest(),
            )
            result = dry_run(a, "sharpa", "battery_size1")
            self.assertTrue(result["mock_sequence_completed"])
            self.assertIsNone(result["physical_success"])

    def test_adapters_share_contract_and_are_disabled(self):
        self.assertTrue(issubclass(vega.VegaAdapter, RobotAdapter))
        self.assertTrue(issubclass(sharpa.SharpaAdapter, RobotAdapter))
        for module in (vega, sharpa):
            with self.assertRaises(HardwareUnavailableError):
                module.connect({})


if __name__ == "__main__":
    unittest.main()
