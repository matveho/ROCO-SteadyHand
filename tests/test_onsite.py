"""Offline checks for configuration errors and honest dry-run outcomes."""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.config import load_bundle, missing_setup, validate_bundle
from steadyhand.runner import dry_run
from steadyhand.sessions import create_session
from steadyhand.adapters import sharpa, vega


class OnsiteTests(unittest.TestCase):
    def test_templates_are_valid_but_incomplete(self):
        for robot in ("vega", "sharpa"):
            self.assertTrue(missing_setup(load_bundle(robot)))

    def test_rejects_wrong_robot_calibration(self):
        bundle = load_bundle("vega")
        bundle["calibration"]["robot_id"] = "sharpa"
        with self.assertRaisesRegex(ValueError, "different robot"):
            validate_bundle(bundle)

    def test_rejects_invalid_pose_and_reflected_frame(self):
        bundle = load_bundle("vega")
        bad = copy.deepcopy(bundle)
        bad["tasks"]["parts"]["pin"]["pick_pose"] = {
            "position_m": [0, 0, float("nan")], "quaternion_wxyz": [1, 0, 0, 0],
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
            events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
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
            self.assertEqual(metadata["submission_sha256"], hashlib.sha256(submission.read_bytes()).hexdigest())
            result = dry_run(a, "sharpa", "battery_size1")
            self.assertTrue(result["mock_sequence_completed"])
            self.assertIsNone(result["physical_success"])

    def test_hardware_adapters_are_disabled(self):
        for adapter in (vega, sharpa):
            with self.assertRaises(NotImplementedError):
                adapter.connect({})


if __name__ == "__main__":
    unittest.main()
