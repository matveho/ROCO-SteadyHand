import ast
import math
from pathlib import Path
import unittest

import numpy as np

from tools.vega_tool_frame_calibration import (
    analyze,
    matrix_to_quat,
    rpy_matrix,
)
from tools.vega_tool_frame_record import (
    build_analyzer_payload,
    make_observation,
    parse_optional_vector,
)


class _Pose:
    def __init__(self, position_m, quaternion_wxyz):
        self.position_m = tuple(position_m)
        self.quaternion_wxyz = tuple(quaternion_wxyz)


def _pose(position, rotation):
    return _Pose(position, matrix_to_quat(rotation))


class VegaToolFrameRecordTests(unittest.TestCase):
    def test_payload_is_directly_analyzable_and_recovers_pivot_offset(self):
        offset = np.asarray([0.018, -0.011, 0.176])
        pivot = np.asarray([0.55, 0.07, 0.50])
        rotations = {
            "A_REFERENCE": np.eye(3),
            "C_PIVOT_PITCH": rpy_matrix((0.0, math.radians(10.0), 0.0)),
            "D_PIVOT_ROLL": rpy_matrix((math.radians(-9.0), 0.0, 0.0)),
        }
        observations = []
        stamp = 100
        for label, rotation in rotations.items():
            tip = pivot - rotation @ offset
            kwargs = {}
            if label == "A_REFERENCE":
                # One measured physical frame is sufficient to establish the
                # fixed orientation transform; repeat observations improve fit.
                physical_claw_rotation = rpy_matrix((math.pi, 0.0, 0.0))
                kwargs["physical_axis_base"] = physical_claw_rotation[:, 2]
                kwargs["physical_x_axis_base"] = physical_claw_rotation[:, 0]
            observations.append(
                make_observation(
                    label,
                    pose=_pose(tip, rotation),
                    joint_positions_rad=[0.1] * 7,
                    joint_timestamp_ns=stamp,
                    **kwargs,
                )
            )
            stamp += 1

        b_tip = np.asarray(observations[0]["modeled_tip_pose"]["position_m"]) + np.asarray([0.08, 0.0, 0.0])
        observations.insert(
            1,
            make_observation(
                "B_FORWARD",
                pose=_pose(b_tip, rotations["A_REFERENCE"]),
                joint_positions_rad=[0.11] * 7,
                joint_timestamp_ns=200,
            ),
        )

        payload = build_analyzer_payload(
            observations,
            explicit_ab_delta_m=(0.08, 0.0, 0.0),
            robot_name="dm/test",
        )
        result = analyze(payload)

        np.testing.assert_allclose(
            result["offset_solution"]["translation_tip_to_claw_center_m"],
            offset,
            atol=1e-10,
        )
        self.assertEqual(
            payload["translation_checks"][0]["measurement_source"],
            "operator_entered_delta",
        )
        self.assertTrue(
            result["translation_checks"][0][
                "fixed_tool_transform_explains_within_tolerance"
            ]
        )

    def test_absolute_a_b_centers_derive_translation_check(self):
        pose = _Pose((0.5, 0.0, 0.6), (1.0, 0.0, 0.0, 0.0))
        a = make_observation(
            "A_REFERENCE",
            pose=pose,
            joint_positions_rad=[0.0] * 7,
            joint_timestamp_ns=1,
            physical_center_base_m=(0.51, 0.02, 0.55),
        )
        b = make_observation(
            "B_FORWARD",
            pose=_Pose((0.58, 0.0, 0.6), (1.0, 0.0, 0.0, 0.0)),
            joint_positions_rad=[0.0] * 7,
            joint_timestamp_ns=2,
            physical_center_base_m=(0.59, 0.02, 0.552),
        )
        payload = build_analyzer_payload([a, b])
        check = payload["translation_checks"][0]
        self.assertEqual(check["measurement_source"], "difference_of_absolute_centers")
        np.testing.assert_allclose(
            check["measured_physical_center_delta_m"],
            [0.08, 0.0, 0.002],
        )

    def test_missing_external_delta_omits_translation_check_but_remains_analyzable(self):
        observations = []
        for index, label in enumerate(
            ("A_REFERENCE", "B_FORWARD", "C_PIVOT_PITCH", "D_PIVOT_ROLL"),
            start=1,
        ):
            observations.append(
                make_observation(
                    label,
                    pose=_Pose(
                        (0.5 + 0.01 * index, 0.0, 0.6),
                        (1.0, 0.0, 0.0, 0.0),
                    ),
                    joint_positions_rad=[0.0] * 7,
                    joint_timestamp_ns=index,
                )
            )
        payload = build_analyzer_payload(observations)
        self.assertEqual(payload["translation_checks"], [])
        result = analyze(payload)
        self.assertIn("assessment", result)

    def test_external_mm_parser_converts_to_metres(self):
        self.assertEqual(
            parse_optional_vector("80, 0, 2", scale=0.001, name="delta"),
            (0.08, 0.0, 0.002),
        )
        self.assertIsNone(parse_optional_vector("skip", name="delta"))

    def test_x_axis_requires_physical_axis(self):
        pose = _Pose((0.5, 0.0, 0.6), (1.0, 0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "requires physical_claw_axis_base"):
            make_observation(
                "A_REFERENCE",
                pose=pose,
                joint_positions_rad=[0.0] * 7,
                joint_timestamp_ns=1,
                physical_x_axis_base=(1.0, 0.0, 0.0),
            )

    def test_recorder_source_has_no_robot_or_motion_command_calls(self):
        source_path = Path(__file__).resolve().parents[1] / "tools" / "vega_tool_frame_record.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))

        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn("dexcontrol.robot", imported_modules)

        forbidden_calls = {
            "set_joint_pos",
            "move_to_joint_pos",
            "move_joint_pos",
            "set_joint_target",
            "set_joint_trajectory",
            "set_modes",
            "activate",
            "deactivate",
            "open_gripper",
            "close_gripper",
            "grip",
        }
        called_attrs = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue(forbidden_calls.isdisjoint(called_attrs))


if __name__ == "__main__":
    unittest.main()
