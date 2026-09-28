import math
import unittest

import numpy as np

from tools.vega_tool_frame_calibration import (
    corrected_vertical_tip_quaternion,
    matrix_to_quat,
    quat_to_matrix,
    rpy_matrix,
    solve_full_orientation_correction,
    solve_pivot_group,
    translation_checks,
)


def pose(position, rotation):
    return {
        "position_m": list(position),
        "quaternion_wxyz": matrix_to_quat(rotation).tolist(),
    }


class VegaToolFrameCalibrationTests(unittest.TestCase):
    def test_same_pivot_recovers_tip_to_physical_center_offset(self):
        offset = np.asarray([0.021, -0.012, 0.181])
        pivot = np.asarray([0.54, 0.08, 0.49])
        rotations = [
            np.eye(3),
            rpy_matrix((math.radians(10), 0, 0)),
            rpy_matrix((0, math.radians(-9), 0)),
            rpy_matrix((math.radians(-7), math.radians(6), 0)),
        ]
        observations = []
        for index, rotation in enumerate(rotations):
            tip_position = pivot - rotation @ offset
            observations.append({
                "label": f"P{index}",
                "pivot_group": "center_mark",
                "modeled_tip_pose": pose(tip_position, rotation),
            })

        result = solve_pivot_group(observations, "center_mark")
        np.testing.assert_allclose(
            result["translation_tip_to_claw_center_m"], offset, atol=1e-10
        )
        np.testing.assert_allclose(result["pivot_base_m"], pivot, atol=1e-10)
        self.assertLess(result["rms_residual_mm"], 1e-6)

    def test_same_orientation_rise_cannot_be_caused_by_fixed_tool_offset(self):
        rotation = rpy_matrix((0, 0, -math.pi / 2))
        observations = [
            {
                "label": "A",
                "modeled_tip_pose": pose((0.50, 0.0, 0.55), rotation),
            },
            {
                "label": "B",
                "modeled_tip_pose": pose((0.60, 0.0, 0.55), rotation),
            },
        ]
        measured = [
            0.100,
            0.0,
            0.100 * math.tan(math.radians(10.0)),
        ]
        rows = translation_checks(
            {
                "translation_checks": [{
                    "from": "A",
                    "to": "B",
                    "measured_physical_center_delta_m": measured,
                    "explanation_tolerance_m": 0.003,
                }]
            },
            observations,
            offset=None,
        )
        row = rows[0]
        self.assertLess(row["modeled_orientation_change_deg"], 1e-6)
        self.assertFalse(row["fixed_offset_alone_can_create_coupling"])
        self.assertFalse(row["fixed_tool_transform_explains_within_tolerance"])
        self.assertAlmostEqual(row["measured_center_path_slope_deg"], 10.0, places=6)

    def test_orientation_change_plus_offset_can_explain_center_coupling(self):
        offset = np.asarray([0.0, 0.0, 0.20])
        r0 = np.eye(3)
        r1 = rpy_matrix((0, math.radians(6.0), 0))
        p0 = np.asarray([0.50, 0.0, 0.55])
        p1 = np.asarray([0.60, 0.0, 0.55])
        predicted = (p1 - p0) + (r1 - r0) @ offset
        observations = [
            {"label": "A", "modeled_tip_pose": pose(p0, r0)},
            {"label": "B", "modeled_tip_pose": pose(p1, r1)},
        ]
        rows = translation_checks(
            {
                "translation_checks": [{
                    "from": "A",
                    "to": "B",
                    "measured_physical_center_delta_m": predicted.tolist(),
                    "explanation_tolerance_m": 0.0001,
                }]
            },
            observations,
            offset={"translation_tip_to_claw_center_m": offset.tolist()},
        )
        self.assertTrue(rows[0]["fixed_tool_transform_explains_within_tolerance"])
        self.assertLess(rows[0]["residual_norm_mm"], 1e-8)

    def test_full_physical_frame_recovers_orientation_and_corrected_vertical(self):
        r_tip_claw = rpy_matrix((
            math.radians(7.0),
            math.radians(-4.0),
            math.radians(18.0),
        ))
        observations = []
        for index, r_base_tip in enumerate([
            np.eye(3),
            rpy_matrix((0, 0, math.radians(30))),
            rpy_matrix((math.radians(8), math.radians(-5), math.radians(20))),
        ]):
            r_base_claw = r_base_tip @ r_tip_claw
            observations.append({
                "label": f"O{index}",
                "modeled_tip_pose": pose((0.5, 0.0, 0.6), r_base_tip),
                "physical_claw_axis_base": r_base_claw[:, 2].tolist(),
                "physical_claw_x_axis_base": r_base_claw[:, 0].tolist(),
            })

        solved = solve_full_orientation_correction(observations)
        np.testing.assert_allclose(
            solved["rotation_tip_to_physical_claw"], r_tip_claw, atol=1e-10
        )

        quat = corrected_vertical_tip_quaternion(
            solved["rotation_tip_to_physical_claw"],
            yaw_deg=0.0,
        )
        r_base_tip = quat_to_matrix(quat)
        r_base_claw = r_base_tip @ r_tip_claw
        desired = rpy_matrix((math.pi, 0.0, 0.0))
        np.testing.assert_allclose(r_base_claw, desired, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
