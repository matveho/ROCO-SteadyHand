import unittest

from steadyhand.models import Pose
from tools.vega_board_corner_height_calibrate import _corner_pose, _measurement_summary


class CornerHeightCalibrationTests(unittest.TestCase):
    def test_corner_targets_use_calibrated_axes_and_center_height(self):
        center = Pose((0.48, -0.03, 0.55), (1.0, 0.0, 0.0, 0.0))
        top = _corner_pose((0.48, -0.03), (1.0, 0.0), (0.0, 1.0), center, 0.2, 0.1)
        for actual, expected in zip(top.position_m, (0.68, 0.07, 0.55)):
            self.assertAlmostEqual(actual, expected)

    def test_board_surface_summary_calculates_global_offset(self):
        samples = {
            "TOP_RIGHT": {"measured_value_mm": 470.0, "tip_r_pose": {"position_m": [0, 0, 0.55]}},
            "BOTTOM_LEFT": {"measured_value_mm": 466.0, "tip_r_pose": {"position_m": [0, 0, 0.548]}},
        }
        summary = _measurement_summary(samples, "board_surface_z_mm", 0.456)
        self.assertAlmostEqual(summary["candidate_global_board_z_offset_m"], 0.012)
        self.assertAlmostEqual(summary["corner_difference_mm"], 4.0)


if __name__ == "__main__":
    unittest.main()
