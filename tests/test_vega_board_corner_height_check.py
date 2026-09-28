import unittest

from tools.vega_board_corner_height_check import summarize_height_samples


class CornerHeightSurveyTests(unittest.TestCase):
    def setUp(self):
        self.samples = {
            "TOP_RIGHT": {
                "measured_value": 470.0,
                "tip_r_pose": {"position_m": [0.6, -0.2, 0.551]},
            },
            "BOTTOM_LEFT": {
                "measured_value": 465.0,
                "tip_r_pose": {"position_m": [0.3, 0.1, 0.549]},
            },
        }

    def test_board_surface_values_produce_global_offset_and_range(self):
        result = summarize_height_samples(
            self.samples,
            measurement_kind="board_surface_z_mm",
            configured_floor_m=0.456,
        )
        self.assertAlmostEqual(result["corner_difference"], 5.0)
        self.assertAlmostEqual(result["modeled_tip_z_difference_m"], 0.002)
        self.assertAlmostEqual(result["candidate_global_board_z_offset_m"], 0.0115)
        self.assertAlmostEqual(result["board_surface_z_range_m"], 0.005)

    def test_clearance_values_do_not_claim_global_offset(self):
        result = summarize_height_samples(
            self.samples,
            measurement_kind="claw_clearance_mm",
            configured_floor_m=0.456,
        )
        self.assertIsNone(result["candidate_global_board_z_offset_m"])
        self.assertIn("independently measured", result["note"])


if __name__ == "__main__":
    unittest.main()
