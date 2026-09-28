import unittest
from unittest import mock

from tools.vega_competition_pipeline import (
    DEFAULT_PIPELINE_SPEED_SCALE,
    POSITION_RECALIBRATE_REQUESTED,
    _prompt_next_location,
)


class CompetitionPipelineTests(unittest.TestCase):
    def test_location_prompt_supports_recalibration_and_exit(self):
        targets = {"board.center": object(), "task.battery_size1.pick": object()}
        with mock.patch("builtins.input", return_value="r"):
            self.assertEqual(_prompt_next_location("board.center", targets), "recalibrate")
        with mock.patch("builtins.input", return_value="e"):
            self.assertEqual(_prompt_next_location("board.center", targets), "exit")

    def test_location_prompt_accepts_numbered_destination(self):
        targets = {"board.center": object(), "task.battery_size1.pick": object()}
        with mock.patch("builtins.input", return_value="2"):
            self.assertEqual(
                _prompt_next_location("board.center", targets),
                "task.battery_size1.pick",
            )

    def test_pipeline_speed_and_recalibration_sentinel_are_defined(self):
        self.assertAlmostEqual(DEFAULT_PIPELINE_SPEED_SCALE, 0.38)
        self.assertEqual(POSITION_RECALIBRATE_REQUESTED, 3)


if __name__ == "__main__":
    unittest.main()
