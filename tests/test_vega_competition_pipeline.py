import unittest
from unittest import mock

from tools.vega_competition_pipeline import (
    DEFAULT_PIPELINE_SPEED_SCALE,
    DEFAULT_PICK_PRIORITY,
    _load_competition_plan,
    _reload_operator_settings,
    _run_competition_action,
    _sequence_indices,
    _prompt_next_location,
)


class CompetitionPipelineTests(unittest.TestCase):
    def test_location_prompt_supports_image_retake_and_exit(self):
        targets = {"board.center": object(), "task.battery_size1.pick": object()}
        with mock.patch("builtins.input", return_value="r"):
            self.assertEqual(_prompt_next_location("board.center", targets), "retake_image")
        with mock.patch("builtins.input", return_value="e"):
            self.assertEqual(_prompt_next_location("board.center", targets), "exit")

    def test_location_prompt_accepts_numbered_destination(self):
        targets = {"board.center": object(), "task.battery_size1.pick": object()}
        with mock.patch("builtins.input", return_value="2"):
            self.assertEqual(
                _prompt_next_location("board.center", targets),
                "task.battery_size1.pick",
            )

    def test_pipeline_speed_default_is_slightly_higher(self):
        self.assertAlmostEqual(DEFAULT_PIPELINE_SPEED_SCALE, 0.38)

    def test_priority_plan_is_complete_and_operator_editable(self):
        plan = _load_competition_plan()
        self.assertEqual(set(plan["pick_priority"]), set(DEFAULT_PICK_PRIORITY))
        self.assertEqual(plan["default_action"], "pick_place")
        self.assertGreaterEqual(plan["max_retries_per_part"], 0)

    def test_menu_reload_applies_master_settings_without_hardware(self):
        args = type("Args", (), {
            "speed_scale": 0.55,
            "clearance_mm": 60.0,
            "clearance_m": 0.060,
            "speed_scale_cli": False,
            "clearance_mm_cli": False,
        })()
        self.assertEqual(_reload_operator_settings(args), 0)
        self.assertAlmostEqual(args.speed_scale, 0.38)
        self.assertAlmostEqual(args.clearance_mm, 100.0)
        self.assertAlmostEqual(args.clearance_m, 0.100)

    def test_sequence_ranges_use_easiest_first_numbering(self):
        self.assertEqual(_sequence_indices("1-3,8,9"), [1, 2, 3, 8, 9])

    def test_failed_action_can_be_retried_once_only_after_operator_choice(self):
        args = type("Args", (), {"speed_scale": 0.38})()
        with mock.patch(
            "tools.vega_competition_pipeline.run_wrist_part_calibration",
            side_effect=[2, 0],
        ) as runner, mock.patch("builtins.input", return_value="r"):
            self.assertEqual(
                _run_competition_action(args, "battery_size1", "pick_place", retries=1),
                0,
            )
        self.assertEqual(runner.call_count, 2)

    def test_possible_held_part_is_never_retried(self):
        args = type("Args", (), {"speed_scale": 0.38})()
        with mock.patch(
            "tools.vega_competition_pipeline.run_wrist_part_calibration",
            return_value=3,
        ) as runner:
            self.assertEqual(
                _run_competition_action(args, "battery_size1", "pick", retries=2),
                3,
            )
        self.assertEqual(runner.call_count, 1)

    def test_exception_from_action_is_bounded_and_operator_controlled(self):
        args = type("Args", (), {"speed_scale": 0.38})()
        with mock.patch(
            "tools.vega_competition_pipeline.run_wrist_part_calibration",
            side_effect=RuntimeError("transient setup failure"),
        ), mock.patch("builtins.input", return_value="s"):
            self.assertEqual(
                _run_competition_action(args, "battery_size1", "pick_place", retries=1),
                -1,
            )


if __name__ == "__main__":
    unittest.main()
