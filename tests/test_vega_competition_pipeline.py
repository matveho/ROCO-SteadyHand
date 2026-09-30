import unittest
import json
import tempfile
from pathlib import Path
from unittest import mock

from tools.vega_competition_pipeline import (
    DEFAULT_PIPELINE_SPEED_SCALE,
    DEFAULT_PICK_PRIORITY,
    _load_competition_plan,
    _load_competition_actions,
    _reload_operator_settings,
    _run_competition_action,
    _priority_competition_actions,
    _all_calibrated_competition_run,
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

    def test_competition_actions_has_bounded_attempts_and_place_cv_switch(self):
        settings = _load_competition_actions()
        self.assertEqual(settings["max_attempts_per_action"], 3)
        self.assertEqual(settings["retries_per_action"], 2)
        self.assertTrue(settings["use_place_cv"])
        self.assertTrue(all(
            all(field in entry for field in (
                "enabled", "pick_enabled", "place_enabled",
                "use_wrist_pick_cv", "use_place_cv", "max_attempts",
            ))
            for entry in settings["parts"].values()
        ))

    def test_malformed_master_plan_is_rejected_before_reload(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "competition_plan.json"
            path.write_text(json.dumps({"schema_version": 1, "max_retries_per_part": 1.5}))
            with mock.patch("tools.vega_competition_pipeline.COMPETITION_PLAN", path):
                with self.assertRaises(ValueError):
                    _load_competition_plan()

    def test_competition_action_booleans_and_attempts_are_strict(self):
        source = json.loads(Path("configs/competition_actions.json").read_text())
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "competition_actions.json"
            bad = json.loads(json.dumps(source))
            bad["parts"]["battery_size1"]["enabled"] = "false"
            path.write_text(json.dumps(bad))
            with mock.patch("tools.vega_competition_pipeline.COMPETITION_ACTIONS", path):
                with self.assertRaisesRegex(ValueError, "enabled"):
                    _load_competition_actions()
            bad = json.loads(json.dumps(source))
            bad["parts"]["battery_size1"]["max_attempts"] = 1.5
            path.write_text(json.dumps(bad))
            with mock.patch("tools.vega_competition_pipeline.COMPETITION_ACTIONS", path):
                with self.assertRaisesRegex(ValueError, "max_attempts"):
                    _load_competition_actions()

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
        def run_once(command):
            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True, exist_ok=True)
            run_once.calls += 1
            if run_once.calls == 2:
                (output / "run_summary.json").write_text(
                    json.dumps({"status": "completed", "holding_may_be_true": False})
                )
                return 0
            (output / "run_summary.json").write_text(
                json.dumps({"status": "failed", "holding_may_be_true": False})
            )
            return 2
        run_once.calls = 0
        with mock.patch(
            "tools.vega_competition_pipeline.run_wrist_part_calibration",
            side_effect=run_once,
        ) as runner, mock.patch("builtins.input", return_value="r"):
            self.assertEqual(
                _run_competition_action(args, "battery_size1", "pick_place", retries=1),
                0,
            )
        self.assertEqual(runner.call_count, 2)

    def test_success_without_completed_run_summary_is_blocked(self):
        args = type("Args", (), {"speed_scale": 0.38})()
        with mock.patch(
            "tools.vega_competition_pipeline.run_wrist_part_calibration",
            return_value=0,
        ) as runner:
            self.assertEqual(
                _run_competition_action(args, "battery_size1", "pick_place", retries=0),
                2,
            )
        self.assertEqual(runner.call_count, 1)

    def test_pick_returned_status_is_success(self):
        args = type("Args", (), {"speed_scale": 0.38})()

        def run_pick(command):
            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "run_summary.json").write_text(
                json.dumps({"status": "pick_complete_returned", "holding_may_be_true": False})
            )
            return 0

        with mock.patch(
            "tools.vega_competition_pipeline.run_wrist_part_calibration",
            side_effect=run_pick,
        ):
            self.assertEqual(
                _run_competition_action(args, "battery_size1", "pick", retries=0),
                0,
            )

    def test_malformed_run_summary_blocks_success(self):
        args = type("Args", (), {"speed_scale": 0.38})()

        def run_bad(command):
            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "run_summary.json").write_text("not-json")
            return 0

        with mock.patch(
            "tools.vega_competition_pipeline.run_wrist_part_calibration",
            side_effect=run_bad,
        ):
            self.assertEqual(
                _run_competition_action(args, "battery_size1", "pick_place", retries=0),
                2,
            )

    def test_priority_check_only_lists_only_verified_actions_without_motion(self):
        args = type("Args", (), {"speed_scale": 0.38, "check_only": True})()
        plan = {
            "pick_priority": ["battery_size1", "gear_20teeth"],
            "default_action": "pick_place",
            "max_retries_per_part": 1,
        }
        profiles = {"parts": {
            "battery_size1": {
                "grasp_verified": True,
                "place": {"offset_board_xy_m": [0.0, 0.0], "clearance_m": 0.05, "yaw_deg": 0.0},
                "place_verified": True,
            },
        }}
        with mock.patch("tools.vega_competition_pipeline._load_competition_plan", return_value=plan), \
             mock.patch("tools.vega_competition_pipeline.load_bundle", return_value={"robot": {}}), \
             mock.patch("tools.vega_competition_pipeline.load_profiles", return_value=profiles), \
             mock.patch("tools.vega_competition_pipeline._run_competition_action") as runner:
            self.assertEqual(_priority_competition_actions(args), 0)
        runner.assert_not_called()

    def test_priority_pick_mode_accepts_grasp_verified_profile_without_place(self):
        args = type("Args", (), {"speed_scale": 0.38, "check_only": False})()
        plan = {
            "pick_priority": ["battery_size1", "gear_20teeth"],
            "default_action": "pick_place",
            "max_retries_per_part": 1,
        }
        profiles = {"parts": {
            "battery_size1": {"grasp_verified": True},
        }}
        with mock.patch("tools.vega_competition_pipeline._load_competition_plan", return_value=plan), \
             mock.patch("tools.vega_competition_pipeline.load_bundle", return_value={"robot": {}}), \
             mock.patch("tools.vega_competition_pipeline.load_profiles", return_value=profiles), \
             mock.patch("tools.vega_competition_pipeline._run_competition_action", return_value=0) as runner:
            self.assertEqual(_priority_competition_actions(args, action="pick"), 0)
        runner.assert_called_once_with(args, "battery_size1", "pick", retries=1)

    def test_priority_run_invokes_only_one_verified_action(self):
        args = type("Args", (), {"speed_scale": 0.38, "check_only": False})()
        plan = {
            "pick_priority": ["battery_size1", "gear_20teeth"],
            "default_action": "pick_place",
            "max_retries_per_part": 1,
        }
        profiles = {"parts": {
            "battery_size1": {
                "grasp_verified": True,
                "place": {"offset_board_xy_m": [0.0, 0.0], "clearance_m": 0.05, "yaw_deg": 0.0},
                "place_verified": True,
            },
        }}
        with mock.patch("tools.vega_competition_pipeline._load_competition_plan", return_value=plan), \
             mock.patch("tools.vega_competition_pipeline.load_bundle", return_value={"robot": {}}), \
             mock.patch("tools.vega_competition_pipeline.load_profiles", return_value=profiles), \
             mock.patch("tools.vega_competition_pipeline._run_competition_action", return_value=0) as runner:
            self.assertEqual(_priority_competition_actions(args), 0)
        runner.assert_called_once_with(args, "battery_size1", "pick_place", retries=1)

    def test_priority_skip_continues_to_next_verified_action(self):
        args = type("Args", (), {"speed_scale": 0.38, "check_only": False})()
        plan = {
            "pick_priority": ["battery_size1", "gear_20teeth"],
            "default_action": "pick_place",
            "max_retries_per_part": 1,
        }
        profile = {
            "grasp_verified": True,
            "place": {"offset_board_xy_m": [0.0, 0.0], "clearance_m": 0.05, "yaw_deg": 0.0},
            "place_verified": True,
        }
        profiles = {"parts": {"battery_size1": profile, "gear_20teeth": profile}}
        with mock.patch("tools.vega_competition_pipeline._load_competition_plan", return_value=plan), \
             mock.patch("tools.vega_competition_pipeline.load_bundle", return_value={"robot": {}}), \
             mock.patch("tools.vega_competition_pipeline.load_profiles", return_value=profiles), \
             mock.patch("tools.vega_competition_pipeline._run_competition_action", side_effect=[-1, 0]) as runner:
            self.assertEqual(_priority_competition_actions(args), 0)
        self.assertEqual(runner.call_count, 2)

    def test_all_calibrated_mode_ignores_config_disable_switches(self):
        args = type("Args", (), {"speed_scale": 0.38, "check_only": True})()
        settings = {
            "order": ["battery_size1"],
            "parts": {"battery_size1": {"enabled": False, "pick_enabled": False, "max_attempts": 1}},
        }
        profile = {"grasp_verified": True}
        with mock.patch("tools.vega_competition_pipeline._load_competition_actions", return_value=settings), \
             mock.patch("tools.vega_competition_pipeline.load_bundle", return_value={"robot": {}}), \
             mock.patch("tools.vega_competition_pipeline.load_profiles", return_value={"parts": {"battery_size1": profile}}):
            self.assertEqual(_all_calibrated_competition_run(args, place_cv=False), 0)

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
