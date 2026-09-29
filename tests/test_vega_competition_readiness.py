import unittest

from tools.vega_competition_readiness import build_report


class CompetitionReadinessTests(unittest.TestCase):
    def test_report_is_hardware_free_and_validates_protected_inputs(self):
        report = build_report(run_tests=False)
        self.assertFalse(report["hardware_accessed"])
        self.assertTrue(report["checks"]["task_coordinates"]["passed"])
        self.assertTrue(report["checks"]["calibration"]["passed"])
        self.assertTrue(report["checks"]["protected_inputs_unchanged"]["passed"])
        eligible = report["checks"]["competition_plan"]["eligible_actions_now"]
        self.assertIn("battery_size1.pick_place", eligible)
        self.assertTrue(report["competition_actions_eligible"])


if __name__ == "__main__":
    unittest.main()
