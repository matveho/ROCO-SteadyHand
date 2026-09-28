import unittest

from tools.vega_motion_trace import JointSample, summarize_post_motion


TARGET = (0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7)
PRE_STAMP = 100
TOL = 0.005
ZERO_VEL = (0.0,) * 7


def sample(t, stamp, q, vel=ZERO_VEL):
    return JointSample(t, stamp, tuple(q), tuple(vel))


class MotionTraceSummaryTests(unittest.TestCase):
    def test_plugin_finished_and_target_reached(self):
        result = summarize_post_motion(
            plugin_state="finished",
            pre_motion_timestamp_ns=PRE_STAMP,
            target_joints_rad=TARGET,
            tolerance_rad=TOL,
            samples=[sample(0.0, 101, TARGET), sample(0.05, 102, TARGET)],
        )
        self.assertTrue(result["plugin_finished"])
        self.assertTrue(result["fresh_feedback"])
        self.assertTrue(result["target_reached"])
        self.assertEqual(result["first_reached_elapsed_s"], 0.0)

    def test_plugin_finished_with_stale_feedback(self):
        result = summarize_post_motion(
            plugin_state="finished",
            pre_motion_timestamp_ns=PRE_STAMP,
            target_joints_rad=TARGET,
            tolerance_rad=TOL,
            samples=[sample(0.0, PRE_STAMP, TARGET), sample(0.05, PRE_STAMP, TARGET)],
        )
        self.assertTrue(result["plugin_finished"])
        self.assertFalse(result["fresh_feedback"])
        self.assertTrue(result["target_reached"])
        self.assertEqual(result["timestamp_progressions"], 0)

    def test_plugin_finished_with_endpoint_outside_tolerance(self):
        q = list(TARGET)
        q[3] += 0.020
        result = summarize_post_motion(
            plugin_state="finished",
            pre_motion_timestamp_ns=PRE_STAMP,
            target_joints_rad=TARGET,
            tolerance_rad=TOL,
            samples=[sample(0.0, 101, q), sample(0.05, 102, q)],
        )
        self.assertTrue(result["plugin_finished"])
        self.assertTrue(result["fresh_feedback"])
        self.assertFalse(result["target_reached"])
        self.assertAlmostEqual(result["final_max_joint_error_rad"], 0.020)

    def test_endpoint_settles_into_tolerance_after_plugin_completion(self):
        q0 = list(TARGET)
        q1 = list(TARGET)
        q2 = list(TARGET)
        q0[0] += 0.020
        q1[0] += 0.008
        q2[0] += 0.004
        result = summarize_post_motion(
            plugin_state="finished",
            pre_motion_timestamp_ns=PRE_STAMP,
            target_joints_rad=TARGET,
            tolerance_rad=TOL,
            samples=[
                sample(0.00, 101, q0, (0.20,) + (0.0,) * 6),
                sample(0.05, 102, q1, (0.08,) + (0.0,) * 6),
                sample(0.10, 103, q2, (0.01,) + (0.0,) * 6),
            ],
        )
        self.assertTrue(result["plugin_finished"])
        self.assertTrue(result["fresh_feedback"])
        self.assertTrue(result["target_reached"])
        self.assertAlmostEqual(result["first_reached_elapsed_s"], 0.10)
        self.assertAlmostEqual(result["initial_max_joint_error_rad"], 0.020)
        self.assertAlmostEqual(result["final_max_joint_error_rad"], 0.004)


if __name__ == "__main__":
    unittest.main()
