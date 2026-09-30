"""End-to-end operator recovery and pickup traces with no hardware access."""
import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from steadyhand.models import Pose
from steadyhand.wrist_part_profiles import load_profiles
from tools import vega_competition_pipeline as pipeline
from tools import vega_wrist_part_calibrate as wrist


class Robot:
    def __init__(self, pose):
        self.pose, self.trace = pose, []
        self.opening = .2
        self.result = {"gripped": True}
        self._kinematics = SimpleNamespace(solve=lambda target, seed: seed)
        self._gripper = SimpleNamespace(move_fraction=self.jaws, last_grip_result=lambda: self.result)

    def get_tcp_pose(self):
        return self.pose

    def _read_joint_positions(self):
        return [0.] * 7

    def move_tcp(self, pose, *, speed_scale):
        self.trace.append(("move", pose, speed_scale))
        self.pose = pose

    def connect_gripper(self):
        pass

    def jaws(self, fraction, *, speed):
        self.opening = fraction
        self.trace.append(("jaws", fraction, speed))
        return fraction

    def gripper_position(self):
        return self.opening

    def grip(self, part):
        self.trace.append(("grip", self.pose, part))

    def release_gripper(self, part):
        self.trace.append(("release", self.pose, part))
        return {"delta_fraction": .05}


class FinalWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = pipeline._load_runtime()
        self.cfg = self.runtime[0]["robot"]
        self.profiles = load_profiles(pipeline.ROOT / "calibration/wrist_part_profiles.json", self.cfg)
        self.output = io.StringIO()
        self.silence = redirect_stdout(self.output)
        self.silence.__enter__()
        self.addCleanup(self.silence.__exit__, None, None, None)

    def session(self, *, mode="test", competition=False, part="battery_size1"):
        args = SimpleNamespace(mode=mode, remote_safe=False, no_cv=False, place_cv=False,
            competition=competition, speed_scale=.38, profiles=str(Path(self.temp.name) / "profiles.json"),
            head_reacquire=False, execution_offsets=None, no_viewer=True)
        s = wrist.PartSession(args, Path(self.temp.name), self.cfg, copy.deepcopy(self.profiles))
        s.runtime = self.runtime
        s.targets = pipeline._task_targets(self.runtime, self.runtime[1], .100, use_profiles=False)
        s.robot = Robot(self.runtime[3])
        s.part = part
        s.event = mock.Mock()
        s.cameras = mock.Mock()
        s.capture = mock.Mock()
        s.frame = mock.Mock(return_value=(np.zeros((1536, 1920, 3), np.uint8), Path(self.temp.name) / "frame.png"))
        s._capture_drop_evidence = mock.Mock()
        return s, s.profiles["parts"][part]

    def test_reported_saved_template_error_does_not_exit_or_grasp(self):
        s, profile = self.session()
        with mock.patch.object(wrist.TemplateTracker, "from_saved_template", side_effect=RuntimeError(
                "Feature lost/ambiguous: score=0.835, margin=0.021")):
            result = s.begin_part(s.part, profile)
        self.assertIsNone(result)
        self.assertFalse(s.alignment_verified)
        self.assertFalse(s.holding)
        self.assertIsNone(s.tracker)
        self.assertNotIn("grip", [v[0] for v in s.robot.trace])
        np.testing.assert_allclose(s.robot.pose.position_m, s.coarse.position_m)
        self.assertIn("Arm left at its current hover", self.output.getvalue())

    def test_no_cv_inspection_starts_camera_and_camera_failure_keeps_prompt(self):
        s, _ = self.session(mode="drop")
        s.capture.return_value = np.zeros((8, 8, 3), np.uint8)
        s.capture.index = 1
        s._publish_live_image = mock.Mock()
        wrist.PartSession.frame(s, "inspection")
        s.cameras.connect.assert_called_once()
        s.capture.assert_called_once()
        s.frame.side_effect = RuntimeError("camera timeout")
        with mock.patch("builtins.input", side_effect=["image", "retry"]):
            self.assertTrue(s._pickup_plan_retry(wrist.PickupPreflightError("unreachable")))
        self.assertEqual(s.robot.trace, [])
        self.assertIn("IMAGE UNAVAILABLE", self.output.getvalue())

    def test_logged_8_45mm_servo_miss_replans_saved_grasp_after_stationary_check(self):
        s, profile = self.session(competition=True, part="gear_20teeth")
        s._load_saved_feature = lambda _: (setattr(s, "tracker", mock.Mock()),
            setattr(s, "reference_feature", tuple(profile["feature_uv"])),
            setattr(s, "goal", tuple(profile["goal_uv"])))
        s.robot.stationary_tcp_pose = mock.Mock(side_effect=s.robot.get_tcp_pose)
        def failed_servo(*args, **kwargs):
            current = s.robot.pose
            s.robot.pose = Pose((current.position_m[0] + .006, *current.position_m[1:]), current.quaternion_wxyz)
            raise wrist.ServoWaypointError("TCP missed servo waypoint by >8 mm", position_error_m=.008452,
                                            measured_pose=s.robot.pose)
        with mock.patch.object(wrist, "run_xy_servo", side_effect=failed_servo), \
                mock.patch("builtins.input", side_effect=AssertionError("competition prompted")):
            self.assertEqual(s.test(s.part, "pick", competition=True), 0)
        s.robot.stationary_tcp_pose.assert_called_once()
        self.assertFalse(s.motion_faulted)
        self.assertFalse(s.holding)
        grip = next(v for v in s.robot.trace if v[0] == "grip")
        np.testing.assert_allclose(grip[1].position_m[:2], s.grasp_target.position_m[:2])

    def test_servo_replan_requires_small_structured_error_and_stopped_arm(self):
        s, _ = self.session(competition=True)
        s.robot.stationary_tcp_pose = mock.Mock(side_effect=RuntimeError("stale state"))
        for error in (RuntimeError("TCP missed servo waypoint"),
                      wrist.ServoWaypointError("miss", position_error_m=.020, measured_pose=s.robot.pose)):
            self.assertFalse(s._recoverable_servo_miss(error))
        s.robot.stationary_tcp_pose.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "stale state"):
            s._recoverable_servo_miss(wrist.ServoWaypointError("miss", position_error_m=.009, measured_pose=s.robot.pose))
        self.assertEqual(s.robot.trace, [])

    def test_logged_bolt_probe_y_failure_runs_real_servo_then_saved_pick_without_prompt(self):
        s, profile = self.session(competition=True, part="bolt_8mm")
        original_profile = copy.deepcopy(profile)
        s._load_saved_feature = lambda _: (setattr(s, "tracker", mock.Mock()),
            setattr(s, "reference_feature", (906., 851.)), setattr(s, "goal", (916., 705.)))
        s.robot.stationary_tcp_pose = mock.Mock(side_effect=s.robot.get_tcp_pose)
        actual_servo = wrist.run_xy_servo
        actual_move = s.robot.move_tcp
        calls = []
        s.capture.return_value = np.zeros((1536, 1920, 3), np.uint8)
        def servo(robot, capture, **kwargs):
            tracker = SimpleNamespace(uv=(906., 851.), locate=lambda rgb: ((906., 851.), .956))
            kwargs["tracker_factory"] = lambda rgb, uv: tracker
            def move(pose, *, speed_scale):
                calls.append(pose)
                actual_move(pose, speed_scale=speed_scale)
                if len(calls) == 3:  # probe_x, return, then the reported missed probe_y
                    requested = (.5422002345160333, -.14552863612351902, .5711560682008909)
                    measured = (.5433259777349304, -.15538316249527073, .5704454995736536)
                    robot.pose = Pose(tuple(p + m - r for p, m, r in zip(pose.position_m, measured, requested)),
                                      pose.quaternion_wxyz)
            robot.move_tcp = move
            try:
                return actual_servo(robot, capture, **kwargs)
            finally:
                robot.move_tcp = actual_move
        with mock.patch.object(wrist, "run_xy_servo", side_effect=servo), \
                mock.patch("steadyhand.vision.wrist_servo.time.sleep"), \
                mock.patch("builtins.input", side_effect=AssertionError("competition prompted")):
            self.assertEqual(s.test(s.part, "pick", competition=True), 0)
        self.assertEqual(len(calls), 3)  # no invalid Jacobian or further visual correction
        s.robot.stationary_tcp_pose.assert_called_once()
        self.assertIn("9.94 mm", self.output.getvalue())
        self.assertFalse(s.motion_faulted)
        self.assertFalse(s.holding)
        self.assertEqual(s.status, "pick_complete_returned")
        self.assertEqual(profile, original_profile)
        grip = next(v for v in s.robot.trace if v[0] == "grip")
        np.testing.assert_allclose(grip[1].position_m[:2], s.grasp_target.position_m[:2])

    def test_drop_unreachable_transfer_keeps_controls_and_returns_held_part(self):
        s, profile = self.session(mode="drop", part="gear_20teeth")
        original = copy.deepcopy(s.profiles)
        target = s.profile_pose(profile, "place")
        original_move = s.move
        def move(pose, **kwargs):
            if pose == target:
                raise wrist.IKError("waypoint 6/7: position_error=0.04226 m")
            return original_move(pose, **kwargs)
        s.move = move
        with mock.patch("builtins.input", side_effect=["image", "status", "return"]):
            self.assertEqual(s.teach_drop(s.part, profile), 1)
        self.assertFalse(s.holding)
        self.assertEqual(s.status, "drop_cancelled_returned")
        self.assertEqual(s.profiles, original)
        self.assertEqual([v[0] for v in s.robot.trace].count("grip"), 1)
        self.assertEqual([v[0] for v in s.robot.trace].count("release"), 1)

    def test_competition_unreachable_place_returns_source_and_keeps_pickup(self):
        s, _ = self.session(competition=True)
        s.place = mock.Mock(side_effect=wrist.PlacementPreflightError("unreachable transfer"))
        with mock.patch("builtins.input", side_effect=AssertionError("competition prompted")):
            self.assertEqual(s.test(s.part, "pick_place", competition=True, no_cv=True), 0)
        self.assertFalse(s.holding)
        self.assertEqual(s.status, "pick_complete_place_blocked_returned")
        self.assertTrue(s.automatic_continuation_safe)

    def test_placement_preflight_rejects_before_transport_and_preserves_hold(self):
        s, profile = self.session()
        s.holding = True
        s.robot._kinematics.solve = mock.Mock(side_effect=wrist.IKError("unreachable"))
        with self.assertRaises(wrist.PlacementPreflightError):
            s.place(profile["place"])
        self.assertTrue(s.holding)
        self.assertEqual(s.robot.trace, [])

    def test_menu4_can_inspect_retry_rejected_feature_then_use_saved_and_grab(self):
        s, profile = self.session()
        original = copy.deepcopy(profile)
        with mock.patch.object(wrist.TemplateTracker, "from_saved_template", side_effect=RuntimeError(
                "Feature lost/ambiguous: score=0.835, margin=0.021")), \
                mock.patch("builtins.input", side_effect=["", "image", "center", "saved", "grab", "yes", "return"]):
            self.assertEqual(s.test(s.part, "pick"), 0)
        self.assertFalse(s.holding)
        self.assertEqual(s.status, "pick_complete_returned")
        self.assertEqual(profile, original)
        self.assertEqual(s.profiles["parts"][s.part], original)
        grip = next(v for v in s.robot.trace if v[0] == "grip")
        np.testing.assert_allclose(grip[1].position_m[:2], s.grasp_target.position_m[:2])
        self.assertAlmostEqual(grip[1].position_m[2] - s.surface(*grip[1].position_m[:2]), profile["grasp_clearance_m"])

    def test_competition_visual_failure_uses_recorded_pose_without_prompt(self):
        s, profile = self.session(competition=True)
        with mock.patch.object(wrist.TemplateTracker, "from_saved_template", side_effect=RuntimeError(
                "Feature lost/ambiguous: score=0.835, margin=0.021")), \
                mock.patch("builtins.input", side_effect=AssertionError("competition prompted")):
            self.assertEqual(s.test(s.part, "pick", competition=True), 0)
        self.assertFalse(s.holding)
        self.assertTrue(s.no_cv_mode)
        self.assertEqual(s.status, "pick_complete_returned")

    def test_camera_and_servo_waypoint_faults_never_turn_into_saved_pose_grabs(self):
        for error in (RuntimeError("Wrist camera timeout"), RuntimeError("joint state is stale")):
            with self.subTest(error=error):
                s, profile = self.session(competition=True)
                s.frame.side_effect = error
                with self.assertRaisesRegex(RuntimeError, str(error)):
                    s.begin_part(s.part, profile, competition=True)
                self.assertNotIn("grip", [v[0] for v in s.robot.trace])
        s, profile = self.session(competition=True)
        s._load_saved_feature = lambda _: (setattr(s, "tracker", mock.Mock()),
            setattr(s, "reference_feature", tuple(profile["feature_uv"])),
            setattr(s, "goal", tuple(profile["goal_uv"])))
        with mock.patch.object(wrist, "run_xy_servo", side_effect=RuntimeError("TCP missed servo waypoint by >8 mm")):
            with self.assertRaisesRegex(RuntimeError, "TCP missed"):
                s.test(s.part, "pick", competition=True)
        self.assertTrue(s.motion_faulted)
        self.assertFalse(s.automatic_continuation_safe)
        self.assertNotIn("grip", [v[0] for v in s.robot.trace])
        s, profile = self.session(competition=True)
        s.motion_faulted = True
        with self.assertRaisesRegex(RuntimeError, "motion fault"):
            s.use_recorded_grasp()
        with self.assertRaisesRegex(RuntimeError, "motion fault"):
            s.grab(profile["grasp_clearance_m"], allow_unverified=True)
        self.assertEqual(s.robot.trace, [])

    def test_visual_failure_after_probe_returns_to_recorded_grasp_before_competition_grip(self):
        s, profile = self.session(competition=True)
        s._load_saved_feature = mock.Mock()
        def failed_localize():
            current = s.robot.pose
            s.robot.pose = Pose((current.position_m[0] + .008, *current.position_m[1:]), current.quaternion_wxyz)
            raise RuntimeError("Feature did not return within 18 px")
        s.localize = failed_localize
        with mock.patch("builtins.input", side_effect=AssertionError("competition prompted")):
            self.assertEqual(s.test(s.part, "pick", competition=True), 0)
        grip = next(v for v in s.robot.trace if v[0] == "grip")
        np.testing.assert_allclose(grip[1].position_m[:2], s.grasp_target.position_m[:2])

    def test_empty_grip_releases_then_retreats_before_automatic_continuation(self):
        s, profile = self.session(competition=True)
        s.robot.result = {"gripped": False}
        with mock.patch("builtins.input", side_effect=AssertionError("competition prompted")):
            self.assertEqual(s.test(s.part, "pick", competition=True, no_cv=True), 2)
        self.assertFalse(s.holding)
        self.assertTrue(s.automatic_continuation_safe)
        self.assertEqual(s.robot.pose, s.pickup_hover)
        self.assertEqual(s.status, "pick_failed_returned")

    def test_unknown_gripper_result_is_not_released_or_retried(self):
        s, profile = self.session(competition=True)
        s.robot.result = None
        with self.assertRaisesRegex(RuntimeError, "unknown"):
            s.test(s.part, "pick", competition=True, no_cv=True)
        self.assertTrue(s.holding)
        self.assertFalse(s.automatic_continuation_safe)
        self.assertNotIn("release", [v[0] for v in s.robot.trace])

    def test_missing_template_offers_saved_pickup_instead_of_blocking_before_connection(self):
        s, profile = self.session()
        profile["template"]["path"] = str(Path(self.temp.name) / "missing.png")
        with mock.patch("builtins.input", return_value="abort"):
            self.assertEqual(s.test(s.part, "pick"), 1)
        self.assertFalse(s.holding)
        self.assertIn("Wrist template unavailable", self.output.getvalue())

    def test_pickup_preflight_rejection_keeps_supervised_test_inspectable(self):
        s, profile = self.session()
        s.grab = mock.Mock(side_effect=wrist.PickupPreflightError("descent is unreachable"))
        with mock.patch("builtins.input", side_effect=["grab", "image", "abort"]):
            self.assertEqual(s.test(s.part, "pick", no_cv=True), 1)
        self.assertEqual(s.status, "cancelled_before_grip")
        self.assertNotIn("grip", [v[0] for v in s.robot.trace])

    def test_all_existing_profiles_keep_no_cv_depth_yaw_jaws_and_return_anchor(self):
        for part in self.profiles["parts"]:
            with self.subTest(part=part):
                s, profile = self.session(competition=True, part=part)
                original = copy.deepcopy(profile)
                with mock.patch("builtins.input", side_effect=AssertionError("competition prompted")):
                    self.assertEqual(s.test(part, "pick", competition=True, no_cv=True), 0)
                np.testing.assert_allclose(s.grasp_target.position_m[:2], original["coarse_xy_m"], atol=1e-12)
                expected_quat = wrist._yaw_pose(self.runtime[3], original["yaw_deg"]).quaternion_wxyz
                np.testing.assert_allclose(s.grasp_target.quaternion_wxyz, expected_quat, atol=1e-12)
                grip = next(v for v in s.robot.trace if v[0] == "grip")
                release = next(v for v in s.robot.trace if v[0] == "release")
                self.assertEqual(grip[1], release[1])
                self.assertAlmostEqual(grip[1].position_m[2], s.surface(*original["coarse_xy_m"]) + original["grasp_clearance_m"])
                openings = [v[1] for v in s.robot.trace if v[0] == "jaws"]
                self.assertTrue(all(v == min(.60, max(.20, original["gripper_open_fraction"])) for v in openings))
                self.assertEqual(s.profiles["parts"][part], original)
                s.frame.assert_not_called()

    def test_task_test_and_competition_cv_paths_have_identical_motion_and_jaw_traces(self):
        traces = []
        for competition in (False, True):
            s, profile = self.session(competition=competition)
            def load_feature(_):
                s.tracker = mock.Mock()
                s.reference_feature = tuple(profile["feature_uv"])
                s.goal = tuple(profile["goal_uv"])
            s._load_saved_feature = load_feature
            def centered(robot, *args, **kwargs):
                x, y, _ = robot.pose.position_m
                robot.pose = Pose((x + .004, y + .002, s.surface(x + .004, y + .002) + .1), robot.pose.quaternion_wxyz)
                return {"status": "converged"}
            with mock.patch.object(wrist, "run_xy_servo", side_effect=centered), \
                    mock.patch("builtins.input", side_effect=(AssertionError("competition prompted") if competition else ["grab", "yes", "return"])):
                self.assertEqual(s.test(s.part, "pick", competition=competition), 0)
            traces.append(s.robot.trace)
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(self.output.getvalue().count("EXECUTION CENTER BACKOFF"), 2)

    def test_next_part_resets_previous_visual_success(self):
        s, profile = self.session()
        s.alignment_verified, s.goal, s.tracker = True, (1, 2), object()
        s.begin_part(s.part, profile, no_cv=True)
        self.assertFalse(s.alignment_verified)
        self.assertIsNone(s.goal)
        self.assertIsNone(s.tracker)

    def test_no_cv_start_does_not_require_wrist_bridge(self):
        s, _ = self.session()
        s.args.no_cv = True
        s.robot.connect = mock.Mock()
        s.retake = mock.Mock()
        s.start()
        s.robot.connect.assert_called_once()
        s.retake.assert_called_once()
        s.cameras.connect.assert_not_called()

    def test_drop_directional_adjustment_and_return(self):
        s, profile = self.session(mode="drop")
        with mock.patch("builtins.input", side_effect=["forward 2", "left 3", "undo", "return"]):
            self.assertEqual(s.teach_drop(s.part, profile), 1)
        self.assertFalse(s.holding)
        self.assertNotIn("COMMAND BLOCKED", self.output.getvalue())
        self.assertEqual(s.status, "drop_cancelled_returned")

    def test_minimal_menu_keeps_known_numbers_without_remote_variants(self):
        with mock.patch("builtins.input", return_value="0"):
            self.assertEqual(pipeline.main(["--check-only"]), 0)
        output = self.output.getvalue()
        self.assertIn("  3. Wrist", output)
        self.assertIn(" 10. Calibrate drop", output)
        self.assertNotIn("Remote-safe", output)
        self.assertNotIn(" 11.", output)
        self.assertNotIn(" 12.", output)
        self.assertNotIn("  9.", output)

    def test_test_command_uses_part_competition_cv_and_speed_settings(self):
        settings = pipeline._load_competition_actions()
        settings["parts"]["battery_size1"]["use_wrist_pick_cv"] = False
        settings["pipeline_speed_scale"] = .31
        with mock.patch.object(pipeline, "_load_competition_actions", return_value=settings):
            command = pipeline._task_test_command(SimpleNamespace(speed_scale=.38), "battery_size1", "pick")
        self.assertIn("--no-cv", command)
        self.assertNotIn("--competition", command)
        self.assertNotIn("--remote-safe", command)
        self.assertEqual(command[command.index("--speed-scale") + 1], "0.31")


if __name__ == "__main__":
    unittest.main()
