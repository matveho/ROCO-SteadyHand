"""End-to-end operator recovery and pickup traces with no hardware access."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from steadyhand.models import Pose
from steadyhand.board_relative import make_record, record_target, snapshot
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

    def stationary_tcp_pose(self):
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
        before = self.opening
        self.opening = min(1., before + .04)
        return {"from_fraction": before, "to_fraction": self.opening,
                "delta_fraction": self.opening - before}


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

    def shifted_pickup_profile(self, s, profile):
        """A taught approach differs from the grasp, and the board has moved."""
        approach = s.profile_pose(profile, "pick")
        grasp = Pose((approach.position_m[0] + .02, approach.position_m[1] - .01,
                      approach.position_m[2]), approach.quaternion_wxyz)
        reference = snapshot(s.runtime[2])
        profile["pickup_board"] = make_record(reference,
            approach=record_target(reference, approach, source="feature_image_TCP"),
            grasp=record_target(reference, grasp, source="confirmed_grasp_hover_TCP"))
        center, ux, uy, plane = s.runtime[2]
        s.runtime = (*s.runtime[:2], ((center[0] + .01, center[1] - .005), ux, uy, plane), s.runtime[3])
        xy = (grasp.position_m[0] + .01, grasp.position_m[1] - .005)
        return Pose((*xy, s.surface(*xy) + .100), grasp.quaternion_wxyz)

    def test_open_pickup_calibration_goes_to_projected_grasp_without_servo_or_probes(self):
        s, profile = self.session(mode="calibrate")
        expected = self.shifted_pickup_profile(s, profile)
        original = copy.deepcopy(profile)
        template_path = wrist.ROOT / profile["template"]["path"]
        template_bytes = template_path.read_bytes()
        profiles_path = Path(s.args.profiles)
        profiles_path.write_text(json.dumps(s.profiles))
        profile_bytes = profiles_path.read_bytes()
        start = s.robot.pose
        # Even a leftover head-reacquire flag cannot override the taught hover.
        s.args.head_reacquire = True
        with mock.patch.object(wrist, "run_xy_servo") as servo, \
                mock.patch.object(s, "_load_saved_feature") as load_feature, \
                mock.patch.object(s, "teach_feature") as teach_feature, \
                mock.patch.object(s, "move", wraps=s.move) as move, \
                mock.patch.object(wrist.time, "sleep"):
            def first_prompt(prompt):
                self.assertEqual(prompt, f"{s.part}> ")
                servo.assert_not_called()
                load_feature.assert_not_called()
                teach_feature.assert_not_called()
                move.assert_called_once_with(expected)
                self.assertIsNone(s.tracker)
                self.assertIsNone(s.reference_feature)
                np.testing.assert_allclose(s.robot.pose.position_m, expected.position_m)
                # Every arm waypoint is on the one approach segment; none
                # leaves the reached hover for a probe or correction.
                moves = [v[1].position_m for v in s.robot.trace if v[0] == "move"]
                for index, xyz in enumerate(moves, 1):
                    np.testing.assert_allclose(xyz, np.array(start.position_m) +
                        (np.array(expected.position_m) - start.position_m) * index / len(moves))
                self.assertFalse(any("probe" in call.args[0] for call in s.event.call_args_list))
                return "abort"
            with mock.patch("builtins.input", side_effect=first_prompt) as prompt:
                self.assertEqual(s.teach(s.part, profile), 1)
            prompt.assert_called_once()
        self.assertEqual(profile, original)
        self.assertEqual(profiles_path.read_bytes(), profile_bytes)
        self.assertEqual(template_path.read_bytes(), template_bytes)

    def test_new_pickup_calibration_opens_manual_controls_at_nominal_hover(self):
        s, _ = self.session(mode="calibrate")
        with mock.patch.object(wrist, "run_xy_servo") as servo, \
                mock.patch.object(s, "_load_saved_feature") as load_feature, \
                mock.patch.object(s, "teach_feature") as teach_feature, \
                mock.patch.object(s, "move", wraps=s.move) as move, \
                mock.patch.object(wrist.time, "sleep"), \
                mock.patch("builtins.input", side_effect=["image", "center", "abort"]):
            self.assertEqual(s.teach(s.part), 1)
        servo.assert_not_called()
        load_feature.assert_not_called()
        teach_feature.assert_not_called()
        move.assert_called_once_with(s.targets[f"task.{s.part}.pick"])
        s.frame.assert_called_once_with("manual wrist capture")
        self.assertTrue(s.no_cv_mode)
        self.assertIn("CENTER NOT STARTED", self.output.getvalue())

    def test_explicit_center_loads_saved_feature_only_after_operator_prompt(self):
        s, profile = self.session(mode="calibrate")
        def load_feature(_):
            s.tracker = mock.Mock()
            s.reference_feature = tuple(profile["feature_uv"])
            s.goal = tuple(profile["goal_uv"])
        with mock.patch.object(s, "_load_saved_feature", side_effect=load_feature) as load, \
                mock.patch.object(wrist, "run_xy_servo", return_value={"status": "converged"}) as servo, \
                mock.patch.object(wrist.time, "sleep"):
            def command(prompt):
                self.assertEqual(prompt, f"{s.part}> ")
                if load.call_count == 0:
                    servo.assert_not_called()
                    return "center"
                servo.assert_called_once()
                self.assertTrue(s.alignment_verified)
                return "abort"
            with mock.patch("builtins.input", side_effect=command) as prompt:
                self.assertEqual(s.teach(s.part, profile), 1)
            self.assertEqual(prompt.call_count, 2)
        load.assert_called_once_with(profile)

    def test_manual_pickup_save_preserves_old_visual_and_placement_calibration(self):
        s, profile = self.session(mode="calibrate")
        self.shifted_pickup_profile(s, profile)
        original = copy.deepcopy(profile)
        approach = s.profile_pose(profile, "pick")
        template_path = wrist.ROOT / profile["template"]["path"]
        template_bytes = template_path.read_bytes()
        commands = ["back 2", "depth 90", "grab manual", "yes", "return", "save"]
        with mock.patch("builtins.input", side_effect=commands), \
                mock.patch.object(wrist, "_crop_template") as crop, \
                mock.patch.object(s, "_load_saved_feature") as load, \
                mock.patch.object(wrist, "run_xy_servo") as servo, \
                mock.patch.object(wrist.time, "sleep"):
            self.assertEqual(s.teach(s.part, profile), 0)
        saved = load_profiles(s.args.profiles, s.cfg)["parts"][s.part]
        for key, value in original.items():
            if key.startswith(("feature", "goal", "reference_match", "final_match", "place")) \
                    or key in ("template", "image_shape"):
                self.assertEqual(saved[key], value, key)
        self.assertAlmostEqual(saved["grasp_clearance_m"], .01)
        np.testing.assert_allclose(s.profile_pose(saved, "pick").position_m, approach.position_m)
        np.testing.assert_allclose(s.profile_pose(saved, "pick", no_cv=True).position_m,
                                   s.successful_pickup_pose.position_m)
        self.assertEqual(profile, original)
        self.assertEqual(template_path.read_bytes(), template_bytes)
        crop.assert_not_called()
        load.assert_not_called()
        servo.assert_not_called()

    def test_feature_is_replaced_only_after_explicit_selection_and_save(self):
        for save in (False, True):
            with self.subTest(save=save):
                s, profile = self.session(mode="calibrate")
                original = copy.deepcopy(profile)
                template_path = wrist.ROOT / profile["template"]["path"]
                template_bytes = template_path.read_bytes()
                profiles_path = Path(s.args.profiles)
                profiles_path.write_text(json.dumps(s.profiles))
                profile_bytes = profiles_path.read_bytes()
                tracker = mock.Mock()
                tracker.locate.return_value = ((200., 210.), .99)
                commands = ["feature"] + (["grab manual", "yes", "return", "save"] if save else ["abort"])
                with mock.patch("builtins.input", side_effect=commands), \
                        mock.patch.object(s, "select", return_value=(200., 210.)), \
                        mock.patch.object(wrist, "TemplateTracker", return_value=tracker), \
                        mock.patch.object(wrist, "run_xy_servo") as servo, \
                        mock.patch.object(wrist, "ROOT", Path(self.temp.name)), \
                        mock.patch.object(wrist, "load_board_calibration", return_value={"sha256": "a" * 64}), \
                        mock.patch.object(wrist.time, "sleep"):
                    self.assertEqual(s.teach(s.part, profile), 0 if save else 1)
                servo.assert_not_called()
                self.assertEqual(profile, original)
                self.assertEqual(template_path.read_bytes(), template_bytes)
                if save:
                    saved = load_profiles(profiles_path, s.cfg)["parts"][s.part]
                    self.assertEqual(saved["feature_uv"], [200., 210.])
                    self.assertEqual(saved["goal_uv"], [200., 210.])
                    self.assertNotEqual(saved["template"]["path"], profile["template"]["path"])
                    self.assertEqual(saved["template"]["sha256"], wrist.file_sha256(
                        Path(self.temp.name) / saved["template"]["path"]))
                    self.assertEqual(saved["place"], profile["place"])
                else:
                    self.assertEqual(profiles_path.read_bytes(), profile_bytes)

    def test_pickup_hover_settles_delayed_readback_without_commanding_more_motion(self):
        s, profile = self.session()
        target = s.profile_pose(profile, "pick")
        s.coarse = s.grasp_target = target
        delayed = Pose((target.position_m[0] + .006, *target.position_m[1:]),
                       target.quaternion_wxyz)
        with mock.patch.object(s.robot, 'get_tcp_pose', side_effect=[delayed, target, target, target]), \
                mock.patch.object(wrist.time, 'sleep') as sleep:
            s._settle_pickup_hover(target)
        self.assertEqual(sleep.call_count, 3)
        self.assertEqual(s.robot.trace, [])
        self.assertEqual(s.event.call_args.args[0], 'pickup_hover_reached')
        self.assertEqual(s.event.call_args.args[1]['position_error_m'], 0.)

    def test_pickup_hover_reports_persistent_error_without_blind_retry(self):
        s, profile = self.session()
        target = s.profile_pose(profile, "pick")
        s.coarse = s.grasp_target = target
        s.robot.pose = Pose((target.position_m[0] + .006, *target.position_m[1:]),
                            target.quaternion_wxyz)
        with mock.patch.object(wrist.time, 'sleep'):
            s._settle_pickup_hover(target)
        self.assertEqual(s.robot.trace, [])
        self.assertAlmostEqual(s.event.call_args.args[1]['position_error_m'], .006)

    def test_localization_uses_forty_corrections_for_both_controller_attempts(self):
        s, _ = self.session(mode='calibrate')
        s.tracker = mock.Mock()
        s.reference_feature = (100, 100)
        s.goal = (110, 110)
        with mock.patch.object(wrist, 'run_xy_servo', side_effect=[
                RuntimeError('Feature lost/ambiguous'), {'status': 'converged'}]) as servo:
            self.assertEqual(s.localize()['status'], 'converged')
        self.assertEqual([call.kwargs['max_iterations'] for call in servo.call_args_list], [40, 40])

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
        s.robot.stationary_tcp_pose = mock.Mock(side_effect=RuntimeError("Software E-stop active"))
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

    def test_unknown_gripper_result_is_released_only_after_stationary_verification(self):
        s, profile = self.session(competition=True)
        s.robot.result = None
        with mock.patch("builtins.input", side_effect=AssertionError("competition prompted")):
            self.assertEqual(s.test(s.part, "pick", competition=True, no_cv=True), 2)
        self.assertFalse(s.holding)
        self.assertTrue(s.automatic_continuation_safe)
        release = next(v[1] for v in s.robot.trace if v[0] == "release")
        self.assertAlmostEqual(release.position_m[2] - s.surface(*release.position_m[:2]), .040)
        self.assertFalse(s.pickup_completed)

    def test_failed_return_or_placement_recovers_at_40mm_and_keeps_pickup_credit(self):
        for action, failed_stage in (("pick", "return_part"), ("pick_place", "place")):
            with self.subTest(action=action):
                s, _ = self.session(competition=True)
                setattr(s, failed_stage, mock.Mock(side_effect=RuntimeError("operation failed")))
                with mock.patch("builtins.input", side_effect=AssertionError("competition prompted")):
                    self.assertEqual(s.test(s.part, action, competition=True, no_cv=True), 0)
                self.assertEqual(s.status, "pick_complete_recovery_released")
                self.assertTrue(s.pickup_completed)
                self.assertTrue(s.automatic_continuation_safe)
                self.assertFalse(s.holding)
                releases = [v[1] for v in s.robot.trace if v[0] == "release"]
                self.assertEqual(len(releases), 1)
                self.assertAlmostEqual(releases[0].position_m[2] - s.surface(*releases[0].position_m[:2]), .04)
                self.assertEqual(sum(v[0] == "grip" for v in s.robot.trace), 1)
                self.assertAlmostEqual(s.robot.pose.position_m[2] - s.surface(*s.robot.pose.position_m[:2]), .1)

    def test_failed_empty_pickup_return_uses_same_recovery_before_retry(self):
        s, _ = self.session(competition=True)
        s.robot.result = {"gripped": False}
        s.return_part = mock.Mock(side_effect=RuntimeError("return failed"))
        self.assertEqual(s.test(s.part, "pick", competition=True, no_cv=True), 2)
        self.assertEqual(s.status, "failed_recovered_empty")
        self.assertTrue(s.automatic_continuation_safe)
        self.assertFalse(s.holding)
        self.assertFalse(s.pickup_completed)

    def test_recovery_never_moves_or_opens_with_stale_moving_or_estop_state(self):
        for reason in ("stale state", "arm moving", "Physical E-stop active", "Software E-stop active",
                       "joint outside limits", "motion handle active", "communication unavailable"):
            with self.subTest(reason=reason):
                s, _ = self.session(competition=True)
                s.robot.result = None
                s.robot.stationary_tcp_pose = mock.Mock(side_effect=RuntimeError(reason))
                with self.assertRaisesRegex(RuntimeError, "unknown"):
                    s.test(s.part, "pick", competition=True, no_cv=True)
                # The only grip is the original attempt. There is no recovery
                # motion after that grip, and no clearing E-stop or retry.
                index = next(i for i, v in enumerate(s.robot.trace) if v[0] == "grip")
                self.assertEqual(s.robot.trace[index + 1:], [])
                self.assertTrue(s.holding)
                self.assertTrue(s.recovery_blocked)
                self.assertFalse(s.automatic_continuation_safe)

    def test_recovery_refuses_outside_board_and_unreachable_release_before_motion(self):
        s, profile = self.session(competition=True)
        s.holding = True
        s.robot.pose = Pose((2., 2., .6), s.runtime[3].quaternion_wxyz)
        with self.assertRaisesRegex(RuntimeError, "outside the board"):
            s.recover_action("failed return")
        self.assertEqual(s.robot.trace, [])
        s.robot.pose = s.profile_pose(profile, "pick", no_cv=True)
        s.robot._kinematics.solve = mock.Mock(side_effect=wrist.IKError("unreachable release"))
        with self.assertRaisesRegex(wrist.IKError, "unreachable release"):
            s.recover_action("failed return")
        self.assertEqual(s.robot.trace, [])
        self.assertTrue(s.holding)

    def test_unconfirmed_recovery_opening_preserves_hold_and_prevents_retreat(self):
        s, profile = self.session(competition=True)
        s.robot.pose = s.profile_pose(profile, "pick", no_cv=True)
        s.holding = True
        s.robot.release_gripper = mock.Mock(return_value={"from_fraction": .2, "to_fraction": .24})
        with self.assertRaisesRegex(RuntimeError, "opening was not confirmed"):
            s.recover_action("release failure")
        s.robot.release_gripper.assert_called_once()
        self.assertTrue(s.holding)
        self.assertTrue(s.recovery_release_attempted)
        self.assertFalse(s.automatic_continuation_safe)
        self.assertAlmostEqual(s.robot.pose.position_m[2] - s.surface(*s.robot.pose.position_m[:2]), .04)

    def test_checkpoint_recovery_skips_camera_clear_and_ready_and_never_clears_estop(self):
        s, profile = self.session(competition=True)
        s.robot.pose = s.profile_pose(profile, "pick", no_cv=True)
        s.robot.connect = mock.Mock()
        s.robot.close = mock.Mock()
        previous = Path(self.temp.name) / "previous"
        previous.mkdir()
        (previous / "run_summary.json").write_text(json.dumps({
            "part": s.part, "action": "pick_place", "holding_may_be_true": True,
            "pickup_completed": True, "board_reference": snapshot(s.runtime[2])}))
        s.output = Path(self.temp.name) / "recovery"
        with mock.patch.object(wrist, "PartSession", return_value=s) as factory:
            summary = wrist.recover_competition_checkpoint(s.part, "pick_place", previous, s.output,
                                                            speed_scale=.38)
        self.assertFalse(factory.call_args.args[2]["auto_clear_software_estop_on_connect"])
        self.assertTrue(factory.call_args.args[2]["gripper"]["require_cached_calibration"])
        s.robot.connect.assert_called_once()
        s.robot.close.assert_called_once()
        self.assertTrue(summary["automatic_continuation_safe"])
        self.assertTrue(summary["recovery_state_verified"])
        self.assertFalse(summary["holding_may_be_true"])
        self.assertTrue(summary["pickup_completed"])
        # Recovery retains XY throughout and never commands the ready preset.
        for entry in s.robot.trace:
            if entry[0] == "move":
                np.testing.assert_allclose(entry[1].position_m[:2], profile["coarse_xy_m"])
        s.frame.assert_not_called()

    def test_shutdown_error_invalidates_safe_summary_even_after_completed_recovery(self):
        s, _ = self.session(competition=True)
        s.status = "pick_complete_recovery_released"
        s.automatic_continuation_safe = s.recovery_state_verified = s.pickup_completed = True
        s.robot.close = mock.Mock(side_effect=RuntimeError("shutdown unavailable"))
        with self.assertRaisesRegex(RuntimeError, "shutdown unavailable"):
            s.close()
        summary = json.loads((s.output / "run_summary.json").read_text())
        self.assertIn("shutdown unavailable", summary["cleanup_error"])
        self.assertFalse(summary["automatic_continuation_safe"])
        self.assertFalse(summary["recovery_state_verified"])
        self.assertTrue(summary["pickup_completed"])

    def test_material_servo_miss_restarts_action_only_after_fresh_stationary_checks(self):
        sessions = []
        def child(command):
            s, profile = self.session(competition=True)
            s.output = Path(command[command.index("--output") + 1])
            s.output.mkdir(parents=True)
            s.robot.close = mock.Mock()
            s._load_saved_feature = lambda _: (setattr(s, "tracker", mock.Mock()),
                setattr(s, "reference_feature", tuple(profile["feature_uv"])),
                setattr(s, "goal", tuple(profile["goal_uv"])))
            sessions.append(s)
            try:
                return s.test(s.part, "pick", competition=True, no_cv="--no-cv" in command)
            finally:
                s.close()
        error = wrist.ServoWaypointError("TCP missed servo waypoint by >8 mm",
                                        position_error_m=.014, measured_pose=self.runtime[3])
        with mock.patch.object(pipeline, "ROOT", Path(self.temp.name)), \
                mock.patch.object(pipeline, "run_wrist_part_calibration", side_effect=child) as runner, \
                mock.patch.object(wrist, "run_xy_servo", side_effect=error), \
                mock.patch.object(wrist.time, "sleep"), \
                mock.patch("builtins.input", side_effect=AssertionError("automatic run prompted")):
            self.assertEqual(pipeline._run_competition_action(SimpleNamespace(speed_scale=.38),
                             "battery_size1", "pick", retries=1), 0)
        self.assertEqual(runner.call_count, 2)
        self.assertNotIn("grip", [v[0] for v in sessions[0].robot.trace])
        self.assertTrue(sessions[0].recovery_state_verified)
        self.assertFalse(sessions[0].motion_faulted)
        self.assertIn("--no-cv", runner.call_args.args[0])
        self.assertEqual(sum(v[0] == "grip" for v in sessions[1].robot.trace), 1)

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
            with mock.patch.object(wrist, "run_xy_servo", side_effect=centered) as servo, \
                    mock.patch("builtins.input", side_effect=(AssertionError("competition prompted") if competition else ["grab", "yes", "return"])):
                self.assertEqual(s.test(s.part, "pick", competition=competition), 0)
            servo.assert_called_once()
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

    def test_manual_pickup_jogs_can_exceed_old_radius_and_still_undo(self):
        s, _ = self.session(mode='calibrate')
        start = s.robot.pose
        s.coarse = start
        commands = ['back 30', 'back 20', 'back 10', 'right 20', 'right 31', 'undo', 'abort']
        with mock.patch.object(s, 'begin_part'), mock.patch.object(s, 'teach_feature'), \
                mock.patch('builtins.input', side_effect=commands):
            self.assertEqual(s.teach(s.part), 1)
        np.testing.assert_allclose(s.robot.pose.position_m[:2],
                                   [start.position_m[0] - .06, start.position_m[1]])
        self.assertTrue(any(np.linalg.norm(np.array(v[1].position_m[:2]) - start.position_m[:2]) > .06
                            for v in s.robot.trace if v[0] == 'move'))
        self.assertEqual(len(s.history), 3)  # four accepted jogs, then undo; 31 mm rejected
        self.assertNotIn('Adjustment exceeds 60 mm', self.output.getvalue())

    def test_faster_motion_preserves_requested_lower_speed_and_trajectory(self):
        traces = []
        for requested, slow, expected in ((.60, False, .60), (.60, True, .42), (.25, True, .25)):
            with self.subTest(requested=requested, slow=slow):
                s, _ = self.session()
                s.args.speed_scale = requested
                start = s.robot.pose
                target = Pose((*start.position_m[:2], start.position_m[2] - .06),
                              start.quaternion_wxyz)
                s.move(target, slow=slow)
                s.move(start, slow=slow)
                moves = [v for v in s.robot.trace if v[0] == 'move']
                self.assertTrue(moves)
                self.assertTrue(all(v[2] == expected for v in moves))
                traces.append([v[1] for v in moves])
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(traces[1], traces[2])

    def test_competition_menu_uses_requested_numbers(self):
        with mock.patch("builtins.input", return_value="0"):
            self.assertEqual(pipeline.main(["--check-only"]), 0)
        output = self.output.getvalue()
        self.assertIn("  3. Calibrate pickup", output)
        self.assertIn("  4. Calibrate placement", output)
        self.assertNotIn("Remote-safe", output)
        self.assertNotIn(" 11.", output)
        self.assertNotIn(" 12.", output)
        self.assertIn("  0. Exit", output)
        self.assertNotIn("Board calibration", output)
        self.assertNotIn("Rollback versions", output)

    def test_test_command_uses_part_competition_cv_and_speed_settings(self):
        settings = pipeline._load_competition_actions()
        settings["parts"]["battery_size1"]["use_wrist_pick_cv"] = False
        settings["pipeline_speed_scale"] = .31
        args = SimpleNamespace(speed_scale=.38, check_only=False)
        with mock.patch.object(pipeline, "_load_competition_actions", return_value=settings), \
                mock.patch.object(pipeline, "_start_competition_progress", return_value=True), \
                mock.patch.object(pipeline, "_run_competition_action", return_value=0) as run:
            self.assertEqual(pipeline._configured_competition_run(
                args, selected_parts=["battery_size1"], action_override="pick"), 0)
        self.assertEqual(run.call_args.args[1:], ("battery_size1", "pick"))
        self.assertTrue(run.call_args.kwargs["no_cv"])
        self.assertEqual(run.call_args.kwargs["retries"], settings["parts"]["battery_size1"]["max_attempts"] - 1)
        self.assertEqual(args.speed_scale, .31)


if __name__ == "__main__":
    unittest.main()
