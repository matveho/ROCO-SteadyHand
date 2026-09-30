import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from steadyhand.board_relative import (base_xy, local_xy, make_record, record_target,
    register, resolve_profile_target, resolve_record, snapshot, validate_record)
from steadyhand.models import Pose
from steadyhand.config import load_bundle
from steadyhand.board_calibration import load_board_calibration
from steadyhand.wrist_part_profiles import load_profiles
from tools import vega_competition_pipeline as pipeline
from tools.vega_wrist_part_calibrate import PartSession
from tools.vega_migrate_board_profiles import migrate


Q = (1., 0., 0., 0.)


class BoardRelativeTests(unittest.TestCase):
    def setUp(self):
        self.frame = ((.45, -.04), (0., -1.), (-1., 0.), {
            "coefficients": (-.2, .01, .6), "anchors": [],
            "registration": {"status": "fresh"}})
        self.reference = snapshot(self.frame)
        self.frame[3]["calibration_board_reference"] = self.reference
        self.frame[3]["calibration_ready_quaternion"] = Q
        self.ready = Pose((.45, -.04, .7), Q)
        self.pose = Pose((.34, -.14, .6), Q)
        self.record = make_record(self.reference,
            approach=record_target(self.reference, self.pose, source="test"),
            grasp=record_target(self.reference, Pose((.33, -.135, .6), Q), source="test"))
        self.profile = {"part": "battery_size1", "coarse_xy_m": [.34, -.14],
            "yaw_deg": 0., "pickup_board": self.record,
            "place": {"offset_board_xy_m": [.01, .02], "clearance_m": .02, "yaw_deg": 0.}}
        self.board = {"corners_base_m_coarse": {k: [*base_xy(self.reference, xy), .456]
            for k, xy in zip(("tl", "tr", "br", "bl"),
                ((-.193, -.193), (.193, -.193), (.193, .193), (-.193, .193)))}}

    def shifted(self, dx, dy, angle=0.):
        reference = copy.deepcopy(self.reference)
        a = math.radians(angle)
        rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        reference["center_base_xy_m"] = (np.array(self.frame[0]) + [dx, dy]).tolist()
        reference["board_x_unit_base_xy"] = (rotation @ self.frame[1]).tolist()
        reference["board_y_unit_base_xy"] = (rotation @ self.frame[2]).tolist()
        return reference, rotation

    def test_roundtrip_and_zero_movement_preserve_exact_target(self):
        np.testing.assert_allclose(base_xy(self.reference, local_xy(self.reference, self.pose.position_m[:2])),
                                   self.pose.position_m[:2], atol=1e-12)
        xy, quat = resolve_record(self.record, self.reference, kind="approach")
        np.testing.assert_allclose(xy, self.pose.position_m[:2], atol=1e-12)
        np.testing.assert_allclose(quat, Q, atol=1e-12)

    def test_cv_approach_and_final_grasp_both_follow_shifted_board(self):
        live, rotation = self.shifted(.01, -.01, 3.)
        frame = (live['center_base_xy_m'], live['board_x_unit_base_xy'],
                 live['board_y_unit_base_xy'], self.frame[3])
        before = copy.deepcopy(self.profile)
        for no_cv, kind in ((False, 'approach'), (True, 'grasp')):
            xy, _ = resolve_profile_target(self.profile, frame, self.ready, self.pose,
                                            action='pick', no_cv=no_cv)
            old_xy = self.record[kind]['tcp_pose']['position_m'][:2]
            expected = np.array(live['center_base_xy_m']) + rotation @ (np.array(old_xy) - self.frame[0])
            np.testing.assert_allclose(xy, expected, atol=1e-12)
        self.assertEqual(self.profile, before)

    def test_translation_grid_and_rotation_preserve_hand_and_distances(self):
        for dx in (-.01, 0., .01):
            for dy in (-.01, 0., .01):
                for angle in (-5., 0., 5.):
                    with self.subTest(dx=dx, dy=dy, angle=angle):
                        live, rotation = self.shifted(dx, dy, angle)
                        xy, quat = resolve_record(self.record, live, kind="approach")
                        expected = np.array(self.frame[0]) + [dx, dy] + rotation @ (np.array(self.pose.position_m[:2])-self.frame[0])
                        np.testing.assert_allclose(xy, expected, atol=1e-12)
                        self.assertAlmostEqual(math.dist(xy, live["center_base_xy_m"]),
                                               math.dist(self.pose.position_m[:2], self.frame[0]))
                        self.assertAlmostEqual(math.degrees(2*math.atan2(quat[3], quat[0])), angle)

    def test_corner_registration_identity_and_translation(self):
        for dx, dy, degrees in ((0, 0, 0), (.01, -.01, 0), (-.01, .01, 4)):
            board = copy.deepcopy(self.board)
            expected, rotation = self.shifted(dx, dy, degrees)
            for point in board["corners_base_m_coarse"].values():
                point[:2] = (np.array(self.frame[0]) + [dx, dy] + rotation @ (np.array(point[:2])-self.frame[0])).tolist()
            actual = register(self.reference, self.board, board)
            for name in ("center_base_xy_m", "board_x_unit_base_xy", "board_y_unit_base_xy"):
                np.testing.assert_allclose(actual[name], expected[name], atol=1e-10)
            self.assertEqual(actual["registration"]["status"], "fresh")

    def test_bad_corner_order_nan_and_large_shift_rejected(self):
        for kind in ("reflected", "half_turn", "nan", "large_shift", "scale", "plane"):
            board = copy.deepcopy(self.board)
            points = board["corners_base_m_coarse"]
            if kind == "reflected":
                points["tl"], points["tr"] = points["tr"], points["tl"]
                points["bl"], points["br"] = points["br"], points["bl"]
            elif kind == "half_turn":
                points["tl"], points["br"] = points["br"], points["tl"]
                points["bl"], points["tr"] = points["tr"], points["bl"]
            elif kind == "nan":
                points["tl"][0] = float("nan")
            else:
                for point in points.values():
                    if kind == "large_shift": point[0] += .2
                    elif kind == "scale": point[0] *= 2
                    else: point[2] += .02
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                register(self.reference, self.board, board)

    def test_record_tampering_and_handedness_fail_closed(self):
        record = copy.deepcopy(self.record)
        record["grasp"]["board_xy_m"][0] += .01
        with self.assertRaises(ValueError): validate_record(record)
        current = copy.deepcopy(self.reference)
        current["board_y_unit_base_xy"] = [1., 0.]
        with self.assertRaisesRegex(ValueError, "handedness"):
            resolve_record(self.record, current, kind="grasp")

    def test_approach_grasp_and_place_use_independent_references(self):
        placement_ref, _ = self.shifted(.01, 0.)
        release = Pose((.5, .1, .52), Q)
        self.profile["placement_board"] = make_record(placement_ref,
            release=record_target(placement_ref, release, source="release"))
        place_before = copy.deepcopy(self.profile["placement_board"])
        current, _ = self.shifted(.01, .01)
        frame = (current["center_base_xy_m"], current["board_x_unit_base_xy"], current["board_y_unit_base_xy"], self.frame[3])
        for no_cv, expected in ((False, [.35, -.13]), (True, [.34, -.125])):
            xy, _ = resolve_profile_target(self.profile, frame, self.ready, self.pose, action="pick", no_cv=no_cv)
            np.testing.assert_allclose(xy, expected)
        xy, _ = resolve_profile_target(self.profile, frame, self.ready, self.pose, action="place")
        np.testing.assert_allclose(xy, [.5, .11])
        self.profile["pickup_board"] = copy.deepcopy(self.record)
        self.assertEqual(self.profile["placement_board"], place_before)

    def test_legacy_position_uses_reference_not_current_frame(self):
        profile = dict(self.profile)
        profile.pop("pickup_board")
        live, _ = self.shifted(.01, .01)
        frame = (live["center_base_xy_m"], live["board_x_unit_base_xy"], live["board_y_unit_base_xy"], self.frame[3])
        xy, _ = resolve_profile_target(profile, frame, self.ready, self.pose, action="pick", no_cv=True)
        np.testing.assert_allclose(xy, [.35, -.13], atol=1e-12)

    def test_failed_registration_is_explicit_and_preserves_surface(self):
        plane = copy.deepcopy(self.frame[3])
        plane["calibration_camera_board"] = self.board
        plane["anchors"] = [{"x_m": .5, "y_m": 0., "residual_m": .003}]
        runtime = ({}, {}, (*self.frame[:3], plane), self.ready)
        fresh = pipeline._runtime_from_board_scene(runtime, {"board": self.board})
        self.assertEqual(fresh[2][3]["anchors"], plane["anchors"])
        np.testing.assert_allclose(fresh[2][0], self.frame[0])
        stale = pipeline._runtime_from_board_scene(fresh, {"board": {}})
        self.assertEqual(stale[2][3]["registration"]["status"], "stale")
        self.assertEqual(stale[2][:3], fresh[2][:3])

    def test_registration_recapture_is_bounded(self):
        runtime = ({}, {}, self.frame, self.ready)
        with mock.patch.object(pipeline, "_capture_downward_head_frame", return_value={}) as capture:
            result, _ = pipeline._capture_registered_board(None, runtime, floor_m=.45)
        self.assertEqual(capture.call_count, 2)
        self.assertEqual(result[2][3]["registration"]["status"], "stale")

    def test_head_motion_compensated_before_registration(self):
        from steadyhand.vision.board import head_left_optical_transform, pixels_to_horizontal_plane
        world = np.array(list(self.board["corners_base_m_coarse"].values()))
        records = []
        for head in ((.52, 0, 0), (.525, .005, -.003)):
            t = head_left_optical_transform(head)
            optical = (world-t[:3, 3]) @ t[:3, :3]
            pixels = optical[:, :2] / optical[:, 2, None] * 700 + [640, 360]
            projected = pixels_to_horizontal_plane(pixels, fx=700, fy=700, cx=640, cy=360,
                                                  T_base_camera=t, plane_z_m=.456)
            records.append({"corners_base_m_coarse": dict(zip(("tl", "tr", "br", "bl"), projected.tolist()))})
        fresh = register(self.reference, *records)
        np.testing.assert_allclose(fresh["center_base_xy_m"], self.frame[0], atol=1e-10)

    def test_position_menu_and_no_cv_session_resolve_identically(self):
        runtime = pipeline._load_runtime()
        profile = load_profiles(pipeline.ROOT / "calibration/wrist_part_profiles.json", runtime[0]["robot"])["parts"]["battery_size1"]
        menu = pipeline._task_targets(runtime, runtime[1], .100)
        session = PartSession.__new__(PartSession)
        session.part, session.runtime = "battery_size1", runtime
        session.targets = pipeline._task_targets(runtime, runtime[1], .100, use_profiles=False)
        for action in ("pick", "place"):
            target = session.profile_pose(profile, action, no_cv=True)
            self.assertEqual(target, menu[f"task.battery_size1.{action}"])

    def test_unchanged_image_preserves_calibrated_corner_hovers_too(self):
        runtime = pipeline._load_runtime()
        before = pipeline._board_targets(runtime, .1)
        after_runtime = pipeline._runtime_from_board_scene(runtime,
            {"board": runtime[2][3]["calibration_camera_board"]})
        after = pipeline._board_targets(after_runtime, .1)
        for name in before:
            np.testing.assert_allclose(after[name].position_m, before[name].position_m, atol=1e-12)

    def test_repository_migration_preserves_legacy_fields_and_reports_assumptions(self):
        root = pipeline.ROOT
        cfg = load_bundle("vega")["robot"]
        original = load_profiles(root / "calibration/wrist_part_profiles.json", cfg)
        cal = load_board_calibration(root / "calibration/vega_board_manual.json", cfg)
        _, ready = pipeline.configured_right_preset(cfg, "right_ready")
        with tempfile.TemporaryDirectory() as empty:
            migrated, report = migrate(Path(empty), original, cal, ready, cfg)
        for part, profile in original["parts"].items():
            self.assertEqual({k: migrated["parts"][part][k] for k in profile}, profile)
            self.assertIn("pickup_board", migrated["parts"][part])
        self.assertTrue(all(r["pickup"] == "assumed_calibration_reference" for r in report))

    def test_teaching_log_migration_has_no_jump_on_the_same_board_observation(self):
        root = pipeline.ROOT
        cfg = load_bundle("vega")["robot"]
        profile = load_profiles(root / "calibration/wrist_part_profiles.json", cfg)["parts"]["battery_size1"]
        cal = load_board_calibration(root / "calibration/vega_board_manual.json", cfg)
        runtime = pipeline._load_runtime()
        scene = {"board": copy.deepcopy(cal["raw"]["camera_board_read"])}
        for point in scene["board"]["corners_base_m_coarse"].values():
            point[0] += .01
        # A successful grasp was adjusted away from the originally saved hover.
        anchor = [.327, -.128]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/wrist_parts_teaching"
            run.mkdir(parents=True)
            (run / "battery_size1_handoff.json").write_text(json.dumps(profile))
            (run / "board_123.json").write_text(json.dumps(scene))
            (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in (
                {"part": "battery_size1", "event": "grasp_approach_readback", "anchor_xy_m": anchor},
                {"part": "battery_size1", "event": "grip_result", "result": {"gripped": True}},
            )))
            migrated, report = migrate(root, {"parts": {"battery_size1": profile}}, cal, runtime[3], cfg)
        self.assertTrue(report[0]["successful_grasp_recovered"])
        current = pipeline._runtime_from_board_scene(runtime, scene)
        record = migrated["parts"]["battery_size1"]["pickup_board"]
        xy, _ = resolve_record(record, snapshot(current[2]), kind="grasp")
        np.testing.assert_allclose(xy, anchor, atol=1e-12)
        xy, _ = resolve_record(record, snapshot(current[2]), kind="approach")
        np.testing.assert_allclose(xy, profile["coarse_xy_m"], atol=1e-12)

    def test_drop_save_keeps_pickup_and_invalidates_only_obsolete_place_cv(self):
        session = PartSession.__new__(PartSession)
        session.runtime, session.part = ({}, {}, self.frame, self.ready), "battery_size1"
        session.profiles = {"parts": {session.part: dict(self.profile, place_cv={"enabled": True})}}
        session.cfg, session.args = {}, SimpleNamespace(profiles="test.json")
        old_pickup = copy.deepcopy(session.profiles["parts"][session.part]["pickup_board"])
        settings = {"offset_board_xy_m": [.02, .03], "clearance_m": .03, "yaw_deg": 0.}
        with mock.patch("tools.vega_wrist_part_calibrate.save_profile", side_effect=lambda p, c, v: {"parts": {v["part"]: v}}):
            saved = session._save_drop_profile(Pose((.5, .1, .6), Q), settings, {})
        self.assertEqual(saved["pickup_board"], old_pickup)
        self.assertNotIn("place_cv", saved)
        xy, _ = resolve_record(saved["placement_board"], self.reference, kind="release")
        np.testing.assert_allclose(xy, [.5, .1])

    def test_successful_grasp_record_is_separate_from_feature_approach(self):
        session = PartSession.__new__(PartSession)
        session.runtime = ({}, {}, self.frame, self.ready)
        session.coarse = self.pose
        session.reference_pose = Pose((.35, -.15, .61), Q)
        session.successful_pickup_pose = Pose((.33, -.13, .6), Q)
        result = session.pickup_record()
        np.testing.assert_allclose(resolve_record(result, self.reference, kind="approach")[0], [.35, -.15])
        np.testing.assert_allclose(resolve_record(result, self.reference, kind="grasp")[0], [.33, -.13])

    def test_head_retry_preserves_taught_jaw_offset_from_part_center(self):
        session = PartSession.__new__(PartSession)
        session.runtime = pipeline._load_runtime()
        session.part = "battery_size1"
        session.cfg = session.runtime[0]["robot"]
        session.args = SimpleNamespace(head_reacquire=True, mode="test", speed_scale=.38)
        session.targets = pipeline._task_targets(session.runtime, session.runtime[1], .1, use_profiles=False)
        session.head_observations = {session.part: {"selection": "head_detection",
            "expected_xy_m": [.34, -.14], "selected_xy_m": [.35, -.135]}}
        profile = load_profiles(pipeline.ROOT / "calibration/wrist_part_profiles.json", session.cfg)["parts"][session.part]
        baseline = session.profile_pose(profile, "pick", no_cv=True)
        session.event = session.move = session.remote_checkpoint = mock.Mock()
        session._set_gripper_fraction = mock.Mock()
        session.holding = False
        with mock.patch.object(session, '_settle_pickup_hover'):
            session.begin_part(session.part, profile, competition=True, no_cv=True)
        np.testing.assert_allclose(session.grasp_target.position_m[:2],
            np.array(baseline.position_m[:2]) + [.01, .005])

    def test_legacy_release_uses_actual_recorded_pose_instead_of_new_task_offsets(self):
        self.profile["place_release_tcp_m"] = [.51, .12, .53]
        xy, _ = resolve_profile_target(self.profile, self.frame, self.ready,
            Pose((.6, .2, .6), Q), action="place")
        np.testing.assert_allclose(xy, [.51, .12])

    @mock.patch.object(pipeline, "_start_competition_progress", return_value=True)
    def test_competition_loops_do_not_continue_past_unknown_hardware_state(self, _progress):
        args = SimpleNamespace(speed_scale=.38, check_only=False)
        with mock.patch.object(pipeline, "_run_competition_action", return_value=2) as run:
            self.assertEqual(pipeline._configured_competition_run(args), 2)
            self.assertEqual(run.call_count, 1)
        with mock.patch.object(pipeline, "_run_competition_action", return_value=2) as run:
            self.assertEqual(pipeline._all_calibrated_competition_run(args, place_cv=False), 2)
            self.assertEqual(run.call_count, 1)

    def test_direct_hover_command_passes_live_registration_context(self):
        with mock.patch.object(pipeline, "_run_motion_targets", return_value=0) as run:
            self.assertEqual(pipeline.main(["--test-positions", "task.battery_size1.pick",
                                           "--confirm-physical-motion"]), 0)
        self.assertIsNotNone(run.call_args.kwargs["runtime"])
        self.assertIsNotNone(run.call_args.kwargs["task_data"])
        self.assertAlmostEqual(run.call_args.kwargs["clearance_m"], .1)


if __name__ == "__main__":
    unittest.main()
