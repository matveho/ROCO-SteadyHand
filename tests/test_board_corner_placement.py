import copy
import math
from pathlib import Path
import unittest
from unittest import mock

import cv2
import numpy as np

from steadyhand.models import Pose
from steadyhand.vision.board_corners import (
    CornerServo, CornerVisualError, correction, make_reference, match_corners,
    rotate_reference,
)


IMAGE = Path(__file__).parent / "fixtures" / "placement_board_corners.jpg"


class BoardCornerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rgb = cv2.cvtColor(cv2.imread(str(IMAGE)), cv2.COLOR_BGR2RGB)

    def translated(self, dx, dy, image=None):
        image = self.rgb if image is None else image
        return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]),
                             (image.shape[1], image.shape[0]))

    def test_actual_fisheye_image_four_corners_not_held_part_or_jaws(self):
        reference = make_reference(self.rgb)
        expected = [(1052, 110), (432, 338), (1180, 867), (511, 925)]
        self.assertEqual(len(reference["corners"]), 4)
        for corner, point in zip(reference["corners"], expected):
            self.assertLess(math.dist(corner["uv"], point), 3)

    def test_translation_and_lighting_preserve_corner_identity(self):
        reference = make_reference(self.rgb)
        image = np.clip(self.translated(26, -18).astype(float) * .85 + 9, 0, 255).astype(np.uint8)
        matched = match_corners(image, reference)
        self.assertEqual(len(matched), 4)
        for corner in reference["corners"]:
            np.testing.assert_allclose(np.asarray(matched[corner["id"]]["uv"]) - corner["uv"],
                                       [26, -18], atol=1.5)

    def test_occluded_corners_are_missing_not_inferred_or_replaced_by_claw(self):
        reference = make_reference(self.rgb)
        image = self.rgb.copy()
        image[750:, 400:800] = 45
        image[:240, 950:1200] = 45
        matched = match_corners(image, reference)
        self.assertNotIn("C1", matched)
        self.assertNotIn("C4", matched)
        self.assertIn("C2", matched)
        self.assertIn("C3", matched)

    def test_single_visible_corner_can_match(self):
        reference = make_reference(self.rgb)
        image = self.rgb.copy()
        image[700:] = 40
        image[:480, :800] = 40
        matched = match_corners(image, reference)
        self.assertEqual(set(matched), {"C1"})

    def test_missing_board_and_changed_resolution_rejected(self):
        reference = make_reference(self.rgb)
        for image in (np.full_like(self.rgb, 40), self.rgb[::2, ::2]):
            with self.assertRaises(CornerVisualError):
                match_corners(image, reference)

    def test_manual_click_selects_real_boundary_not_gear_tooth(self):
        reference = make_reference(self.rgb, [(1051, 110)])
        self.assertEqual(len(reference["corners"]), 1)
        with self.assertRaises(CornerVisualError):
            make_reference(self.rgb, [(870, 142)])

    def calibrated(self):
        reference = make_reference(self.rgb)
        for corner in reference["corners"]:
            corner["jacobian_px_per_m"] = [[0, 2000], [1800, 0]]
        reference.update(reference_clearance_m=.1, reference_quaternion_wxyz=[1, 0, 0, 0],
                         reference_board={"board_x_unit_base_xy": [1, 0]})
        return reference

    def test_fisheye_corner_specific_jacobians_not_average_pixel_translation(self):
        reference = self.calibrated()
        desired = np.array([.004, -.003])
        matches = {}
        for i, corner in enumerate(reference["corners"]):
            matrix = np.asarray(corner["jacobian_px_per_m"]) * (1+i*.3)
            corner["jacobian_px_per_m"] = matrix.tolist()
            matches[corner["id"]] = {"uv": np.asarray(corner["uv"]) - matrix @ desired}
        delta, _, keys = correction(reference, matches)
        np.testing.assert_allclose(delta, desired)
        self.assertEqual(len(keys), 4)

    def test_disagreeing_corners_cannot_authorize_motion(self):
        reference = self.calibrated()
        matches = {c["id"]: {"uv": np.asarray(c["uv"]) + [i*25, 0]}
                   for i, c in enumerate(reference["corners"][:2])}
        with self.assertRaises(CornerVisualError):
            correction(reference, matches)

    def test_single_corner_has_smaller_correction_bound(self):
        reference = self.calibrated()
        corner = reference["corners"][0]
        with self.assertRaisesRegex(CornerVisualError, "10 mm"):
            correction(reference, {corner["id"]: {"uv": np.asarray(corner["uv"]) + [25, 0]}})

    def test_board_yaw_rotates_mapping_without_mutating_profile(self):
        reference = self.calibrated()
        before = copy.deepcopy(reference)
        result = rotate_reference(reference, {"board_x_unit_base_xy": [0, 1]})
        np.testing.assert_allclose(result["corners"][0]["jacobian_px_per_m"],
                                   [[-2000, 0], [0, 1800]], atol=1e-10)
        self.assertEqual(reference, before)

    def simulated(self, offset=(0, 0)):
        robot = mock.Mock()
        robot.pose = Pose((.4, -.1, .6), (1, 0, 0, 0))
        robot.stationary_tcp_pose.side_effect = lambda **kwargs: robot.pose
        def move(pose):
            robot.pose = pose
        motion = mock.Mock(side_effect=move)
        def capture():
            shift = np.array([[0, 2000], [1800, 0]]) @ (np.asarray(robot.pose.position_m[:2]) - [.4, -.1]) + offset
            return self.translated(*shift)
        return robot, motion, capture

    def test_runtime_direct_corrections_converge_without_probes(self):
        robot, motion, capture = self.simulated(offset=[-12, 9])
        servo = CornerServo(robot, motion, capture, lambda x, y: .5)
        result = servo.align(self.calibrated())
        self.assertEqual(result["status"], "converged")
        np.testing.assert_allclose(robot.pose.position_m[:2], [.395, -.094], atol=.002)
        self.assertLessEqual(motion.call_count, 4)
        for call in motion.call_args_list:
            self.assertAlmostEqual(call.args[0].position_m[2], .6)

    def test_already_aligned_run_commands_no_movement(self):
        robot, motion, capture = self.simulated()
        CornerServo(robot, motion, capture, lambda x, y: .5).align(self.calibrated())
        motion.assert_not_called()

    def test_reported_hdmi_residual_converges_despite_controller_deadband(self):
        # Exact C3 mapping and final residual from the 20260913T112707 HDMI
        # log. The old 0.6 gain issued 1.29 mm repeatedly with <0.04 mm motion.
        matrix = np.array([[-302.7382252138453, 1166.024446832604],
                           [1584.0785391979039, -149.55502599383064]])
        residual = np.array([-.0007186840509216132, -.0020243618950434813])
        for deadband in (.0015, .0025):
            with self.subTest(deadband=deadband):
                robot, motion, capture = self.simulated()
                desired = np.asarray(robot.pose.position_m[:2]) + residual
                reference = self.calibrated()
                reference["corners"] = reference["corners"][:1]
                corner = reference["corners"][0]
                corner["jacobian_px_per_m"] = matrix.tolist()
                def move(pose):
                    if math.dist(robot.pose.position_m, pose.position_m) > deadband:
                        robot.pose = pose
                motion.side_effect = move
                events = mock.Mock()
                servo = CornerServo(robot, motion, capture, lambda x, y: .5, events)
                def observe(_):
                    uv = np.asarray(corner["uv"]) + matrix @ (np.asarray(robot.pose.position_m[:2]) - desired)
                    return {corner["id"]: {"uv": uv.tolist(), "score": .997}}, robot.pose
                servo.observe = mock.Mock(side_effect=observe)
                before = copy.deepcopy(reference)
                result = servo.align(reference)
                self.assertEqual(result["status"], "converged")
                self.assertLessEqual(result["estimated_error_m"], .001)
                self.assertLessEqual(result["error_px"], 2.)
                self.assertLessEqual(motion.call_count, 2)
                self.assertLessEqual(math.dist(robot.pose.position_m[:2], desired), .001)
                self.assertEqual(reference, before)
                controls = [c.args[1] for c in events.call_args_list if c.args[0] == "place_corner_control"]
                self.assertTrue(all(c["deadband_compensation_m"] <= .00075 for c in controls))

    def test_unresponsive_arm_does_not_accumulate_unbounded_corner_commands(self):
        robot, motion, capture = self.simulated(offset=[4, 0])
        motion.side_effect = lambda target: None
        events = mock.Mock()
        with self.assertRaisesRegex(CornerVisualError, "stalled"):
            CornerServo(robot, motion, capture, lambda x, y: .5, events).align(self.calibrated())
        self.assertEqual(motion.call_count, 3)
        self.assertLessEqual(max(math.dist(c.args[0].position_m, robot.pose.position_m)
                                 for c in motion.call_args_list), .0028)

    def test_large_real_motion_without_visual_progress_does_not_get_compensation(self):
        robot, motion, capture = self.simulated()
        reference = self.calibrated()
        events = mock.Mock()
        servo = CornerServo(robot, motion, capture, lambda x, y: .5, events)
        fixed = {c["id"]: {"uv": (np.asarray(c["uv"]) + [4, 0]).tolist()}
                 for c in reference["corners"]}
        servo.observe = mock.Mock(side_effect=lambda _: (fixed, robot.pose))
        with self.assertRaisesRegex(CornerVisualError, "stalled"):
            servo.align(reference)
        controls = [c.args[1] for c in events.call_args_list if c.args[0] == "place_corner_control"]
        self.assertTrue(all(c["deadband_compensation_m"] == 0 for c in controls))

    def test_one_time_teaching_uses_measured_movement_and_returns_to_anchor(self):
        robot, motion, capture = self.simulated()
        reference, _ = CornerServo(robot, motion, capture, lambda x, y: .5).teach(anchor_xy=[.4, -.1])
        self.assertEqual(len(reference["corners"]), 4)
        self.assertEqual(motion.call_count, 4)
        np.testing.assert_allclose(robot.pose.position_m, [.4, -.1, .6], atol=1e-8)
        for corner in reference["corners"]:
            np.testing.assert_allclose(corner["jacobian_px_per_m"], [[0, 2000], [1800, 0]], atol=70)

    def test_teaching_probes_exceed_small_motion_deadband_without_expanding_radius(self):
        robot, motion, capture = self.simulated()
        def move(pose):
            # Reproduce a controller accepting a small target without moving:
            # the old 6 mm probes could never learn an image Jacobian here.
            if math.dist(pose.position_m, robot.pose.position_m) > .007:
                robot.pose = pose
        motion.side_effect = move
        events = mock.Mock()
        reference, _ = CornerServo(robot, motion, capture, lambda x, y: .5, events).teach()
        self.assertEqual(len(reference["corners"]), 4)
        self.assertEqual(motion.call_count, 4)
        self.assertAlmostEqual(max(math.dist(c.args[0].position_m[:2], [.4, -.1])
                                   for c in motion.call_args_list), .010)
        measurements = [c.args[1] for c in events.call_args_list if c.args[0] == "place_corner_measurement"]
        self.assertEqual([m["stage"] for m in measurements],
                         ["reference", "probe_x", "return_x", "probe_y", "return_y"])

    def test_static_corner_images_cannot_teach_a_mapping_and_report_pixel_deltas(self):
        robot, motion, _ = self.simulated()
        events = mock.Mock()
        with self.assertRaisesRegex(CornerVisualError, "Feature barely moved"):
            CornerServo(robot, motion, lambda: self.rgb, lambda x, y: .5, events).teach()
        rejected = [c.args[1] for c in events.call_args_list if c.args[0] == "place_corner_rejected"]
        self.assertEqual(len(rejected), 4)
        for r in rejected:
            np.testing.assert_allclose(r["probe_pixel_deltas"], 0)
            np.testing.assert_allclose(r["probe_xy_deltas_m"], [[.01, 0], [0, .01]], atol=1e-12)
        self.assertFalse(any(c.args[0] == "place_corner_calibrated" for c in events.call_args_list))

    def test_material_probe_miss_still_stops_with_measured_diagnostics(self):
        robot, motion, capture = self.simulated()
        motion.side_effect = lambda pose: None
        events = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "missed board-corner waypoint"):
            CornerServo(robot, motion, capture, lambda x, y: .5, events).teach()
        self.assertEqual(motion.call_count, 1)
        record = next(c.args[1] for c in events.call_args_list if c.args[0] == "place_corner_motion")
        np.testing.assert_allclose(record["measured_delta_xy_m"], [0, 0])
        self.assertAlmostEqual(record["position_error_m"], .01)

    def test_hover_arrival_error_is_not_baked_into_new_goal(self):
        robot, motion, capture = self.simulated()
        robot.pose = Pose((.403, -.1, .6), (1, 0, 0, 0))
        reference, _ = CornerServo(robot, motion, capture, lambda x, y: .5).teach(anchor_xy=[.4, -.1])
        np.testing.assert_allclose(robot.pose.position_m[:2], [.4, -.1], atol=.002)
        for corner in reference["corners"]:
            self.assertNotEqual(corner["uv"], corner["observed_uv"])

    def test_missing_image_or_stationary_feedback_never_issues_correction(self):
        robot, motion, capture = self.simulated()
        servo = CornerServo(robot, motion, lambda: np.zeros_like(self.rgb), lambda x, y: .5)
        with self.assertRaises(CornerVisualError):
            servo.align(self.calibrated())
        motion.assert_not_called()
        robot.stationary_tcp_pose.side_effect = RuntimeError("E-stop is active")
        with self.assertRaisesRegex(RuntimeError, "E-stop"):
            CornerServo(robot, motion, capture, lambda x, y: .5)
        motion.assert_not_called()

    def test_reported_battery_failure_tracks_board_not_larger_venue_floor_or_jaw(self):
        path = IMAGE.with_name("placement_battery_failed_corners.png")
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        reference = make_reference(rgb)
        expected = [(1182, 136), (486, 297), (1275, 892)]
        self.assertEqual(len(reference["corners"]), 3)
        for corner, uv in zip(reference["corners"], expected):
            self.assertLess(math.dist(corner["uv"], uv), 2.)
        # C4 is hidden by the jaw. Its hull/occlusion intersection near (542,
        # 896) is not a board corner and must not acquire a tracking identity.
        for dx, dy, gain in ((-20, 14, .8), (15, -21, 1.1), (.4, -.7, 1.)):
            image = np.clip(self.translated(dx, dy, rgb).astype(float) * gain + 4,
                            0, 255).astype(np.uint8)
            matches = match_corners(image, reference)
            self.assertGreaterEqual(len(matches), 2)
            for corner in reference["corners"]:
                if corner["id"] in matches:
                    np.testing.assert_allclose(np.asarray(matches[corner["id"]]["uv"]) - corner["uv"],
                                               [dx, dy], atol=1.)

    def test_actual_adjacent_retreat_frame_does_not_switch_corner_identity(self):
        rgb = cv2.cvtColor(cv2.imread(str(IMAGE.with_name("placement_battery_failed_corners.png"))),
                           cv2.COLOR_BGR2RGB)
        adjacent = cv2.cvtColor(cv2.imread(str(IMAGE.with_name("placement_battery_retreat.png"))),
                               cv2.COLOR_BGR2RGB)
        reference = make_reference(rgb)
        matched = match_corners(adjacent, reference)
        self.assertEqual(set(matched), {"C2", "C3"})
        for corner in reference["corners"]:
            if corner["id"] in matched:
                self.assertLess(math.dist(corner["uv"], matched[corner["id"]]["uv"]), 2.)

    def test_battery_image_feedback_converges_to_one_mm_without_probes(self):
        rgb = cv2.cvtColor(cv2.imread(str(IMAGE.with_name("placement_battery_failed_corners.png"))),
                           cv2.COLOR_BGR2RGB)
        robot, motion, _ = self.simulated()
        reference = make_reference(rgb)
        matrix = np.array([[0, 2000], [1800, 0]])
        for corner in reference["corners"]:
            corner["jacobian_px_per_m"] = matrix.tolist()
        reference.update(reference_clearance_m=.1, reference_quaternion_wxyz=[1, 0, 0, 0])
        def capture():
            shift = matrix @ (np.asarray(robot.pose.position_m[:2]) - [.4, -.1]) + [-12, 9]
            return self.translated(*shift, image=rgb)
        result = CornerServo(robot, motion, capture, lambda x, y: .5).align(reference)
        self.assertEqual(result["status"], "converged")
        self.assertLess(np.linalg.norm(np.asarray(robot.pose.position_m[:2]) - [.395, -.094]), .001)
        self.assertLessEqual(result["error_px"], 2.)

    def test_motion_fault_does_not_trigger_blind_return(self):
        robot, motion, capture = self.simulated(offset=[-12, 9])
        motion.side_effect = RuntimeError("joint timeout")
        with self.assertRaisesRegex(RuntimeError, "joint timeout"):
            CornerServo(robot, motion, capture, lambda x, y: .5).align(self.calibrated())
        self.assertEqual(motion.call_count, 1)

    def test_height_and_orientation_mismatch_stop_before_correction(self):
        robot, motion, capture = self.simulated()
        reference = self.calibrated()
        reference["reference_clearance_m"] = .08
        with self.assertRaises(CornerVisualError):
            CornerServo(robot, motion, capture, lambda x, y: .5).align(reference)
        reference["reference_clearance_m"] = .1
        reference["reference_quaternion_wxyz"] = [.999, 0, 0, .04]
        with self.assertRaises(RuntimeError):
            CornerServo(robot, motion, capture, lambda x, y: .5).align(reference)
        motion.assert_not_called()

    def test_persisted_reference_rejects_corrupt_motion_mapping(self):
        from steadyhand.wrist_part_profiles import _validate_place_corners
        reference = self.calibrated()
        reference["camera"] = "wrist_a"
        _validate_place_corners(reference)
        reference["corners"][0]["jacobian_px_per_m"] = [[0, 0], [0, 0]]
        with self.assertRaises(ValueError):
            _validate_place_corners(reference)

    def test_low_reference_requires_explicit_height_and_preserves_legacy_validation(self):
        from steadyhand.wrist_part_profiles import _validate_place_corners
        reference = self.calibrated()
        reference["camera"] = "wrist_a"
        _validate_place_corners(reference)  # Existing 100 mm calibration.
        reference["reference_clearance_m"] = .020
        with self.assertRaises(ValueError):
            _validate_place_corners(reference)  # Cannot silently reinterpret old data.
        reference["alignment_clearance_m"] = .020
        _validate_place_corners(reference)
        for invalid in (.005, .040, .100):
            reference["reference_clearance_m"] = invalid
            with self.assertRaises(ValueError):
                _validate_place_corners(reference)

    def test_low_corner_teaching_stays_at_20_with_smaller_bounded_probes(self):
        robot, motion, capture = self.simulated()
        robot.pose = Pose((.4, -.1, .520), (1, 0, 0, 0))
        servo = CornerServo(robot, motion, capture, lambda x, y: .5, low_clearance=True)
        reference, _ = servo.teach()
        self.assertAlmostEqual(reference["reference_clearance_m"], .020)
        for call in motion.call_args_list:
            self.assertAlmostEqual(call.args[0].position_m[2], .520)
            self.assertLessEqual(math.dist(call.args[0].position_m[:2], [.4, -.1]), .006001)
        with self.assertRaisesRegex(CornerVisualError, "10 mm travel"):
            servo.go((.411, -.1))

    def test_low_alignment_requires_explicit_opt_in_and_rejects_wrong_reference_height(self):
        robot, motion, capture = self.simulated()
        robot.pose = Pose((.4, -.1, .520), (1, 0, 0, 0))
        with self.assertRaises(ValueError):
            CornerServo(robot, motion, capture, lambda x, y: .5)
        servo = CornerServo(robot, motion, capture, lambda x, y: .5, low_clearance=True)
        with self.assertRaisesRegex(CornerVisualError, "different hover height"):
            servo.align(self.calibrated())
        motion.assert_not_called()

    def test_unstable_corner_frames_stop_before_motion(self):
        robot, motion, _ = self.simulated()
        capture = mock.Mock(side_effect=[self.rgb, self.translated(6, 0)] * 2)
        with self.assertRaisesRegex(CornerVisualError, "unstable"):
            CornerServo(robot, motion, capture, lambda x, y: .5).align(self.calibrated())
        self.assertEqual(capture.call_count, 4)
        motion.assert_not_called()

    def test_one_unstable_pair_retries_images_without_moving_or_loosening_jitter(self):
        robot, motion, _ = self.simulated()
        capture = mock.Mock(side_effect=[self.rgb, self.translated(6, 0), self.translated(6.2, .1)])
        matches, _ = CornerServo(robot, motion, capture, lambda x, y: .5).observe(self.calibrated())
        self.assertEqual(capture.call_count, 3)
        self.assertEqual(len(matches), 4)
        motion.assert_not_called()

    def test_teaching_gives_intermittently_missing_corners_an_extra_image_pair(self):
        robot, motion, capture = self.simulated()
        reference = self.calibrated()
        all_corners = {c["id"]: {"uv": c["uv"]} for c in reference["corners"]}
        one_corner = {"C3": all_corners["C3"]}
        with mock.patch("steadyhand.vision.board_corners.match_corners",
                        side_effect=[all_corners, one_corner, all_corners, all_corners]) as match:
            matches, _ = CornerServo(robot, motion, capture, lambda x, y: .5).observe(reference, prefer_all=True)
        self.assertEqual(set(matches), set(all_corners))
        self.assertEqual(match.call_count, 4)
        motion.assert_not_called()

    def test_camera_fault_is_not_retried_as_an_unstable_image(self):
        robot, motion, _ = self.simulated()
        capture = mock.Mock(side_effect=RuntimeError("camera disconnected"))
        with self.assertRaisesRegex(RuntimeError, "camera disconnected"):
            CornerServo(robot, motion, capture, lambda x, y: .5).observe(self.calibrated())
        capture.assert_called_once()
        motion.assert_not_called()


if __name__ == "__main__":
    unittest.main()
