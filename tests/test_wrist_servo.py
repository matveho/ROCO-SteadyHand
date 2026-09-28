"""Offline servo sign, convergence and bounded-failure checks; no hardware access."""
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steadyhand.models import Pose
from steadyhand.vision.wrist_servo import (
    PixelJacobian, TemplateTracker, jacobian_from_measured_probes,
    jacobian_from_probes, run_xy_servo,
)
from tools.vega_wrist_servo import WristCapture, main
from tools.vega_wrist_fine_center import WristAOnlyCapture, main as fine_main

try:
    import numpy as np
    import cv2
except ImportError:
    np = cv2 = None


class JacobianTests(unittest.TestCase):
    def test_rotated_camera_negative_feedback_and_step_limit(self):
        jac = jacobian_from_probes((100, 200), (88, 224), (136, 212), 0.012)
        delta = jac.base_delta_for_pixel_error((12, -20), gain=1, max_step_m=1)
        self.assertAlmostEqual(jac.du_dx*delta[0]+jac.du_dy*delta[1], -12)
        self.assertAlmostEqual(jac.dv_dx*delta[0]+jac.dv_dy*delta[1], 20)
        self.assertAlmostEqual(math.hypot(*jac.base_delta_for_pixel_error((500, 200))), 0.025)

    def test_bad_calibration_and_controls_rejected(self):
        for jac in (PixelJacobian(1, 2, 2, 4), PixelJacobian(10000, 0, 0, 1)):
            with self.assertRaises(ValueError):
                jac.base_delta_for_pixel_error((1, 1))
        with self.assertRaises(ValueError):
            PixelJacobian(float('nan'), 1, 1, 1)
        jac = PixelJacobian(1000, 0, 0, 1000)
        for settings in ({'gain': -1}, {'max_step_m': -1}, {'gain': float('nan')}):
            with self.assertRaises(ValueError):
                jac.base_delta_for_pixel_error((1, 1), **settings)

    def test_execute_gate_precedes_hardware_import(self):
        with self.assertRaises(SystemExit) as cm:
            main(['--execute'])
        self.assertEqual(cm.exception.code, 2)

    def test_fine_tool_requires_confirmation_and_explicit_coarse_xy(self):
        with self.assertRaises(SystemExit) as cm:
            fine_main(['--coarse-xy', '.56', '0'])
        self.assertEqual(cm.exception.code, 2)
        with self.assertRaises(SystemExit) as cm:
            fine_main(['--confirm-physical-motion'])
        self.assertEqual(cm.exception.code, 2)


class FakeRobot:
    def __init__(self):
        self.pose = Pose((0.56, 0.0, 0.64), (math.sqrt(.5), 0, 0, -math.sqrt(.5)))
        self.moves = []
        self.bias = (0., 0.)
        self.z_drift = 0

    def get_tcp_pose(self):
        return self.pose

    def move_tcp(self, pose, *, speed_scale):
        self.moves.append(pose)
        self.pose = Pose((pose.position_m[0]+self.bias[0], pose.position_m[1]+self.bias[1],
                          pose.position_m[2]+self.z_drift), pose.quaternion_wxyz)


@unittest.skipIf(np is None or cv2 is None, 'requires NumPy and OpenCV')
class ImageServoTests(unittest.TestCase):
    def setUp(self):
        self.robot = FakeRobot()
        self.origin = np.array(self.robot.pose.position_m[:2])
        self.jac = np.array([[-1700, 900], [600, 1900.]])
        self.feature = np.array([270., 165.])
        self.patch = np.random.default_rng(27).integers(0, 256, (51, 51, 3), dtype=np.uint8)
        self.frames = 0

    def capture(self):
        self.frames += 1
        position = self.feature + self.jac @ (np.array(self.robot.pose.position_m[:2])-self.origin)
        u, v = np.rint(position).astype(int)
        image = np.full((360, 480, 3), 80, np.uint8)
        if 25 <= u < 455 and 25 <= v < 335:
            image[v-25:v+26, u-25:u+26] = self.patch
        return image

    def run_servo(self, capture=None, **kwargs):
        return run_xy_servo(self.robot, capture or self.capture, floor_m=.456,
                            feature_uv=self.feature, **kwargs)

    def test_real_template_and_loop_converge_with_camera_rotation(self):
        result = self.run_servo()
        self.assertEqual(result['status'], 'converged')
        self.assertLessEqual(result['error_px'], 5)
        self.assertGreater(result['iterations'], 0)
        for move in self.robot.moves:
            self.assertEqual(move.position_m[2], .64)
            self.assertLessEqual(math.dist(move.position_m[:2], self.origin), .06)

    def test_automatic_feature_selection(self):
        result = run_xy_servo(self.robot, self.capture, floor_m=.456)
        self.assertEqual(result['status'], 'converged')

    def test_measured_cross_axis_probes_recover_jacobian(self):
        offsets = np.array([[.011, .001], [-.002, .013]])
        pixels = np.array([100, 200]) + offsets @ self.jac.T
        jac = jacobian_from_measured_probes((100, 200), pixels, (0, 0), offsets)
        np.testing.assert_allclose(jac.matrix(), self.jac)
        with self.assertRaisesRegex(ValueError, 'small or collinear'):
            jacobian_from_measured_probes((100, 200), pixels, (0, 0), [[.012, 0], [.011, 0]])

    def test_wrong_static_camera_never_commands_correction(self):
        static = self.capture()
        with self.assertRaisesRegex(ValueError, 'barely moved'):
            self.run_servo(lambda: static.copy())
        self.assertEqual(len(self.robot.moves), 4)  # probes and returns only

    def test_lost_feature_stops_after_first_probe(self):
        initial = self.capture()
        images = iter([initial, np.zeros_like(initial)])
        with self.assertRaisesRegex(RuntimeError, 'lost/ambiguous'):
            self.run_servo(lambda: next(images))
        self.assertEqual(len(self.robot.moves), 1)

    def test_repeated_texture_rejected_before_motion(self):
        image = self.capture()
        image[140:191, 335:386] = self.patch
        with self.assertRaisesRegex(RuntimeError, 'lost/ambiguous'):
            self.run_servo(lambda: image)
        self.assertEqual(self.robot.moves, [])

    def test_textureless_feature_rejected_before_motion(self):
        with self.assertRaisesRegex(ValueError, 'textureless'):
            self.run_servo(lambda: np.zeros((360, 480, 3), np.uint8))
        self.assertEqual(self.robot.moves, [])

    def test_tcp_drift_stops_without_return_motion(self):
        self.robot.z_drift = .006
        with self.assertRaisesRegex(RuntimeError, 'drifted'):
            self.run_servo()
        self.assertEqual(len(self.robot.moves), 1)

    def test_correction_rejected_outside_local_radius(self):
        self.feature = np.array([350., 180.])
        with self.assertRaisesRegex(RuntimeError, 'local servo radius'):
            self.run_servo(max_radius_m=.012)
        self.assertEqual(len(self.robot.moves), 4)

    def test_nonvertical_or_low_tcp_rejected_before_capture(self):
        for pose in (Pose((.56, 0, .49), (1, 0, 0, 0)), Pose((.56, 0, .64), (0, 1, 0, 0))):
            self.robot.pose = pose
            with self.assertRaises(ValueError):
                self.run_servo()
        self.assertEqual(self.frames, 0)
        self.assertEqual(self.robot.moves, [])

    def test_fine_servo_logs_every_motion_and_correction(self):
        events = []
        result = self.run_servo(
            event=lambda kind, fields: events.append((kind, fields))
        )
        self.assertEqual(result['status'], 'converged')
        motions = [fields for kind, fields in events if kind == 'motion']
        self.assertGreaterEqual(len(motions), 4)
        for fields in motions:
            self.assertIn('requested_tcp', fields)
            self.assertIn('measured_tcp', fields)
            self.assertIn('position_error_m', fields)
        calibrated = [fields for kind, fields in events if kind == 'calibrated']
        self.assertEqual(len(calibrated), 1)
        self.assertIn('jacobian_px_per_m', calibrated[0])
        corrections = [fields for kind, fields in events if kind == 'correction']
        self.assertGreaterEqual(len(corrections), 1)
        for fields in corrections:
            self.assertIn('measured_tcp', fields)
            self.assertIn('requested_tcp', fields)

    def test_wrist_a_only_capture_retries_black_and_ignores_left_black(self):
        good = self.capture()
        black = np.zeros_like(good)
        calls = 0

        def read(**kwargs):
            nonlocal calls
            calls += 1
            right = black if calls == 1 else good
            frame_b = types.SimpleNamespace(
                rgb=right, frame_id=calls, timestamp_ns=calls * 10,
                received_monotonic_ns=calls * 100,
            )
            frame_a = types.SimpleNamespace(
                rgb=black, frame_id=calls, timestamp_ns=calls * 10,
                received_monotonic_ns=calls * 100,
            )
            return types.SimpleNamespace(wrist_a=frame_a, wrist_b=frame_b)

        cameras = types.SimpleNamespace(read=read)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            capture = WristAOnlyCapture(
                cameras, output, settle_s=0, warmup_attempts=2
            )
            image = capture()
            self.assertEqual(image.shape, good.shape)
            self.assertTrue((output / '000_wrist_a.png').is_file())
            self.assertFalse((output / '000_wrist_b.png').exists())
            records = [
                json.loads(line)
                for line in (output / 'capture_events.jsonl').read_text().splitlines()
            ]
            self.assertEqual([record['accepted'] for record in records], [False, True])

    def test_wrist_a_only_capture_rejects_stale_identity(self):
        image = self.capture()
        frame = types.SimpleNamespace(
            rgb=image, frame_id=7, timestamp_ns=99, received_monotonic_ns=101
        )
        cameras = types.SimpleNamespace(
            read=lambda **kwargs: types.SimpleNamespace(wrist_a=frame, wrist_b=frame)
        )
        with tempfile.TemporaryDirectory() as directory:
            capture = WristAOnlyCapture(
                cameras, Path(directory), settle_s=0, warmup_attempts=2
            )
            capture()
            with self.assertRaisesRegex(RuntimeError, 'stale'):
                capture()

    def test_fine_tool_rejects_wrong_start_xy_before_servo(self):
        from steadyhand.config import load_bundle

        good = self.capture()
        frame_id = 0

        def read(**kwargs):
            nonlocal frame_id
            frame_id += 1
            frame = types.SimpleNamespace(
                rgb=good, frame_id=frame_id, timestamp_ns=frame_id * 10,
                received_monotonic_ns=frame_id * 100,
            )
            return types.SimpleNamespace(wrist_a=frame, wrist_b=frame)

        cameras = types.SimpleNamespace(
            connect=lambda: None, close=lambda: None, read=read
        )

        class FineRobot:
            def __init__(self):
                self.config = load_bundle('vega')['robot']
                self._kinematics = types.SimpleNamespace(
                    config=dict(self.config['kinematics'])
                )
                self.stopped = False

            def connect(self):
                pass

            def get_tcp_pose(self):
                return Pose(
                    (.56, 0., .55),
                    (math.sqrt(.5), 0., 0., -math.sqrt(.5)),
                )

            def stop(self):
                self.stopped = True

            def close(self):
                pass

        robot = FineRobot()
        with tempfile.TemporaryDirectory() as directory, \
                patch('tools.vega_wrist_fine_center.VegaWristCameras', return_value=cameras), \
                patch('tools.vega_wrist_fine_center.VegaAdapter', return_value=robot), \
                patch('tools.vega_wrist_fine_center.run_xy_servo') as servo, \
                patch('tools.vega_wrist_fine_center.time.sleep'):
            output = Path(directory) / 'fine'
            with self.assertRaisesRegex(RuntimeError, 'establish the coarse hover'):
                fine_main([
                    '--coarse-xy', '.60', '0',
                    '--confirm-physical-motion',
                    '--output', str(output),
                ])
            servo.assert_not_called()
            # A pre-motion validation failure must not assert software E-stop.
            self.assertFalse(robot.stopped)

    def test_repeated_frame_identity_rejected(self):
        frame = types.SimpleNamespace(rgb=self.capture(), frame_id=1, timestamp_ns=9, received_monotonic_ns=10)
        cameras = types.SimpleNamespace(read=lambda **kw: types.SimpleNamespace(wrist_a=frame, wrist_b=frame))
        with tempfile.TemporaryDirectory() as directory:
            capture = WristCapture(cameras, Path(directory), 'wrist_a', settle_s=0)
            capture()
            with self.assertRaisesRegex(RuntimeError, 'stale feedback'):
                capture()

    def test_capture_only_never_constructs_robot(self):
        frame = types.SimpleNamespace(rgb=self.capture(), frame_id=1, timestamp_ns=9, received_monotonic_ns=10)
        cameras = types.SimpleNamespace(connect=lambda: None, close=lambda: None,
            read=lambda **kw: types.SimpleNamespace(wrist_a=frame, wrist_b=frame))
        with tempfile.TemporaryDirectory() as directory, patch('tools.vega_wrist_servo.VegaWristCameras', return_value=cameras), patch('tools.vega_wrist_servo.VegaAdapter') as adapter:
            output = Path(directory) / 'capture'
            self.assertEqual(main(['--output', str(output)]), 0)
            adapter.assert_not_called()
            self.assertTrue((output / '000_wrist_a.png').is_file())
            self.assertTrue((output / '000_wrist_b.png').is_file())

    def test_cli_coarse_entry_and_servo_use_current_planner_contract(self):
        from steadyhand.config import load_bundle

        self.robot.config = load_bundle('vega')['robot']
        self.robot._kinematics = types.SimpleNamespace(
            config=dict(self.robot.config['kinematics']), solve=lambda *a: (0,)*7)
        self.robot._read_joint_positions = lambda: (0,)*7
        head = types.SimpleNamespace(get_joint_pos=lambda: [0., 0., 0.],
                                     set_joint_pos=lambda *a, **kw: None)
        self.robot._robot = types.SimpleNamespace(head=head)
        self.robot.connect = lambda: None
        self.robot.close = lambda: None
        frame_count = 0

        def read(**kwargs):
            nonlocal frame_count
            frame_count += 1
            frame = types.SimpleNamespace(rgb=self.capture(), frame_id=frame_count,
                timestamp_ns=frame_count*1000000, received_monotonic_ns=frame_count*1000000)
            return types.SimpleNamespace(wrist_a=frame, wrist_b=frame)

        cameras = types.SimpleNamespace(connect=lambda: None, close=lambda: None, read=read)
        with tempfile.TemporaryDirectory() as directory, \
                patch('tools.vega_wrist_servo.VegaWristCameras', return_value=cameras), \
                patch('tools.vega_wrist_servo.VegaAdapter', return_value=self.robot), \
                patch('tools.vega_wrist_servo.time.sleep'):
            output = Path(directory) / 'servo'
            code = main(['--output', str(output), '--execute', '--confirm-physical-motion',
                         '--camera', 'wrist_a', '--move-to-board', '--board-xy', '.56', '0',
                         '--feature', '270', '165'])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads((output / 'result.json').read_text())['status'], 'converged')
            self.assertEqual(self.robot.pose.position_m[2], .55)
            self.assertEqual(self.robot._kinematics.config['position_tolerance_m'], .0007)
            self.assertTrue(list(output.glob('*_tracked.png')))


if __name__ == '__main__':
    unittest.main()
