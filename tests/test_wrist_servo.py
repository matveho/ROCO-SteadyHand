import math
import unittest

from steadyhand.vision.wrist_servo import PixelJacobian, jacobian_from_probes


class WristServoTests(unittest.TestCase):
    def test_cross_axis_sign_and_bounded_correction(self):
        j = jacobian_from_probes((100, 200), (100, 190), (120, 200), 0.01)
        dx, dy = j.base_delta_for_pixel_error((20, -10), gain=1, max_step_m=1)
        self.assertAlmostEqual(dx, -0.01)
        self.assertAlmostEqual(dy, -0.01)
        dx, dy = j.base_delta_for_pixel_error((200, -100), max_step_m=0.01)
        self.assertAlmostEqual(math.hypot(dx, dy), 0.01)

    def test_bad_calibration_and_inputs_rejected(self):
        for j in (PixelJacobian(1, 1, 1, 1), PixelJacobian(10000, 0, 0, 1)):
            with self.assertRaises(ValueError):
                j.base_delta_for_pixel_error((1, 1))
        j = PixelJacobian(1000, 0, 0, 1000)
        for kwargs in ({'gain': -1}, {'max_step_m': -1}, {'gain': float('nan')}):
            with self.assertRaises(ValueError):
                j.base_delta_for_pixel_error((1, 1), **kwargs)

    def test_measured_probes_account_for_cross_coupling(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('NumPy unavailable')
        from steadyhand.vision.wrist_servo import jacobian_from_measured_probes
        xy = np.array([[.5, .1], [.510, .101], [.502, .112]])
        expected = np.array([[200, -1100], [1200, 300]])
        uv = (xy-xy[0]) @ expected.T + [300, 400]
        actual = jacobian_from_measured_probes(uv, xy)
        np.testing.assert_allclose(actual.matrix(), expected)
        with self.assertRaises(ValueError):
            jacobian_from_measured_probes(uv, [[0, 0], [.01, 0], [.02, 0]])


class BoardBenchmarkTests(unittest.TestCase):
    def test_execute_passes_yaw_and_stops_on_failure(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch
        try:
            import numpy
        except ImportError:
            self.skipTest('NumPy unavailable')
        from tools import vega_board_benchmark as benchmark
        robot = MagicMock()
        robot._robot.head.get_joint_pos.return_value = [0., 0., 0.]
        cfg = {'robot_name': 'test', 'motion': {'max_step_rad': .12}}
        def planner(robot, label, point, center, hover_z, floor, yaw):
            self.assertAlmostEqual(yaw, math.pi/6)
            raise RuntimeError('IK failed')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'board.json'
            path.write_text(json.dumps({'corners_base_m': {k:[.5,.1,.456] for k in ('tl','tr','br','bl')},
                                        'center_base_m': [.5,.1,.456]}))
            with patch.object(benchmark, 'load_bundle', return_value={'robot':cfg}), \
                 patch.object(benchmark, 'load_vega_skills', return_value={'safety':{'min_tcp_z_m':.456}}), \
                 patch.object(benchmark, 'VegaAdapter', return_value=robot), \
                 patch.object(benchmark, '_vertical_target_for_point', side_effect=planner):
                with self.assertRaisesRegex(RuntimeError, 'IK failed'):
                    benchmark.main(['--reuse-registration', '--output', str(path), '--execute',
                                    '--confirm-physical-motion', '--claw-yaw-deg', '30'])
        robot.stop.assert_called_once()
        robot.close.assert_called_once()
