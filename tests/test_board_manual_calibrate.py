import math
import unittest

from tools.vega_board_manual_calibrate import _board_parallel_jog_delta


class BoardManualCalibrationTests(unittest.TestCase):
    def test_forward_rise_compensation_moves_tcp_down(self):
        dx, dy, dz = _board_parallel_jog_delta("forward", 50.0, 10.0)
        self.assertAlmostEqual(dx, 0.050)
        self.assertAlmostEqual(dy, 0.0)
        self.assertAlmostEqual(dz, -0.050 * math.tan(math.radians(10.0)))

    def test_back_compensation_is_exact_inverse(self):
        forward = _board_parallel_jog_delta("forward", 20.0, 10.0)
        back = _board_parallel_jog_delta("back", 20.0, 10.0)
        for a, b in zip(forward, back):
            self.assertAlmostEqual(a, -b)

    def test_left_right_have_no_unmeasured_z_compensation(self):
        self.assertEqual(_board_parallel_jog_delta("left", 10.0, 10.0), (0.0, 0.010, 0.0))
        self.assertEqual(_board_parallel_jog_delta("right", 10.0, 10.0), (0.0, -0.010, 0.0))


if __name__ == "__main__":
    unittest.main()
