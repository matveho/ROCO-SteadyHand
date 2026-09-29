import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from steadyhand.board_calibration import (
    board_geometry_signature,
    compare_board_geometry,
    load_board_calibration,
    orthonormalize_xy_axes,
)
from steadyhand.config import load_bundle


class BoardCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = copy.deepcopy(load_bundle("vega")["robot"])
        self.fallback = Path(__file__).parents[1] / "calibration" / "vega_board_manual_fallback.json"

    def test_fallback_is_explicitly_freshness_exempt_and_axes_are_rigid(self):
        loaded = load_board_calibration(self.fallback, self.cfg)
        self.assertTrue(loaded["is_permanent_fallback"])
        self.assertAlmostEqual(
            loaded["board_x_unit_base_xy"][0] * loaded["board_y_unit_base_xy"][0]
            + loaded["board_x_unit_base_xy"][1] * loaded["board_y_unit_base_xy"][1],
            0.0,
            places=12,
        )
        self.assertGreater(loaded["axis_angle_error_deg"], 2.0)
        self.assertIn("surface_plane", loaded)

    def test_non_fallback_v2_stale_record_is_rejected(self):
        value = json.loads(self.fallback.read_text())
        value["permanent_fallback"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manual.json"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "stale"):
                load_board_calibration(path, self.cfg, max_age_minutes=720.0)

    def test_measured_axes_preserve_y_sign(self):
        ux, uy, dot, angle = orthonormalize_xy_axes((1.0, 0.0), (0.1, 1.0))
        self.assertAlmostEqual(sum(v * v for v in ux), 1.0)
        self.assertAlmostEqual(sum(v * v for v in uy), 1.0)
        self.assertGreater(uy[1], 0.90)
        self.assertAlmostEqual(ux[0] * uy[0] + ux[1] * uy[1], 0.0)
        self.assertGreater(dot, 0.0)
        self.assertGreater(angle, 0.0)

    def test_retake_geometry_rejects_rotation_and_accepts_translation(self):
        base = {"tl": [0, 0], "tr": [200, 0], "br": [200, 200], "bl": [0, 200]}
        translated = {k: [p[0] + 40, p[1] + 25] for k, p in base.items()}
        rotated = {"tl": [0, 0], "tr": [180, 110], "br": [70, 290], "bl": [-110, 180]}
        reference = board_geometry_signature(base)
        self.assertTrue(compare_board_geometry(reference, board_geometry_signature(translated))["valid"])
        self.assertFalse(compare_board_geometry(reference, board_geometry_signature(rotated))["valid"])


if __name__ == "__main__":
    unittest.main()
