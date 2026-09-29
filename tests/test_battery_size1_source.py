"""Offline tests for operator-explicit battery_size1 source localization."""

from datetime import datetime, timedelta, timezone
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from steadyhand.battery_size1_source import (
    MEASURED_BOARD_WIDTH_MM,
    base_xy_from_board_offset,
    dimensions_from_measured_width,
    homography_board_fraction,
    load_manual_board_calibration,
    metric_offset_from_fraction,
)
from steadyhand.config import load_bundle
from tools.vega_battery_size1_source_localize import main

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None


def manual_record(cfg, *, generated=None):
    generated = generated or datetime.now(timezone.utc)
    return {
        "schema_version": 1,
        "generated_at_utc": generated.isoformat(),
        "robot_name": cfg["robot_name"],
        "base_frame": cfg["kinematics"]["base_frame"],
        "tcp_frame": cfg["kinematics"]["ee_frame"],
        "floor_m": 0.456,
        "nominal_axis_offset_m": 0.100,
        "corrected_board_frame_xy": {
            "center_base_xy_m": [0.500, 0.100],
            "board_x_unit_base_xy": [1.0, 0.0],
            "board_y_unit_base_xy": [0.0, 1.0],
            "x_reference_distance_m": 0.100,
            "y_reference_distance_m": 0.100,
        },
    }


class SourceGeometryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = copy.deepcopy(load_bundle("vega")["robot"])

    def test_manual_calibration_rejects_wrong_robot_and_stale_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.json"
            wrong = manual_record(self.cfg)
            wrong["robot_name"] = "dm/not-this-robot"
            path.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, "different robot"):
                load_manual_board_calibration(path, self.cfg)

            old = manual_record(
                self.cfg,
                generated=datetime.now(timezone.utc) - timedelta(hours=13),
            )
            path.write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError, "stale"):
                load_manual_board_calibration(
                    path,
                    self.cfg,
                    max_age_minutes=720,
                )

    def test_current_five_point_fallback_is_accepted_by_source_localizer(self):
        path = Path(__file__).parents[1] / "calibration" / "vega_board_manual_fallback.json"
        loaded = load_manual_board_calibration(path, self.cfg)
        self.assertTrue(loaded["is_permanent_fallback"])
        self.assertEqual(loaded["schema_version"], 2)
        base = base_xy_from_board_offset(loaded, (0.0, 0.0))
        self.assertEqual(base, loaded["center_base_xy_m"])

    def test_measured_width_requires_explicit_axis_and_second_dimension(self):
        self.assertEqual(MEASURED_BOARD_WIDTH_MM, 386.0)
        self.assertEqual(
            dimensions_from_measured_width(
                width_axis="x",
                other_dimension_mm=301.0,
            ),
            (386.0, 301.0),
        )
        self.assertEqual(
            dimensions_from_measured_width(
                width_axis="y",
                other_dimension_mm=301.0,
            ),
            (301.0, 386.0),
        )
        with self.assertRaises(ValueError):
            dimensions_from_measured_width(
                width_axis="unknown",
                other_dimension_mm=301.0,
            )
        with self.assertRaises(ValueError):
            dimensions_from_measured_width(
                width_axis="x",
                other_dimension_mm=None,
            )

    @unittest.skipIf(np is None or cv2 is None, "requires NumPy/OpenCV")
    def test_explicit_four_corner_homography_maps_metric_board_offset(self):
        corners = {
            "xm_ym": (100, 100),
            "xp_ym": (500, 100),
            "xp_yp": (500, 300),
            "xm_yp": (100, 300),
        }
        fraction, _ = homography_board_fraction((400, 150), corners)
        self.assertAlmostEqual(fraction[0], 0.75, places=5)
        self.assertAlmostEqual(fraction[1], 0.25, places=5)
        offset = metric_offset_from_fraction(
            fraction,
            board_x_mm=386.0,
            board_y_mm=300.0,
        )
        self.assertAlmostEqual(offset[0], 0.0965, places=6)
        self.assertAlmostEqual(offset[1], -0.075, places=6)
        manual = {
            "center_base_xy_m": (0.5, 0.1),
            "board_x_unit_base_xy": (1.0, 0.0),
            "board_y_unit_base_xy": (0.0, 1.0),
        }
        base = base_xy_from_board_offset(manual, offset)
        self.assertAlmostEqual(base[0], 0.5965, places=6)
        self.assertAlmostEqual(base[1], 0.025, places=6)

    @unittest.skipIf(np is None or cv2 is None, "requires NumPy/OpenCV")
    def test_crossed_corners_and_outside_battery_are_rejected(self):
        crossed = {
            "xm_ym": (100, 100),
            "xp_ym": (500, 300),
            "xp_yp": (500, 100),
            "xm_yp": (100, 300),
        }
        with self.assertRaises(ValueError):
            homography_board_fraction((300, 200), crossed)

        corners = {
            "xm_ym": (100, 100),
            "xp_ym": (500, 100),
            "xp_yp": (500, 300),
            "xm_yp": (100, 300),
        }
        with self.assertRaisesRegex(ValueError, "outside board"):
            homography_board_fraction((590, 200), corners)


@unittest.skipIf(np is None or cv2 is None, "requires NumPy/OpenCV")
class SourceCliTests(unittest.TestCase):
    def setUp(self):
        self.cfg = copy.deepcopy(load_bundle("vega")["robot"])

    def write_image(self, path):
        image = np.full((400, 600, 3), 180, dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (500, 300), (245, 245, 245), -1)
        cv2.imwrite(str(path), image)

    def write_manual(self, path):
        path.write_text(json.dumps(manual_record(self.cfg)))

    def test_homography_cli_outputs_exact_coarse_xy_and_audit_record(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image = directory / "head.png"
            manual = directory / "manual.json"
            output = directory / "run"
            self.write_image(image)
            self.write_manual(manual)

            code = main(
                [
                    "homography",
                    "--manual-calibration", str(manual),
                    "--image", str(image),
                    "--battery-pixel", "400", "150",
                    "--corner-xm-ym", "100", "100",
                    "--corner-xp-ym", "500", "100",
                    "--corner-xp-yp", "500", "300",
                    "--corner-xm-yp", "100", "300",
                    "--width-axis", "x",
                    "--other-dimension-mm", "300",
                    "--confirm-board-unchanged-since-calibration",
                    "--output", str(output),
                ]
            )
            self.assertEqual(code, 0)
            record = json.loads((output / "localization.json").read_text())
            self.assertEqual(record["part"], "battery_size1")
            self.assertEqual(
                record["method"],
                "explicit_four_corner_homography",
            )
            self.assertEqual(
                record["board_geometry"]["measured_width_mm"],
                386.0,
            )
            self.assertTrue(record["board_geometry"]["no_400mm_assumption"])
            self.assertEqual(record["board_geometry"]["board_size_contract"], "386_mm_span")
            self.assertAlmostEqual(record["coarse_base_xy_m"][0], 0.5965, places=5)
            self.assertAlmostEqual(record["coarse_base_xy_m"][1], 0.025, places=5)
            self.assertEqual(
                record["battery_size1"]["identity_source"],
                "operator_explicit",
            )
            self.assertIn("sha256", record["manual_board_calibration"])
            self.assertIn("sha256", record["source_image"])
            serialized = json.dumps(record).lower()
            self.assertNotIn("part_count", serialized)
            self.assertNotIn("final_layout", serialized)
            self.assertTrue((output / "head_source.png").is_file())
            self.assertTrue((output / "head_selection_overlay.png").is_file())

    def test_board_offset_mode_works_without_second_board_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image = directory / "head.png"
            manual = directory / "manual.json"
            output = directory / "run"
            self.write_image(image)
            self.write_manual(manual)

            code = main(
                [
                    "board-offset",
                    "--manual-calibration", str(manual),
                    "--image", str(image),
                    "--battery-pixel", "350", "200",
                    "--board-offset-mm", "35", "-20",
                    "--confirm-board-unchanged-since-calibration",
                    "--output", str(output),
                ]
            )
            self.assertEqual(code, 0)
            record = json.loads((output / "localization.json").read_text())
            self.assertEqual(record["method"], "operator_board_offset")
            self.assertEqual(record["board_geometry"]["other_dimension_mm"], None)
            self.assertAlmostEqual(record["coarse_base_xy_m"][0], 0.535)
            self.assertAlmostEqual(record["coarse_base_xy_m"][1], 0.080)

    def test_homography_cli_refuses_missing_second_dimension(self):
        with self.assertRaises(SystemExit) as cm:
            main(
                [
                    "homography",
                    "--manual-calibration", "manual.json",
                    "--image", "head.png",
                    "--battery-pixel", "1", "1",
                    "--corner-xm-ym", "1", "1",
                    "--corner-xp-ym", "2", "1",
                    "--corner-xp-yp", "2", "2",
                    "--corner-xm-yp", "1", "2",
                    "--width-axis", "x",
                    "--confirm-board-unchanged-since-calibration",
                ]
            )
        self.assertEqual(cm.exception.code, 2)

    def test_localize_requires_explicit_board_unchanged_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image = directory / "head.png"
            manual = directory / "manual.json"
            self.write_image(image)
            self.write_manual(manual)
            with self.assertRaises(SystemExit) as cm:
                main(
                    [
                        "board-offset",
                        "--manual-calibration", str(manual),
                        "--image", str(image),
                        "--battery-pixel", "300", "200",
                        "--board-offset-mm", "0", "0",
                    ]
                )
            self.assertNotEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
