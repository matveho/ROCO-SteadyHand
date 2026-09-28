"""Offline tests for battery_size1 source-localization -> coarse-hover bridge."""

from datetime import datetime, timedelta, timezone
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from steadyhand.battery_size1_hover import (
    accepted_hover_from_manual,
    load_source_localization,
    plan_hover_stages,
)
from steadyhand.battery_size1_source import (
    file_sha256,
    load_manual_board_calibration,
)
from steadyhand.config import load_bundle
from steadyhand.models import Pose


def manual_record(cfg, *, generated=None):
    generated = generated or datetime.now(timezone.utc)
    q = [math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)]
    return {
        "schema_version": 1,
        "generated_at_utc": generated.isoformat(),
        "robot_name": cfg["robot_name"],
        "base_frame": cfg["kinematics"]["base_frame"],
        "tcp_frame": cfg["kinematics"]["ee_frame"],
        "floor_m": 0.456,
        "nominal_axis_offset_m": 0.100,
        "manual_corrected": {
            "CENTER": {
                "position_m": [0.500, 0.100, 0.550],
                "quaternion_wxyz": q,
            },
            "BOARD_X_PLUS": {
                "position_m": [0.600, 0.100, 0.550],
                "quaternion_wxyz": q,
            },
            "BOARD_Y_PLUS": {
                "position_m": [0.500, 0.200, 0.550],
                "quaternion_wxyz": q,
            },
        },
        "corrected_board_frame_xy": {
            "center_base_xy_m": [0.500, 0.100],
            "board_x_unit_base_xy": [1.0, 0.0],
            "board_y_unit_base_xy": [0.0, 1.0],
            "x_reference_distance_m": 0.100,
            "y_reference_distance_m": 0.100,
        },
        "orientation_status": "operator-accepted hover orientation",
    }


def localization_record(cfg, manual, *, generated=None):
    generated = generated or datetime.now(timezone.utc)
    return {
        "schema_version": 1,
        "generated_at_utc": generated.isoformat(),
        "part": "battery_size1",
        "method": "operator_board_offset",
        "robot_name": cfg["robot_name"],
        "base_frame": cfg["kinematics"]["base_frame"],
        "manual_board_calibration": {
            "path": "calibration/vega_board_manual.json",
            "sha256": manual["sha256"],
            "generated_at_utc": manual["generated_at_utc"],
            "age_minutes": 0.1,
            "operator_confirmed_board_unchanged": True,
        },
        "source_image": {
            "input_path": "head.png",
            "copied_path": "head_source.png",
            "sha256": "a" * 64,
        },
        "battery_size1": {
            "operator_selected_head_pixel_uv": [320.0, 240.0],
            "identity_source": "operator_explicit",
        },
        "coarse_base_xy_m": [0.535, 0.080],
    }


class BatteryCoarseHoverContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg = copy.deepcopy(load_bundle("vega")["robot"])

    def _write_manual(self, directory, *, generated=None):
        path = Path(directory) / "manual.json"
        path.write_text(
            json.dumps(
                manual_record(self.cfg, generated=generated)
            )
        )
        return path

    def _validated_manual(self, path):
        return load_manual_board_calibration(path, self.cfg)

    def test_localization_requires_exact_current_manual_hash_and_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            manual_path = self._write_manual(directory)
            manual = self._validated_manual(manual_path)
            loc = localization_record(self.cfg, manual)
            loc_path = Path(directory) / "localization.json"
            loc_path.write_text(json.dumps(loc))

            loaded = load_source_localization(
                loc_path,
                self.cfg,
                manual,
            )
            self.assertEqual(
                loaded["coarse_base_xy_m"],
                (0.535, 0.080),
            )
            self.assertEqual(
                loaded["manual_calibration_sha256"],
                manual["sha256"],
            )
            self.assertEqual(
                loaded["sha256"],
                file_sha256(loc_path),
            )

            changed = manual_record(self.cfg)
            manual_path.write_text(json.dumps(changed))
            changed_manual = self._validated_manual(manual_path)
            with self.assertRaisesRegex(
                ValueError,
                "different manual board calibration",
            ):
                load_source_localization(
                    loc_path,
                    self.cfg,
                    changed_manual,
                )

    def test_localization_rejects_wrong_robot_stale_or_unconfirmed_board(self):
        with tempfile.TemporaryDirectory() as directory:
            manual_path = self._write_manual(directory)
            manual = self._validated_manual(manual_path)
            loc_path = Path(directory) / "localization.json"

            wrong = localization_record(self.cfg, manual)
            wrong["robot_name"] = "dm/not-this-robot"
            loc_path.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(
                ValueError,
                "different robot",
            ):
                load_source_localization(
                    loc_path,
                    self.cfg,
                    manual,
                )

            stale = localization_record(
                self.cfg,
                manual,
                generated=(
                    datetime.now(timezone.utc)
                    - timedelta(hours=3)
                ),
            )
            loc_path.write_text(json.dumps(stale))
            with self.assertRaisesRegex(ValueError, "stale"):
                load_source_localization(
                    loc_path,
                    self.cfg,
                    manual,
                    max_age_minutes=120,
                )

            unconfirmed = localization_record(self.cfg, manual)
            unconfirmed["manual_board_calibration"][
                "operator_confirmed_board_unchanged"
            ] = False
            loc_path.write_text(json.dumps(unconfirmed))
            with self.assertRaisesRegex(
                ValueError,
                "board-unchanged",
            ):
                load_source_localization(
                    loc_path,
                    self.cfg,
                    manual,
                )

    def test_accepted_hover_comes_from_manual_center_and_enforces_band(self):
        with tempfile.TemporaryDirectory() as directory:
            manual_path = self._write_manual(directory)
            manual = self._validated_manual(manual_path)
            hover = accepted_hover_from_manual(
                manual,
                floor_m=0.456,
            )
            self.assertAlmostEqual(hover["z_m"], 0.550)
            self.assertEqual(
                hover["quaternion_wxyz"],
                (
                    math.sqrt(0.5),
                    0.0,
                    0.0,
                    -math.sqrt(0.5),
                ),
            )

            bad = manual_record(self.cfg)
            bad["manual_corrected"]["CENTER"][
                "position_m"
            ][2] = 0.500
            manual_path.write_text(json.dumps(bad))
            manual_bad = self._validated_manual(manual_path)
            with self.assertRaisesRegex(
                ValueError,
                "safe-hover band",
            ):
                accepted_hover_from_manual(
                    manual_bad,
                    floor_m=0.456,
                )

    def test_hover_plan_is_strict_xy_only_at_accepted_hover(self):
        q = (
            math.sqrt(0.5),
            0.0,
            0.0,
            -math.sqrt(0.5),
        )
        current = Pose((0.45, 0.02, 0.550), q)
        stages = plan_hover_stages(
            current,
            (0.535, 0.080),
            hover_z_m=0.550,
            floor_m=0.456,
        )
        self.assertEqual(len(stages), 1)
        label, target = stages[0]
        self.assertEqual(label, "PLANAR_TO_SOURCE_XY")
        self.assertEqual(target.position_m, (0.535, 0.080, 0.550))
        self.assertEqual(target.quaternion_wxyz, q)

    def test_hover_plan_preserves_live_z_within_hover_tolerance(self):
        q = (
            math.sqrt(0.5),
            0.0,
            0.0,
            -math.sqrt(0.5),
        )
        current = Pose((0.45, 0.02, 0.556), q)
        stages = plan_hover_stages(
            current,
            (0.535, 0.080),
            hover_z_m=0.550,
            floor_m=0.456,
            hover_z_tolerance_m=0.008,
        )
        self.assertEqual(len(stages), 1)
        _, target = stages[0]
        self.assertEqual(target.position_m, (0.535, 0.080, 0.556))
        self.assertEqual(target.quaternion_wxyz, q)

    def test_hover_plan_refuses_any_vertical_reposition(self):
        q = (1.0, 0.0, 0.0, 0.0)
        for z in (0.620, 0.520):
            with self.subTest(z=z):
                current = Pose((0.45, 0.02, z), q)
                with self.assertRaisesRegex(
                    ValueError,
                    "no vertical motion is allowed",
                ):
                    plan_hover_stages(
                        current,
                        (0.535, 0.080),
                        hover_z_m=0.550,
                        floor_m=0.456,
                    )

    def test_hover_plan_refuses_below_floor_start(self):
        current = Pose(
            (0.45, 0.02, 0.450),
            (1.0, 0.0, 0.0, 0.0),
        )
        with self.assertRaisesRegex(
            ValueError,
            "below configured floor",
        ):
            plan_hover_stages(
                current,
                (0.535, 0.080),
                hover_z_m=0.550,
                floor_m=0.456,
            )


if __name__ == "__main__":
    unittest.main()
