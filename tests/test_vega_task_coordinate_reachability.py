import json
import unittest
from pathlib import Path

from steadyhand.models import Pose
from steadyhand.board_geometry import (
    BOARD_SIZE_M,
    configured_board_plane_z,
    validate_task_coordinate_extent,
    validate_task_board_geometry,
)
from tools.vega_task_coordinate_reachability import calibrated_surface_z, _live_pose, _resolve_point


class TaskCoordinateReachabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / "configs/task_coordinates.json"
        cls.data = json.loads(path.read_text())

    def test_source_point_is_registered_about_declared_board_center(self):
        pose = _live_pose(
            (0.193 + 0.1, 0.037 - 0.05, 1.04),
            source_center=(0.193, 0.037), live_center=(0.4, -0.2),
            ux=(1.0, 0.0), uy=(0.0, 1.0), surface_plane=(0.0, 0.0, 0.50),
            clearance_m=0.05,
            quat=(1.0, 0.0, 0.0, 0.0),
        )
        self.assertEqual(pose, Pose((0.5, -0.25, 0.55), (1.0, 0.0, 0.0, 0.0)))

    def test_task_rotation_is_ccw_about_source_board_center(self):
        pose = _live_pose(
            (0.193 + 0.1, 0.037 - 0.05, 1.04),
            source_center=(0.193, 0.037), live_center=(0.4, -0.2),
            ux=(1.0, 0.0), uy=(0.0, 1.0), surface_plane=(0.0, 0.0, 0.50),
            clearance_m=0.05,
            quat=(1.0, 0.0, 0.0, 0.0), rotation_deg=90.0,
        )
        self.assertAlmostEqual(pose.position_m[0], 0.45)
        self.assertAlmostEqual(pose.position_m[1], -0.10)
        self.assertAlmostEqual(pose.position_m[2], 0.55)
        self.assertEqual(pose.quaternion_wxyz, (1.0, 0.0, 0.0, 0.0))

    def test_default_first_points_exist(self):
        for name in ("battery_size1.pick", "usb_a.pick", "rod_16mm.place", "gear_20teeth.place"):
            part, kind, xyz = _resolve_point(name, self.data)
            self.assertEqual(name, f"{part}.{kind}")
            self.assertEqual(len(xyz), 3)

    def test_board_geometry_is_fixed_386mm_horizontal_table_plane(self):
        self.assertEqual(BOARD_SIZE_M, 0.386)
        cfg = {"board_calibration": {"board_plane_z_m": 0.456}}
        self.assertEqual(configured_board_plane_z(cfg, 0.5), 0.456)
        validate_task_board_geometry(self.data)
        with self.assertRaises(ValueError):
            validate_task_board_geometry({"board_width_m": 0.4, "board_height_m": 0.386})

    def test_declared_physical_layout_matches_robot_perspective(self):
        validate_task_board_geometry(self.data)
        validate_task_coordinate_extent(self.data, names=("battery_size1.pick", "rod_16mm.pick"))

    def test_secondary_connect_and_grade_points_are_preserved(self):
        for name in ("rod_16mm.connect", "gear_20teeth.grade"):
            _, _, xyz = _resolve_point(name, self.data)
            self.assertEqual(len(xyz), 3)

    def test_surface_residual_correction_hits_measured_anchor(self):
        model = {
            "coefficients": (0.0, 0.0, 0.50),
            "anchors": [{"x_m": 0.1, "y_m": -0.2, "residual_m": 0.004}],
        }
        self.assertAlmostEqual(calibrated_surface_z(0.1, -0.2, model), 0.504)


if __name__ == "__main__":
    unittest.main()
