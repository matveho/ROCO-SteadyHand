import json
import unittest
from pathlib import Path

from steadyhand.models import Pose
from tools.vega_task_coordinate_reachability import _live_pose, _resolve_point


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

    def test_default_first_points_exist(self):
        for name in ("battery_size1.pick", "usb_a.pick", "rod_16mm.place", "gear_20teeth.place"):
            part, kind, xyz = _resolve_point(name, self.data)
            self.assertEqual(name, f"{part}.{kind}")
            self.assertEqual(len(xyz), 3)

    def test_secondary_connect_and_grade_points_are_preserved(self):
        for name in ("rod_16mm.connect", "gear_20teeth.grade"):
            _, _, xyz = _resolve_point(name, self.data)
            self.assertEqual(len(xyz), 3)


if __name__ == "__main__":
    unittest.main()
