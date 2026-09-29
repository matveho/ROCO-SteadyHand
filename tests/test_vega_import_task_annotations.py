import json
import unittest
from pathlib import Path

from tools.vega_import_task_annotations import convert
from steadyhand.board_geometry import board_relative_task_xy


class AnnotationImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.project = json.loads((root / "tools/outputs/initial.json").read_text(encoding="utf-8"))
        cls.task = convert(cls.project)

    def test_import_is_fixed_386_and_rotated_for_robot_view(self):
        self.assertEqual(self.task["board_width_m"], 0.386)
        self.assertEqual(self.task["board_height_m"], 0.386)
        self.assertEqual(self.task["task_coordinate_rotation_deg"], 180)
        self.assertFalse(self.task["task_coordinate_mirror_x"])
        self.assertEqual(self.task["source_pose_frame"], "board_local_annotation")

    def test_physical_layout_matches_operator_orientation(self):
        center = self.task["source_board_center_xy_m"]
        rotation = self.task["task_coordinate_rotation_deg"]
        part = self.task["parts"]

        def xy(name):
            part_name, kind = name.split(".")
            return board_relative_task_xy(
                part[part_name][kind][:2], center, rotation_deg=rotation,
            )

        small = xy("battery_size5.pick")
        large = xy("battery_size1.pick")
        rod = xy("rod_16mm.place")
        self.assertGreater(small[0], large[0])
        self.assertGreater(large[0], rod[0])
        self.assertGreater(small[1], 0)
        self.assertGreater(large[1], 0)
        self.assertIn("rod_16mm", self.task["legacy_secondary_points"])


if __name__ == "__main__":
    unittest.main()
