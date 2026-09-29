import json
import tempfile
import unittest
from pathlib import Path


from tools.board_annotation_tool import (
    apply_homography,
    export_project,
    invert_homography,
    polygon_is_simple,
    polygon_centroid,
    solve_homography,
)


class BoardAnnotationGeometryTests(unittest.TestCase):
    def test_four_point_homography_round_trip(self):
        source = [(0, 0), (100, 0), (100, 50), (0, 50)]
        destination = [(12, 17), (212, 22), (202, 117), (8, 106)]
        homography = solve_homography(source, destination)
        inverse = invert_homography(homography)
        for point, expected in zip(source, destination):
            mapped = apply_homography(homography, point)
            self.assertAlmostEqual(mapped[0], expected[0], places=7)
            self.assertAlmostEqual(mapped[1], expected[1], places=7)
        for point in ((4, 8), (42, 31), (97, 12)):
            self.assertAlmostEqual(
                apply_homography(inverse, apply_homography(homography, point))[0],
                point[0], places=6,
            )
            self.assertAlmostEqual(
                apply_homography(inverse, apply_homography(homography, point))[1],
                point[1], places=6,
            )

    def test_polygon_centroid_is_area_weighted(self):
        self.assertEqual(polygon_centroid([(0, 0), (10, 0), (10, 10), (0, 10)]), (5.0, 5.0))
        self.assertEqual(polygon_centroid([(0, 0), (10, 0), (0, 10)]), (10 / 3, 10 / 3))
        self.assertTrue(polygon_is_simple([(0, 0), (10, 0), (10, 10), (0, 10)]))
        self.assertFalse(polygon_is_simple([(0, 0), (10, 10), (0, 10), (10, 0)]))

    def test_export_keeps_legacy_top_left_task_coordinates_and_audit_center(self):
        state = {
            "image_path": "/tmp/initial.png",
            "rectified_size_px": [100, 100],
            "board_points_source_px": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "homography_image_to_board": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "homography_board_to_image": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "annotations": [{
                "id": "a1",
                "part_name": "gear_60teeth",
                "kind": "circle",
                "closed": True,
                "center_board_px": [25, 75],
                "radius_board_px": 5,
                "points_board_px": [[25, 75], [30, 75]],
            }],
        }
        final = dict(state)
        final["image_path"] = "/tmp/final.png"
        final["annotations"] = [{
            "id": "a2",
            "part_name": "gear_60teeth",
            "kind": "polygon",
            "closed": True,
            "center_board_px": [75, 25],
            "points_board_px": [[70, 20], [80, 20], [75, 30]],
        }]
        with tempfile.TemporaryDirectory() as directory:
            project_path, task_path, task = export_project(
                Path(directory) / "board_annotations.json",
                states={"initial": state, "final": final},
                part_order=["gear_60teeth"],
                board_width_m=0.386,
                board_height_m=0.400,
            )
            self.assertTrue(project_path.is_file())
            self.assertTrue(task_path.is_file())
            for actual, expected in zip(task["parts"]["gear_60teeth"]["pick"], [0.386 * 25 / 99, 0.400 * 75 / 99, 0.0]):
                self.assertAlmostEqual(actual, expected)
            for actual, expected in zip(task["parts"]["gear_60teeth"]["place"], [0.386 * 75 / 99, 0.400 * 25 / 99, 0.0]):
                self.assertAlmostEqual(actual, expected)
            saved = json.loads(project_path.read_text())
            annotation = saved["states"]["initial"]["annotations"][0]
            self.assertAlmostEqual(annotation["center_board_m"][0], 0.386 * 25 / 99 - 0.386 / 2)
            self.assertEqual(saved["task_coordinates"]["source_pose_frame"], "board_local_annotation")
            self.assertEqual(saved["task_coordinates"]["board_motion_model"], "horizontal_translation_only_fixed_table_plane")


if __name__ == "__main__":
    unittest.main()
