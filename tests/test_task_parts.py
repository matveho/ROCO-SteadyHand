import unittest


class TaskPartVisionTests(unittest.TestCase):
    def setUp(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("NumPy/OpenCV unavailable")
        self.cv2 = cv2
        self.np = np

    def _image(self, boxes):
        image = self.np.full((800, 800, 3), 235, dtype=self.np.uint8)
        for x0, y0, x1, y1 in boxes:
            self.cv2.rectangle(image, (x0, y0), (x1, y1), (20, 20, 20), -1)
        return image

    def _separated_nine(self):
        return [
            (100, 80, 160, 150),
            (250, 90, 320, 180),
            (400, 100, 455, 180),
            (520, 120, 575, 200),
            (520, 260, 610, 350),
            (650, 300, 710, 370),
            (120, 430, 260, 525),
            (125, 575, 245, 615),
            (500, 650, 535, 705),
        ]

    def test_separated_final_layout_can_still_be_labeled(self):
        from steadyhand.vision.task_parts import (
            FINAL_LAYOUT_ORDER,
            detect_dark_part_boxes,
            label_final_layout,
            project_part_boxes_to_image,
        )

        parts = detect_dark_part_boxes(self._image(self._separated_nine()))
        self.assertEqual(len(parts), 9)
        labeled = label_final_layout(parts)
        self.assertEqual(tuple(p["name"] for p in labeled), FINAL_LAYOUT_ORDER)

        projected = project_part_boxes_to_image(labeled, self.np.eye(3))
        self.assertEqual(len(projected), 9)
        for part in projected:
            self.assertEqual(len(part["center_image_px"]), 2)
            self.assertEqual(len(part["quad_image_px"]), 4)
            self.assertTrue(0.0 <= part["center_board_fraction"][0] <= 1.0)
            self.assertTrue(0.0 <= part["center_board_fraction"][1] <= 1.0)

    def test_missing_part_is_not_an_error(self):
        from steadyhand.vision.task_parts import detect_dark_part_boxes, label_final_layout

        parts = detect_dark_part_boxes(self._image(self._separated_nine()[:-1]))
        self.assertEqual(len(parts), 8)
        labeled = label_final_layout(parts)
        self.assertEqual(len(labeled), 8)
        self.assertTrue(all("name" not in part for part in labeled))

    def test_touching_parts_are_not_forced_to_split_or_rejected(self):
        from steadyhand.vision.task_parts import detect_dark_part_boxes

        boxes = self._separated_nine()
        boxes[4] = (540, 260, 660, 390)
        boxes[5] = (660, 300, 710, 370)
        parts = detect_dark_part_boxes(self._image(boxes))
        self.assertEqual(len(parts), 8)

    def test_extra_component_is_not_an_error(self):
        from steadyhand.vision.task_parts import detect_dark_part_boxes, label_final_layout

        boxes = self._separated_nine() + [(300, 500, 350, 550)]
        parts = detect_dark_part_boxes(self._image(boxes))
        self.assertEqual(len(parts), 10)
        labeled = label_final_layout(parts)
        self.assertEqual(len(labeled), 10)
        self.assertTrue(all("name" not in part for part in labeled))


if __name__ == "__main__":
    unittest.main()
