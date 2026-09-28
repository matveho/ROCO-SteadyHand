import unittest


class TaskPartVisionTests(unittest.TestCase):
    def test_detect_nine_parts_and_final_labels(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("NumPy/OpenCV unavailable")

        from steadyhand.vision.task_parts import (
            FINAL_LAYOUT_ORDER,
            detect_dark_part_boxes,
            label_final_layout,
            project_part_boxes_to_image,
        )

        image = np.full((800, 800, 3), 235, dtype=np.uint8)
        boxes = [
            (100, 80, 160, 150),
            (250, 90, 320, 180),
            (400, 100, 455, 180),
            (520, 120, 575, 200),
            # touching pair to exercise merged-component split
            (540, 260, 660, 390),
            (660, 300, 710, 370),
            (120, 430, 260, 525),
            (125, 575, 245, 615),
            (500, 650, 535, 705),
        ]
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(image, (x0, y0), (x1, y1), (20, 20, 20), -1)

        parts = detect_dark_part_boxes(image)
        self.assertEqual(len(parts), 9)
        labeled = label_final_layout(parts)
        self.assertEqual(tuple(p["name"] for p in labeled), FINAL_LAYOUT_ORDER)

        projected = project_part_boxes_to_image(labeled, np.eye(3))
        self.assertEqual(len(projected), 9)
        for part in projected:
            self.assertEqual(len(part["center_image_px"]), 2)
            self.assertEqual(len(part["quad_image_px"]), 4)
            self.assertTrue(0.0 <= part["center_board_fraction"][0] <= 1.0)
            self.assertTrue(0.0 <= part["center_board_fraction"][1] <= 1.0)


if __name__ == "__main__":
    unittest.main()
