import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
except ImportError:
    np = None

from steadyhand.vision.placement import placement_digest, placement_tracker


@unittest.skipIf(np is None, "requires NumPy")
class PlacementVisionTests(unittest.TestCase):
    def test_digest_changes_when_release_geometry_changes(self):
        settings = {
            "offset_board_xy_m": [0.01, -0.02],
            "clearance_m": 0.012,
            "yaw_deg": 4.0,
        }
        altered = dict(settings, clearance_m=0.013)
        self.assertNotEqual(placement_digest(settings), placement_digest(altered))
        self.assertEqual(placement_digest(settings), placement_digest(dict(settings)))

    def test_tracker_stays_anchored_to_taught_goal(self):
        # A textured patch represents a hole/peg edge in a white release image.
        rng = np.random.default_rng(19)
        image = np.full((180, 240, 3), 235, dtype=np.uint8)
        patch = rng.integers(0, 255, (41, 41, 3), dtype=np.uint8)
        image[70:111, 90:131] = patch
        settings = {
            "image_shape": [180, 240],
            "template_uv": [20, 20],
            "goal_uv": [110, 90],
        }
        tracker = placement_tracker(image, patch, settings)
        located, score = tracker.locate(image)
        self.assertAlmostEqual(located[0], 110.0, delta=1.0)
        self.assertAlmostEqual(located[1], 90.0, delta=1.0)
        self.assertGreater(score, 0.7)


if __name__ == "__main__":
    unittest.main()
