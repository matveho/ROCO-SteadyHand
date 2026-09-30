"""Local placement-image matching; never searches globally across other holes."""

import hashlib
import json

from .wrist_servo import TemplateTracker


def placement_digest(settings):
    geometry = {key: settings[key] for key in ("offset_board_xy_m", "clearance_m", "yaw_deg")}
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


class PlacementTracker(TemplateTracker):
    """Fixed release-target patch with grayscale, color, edge and mask scores.

    Ambiguous local features fail closed: an identical-looking distant peg is
    never used as a replacement. Existing pickup trackers are unaffected.
    """

    enable_edge_matching = True

    def _locate_global(self, rgb):
        raise RuntimeError("Placement feature not unique in the local search region")

    def _edge_match(self, rgb, *, x_offset=0, y_offset=0):
        try:
            return super()._edge_match(rgb, x_offset=x_offset, y_offset=y_offset)
        except RuntimeError:
            # Dark holes/pegs on the white field often have almost no texture.
            # Compare binary shape occupancy, retaining a distinct-peak gate.
            gray = self._gray(rgb)
            threshold, mask = self.cv2.threshold(
                self.template, 0, 255, self.cv2.THRESH_BINARY_INV | self.cv2.THRESH_OTSU)
            fraction = float((mask > 0).mean())
            if not .08 < fraction < .92:
                raise RuntimeError("Placement feature has no distinct shape")
            image_mask = self.cv2.threshold(gray, threshold, 255, self.cv2.THRESH_BINARY_INV)[1]
            scores = self.cv2.matchTemplate(image_mask, mask, self.cv2.TM_CCOEFF_NORMED)
            return self._locate_from_map(
                scores, x_offset=x_offset, y_offset=y_offset,
                min_score=.70, min_margin=.05, label="dark_shape")


def placement_tracker(rgb, template, settings):
    if list(rgb.shape[:2]) != settings["image_shape"]:
        raise ValueError("Placement image resolution changed; re-teach this reference")
    return PlacementTracker.from_saved_template(
        rgb, template, template_uv=settings["template_uv"],
        initial_uv=settings["goal_uv"], search_radius=100,
        min_score=.80, min_margin=.06)
