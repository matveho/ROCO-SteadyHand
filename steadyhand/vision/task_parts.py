"""Task-board part segmentation after coarse board-corner detection.

The competition board is 386 x 386 mm. We first
perspective-rectify the detected board to a square, then segment dark task
parts from the bright board surface.  This deliberately exploits the fixed
competition layout instead of attempting general object recognition.

Part segmentation is deliberately arrangement-agnostic. The board may be in an
initial, intermediate, or final state; parts may be missing, already moved,
separated, touching, or temporarily occluded. Detection therefore returns all
valid dark connected components it sees and never requires a specific count.
"""

from __future__ import annotations

from steadyhand.board_geometry import BOARD_SIZE_M


# Confirmed by operator against the Sep-27 physical final-state head image.
# Ordering matches detect_dark_part_boxes(): rectified board top-to-bottom,
# then left-to-right. This mapping is ONLY valid for the assembled/final
# layout. The source/pre-pick layout will get its own board-relative reference
# after today's fresh initial-state capture.
FINAL_LAYOUT_ORDER = (
    "hdmi",
    "rod_16mm",
    "bolt_8mm",
    "usb_a",
    "gear_60teeth",
    "gear_20teeth",
    "battery_size1",
    "battery_size5",
    "pin",
)


def label_final_layout(parts):
    """Attach final-layout names only when the exact final layout is present.

    This helper is advisory metadata, never an execution gate. If the current
    board does not contain exactly the confirmed final-layout detection count,
    return the detections unchanged and unlabeled.
    """
    if len(parts) != len(FINAL_LAYOUT_ORDER):
        return [dict(part) for part in parts]

    out = []
    for part, name in zip(parts, FINAL_LAYOUT_ORDER):
        item = dict(part)
        item["name"] = name
        item["identity_source"] = "confirmed_final_layout_spatial_order"
        out.append(item)
    return out

def rectify_board(rgb, corners_px, *, canonical_size=800):
    """Return (rectified_rgb, H_image_to_board).

    corners_px order must be TL, TR, BR, BL.
    """
    import cv2
    import numpy as np

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("rgb must have shape HxWx3")
    n = int(canonical_size)
    if n < 100:
        raise ValueError("canonical_size must be >= 100")

    src = np.asarray(corners_px, dtype=np.float32)
    if src.shape != (4, 2):
        raise ValueError("corners_px must be 4x2 in TL,TR,BR,BL order")
    dst = np.asarray(
        ((0, 0), (n - 1, 0), (n - 1, n - 1), (0, n - 1)),
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)
    bgr = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2BGR)
    warped_bgr = cv2.warpPerspective(bgr, H, (n, n))
    warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
    return warped_rgb, H


def detect_dark_part_boxes(
    rectified_rgb,
    *,
    dark_threshold=140,
    border_margin_px=20,
    min_area_px=600,
    max_area_px=30000,
):
    """Detect task-part boxes in an already rectified board image.

    Returns a list sorted top-to-bottom then left-to-right.  Each item contains
    board-pixel box/center, normalized center, and board millimetres measured
    from rectified top-left.

    The detector is intentionally count-agnostic. It returns every filtered
    connected component and never assumes all task parts are present or that
    any particular pair is touching. Touching objects may therefore appear as
    one component; separated objects appear separately.
    """
    import cv2
    import numpy as np

    image = np.asarray(rectified_rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("rectified_rgb must have shape HxWx3")
    h, w = image.shape[:2]
    if h != w:
        raise ValueError("rectified board must be square")

    gray = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY)
    dark = (gray < int(dark_threshold)).astype(np.uint8)

    m = int(border_margin_px)
    if m < 0 or 2 * m >= min(h, w):
        raise ValueError("invalid border_margin_px")
    if m:
        dark[:m, :] = 0
        dark[-m:, :] = 0
        dark[:, :m] = 0
        dark[:, -m:] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    components = []
    for i in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if int(min_area_px) <= area <= int(max_area_px):
            components.append(
                dict(
                    component_id=i,
                    x=x,
                    y=y,
                    w=bw,
                    h=bh,
                    area=area,
                    cx=float(centroids[i][0]),
                    cy=float(centroids[i][1]),
                )
            )

    boxes = [
        [c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"]]
        for c in components
    ]

    # Never split or reject detections based on an expected task-part count.
    # Board localization/calibration must work for arbitrary task state.
    boxes.sort(key=lambda b: ((b[1] + b[3]) / 2.0, (b[0] + b[2]) / 2.0))
    result = []
    denom = float(w - 1)
    for index, (x0, y0, x1, y1) in enumerate(boxes, 1):
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        result.append(
            {
                "index": index,
                "box_board_px": [int(x0), int(y0), int(x1), int(y1)],
                "center_board_px": [float(cx), float(cy)],
                "center_board_fraction": [float(cx / denom), float(cy / denom)],
                "center_board_mm_from_tl": [
                    float(1000.0 * BOARD_SIZE_M * cx / denom),
                    float(1000.0 * BOARD_SIZE_M * cy / denom),
                ],
            }
        )
    return result


def project_part_boxes_to_image(parts, H_image_to_board, *, padding_px=6):
    """Add image-space quadrilaterals and exact projected centers."""
    import cv2
    import numpy as np

    H = np.asarray(H_image_to_board, dtype=float)
    if H.shape != (3, 3):
        raise ValueError("H_image_to_board must be 3x3")
    Hinv = np.linalg.inv(H)

    out = []
    for part in parts:
        x0, y0, x1, y1 = (float(v) for v in part["box_board_px"])
        p = float(padding_px)
        quad = np.asarray(
            [[[x0 - p, y0 - p], [x1 + p, y0 - p],
              [x1 + p, y1 + p], [x0 - p, y1 + p]]],
            dtype=np.float32,
        )
        projected = cv2.perspectiveTransform(quad, Hinv)[0]

        cx, cy = (float(v) for v in part["center_board_px"])
        center = cv2.perspectiveTransform(
            np.asarray([[[cx, cy]]], dtype=np.float32), Hinv
        )[0, 0]

        item = dict(part)
        item["quad_image_px"] = [
            [float(x), float(y)] for x, y in projected
        ]
        item["center_image_px"] = [float(center[0]), float(center[1])]
        out.append(item)
    return out
