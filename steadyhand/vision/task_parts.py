"""Task-board part segmentation after coarse board-corner detection.

The competition board is known and approximately 400 x 400 mm.  We first
perspective-rectify the detected board to a square, then segment dark task
parts from the bright board surface.  This deliberately exploits the fixed
competition layout instead of attempting general object recognition.

The current physical board has nine black/dark task parts.  In the head view
the two gears touch in the binary mask, so the common 8-component case is
split at the horizontal occupancy valley of the largest wide component.
"""

from __future__ import annotations


BOARD_SIZE_M = 0.400


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
    expected_parts=9,
):
    """Detect task-part boxes in an already rectified board image.

    Returns a list sorted top-to-bottom then left-to-right.  Each item contains
    board-pixel box/center, normalized center, and board millimetres measured
    from rectified top-left.

    This is intentionally a board-specific detector.  If exactly one pair has
    merged (the normal touching-gear case), the largest wide component is split
    at its column-occupancy valley.
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

    if len(boxes) == int(expected_parts) - 1:
        largest = max(components, key=lambda c: c["area"])
        # The touching-gear cluster is both the largest dark object and wider
        # than it is tall.  Refuse an arbitrary split if that signature is gone.
        if largest["w"] < 1.05 * largest["h"]:
            raise RuntimeError(
                f"Found {len(boxes)} components but largest component does not "
                "look like the expected touching-gear cluster"
            )

        x, y, bw, bh = (
            largest["x"], largest["y"], largest["w"], largest["h"]
        )
        local = (
            labels[y:y + bh, x:x + bw] == largest["component_id"]
        ).astype(np.uint8)
        occupancy = local.sum(axis=0).astype(float)
        smooth = np.convolve(occupancy, np.ones(9) / 9.0, mode="same")

        lo = int(round(0.55 * bw))
        hi = int(round(0.88 * bw))
        if hi <= lo + 2:
            raise RuntimeError("Touching-part component is too narrow to split")
        split_local = lo + int(np.argmin(smooth[lo:hi]))
        split_x = x + split_local

        yy, xx = np.nonzero(labels == largest["component_id"])
        left = xx < split_x
        right = ~left
        if int(left.sum()) < int(min_area_px) or int(right.sum()) < int(min_area_px):
            raise RuntimeError("Touching-part split produced an implausibly small part")

        def bbox(sel):
            sx, sy = xx[sel], yy[sel]
            return [
                int(sx.min()), int(sy.min()),
                int(sx.max()) + 1, int(sy.max()) + 1,
            ]

        merged_box = [
            largest["x"], largest["y"],
            largest["x"] + largest["w"], largest["y"] + largest["h"],
        ]
        boxes.remove(merged_box)
        boxes.extend((bbox(left), bbox(right)))

    if len(boxes) != int(expected_parts):
        raise RuntimeError(
            f"Expected {expected_parts} task parts, detected {len(boxes)} "
            f"after filtering/splitting"
        )

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
    """Add image-space quadrilaterals for board-space part boxes."""
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
        item = dict(part)
        item["quad_image_px"] = [
            [float(x), float(y)] for x, y in projected
        ]
        out.append(item)
    return out
