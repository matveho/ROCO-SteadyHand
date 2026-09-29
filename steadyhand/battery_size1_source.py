"""battery_size1 source localization from manual board registration.

This module is intentionally independent of shared board/part perception.
It never infers part identity or part count. The operator explicitly identifies
battery_size1.

Reliable mapping for the first physical bring-up is:
  head image pixel
    -> operator-supplied four-corner planar homography
    -> metric board offset
    -> manually calibrated board XY frame
    -> coarse Vega base XY

No head intrinsics, depth, head kinematics or final-layout ordering are used.
"""

from __future__ import annotations

import math

from steadyhand.board_geometry import BOARD_SIZE_MM
from steadyhand.board_calibration import file_sha256, load_board_calibration


PART_NAME = "battery_size1"
MEASURED_BOARD_WIDTH_MM = BOARD_SIZE_MM


def load_manual_board_calibration(
    path,
    robot_config,
    *,
    max_age_minutes=720.0,
    now=None,
):
    """Validate either the current five-point or legacy manual calibration.

    The public return shape remains compatible with the battery pipeline while
    exposing the normalized surface/axis provenance used by newer callers.
    """
    manual = load_board_calibration(
        path, robot_config, max_age_minutes=max_age_minutes, now=now,
    )
    return manual


def dimensions_from_measured_width(
    *,
    width_axis,
    other_dimension_mm,
    measured_width_mm=MEASURED_BOARD_WIDTH_MM,
):
    """Map the fixed 386 mm board span onto explicit board X or board Y."""
    width = _finite_positive(measured_width_mm, "measured board width")
    other = _finite_positive(other_dimension_mm, "other board dimension")
    if width_axis not in ("x", "y"):
        raise ValueError("width_axis must be explicitly 'x' or 'y'")
    if width_axis == "x":
        return width, other
    return other, width


def homography_board_fraction(
    battery_pixel_uv,
    corners_by_board_sign,
    *,
    outside_margin=0.03,
):
    """Return normalized board (x_fraction,y_fraction) from explicit corners.

    Corner order is physical board coordinates, not screen order:
      xm_ym, xp_ym, xp_yp, xm_yp

    Fractions map those corners to (0,0), (1,0), (1,1), (0,1).
    """
    import cv2
    import numpy as np

    point = _finite_vector(battery_pixel_uv, 2, "battery pixel")
    required = ("xm_ym", "xp_ym", "xp_yp", "xm_yp")
    if set(corners_by_board_sign) != set(required):
        raise ValueError(
            "four corners must be named xm_ym,xp_ym,xp_yp,xm_yp"
        )
    src = np.asarray(
        [_finite_vector(corners_by_board_sign[name], 2, name) for name in required],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(src)):
        raise ValueError("board corner pixels must be finite")

    # Reject degenerate/self-crossed quadrilaterals before OpenCV produces an
    # unstable mapping. In the required physical order the signed cross
    # products must keep one sign.
    crosses = []
    for i in range(4):
        a = src[(i + 1) % 4] - src[i]
        b = src[(i + 2) % 4] - src[(i + 1) % 4]
        crosses.append(float(a[0] * b[1] - a[1] * b[0]))
    if min(abs(v) for v in crosses) < 25.0:
        raise ValueError("board corner quadrilateral is degenerate")
    if not (all(v > 0 for v in crosses) or all(v < 0 for v in crosses)):
        raise ValueError("board corners are crossed or not in physical perimeter order")

    dst = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    if not np.all(np.isfinite(H)) or abs(float(np.linalg.det(H))) < 1e-12:
        raise ValueError("board homography is singular")

    p = np.asarray([[[point[0], point[1]]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(p, H)[0, 0]
    fx, fy = float(mapped[0]), float(mapped[1])
    margin = float(outside_margin)
    if not math.isfinite(margin) or not 0 <= margin <= 0.10:
        raise ValueError("outside_margin must be in [0,0.10]")
    if not (-margin <= fx <= 1 + margin and -margin <= fy <= 1 + margin):
        raise ValueError(
            f"selected battery pixel maps outside board: fractions=({fx:.4f},{fy:.4f})"
        )
    return (fx, fy), H


def metric_offset_from_fraction(fraction_xy, *, board_x_mm, board_y_mm):
    fx, fy = _finite_vector(fraction_xy, 2, "board fraction")
    x_mm = _finite_positive(board_x_mm, "board X dimension")
    y_mm = _finite_positive(board_y_mm, "board Y dimension")
    return (
        (fx - 0.5) * x_mm / 1000.0,
        (fy - 0.5) * y_mm / 1000.0,
    )


def base_xy_from_board_offset(manual, board_offset_m):
    dx, dy = _finite_vector(board_offset_m, 2, "board offset")
    cx, cy = manual["center_base_xy_m"]
    ux = manual["board_x_unit_base_xy"]
    uy = manual["board_y_unit_base_xy"]
    return (
        cx + dx * ux[0] + dy * uy[0],
        cy + dx * ux[1] + dy * uy[1],
    )


def _finite_vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} finite numbers")
    result = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


def _finite_positive(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and > 0")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and > 0") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return number
