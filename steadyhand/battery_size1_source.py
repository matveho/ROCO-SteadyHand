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

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


PART_NAME = "battery_size1"
MEASURED_BOARD_WIDTH_MM = 383.0
SCHEMA_VERSION = 1


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manual_board_calibration(
    path,
    robot_config,
    *,
    max_age_minutes=720.0,
    now=None,
):
    """Validate the manually corrected board XY frame for this competition robot."""
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manual board calibration must be a JSON object")
    if value.get("schema_version") != 1:
        raise ValueError("manual board calibration schema_version must be 1")
    if value.get("robot_name") != robot_config.get("robot_name"):
        raise ValueError("manual board calibration belongs to a different robot")
    kin = robot_config.get("kinematics") or {}
    if value.get("base_frame") != kin.get("base_frame"):
        raise ValueError("manual board calibration has the wrong base frame")
    if value.get("tcp_frame") != kin.get("ee_frame"):
        raise ValueError("manual board calibration has the wrong TCP frame")

    generated = value.get("generated_at_utc")
    if not isinstance(generated, str):
        raise ValueError("manual board calibration has no generated_at_utc")
    try:
        stamp = datetime.fromisoformat(generated.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("manual board calibration has invalid generated_at_utc") from exc
    if stamp.tzinfo is None:
        raise ValueError("manual board calibration timestamp must include timezone")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    age_s = (current.astimezone(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds()
    max_age_s = _finite_positive(max_age_minutes, "max calibration age") * 60.0
    if age_s < -300:
        raise ValueError("manual board calibration timestamp is unexpectedly in the future")
    if age_s > max_age_s:
        raise ValueError(
            f"manual board calibration is stale: age={age_s/60.0:.1f} min, "
            f"limit={max_age_s/60.0:.1f} min"
        )

    frame = value.get("corrected_board_frame_xy")
    if not isinstance(frame, dict):
        raise ValueError("manual board calibration lacks corrected_board_frame_xy")
    center = _finite_vector(frame.get("center_base_xy_m"), 2, "board center")
    ux = _finite_vector(frame.get("board_x_unit_base_xy"), 2, "board +X unit")
    uy = _finite_vector(frame.get("board_y_unit_base_xy"), 2, "board +Y unit")
    nx = math.hypot(*ux)
    ny = math.hypot(*uy)
    if abs(nx - 1.0) > 0.03 or abs(ny - 1.0) > 0.03:
        raise ValueError("manual board XY axes are not unit vectors")
    dot = ux[0] * uy[0] + ux[1] * uy[1]
    if abs(dot) > 0.25:
        raise ValueError(
            f"manual board XY axes are implausibly non-perpendicular (dot={dot:.3f})"
        )

    nominal = _finite_positive(
        value.get("nominal_axis_offset_m"),
        "manual nominal_axis_offset_m",
    )
    for field in ("x_reference_distance_m", "y_reference_distance_m"):
        distance = _finite_positive(frame.get(field), field)
        if not 0.45 * nominal <= distance <= 1.75 * nominal:
            raise ValueError(
                f"manual {field}={distance:.4f} m is inconsistent with "
                f"nominal offset {nominal:.4f} m"
            )

    return {
        "raw": value,
        "path": str(source),
        "sha256": file_sha256(source),
        "generated_at_utc": generated,
        "age_minutes": max(0.0, age_s / 60.0),
        "center_base_xy_m": center,
        "board_x_unit_base_xy": ux,
        "board_y_unit_base_xy": uy,
    }


def dimensions_from_measured_width(
    *,
    width_axis,
    other_dimension_mm,
    measured_width_mm=MEASURED_BOARD_WIDTH_MM,
):
    """Map the measured 383 mm width onto explicit board X or board Y."""
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
