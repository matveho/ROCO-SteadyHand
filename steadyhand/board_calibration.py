"""Canonical, hardware-free validation of Vega board calibration records.

The competition pipeline and the battery source localizer used to validate
different JSON contracts.  This module accepts the current five-point record
and the older manual record, then exposes one normalized representation.  It
also turns the two measured board directions into the nearest rigid 2-D frame;
the original vectors and their angular residual remain available for audit.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FALLBACK = ROOT / "calibration" / "vega_board_manual_fallback.json"
CANONICAL_KIND = "vega_board_five_point_surface"


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def orthonormalize_xy_axes(raw_x, raw_y):
    """Return the nearest rigid +X/+Y frame and measured non-orthogonality.

    The two measured unit vectors are fitted symmetrically to a 2-D rotation
    (or reflection, preserving their measured handedness), rather than making
    one operator direction exact and moving the other by the whole residual.
    """
    x = _vector(raw_x, 2, "raw board X vector")
    y = _vector(raw_y, 2, "raw board Y vector")
    nx = math.hypot(*x)
    ny = math.hypot(*y)
    if nx < 1e-9 or ny < 1e-9:
        raise ValueError("board axes must have non-zero XY length")
    ux = (x[0] / nx, x[1] / nx)
    raw_uy = (y[0] / ny, y[1] / ny)
    measured_dot = ux[0] * raw_uy[0] + ux[1] * raw_uy[1]
    handedness = 1.0 if ux[0] * raw_uy[1] - ux[1] * raw_uy[0] >= 0 else -1.0
    if abs(ux[0] * raw_uy[1] - ux[1] * raw_uy[0]) < 1e-9:
        raise ValueError("board axes are collinear")
    # For r_y = handedness * perp(r_x), maximize r_x·x + r_y·y.
    a = ux[0] + handedness * raw_uy[1]
    b = ux[1] - handedness * raw_uy[0]
    theta = math.atan2(b, a)
    rx = (math.cos(theta), math.sin(theta))
    uy = (
        handedness * -math.sin(theta),
        handedness * math.cos(theta),
    )
    ux = rx
    angle_error_deg = math.degrees(math.asin(max(-1.0, min(1.0, measured_dot))))
    return ux, uy, measured_dot, angle_error_deg


def board_geometry_signature(corners_px):
    """Return translation-invariant image geometry for a four-corner board."""
    if not isinstance(corners_px, dict):
        return None
    try:
        points = {
            key: _vector(corners_px[key], 2, f"camera corner {key}")
            for key in ("tl", "tr", "br", "bl")
        }
    except (KeyError, ValueError):
        return None
    width = (math.dist(points["tl"], points["tr"]) + math.dist(points["bl"], points["br"])) / 2.0
    height = (math.dist(points["tl"], points["bl"]) + math.dist(points["tr"], points["br"])) / 2.0
    if width <= 1e-6 or height <= 1e-6:
        return None
    edge = (points["tr"][0] - points["tl"][0], points["tr"][1] - points["tl"][1])
    return {
        "width_px": width,
        "height_px": height,
        "aspect_ratio": width / height,
        "x_edge_angle_deg": math.degrees(math.atan2(edge[1], edge[0])),
    }


def compare_board_geometry(reference, current, *, max_scale_change=0.15,
                           max_aspect_change=0.10, max_angle_change_deg=10.0):
    """Validate that a new image is an XY board translation, not a new pose."""
    if not reference or not current:
        return {"valid": True, "reason": "geometry signature unavailable"}
    scale_w = float(current["width_px"]) / float(reference["width_px"])
    scale_h = float(current["height_px"]) / float(reference["height_px"])
    aspect_delta = abs(float(current["aspect_ratio"]) / float(reference["aspect_ratio"]) - 1.0)
    angle_delta = _angle_delta_deg(float(current["x_edge_angle_deg"]), float(reference["x_edge_angle_deg"]))
    checks = {
        "width_scale": scale_w,
        "height_scale": scale_h,
        "aspect_delta": aspect_delta,
        "angle_delta_deg": angle_delta,
    }
    valid = (
        abs(scale_w - 1.0) <= max_scale_change
        and abs(scale_h - 1.0) <= max_scale_change
        and aspect_delta <= max_aspect_change
        and angle_delta <= max_angle_change_deg
    )
    if valid:
        checks["reason"] = "compatible XY retake"
    else:
        checks["reason"] = "material board scale, aspect, or rotation change; full recalibration required"
    checks["valid"] = valid
    return checks


def load_board_calibration(path, robot_config, *, max_age_minutes=720.0,
                           now=None, fallback_path=DEFAULT_FALLBACK):
    """Load schema 2 or legacy schema 1 into one normalized dictionary."""
    source = Path(path)
    if not source.is_file() and source.name == "vega_board_manual.json" and Path(fallback_path).is_file():
        source = Path(fallback_path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manual board calibration must be a JSON object")
    _validate_identity(value, robot_config)
    generated = _parse_timestamp(value.get("generated_at_utc"))
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    age_s = (current.astimezone(timezone.utc) - generated.astimezone(timezone.utc)).total_seconds()
    if age_s < -300:
        raise ValueError("manual board calibration timestamp is unexpectedly in the future")
    permanent = value.get("permanent_fallback") is True
    max_age_s = _positive(max_age_minutes, "max calibration age") * 60.0
    if not permanent and age_s > max_age_s:
        raise ValueError(f"manual board calibration is stale: age={age_s / 60.0:.1f} min, limit={max_age_s / 60.0:.1f} min")

    frame = value.get("corrected_board_frame_xy")
    if not isinstance(frame, dict):
        raise ValueError("manual board calibration lacks corrected_board_frame_xy")
    center = _vector(frame.get("center_base_xy_m"), 2, "board center")
    raw_x = _vector(frame.get("raw_board_x_unit_base_xy", frame.get("board_x_unit_base_xy")), 2, "board +X axis")
    raw_y = _vector(frame.get("raw_board_y_unit_base_xy", frame.get("board_y_unit_base_xy")), 2, "board +Y axis")
    ux, uy, dot, angle_error = orthonormalize_xy_axes(raw_x, raw_y)

    if value.get("schema_version") == 1:
        nominal = _positive(value.get("nominal_axis_offset_m"), "manual nominal_axis_offset_m")
        for field in ("x_reference_distance_m", "y_reference_distance_m"):
            distance = _positive(frame.get(field), field)
            if not 0.45 * nominal <= distance <= 1.75 * nominal:
                raise ValueError(f"manual {field}={distance:.4f} m is inconsistent with nominal offset {nominal:.4f} m")
        plane = None
    else:
        plane = _surface_plane(value)
    signature = board_geometry_signature((value.get("camera_board_read") or {}).get("corners_px"))
    anchors = _anchors_from_samples(value.get("samples"), plane) if plane is not None else []
    return {
        "raw": value,
        "path": str(source),
        "sha256": file_sha256(source),
        "generated_at_utc": value["generated_at_utc"],
        "age_minutes": max(0.0, age_s / 60.0),
        "is_permanent_fallback": permanent,
        "schema_version": value.get("schema_version"),
        "calibration_kind": value.get("calibration_kind"),
        "center_base_xy_m": center,
        "board_x_unit_base_xy": ux,
        "board_y_unit_base_xy": uy,
        "raw_board_x_unit_base_xy": tuple(raw_x),
        "raw_board_y_unit_base_xy": tuple(raw_y),
        "axis_dot_raw": dot,
        "axis_angle_error_deg": angle_error,
        "surface_plane": ({"coefficients": plane, "anchors": anchors} if plane is not None else None),
        "camera_geometry_signature": signature,
        "calibration_camera_geometry_signature": signature,
    }


def _validate_identity(value, robot_config):
    if value.get("robot_name") != robot_config.get("robot_name"):
        raise ValueError("manual board calibration belongs to a different robot")
    kin = robot_config.get("kinematics") or {}
    if value.get("base_frame") != kin.get("base_frame"):
        raise ValueError("manual board calibration has the wrong base frame")
    if value.get("tcp_frame") != kin.get("ee_frame"):
        raise ValueError("manual board calibration has the wrong TCP frame")
    if value.get("schema_version") == 2 and value.get("calibration_kind") != CANONICAL_KIND:
        raise ValueError("task motion requires a completed five-point surface calibration")
    if value.get("schema_version") not in (1, 2):
        raise ValueError("unsupported manual board calibration schema_version")


def _surface_plane(value):
    plane = value.get("board_surface_plane_base") or {}
    coefficients = plane.get("coefficients") or {}
    result = tuple(_finite(coefficients.get(key), f"board plane {key}") for key in ("a", "b", "c"))
    return result


def _anchors_from_samples(samples, plane):
    anchors = []
    if not isinstance(samples, dict):
        return anchors
    a, b, c = plane
    for label in ("CENTER", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT"):
        sample = samples.get(label) or {}
        position = (sample.get("tip_r_pose") or {}).get("position_m")
        if not isinstance(position, list) or len(position) != 3:
            continue
        try:
            x, y, tip_z = (_finite(position[i], f"{label} position") for i in range(3))
            if sample.get("measured_clearance_mm") is not None:
                surface_z = tip_z - _finite(sample["measured_clearance_mm"], f"{label} clearance") / 1000.0
            elif sample.get("measured_surface_z_mm") is not None:
                value = _finite(sample["measured_surface_z_mm"], f"{label} surface")
                surface_z = value / 1000.0 if value >= 200.0 else tip_z - value / 1000.0
            else:
                continue
            anchors.append({"label": label, "x_m": x, "y_m": y, "residual_m": surface_z - (a * x + b * y + c)})
        except (TypeError, ValueError):
            continue
    return anchors


def _parse_timestamp(value):
    if not isinstance(value, str):
        raise ValueError("manual board calibration has no generated_at_utc")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("manual board calibration has invalid generated_at_utc") from exc
    if stamp.tzinfo is None:
        raise ValueError("manual board calibration timestamp must include timezone")
    return stamp


def _vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} finite numbers")
    result = tuple(_finite(v, name) for v in value)
    return result


def _finite(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _positive(value, name):
    result = _finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _angle_delta_deg(a, b):
    delta = (a - b + 180.0) % 360.0 - 180.0
    return abs(delta)
