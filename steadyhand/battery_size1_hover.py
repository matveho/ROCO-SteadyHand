"""battery_size1 coarse-hover provenance and safety contracts.

This module contains no robot I/O. It validates that a battery source
localization is tied to the same completed manual board calibration present at
motion time, and validates an XY-only coarse-hover plan without inventing a new
orientation or changing hover height.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .battery_size1_source import PART_NAME, file_sha256


SCHEMA_VERSION = 1
LOCALIZATION_METHODS = (
    "explicit_four_corner_homography",
    "operator_board_offset",
)


def load_source_localization(
    path,
    robot_config,
    manual_calibration,
    *,
    max_age_minutes=120.0,
    now=None,
):
    """Validate source-localizer output against current robot/manual calibration."""
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("battery source localization must be a JSON object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("battery source localization schema_version must be 1")
    if value.get("part") != PART_NAME:
        raise ValueError("source localization is not for battery_size1")
    if value.get("method") not in LOCALIZATION_METHODS:
        raise ValueError(
            "source localization method is not an accepted battery method"
        )
    if value.get("robot_name") != robot_config.get("robot_name"):
        raise ValueError("source localization belongs to a different robot")
    kin = robot_config.get("kinematics") or {}
    if value.get("base_frame") != kin.get("base_frame"):
        raise ValueError("source localization has the wrong base frame")

    age_minutes = _record_age_minutes(
        value.get("generated_at_utc"),
        max_age_minutes=max_age_minutes,
        now=now,
        name="source localization",
    )

    identity = value.get("battery_size1")
    if (
        not isinstance(identity, dict)
        or identity.get("identity_source") != "operator_explicit"
    ):
        raise ValueError("battery_size1 identity must be operator-explicit")

    provenance = value.get("manual_board_calibration")
    if not isinstance(provenance, dict):
        raise ValueError(
            "source localization lacks manual board calibration provenance"
        )
    if provenance.get("operator_confirmed_board_unchanged") is not True:
        raise ValueError(
            "localization did not record board-unchanged confirmation"
        )
    expected_hash = manual_calibration.get("sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError("current manual board calibration has no SHA256")
    if provenance.get("sha256") != expected_hash:
        raise ValueError(
            "source localization was produced from a different manual board calibration"
        )
    if (
        provenance.get("generated_at_utc")
        != manual_calibration.get("generated_at_utc")
    ):
        raise ValueError(
            "source localization manual-calibration timestamp does not match "
            "current calibration"
        )

    coarse_xy = _finite_vector(
        value.get("coarse_base_xy_m"), 2, "coarse base XY"
    )

    source_image = value.get("source_image")
    if (
        not isinstance(source_image, dict)
        or not isinstance(source_image.get("sha256"), str)
        or not source_image.get("sha256")
    ):
        raise ValueError(
            "source localization lacks source-image hash provenance"
        )

    return {
        "raw": value,
        "path": str(source),
        "sha256": file_sha256(source),
        "generated_at_utc": value["generated_at_utc"],
        "age_minutes": age_minutes,
        "method": value["method"],
        "coarse_base_xy_m": coarse_xy,
        "manual_calibration_sha256": expected_hash,
        "source_image_sha256": source_image["sha256"],
    }


def accepted_hover_from_manual(manual_calibration, *, floor_m):
    """Return the measured low-hover Z/quaternion from manual CENTER.

    This does not claim the quaternion is a newly calibrated physical vertical.
    It is only the orientation that the completed manual board calibration
    actually used at its accepted hover reference.
    """
    raw = manual_calibration.get("raw")
    if not isinstance(raw, dict):
        raise ValueError(
            "manual calibration validator did not provide raw record"
        )
    corrected = raw.get("manual_corrected")
    if not isinstance(corrected, dict):
        raise ValueError("manual calibration lacks manual_corrected poses")
    center = corrected.get("CENTER")
    if not isinstance(center, dict):
        raise ValueError("manual calibration lacks corrected CENTER pose")

    position = _finite_vector(
        center.get("position_m"), 3, "manual CENTER position"
    )
    quat = _unit_quaternion(
        center.get("quaternion_wxyz"),
        "manual CENTER quaternion",
    )
    floor = _finite_scalar(floor_m, "task floor")
    hover_z = position[2]
    if not floor + 0.060 <= hover_z <= floor + 0.120:
        raise ValueError(
            "manual CENTER hover Z is outside the 60-120 mm safe-hover band"
        )
    return {
        "z_m": hover_z,
        "quaternion_wxyz": quat,
        "orientation_status": raw.get("orientation_status"),
    }


def plan_hover_stages(
    current_pose,
    target_xy,
    *,
    hover_z_m,
    floor_m,
    hover_z_tolerance_m=0.008,
):
    """Plan a strictly XY-only move at the current accepted safe hover.

    The current measured Z must already match the completed manual calibration's
    hover reference. The planner never raises or lowers the TCP and preserves
    the exact live quaternion.
    """
    current_xyz = _finite_vector(
        current_pose.position_m, 3, "current TCP position"
    )
    quat = _unit_quaternion(
        current_pose.quaternion_wxyz,
        "current TCP quaternion",
    )
    target_x, target_y = _finite_vector(target_xy, 2, "target XY")
    floor = _finite_scalar(floor_m, "task floor")
    hover_z = _finite_scalar(hover_z_m, "hover Z")
    tolerance = _finite_positive(
        hover_z_tolerance_m, "hover Z tolerance"
    )

    if hover_z < floor:
        raise ValueError("hover Z is below configured floor")
    if current_xyz[2] < floor:
        raise ValueError("current TCP is below configured floor")
    if abs(current_xyz[2] - hover_z) > tolerance:
        raise ValueError(
            "current TCP is not at the accepted hover Z; "
            "no vertical motion is allowed"
        )

    from .models import Pose

    label = (
        "HOLD_SAFE_HOVER"
        if (
            abs(current_xyz[0] - target_x) <= 1e-9
            and abs(current_xyz[1] - target_y) <= 1e-9
        )
        else "PLANAR_TO_SOURCE_XY"
    )
    target = Pose(
        (target_x, target_y, current_xyz[2]),
        quat,
    )

    if target.position_m[2] < floor:
        raise AssertionError(
            "planned hover target crossed configured floor"
        )
    if target.position_m[2] != current_xyz[2]:
        raise AssertionError("hover plan changed Z")
    if target.quaternion_wxyz != quat:
        raise AssertionError("hover plan changed orientation")

    return [(label, target)]


def _record_age_minutes(value, *, max_age_minutes, now, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} has no generated_at_utc")
    try:
        stamp = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(
            f"{name} has invalid generated_at_utc"
        ) from exc
    if stamp.tzinfo is None:
        raise ValueError(f"{name} timestamp must include timezone")

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    age_s = (
        current.astimezone(timezone.utc)
        - stamp.astimezone(timezone.utc)
    ).total_seconds()
    limit = _finite_positive(
        max_age_minutes, "max localization age"
    ) * 60.0

    if age_s < -300.0:
        raise ValueError(
            f"{name} timestamp is unexpectedly in the future"
        )
    if age_s > limit:
        raise ValueError(
            f"{name} is stale: age={age_s/60.0:.1f} min, "
            f"limit={limit/60.0:.1f} min"
        )
    return max(0.0, age_s / 60.0)


def _finite_scalar(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _finite_positive(value, name):
    number = _finite_scalar(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be > 0")
    return number


def _finite_vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(
            f"{name} must contain {size} finite numbers"
        )
    return tuple(_finite_scalar(v, name) for v in value)


def _unit_quaternion(value, name):
    quat = _finite_vector(value, 4, name)
    norm2 = sum(v * v for v in quat)
    if abs(norm2 - 1.0) > 0.001:
        raise ValueError(f"{name} must be unit length")
    return quat
