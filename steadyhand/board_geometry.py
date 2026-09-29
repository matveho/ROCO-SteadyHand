"""Shared physical board geometry assumptions for offline and live perception."""

from __future__ import annotations

import math


# The competition field is a 386 mm board span. Keep this in one module so
# head perception, offline snapshots, and source localization cannot silently
# drift to the old 383 mm measurement or the generic 400 mm detector square.
BOARD_SIZE_M = 0.386
BOARD_SIZE_MM = 386.0
BOARD_MOTION_MODEL = "horizontal_translation_only_fixed_table_plane"


def board_relative_task_xy(source_xy, source_center_xy, *, rotation_deg=0.0,
                            mirror_x=False, mirror_y=False):
    """Map a task source point into the live board-relative XY convention.

    ``mirror_x`` is used for the legacy organizer coordinate export: its
    source X direction is reversed relative to the physical board photo.
    ``mirror_y`` applies the verified robot-facing forward/back reflection
    while preserving left/right.
    """
    dx = float(source_xy[0]) - float(source_center_xy[0])
    dy = float(source_xy[1]) - float(source_center_xy[1])
    if mirror_x:
        dx = -dx
    if mirror_y:
        dy = -dy
    angle = math.radians(float(rotation_deg))
    return (
        dx * math.cos(angle) - dy * math.sin(angle),
        dx * math.sin(angle) + dy * math.cos(angle),
    )


def validate_declared_task_layout(task_data):
    """Check operator-declared physical ordering before any target is used."""
    expected = task_data.get("physical_layout_expectations")
    if not expected:
        return
    parts = task_data.get("parts") or {}
    center = task_data.get("source_board_center_xy_m")
    rotation = task_data.get("task_coordinate_rotation_deg", 0.0)
    mirror = bool(task_data.get("task_coordinate_mirror_x", False))
    mirror_y = bool(task_data.get("task_coordinate_mirror_y", False))

    def point(name):
        part, kind = name.rsplit(".", 1)
        value = parts[part][kind]
        return board_relative_task_xy(
            value[:2], center, rotation_deg=rotation,
            mirror_x=mirror, mirror_y=mirror_y,
        )

    try:
        small = point(expected["small_battery"])
        large = point(expected["large_battery"])
        rod = point(expected["rod"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("physical_layout_expectations reference missing task points") from exc
    if not small[0] > large[0]:
        raise ValueError("task orientation puts battery_size5 left of battery_size1")
    if not large[0] > rod[0]:
        raise ValueError("task orientation puts the rod right of the batteries")
    near_robot = expected.get("near_robot") or (
        expected.get("small_battery"), expected.get("large_battery"),
    )
    try:
        near_sign = -1.0 if mirror_y else 1.0
        if any(point(name)[1] * near_sign <= 0 for name in near_robot):
            raise ValueError("task orientation does not put the declared near-robot parts on the near half")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("task orientation"):
            raise
        raise ValueError("physical_layout_expectations near_robot reference is invalid") from exc


def configured_board_plane_z(robot_config, fallback_m):
    """Return the fixed table-plane height used for board ray intersection."""
    board_config = robot_config.get("board_calibration") or {}
    value = board_config.get("board_plane_z_m", fallback_m)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("board_plane_z_m must be finite")
    return value


def validate_task_board_geometry(task_data):
    """Reject task-coordinate files that drift from the physical board contract."""
    try:
        width = float(task_data.get("board_width_m"))
        height = float(task_data.get("board_height_m", BOARD_SIZE_M))
    except (TypeError, ValueError) as exc:
        raise ValueError("task board dimensions must be finite numbers") from exc
    if not math.isfinite(width) or not math.isfinite(height):
        raise ValueError("task board dimensions must be finite numbers")
    if not math.isclose(width, BOARD_SIZE_M, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"task board width must be {BOARD_SIZE_M:.3f} m")
    if not math.isclose(height, BOARD_SIZE_M, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"task board height must be {BOARD_SIZE_M:.3f} m")
    model = task_data.get("board_motion_model", BOARD_MOTION_MODEL)
    if model != BOARD_MOTION_MODEL:
        raise ValueError(f"unsupported board motion model: {model!r}")
    validate_declared_task_layout(task_data)


def validate_task_coordinate_extent(task_data, *, names=None, tolerance_m=0.0):
    """Reject a target center that lies outside the declared board square."""
    center = task_data.get("source_board_center_xy_m")
    rotation = task_data.get("task_coordinate_rotation_deg", 0.0)
    mirror = bool(task_data.get("task_coordinate_mirror_x", False))
    mirror_y = bool(task_data.get("task_coordinate_mirror_y", False))
    half = BOARD_SIZE_M / 2.0 + float(tolerance_m)
    outside = []
    selected = None if names is None else set(names)
    for part in task_data.get("official_order") or []:
        for kind, value in (task_data.get("parts", {}).get(part) or {}).items():
            if selected is not None and f"{part}.{kind}" not in selected:
                continue
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            x, y = board_relative_task_xy(
                value[:2], center, rotation_deg=rotation,
                mirror_x=mirror, mirror_y=mirror_y,
            )
            if abs(x) > half or abs(y) > half:
                outside.append(f"{part}.{kind}")
    if outside:
        raise ValueError(
            "task coordinate center outside the 386 mm board: " + ", ".join(outside)
        )
