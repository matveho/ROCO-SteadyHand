"""Shared physical board geometry assumptions for offline and live perception."""

from __future__ import annotations

import math


# The competition field is a 386 mm board span. Keep this in one module so
# head perception, offline snapshots, and source localization cannot silently
# drift to the old 383 mm measurement or the generic 400 mm detector square.
BOARD_SIZE_M = 0.386
BOARD_SIZE_MM = 386.0
BOARD_MOTION_MODEL = "horizontal_translation_only_fixed_table_plane"


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
