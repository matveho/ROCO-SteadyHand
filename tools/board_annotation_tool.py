#!/usr/bin/env python3
"""Desktop board-image annotation and coordinate export tool.

The tool is intentionally usable without NumPy/OpenCV.  Pillow and Tk are
used only by the desktop front end; the homography, polygon-centroid, and
export code below are dependency-free and can be tested on a robot computer.

The workflow is:

1. Load the initial and final board photographs.
2. For each photograph, click board corners in TL, TR, BR, BL order and
   rectify the board to a front-facing image.
3. Optionally mark a measured rectangle and/or a ruler on the rectified board.
4. Annotate each part with a circle (center then edge) or a closed polygon.
5. Export an audit project JSON and a board-local ``task_coordinates`` JSON.

The generated task coordinates deliberately remain in a named board-local
frame.  They are not silently converted into robot-base coordinates: the
existing live calibration in ``steadyhand`` owns that transformation.

Run from the repository root with::

    python3 tools/board_annotation_tool.py

Optional image paths may be supplied on the command line::

    python3 tools/board_annotation_tool.py --initial initial.jpg --final final.jpg

Dependencies for the desktop UI are Pillow and a Tk installation.  The
geometry and JSON exporter do not require either dependency.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


TOOL_VERSION = "1.0"
DEFAULT_BOARD_WIDTH_M = 0.386
DEFAULT_BOARD_HEIGHT_M = 0.400
DEFAULT_RECTIFIED_LONG_EDGE_PX = 1100
DEFAULT_PART_NAMES = (
    "gear_60teeth",
    "gear_20teeth",
    "rod_16mm",
    "bolt_8mm",
    "usb_a",
    "hdmi",
    "pin",
    "battery_size1",
    "battery_size5",
)


def repository_part_order() -> list[str]:
    """Read the configured competition order when run inside this checkout."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "task_board.json"
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
        order = value.get("part_order")
        if isinstance(order, list) and order and all(isinstance(name, str) and name.strip() for name in order):
            return list(order)
    except (OSError, ValueError, TypeError):
        pass
    return list(DEFAULT_PART_NAMES)


class GeometryError(ValueError):
    """Raised when a calibration or annotation geometry is unusable."""


def _finite_number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise GeometryError(f"{label} must be a finite number")
    return result


def _point(value: Sequence[Any], label: str = "point") -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise GeometryError(f"{label} must contain two numbers")
    return (_finite_number(value[0], f"{label}.x"),
            _finite_number(value[1], f"{label}.y"))


def _points(values: Iterable[Sequence[Any]], label: str) -> list[tuple[float, float]]:
    return [_point(value, f"{label}[{index}]") for index, value in enumerate(values)]


def distance(a: Sequence[Any], b: Sequence[Any]) -> float:
    ax, ay = _point(a, "a")
    bx, by = _point(b, "b")
    return math.hypot(bx - ax, by - ay)


def polygon_signed_area(vertices: Sequence[Sequence[Any]]) -> float:
    points = _points(vertices, "vertices")
    if len(points) < 3:
        return 0.0
    return 0.5 * sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
    )


def polygon_centroid(vertices: Sequence[Sequence[Any]]) -> tuple[float, float]:
    """Return the area-weighted centroid of a closed polygon.

    A degenerate polygon falls back to its arithmetic mean, which keeps the
    exporter useful while the UI can still report a clear zero-area warning.
    """
    points = _points(vertices, "vertices")
    if len(points) < 3:
        raise GeometryError("a polygon needs at least three vertices")
    cross_sum = sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
    )
    if abs(cross_sum) < 1e-12:
        return (
            sum(x for x, _ in points) / len(points),
            sum(y for _, y in points) / len(points),
        )
    cx = sum((x0 + x1) * (x0 * y1 - x1 * y0)
             for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]))
    cy = sum((y0 + y1) * (x0 * y1 - x1 * y0)
             for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]))
    return (cx / (3.0 * cross_sum), cy / (3.0 * cross_sum))


def polygon_is_simple(vertices: Sequence[Sequence[Any]]) -> bool:
    """Return false when non-adjacent polygon edges cross."""
    points = _points(vertices, "vertices")
    if len(points) < 3:
        return False

    def orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
        return (min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9 and
                min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9)

    def intersects(a, b, c, d) -> bool:
        ab_c = orientation(a, b, c)
        ab_d = orientation(a, b, d)
        cd_a = orientation(c, d, a)
        cd_b = orientation(c, d, b)
        if ((ab_c > 1e-9 and ab_d < -1e-9) or (ab_c < -1e-9 and ab_d > 1e-9)) and ((cd_a > 1e-9 and cd_b < -1e-9) or (cd_a < -1e-9 and cd_b > 1e-9)):
            return True
        return ((abs(ab_c) <= 1e-9 and on_segment(a, c, b)) or
                (abs(ab_d) <= 1e-9 and on_segment(a, d, b)) or
                (abs(cd_a) <= 1e-9 and on_segment(c, a, d)) or
                (abs(cd_b) <= 1e-9 and on_segment(c, b, d)))

    edge_count = len(points)
    for first in range(edge_count):
        first_end = (first + 1) % edge_count
        for second in range(first + 1, edge_count):
            second_end = (second + 1) % edge_count
            if first in (second, second_end) or first_end in (second, second_end):
                continue
            if intersects(points[first], points[first_end], points[second], points[second_end]):
                return False
    return True


def _solve_linear(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    """Solve a small dense system with partial pivoting."""
    n = len(vector)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise GeometryError("linear system must be square")
    a = [[float(value) for value in row] + [float(vector[i])] for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(a[row][column]))
        pivot_value = abs(a[pivot][column])
        if pivot_value < 1e-12:
            raise GeometryError("calibration points are degenerate")
        if pivot != column:
            a[column], a[pivot] = a[pivot], a[column]
        scale = a[column][column]
        a[column] = [value / scale for value in a[column]]
        for row in range(n):
            if row == column:
                continue
            factor = a[row][column]
            if abs(factor) < 1e-15:
                continue
            a[row] = [left - factor * right for left, right in zip(a[row], a[column])]
    return [a[index][-1] for index in range(n)]


def solve_homography(
    source_points: Sequence[Sequence[Any]],
    destination_points: Sequence[Sequence[Any]],
) -> tuple[float, ...]:
    """Return the exact 3x3 homography mapping four source points to four targets.

    The result is row-major and normalized so ``h[8] == 1``.  Four-point
    calibration is sufficient for a planar board and avoids a hidden OpenCV
    dependency in the coordinate exporter.
    """
    source = _points(source_points, "source_points")
    destination = _points(destination_points, "destination_points")
    if len(source) != 4 or len(destination) != 4:
        raise GeometryError("a homography needs exactly four point pairs")
    matrix: list[list[float]] = []
    vector: list[float] = []
    for (x, y), (u, v) in zip(source, destination):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        vector.append(u)
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        vector.append(v)
    result = _solve_linear(matrix, vector) + [1.0]
    if not all(math.isfinite(value) for value in result):
        raise GeometryError("homography contains a non-finite value")
    return tuple(result)


def apply_homography(homography: Sequence[Any], point: Sequence[Any]) -> tuple[float, float]:
    if len(homography) != 9:
        raise GeometryError("homography must contain nine values")
    x, y = _point(point)
    h = [float(value) for value in homography]
    denominator = h[6] * x + h[7] * y + h[8]
    if abs(denominator) < 1e-12:
        raise GeometryError("homography maps point to infinity")
    return (
        (h[0] * x + h[1] * y + h[2]) / denominator,
        (h[3] * x + h[4] * y + h[5]) / denominator,
    )


def invert_homography(homography: Sequence[Any]) -> tuple[float, ...]:
    if len(homography) != 9:
        raise GeometryError("homography must contain nine values")
    a, b, c, d, e, f, g, h, i = [float(value) for value in homography]
    cofactors = (
        e * i - f * h,
        c * h - b * i,
        b * f - c * e,
        f * g - d * i,
        a * i - c * g,
        c * d - a * f,
        d * h - e * g,
        b * g - a * h,
        a * e - b * d,
    )
    determinant = a * cofactors[0] + b * cofactors[3] + c * cofactors[6]
    if abs(determinant) < 1e-12:
        raise GeometryError("homography is not invertible")
    return tuple(value / determinant for value in cofactors)


def _quad_is_valid(points: Sequence[Sequence[Any]]) -> bool:
    """Check that four ordered points form a non-self-intersecting quad."""
    try:
        p = _points(points, "quad")
    except GeometryError:
        return False
    if len(p) != 4:
        return False
    turns = []
    for a, b, c in zip(p, p[1:] + p[:1], p[2:] + p[:2]):
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        turns.append(cross)
    if any(abs(value) < 1e-8 for value in turns):
        return False
    return all(value > 0 for value in turns) or all(value < 0 for value in turns)


def _unit_to_meters(unit: str) -> float:
    values = {"m": 1.0, "meter": 1.0, "meters": 1.0,
              "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
              "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001}
    key = str(unit).strip().lower()
    if key not in values:
        raise GeometryError("length unit must be mm, cm, or m")
    return values[key]


def length_to_meters(value: Any, unit: str) -> float:
    result = _finite_number(value, "length") * _unit_to_meters(unit)
    if result <= 0.0:
        raise GeometryError("length must be greater than zero")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GeometryError("cannot export a non-finite value")
        return value
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _effective_dimensions(state: Mapping[str, Any], target_size: Sequence[int]) -> tuple[float, float]:
    width = _finite_number(state.get("board_width_m", DEFAULT_BOARD_WIDTH_M), "board_width_m")
    height = _finite_number(state.get("board_height_m", DEFAULT_BOARD_HEIGHT_M), "board_height_m")
    reference = state.get("reference_rectangle")
    if isinstance(reference, Mapping):
        scale_x = reference.get("scale_x_m_per_px")
        scale_y = reference.get("scale_y_m_per_px")
        if scale_x is not None and scale_y is not None:
            width = float(target_size[0] - 1) * _finite_number(scale_x, "reference scale_x")
            height = float(target_size[1] - 1) * _finite_number(scale_y, "reference scale_y")
    elif isinstance(state.get("ruler"), Mapping) and state["ruler"].get("use_for_coordinates", True):
        scale = _finite_number(state["ruler"].get("meters_per_pixel"), "ruler scale")
        width = float(target_size[0] - 1) * scale
        height = float(target_size[1] - 1) * scale
    if width <= 0.0 or height <= 0.0:
        raise GeometryError("board dimensions must be greater than zero")
    return width, height


def _coordinate_from_board_pixel(
    point: Sequence[Any], target_size: Sequence[int], dimensions_m: Sequence[float]
) -> tuple[float, float]:
    x, y = _point(point)
    width_px, height_px = (int(target_size[0]), int(target_size[1]))
    width_m, height_m = (_finite_number(dimensions_m[0], "width_m"),
                         _finite_number(dimensions_m[1], "height_m"))
    if width_px < 2 or height_px < 2:
        raise GeometryError("rectified board is too small")
    return (x / float(width_px - 1) * width_m,
            y / float(height_px - 1) * height_m)


def build_task_coordinates(project: Mapping[str, Any]) -> dict[str, Any]:
    """Build the robot-side board-local ``task_coordinates`` artifact."""
    states = project.get("states") or {}
    initial = states.get("initial") or {}
    final = states.get("final") or {}
    parts: dict[str, dict[str, Any]] = {}
    annotations_by_state: dict[str, list[Mapping[str, Any]]] = {}
    for state_name, state in (("initial", initial), ("final", final)):
        for annotation in state.get("annotations") or []:
            name = str(annotation.get("part_name") or "").strip()
            if not name:
                continue
            annotations_by_state.setdefault(name, []).append(annotation)
    warnings: list[str] = []
    known_parts = set(project.get("part_order") or [])
    for state_name, state in (("initial", initial), ("final", final)):
        for annotation in state.get("annotations") or []:
            name = str(annotation.get("part_name") or "").strip()
            if name and name not in known_parts:
                warnings.append(f"{state_name}: annotation {name!r} is not in part_order; audit only")
    for name in project.get("part_order") or []:
        records = annotations_by_state.get(name, [])
        initial_records = [record for record in records if record.get("state") == "initial"]
        final_records = [record for record in records if record.get("state") == "final"]
        # The state field is normally injected by export_project; retain a
        # fallback for callers passing state-local annotation records.
        if not initial_records:
            initial_records = list(initial.get("annotations_by_part", {}).get(name, []))
        if not final_records:
            final_records = list(final.get("annotations_by_part", {}).get(name, []))
        record: dict[str, Any] = {}
        if initial_records:
            # Existing SteadyHand consumers subtract source_board_center_xy_m
            # before applying the live board axes.  Keep task points in the
            # historical top-left-origin source convention; the audit record
            # also retains center-origin coordinates for direct consumers.
            record["pick"] = list(initial_records[0]["center_board_m_from_tl"])
            if len(initial_records) > 1:
                warnings.append(f"{name}: multiple initial annotations; first used for pick")
        if final_records:
            record["place"] = list(final_records[0]["center_board_m_from_tl"])
            if len(final_records) > 1:
                warnings.append(f"{name}: multiple final annotations; first used for place")
        if "pick" not in record:
            warnings.append(f"{name}: no initial annotation")
        if "place" not in record:
            warnings.append(f"{name}: no final annotation")
        # Existing consumers expect three source coordinates.  Z is explicitly
        # metadata only here; live board-plane calibration owns robot Z.
        if "pick" in record:
            record["pick"] = [record["pick"][0], record["pick"][1], 0.0]
        if "place" in record:
            record["place"] = [record["place"][0], record["place"][1], 0.0]
        parts[name] = record

    board = project.get("board") or {}
    width_m = _finite_number(board.get("width_m"), "board.width_m")
    height_m = _finite_number(board.get("height_m"), "board.height_m")
    task = {
        "schema_version": 1,
        "generated_at_utc": project.get("exported_at_utc"),
        "source_pose_frame": "board_local_annotation",
        "position_units": "m",
        "quaternion_order": "wxyz",
        "board_width_m": width_m,
        "board_height_m": height_m,
        "board_coordinate_convention": {
            "origin": "board center",
            "x_positive": "rectified image right",
            "y_positive": "rectified image down",
            "center_coordinates_are_m": True,
        },
        "task_coordinate_rotation_deg": 0.0,
        "source_board_center_xy_m": [width_m / 2.0, height_m / 2.0],
        "official_order": list(project.get("part_order") or []),
        "parts": parts,
        "warnings": warnings,
        "annotation_project_version": TOOL_VERSION,
    }
    return task


def export_project(
    path: str | os.PathLike[str],
    *,
    states: Mapping[str, Mapping[str, Any]],
    part_order: Sequence[str],
    board_width_m: float,
    board_height_m: float,
    source_root: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write the full audit project and compatible board-local task file."""
    project_path = Path(path).expanduser().resolve()
    serialized_states: dict[str, dict[str, Any]] = {}
    for state_name in ("initial", "final"):
        raw = dict(states.get(state_name) or {})
        annotations = []
        target_size = raw.get("rectified_size_px") or [0, 0]
        dimensions = _effective_dimensions(raw, target_size) if target_size[0] and target_size[1] else (board_width_m, board_height_m)
        for annotation in raw.get("annotations") or []:
            item = dict(annotation)
            item["state"] = state_name
            item["center_board_m_from_tl"] = list(
                _coordinate_from_board_pixel(item["center_board_px"], target_size, dimensions)
            )
            item["center_board_m"] = [
                item["center_board_m_from_tl"][0] - dimensions[0] / 2.0,
                item["center_board_m_from_tl"][1] - dimensions[1] / 2.0,
            ]
            if item.get("kind") == "circle":
                radius_px = _finite_number(item.get("radius_board_px"), "circle radius")
                item["radius_m"] = radius_px * ((dimensions[0] / max(1, target_size[0] - 1) + dimensions[1] / max(1, target_size[1] - 1)) / 2.0)
            annotations.append(item)
        raw["effective_dimensions_m"] = list(dimensions)
        raw["annotations"] = annotations
        serialized_states[state_name] = raw

    board = {
        "width_m": float(board_width_m),
        "height_m": float(board_height_m),
        "coordinate_frame": "board_local_annotation",
        "origin": "board center for exported task points; top-left retained in audit fields",
        "x_positive": "rectified image right",
        "y_positive": "rectified image down",
    }
    if source_root is not None:
        root = Path(source_root).expanduser().resolve()
        for state in serialized_states.values():
            image_path = state.get("image_path")
            if image_path:
                try:
                    state["image_path_relative"] = str(Path(image_path).resolve().relative_to(root))
                except ValueError:
                    pass
    project = {
        "schema_version": 1,
        "tool": {"name": "steadyhand-board-annotation", "version": TOOL_VERSION},
        "exported_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "part_order": list(part_order),
        "board": board,
        "states": serialized_states,
    }
    task = build_task_coordinates(project)
    initial_dimensions = serialized_states.get("initial", {}).get("effective_dimensions_m")
    final_dimensions = serialized_states.get("final", {}).get("effective_dimensions_m")
    if initial_dimensions and final_dimensions and any(
        abs(float(a) - float(b)) > 1e-6 for a, b in zip(initial_dimensions, final_dimensions)
    ):
        task["warnings"].append(
            "initial/final effective board scales differ; verify the two photographs use the same physical calibration"
        )
    project["task_coordinates"] = task
    _atomic_write_json(project_path, project)
    task_path = project_path.with_name(project_path.stem + ".task_coordinates.json")
    _atomic_write_json(task_path, task)
    return project_path, task_path, task


class AnnotationApp:
    """Tk/Pillow user interface.  Imported lazily so headless tests work."""

    def __init__(self, root: Any, *, initial: str | None = None, final: str | None = None, output: str | None = None):
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox, simpledialog, ttk
            from PIL import Image, ImageDraw, ImageTk
        except ImportError as exc:
            raise RuntimeError(
                "The desktop UI needs Tk and Pillow. Install Pillow and a Tk package "
                "(for example python3-tk) on the computer that will run it."
            ) from exc
        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.simpledialog = simpledialog
        self.ttk = ttk
        self.Image = Image
        self.ImageDraw = ImageDraw
        self.ImageTk = ImageTk
        self.root = root
        self.root.title("SteadyHand Board Annotation")
        self.root.minsize(1100, 700)
        self.root.geometry("1500x900")
        self.output_path = Path(output).expanduser() if output else None
        self.current_state_name = "initial"
        self.view = "source"
        self.mode = "idle"
        self.pending_points: list[tuple[float, float]] = []
        self.pending_polygon: list[tuple[float, float]] = []
        self.next_id = 1
        self.display_scale = 1.0
        self.zoom_factor = 1.0
        self.pan_offset = (0.0, 0.0)
        self.pan_anchor = None
        self.undo_stack: list[dict[str, Any]] = []
        self.undo_limit = 100
        self.display_offset = (0.0, 0.0)
        self.display_image_size = (1, 1)
        self.tk_image = None
        self.states: dict[str, dict[str, Any]] = {
            "initial": self._new_state(),
            "final": self._new_state(),
        }
        self.part_order = repository_part_order()
        self._build_ui()
        if initial:
            self._load_image(Path(initial), "initial")
        if final:
            self._load_image(Path(final), "final")
        self._refresh()

    @staticmethod
    def _new_state() -> dict[str, Any]:
        return {
            "image": None,
            "image_path": None,
            "board_width_m": DEFAULT_BOARD_WIDTH_M,
            "board_height_m": DEFAULT_BOARD_HEIGHT_M,
            "board_points_source_px": [],
            "homography_image_to_board": None,
            "homography_board_to_image": None,
            "rectified": None,
            "rectified_size_px": None,
            "ruler": None,
            "reference_rectangle": None,
            "annotations": [],
            "pixel_operations": [],
        }

    def _image_resample(self, name: str) -> Any:
        """Support both Pillow's enum API and older module constants."""
        enum = getattr(self.Image, "Resampling", self.Image)
        return getattr(enum, name)

    def _perspective_transform(self) -> Any:
        enum = getattr(self.Image, "Transform", self.Image)
        return getattr(enum, "PERSPECTIVE")

    def _build_ui(self) -> None:
        tk, ttk = self.tk, self.ttk
        toolbar = ttk.Frame(self.root, padding=(8, 6))
        toolbar.pack(fill="x")
        self._button(toolbar, "Open initial", lambda: self._choose_image("initial"))
        self._button(toolbar, "Open final", lambda: self._choose_image("final"))
        self._button(toolbar, "Crop", lambda: self._set_mode("crop"))
        self._button(toolbar, "Rotate 90°", self._rotate_current)
        self._button(toolbar, "Move image", lambda: self._set_mode("move"))
        self._button(toolbar, "Zoom +", lambda: self._zoom(1.25))
        self._button(toolbar, "Zoom −", lambda: self._zoom(1 / 1.25))
        self._button(toolbar, "Fit view", self._fit_view)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=7)
        ttk.Label(toolbar, text="View:").pack(side="left")
        self.view_var = tk.StringVar(value="source")
        for value, label in (("source", "Source"), ("board", "Rectified board")):
            ttk.Radiobutton(toolbar, text=label, variable=self.view_var, value=value,
                            command=self._change_view).pack(side="left")
        ttk.Label(toolbar, text="   Image:").pack(side="left")
        self.state_var = tk.StringVar(value="initial")
        for value, label in (("initial", "Initial"), ("final", "Final")):
            ttk.Radiobutton(toolbar, text=label, variable=self.state_var, value=value,
                            command=self._change_state).pack(side="left")
        self._button(toolbar, "Save/export", self._export_dialog)

        body = ttk.PanedWindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        canvas_frame = ttk.Frame(body)
        side = ttk.Frame(body, width=360, padding=(10, 4))
        body.add(canvas_frame, weight=5)
        body.add(side, weight=2)

        self.canvas = tk.Canvas(canvas_frame, background="#1d2329", highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._refresh())
        self.canvas.bind("<ButtonPress-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self.canvas.bind("<Motion>", self._canvas_motion)
        self.canvas.bind("<ButtonPress-2>", self._pan_start)
        self.canvas.bind("<B2-Motion>", self._pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._pan_release)
        self.canvas.bind("<MouseWheel>", self._mouse_wheel)
        self.canvas.bind("<Button-4>", lambda _event: self._zoom(1.1))
        self.canvas.bind("<Button-5>", lambda _event: self._zoom(1 / 1.1))
        self.canvas.bind("<Return>", lambda _event: self._close_polygon())
        self.canvas.bind("<Escape>", lambda _event: self._set_mode("idle"))
        self.root.bind_all("<Control-z>", self._undo_key)
        self.root.bind_all("<Control-Z>", self._undo_key)

        title = ttk.Label(side, text="Calibration and annotation", font=("TkDefaultFont", 12, "bold"))
        title.pack(anchor="w", pady=(0, 8))
        self.status_var = tk.StringVar()
        ttk.Label(side, textvariable=self.status_var, wraplength=330, foreground="#9a3c00").pack(anchor="w", fill="x", pady=(0, 8))

        dimensions = ttk.LabelFrame(side, text="Board dimensions")
        dimensions.pack(fill="x", pady=4)
        self.width_var = tk.StringVar(value=f"{DEFAULT_BOARD_WIDTH_M:.4f}")
        self.height_var = tk.StringVar(value=f"{DEFAULT_BOARD_HEIGHT_M:.4f}")
        self._labeled_entry(dimensions, "Width (m)", self.width_var, 0)
        self._labeled_entry(dimensions, "Height (m)", self.height_var, 1)
        self._grid_button(dimensions, "Set 4 board corners", lambda: self._set_mode("board"), row=2)
        self._grid_button(dimensions, "Rectify board", self._rectify_current, row=3)
        self._grid_button(dimensions, "Measured rectangle", lambda: self._set_mode("reference_rectangle"), row=4)
        self._grid_button(dimensions, "Set ruler scale", lambda: self._set_mode("ruler"), row=5)

        annotation_frame = ttk.LabelFrame(side, text="Annotation")
        annotation_frame.pack(fill="x", pady=4)
        ttk.Label(annotation_frame, text="Part name").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.part_var = tk.StringVar(value=self.part_order[0])
        self.part_combo = ttk.Combobox(annotation_frame, textvariable=self.part_var,
                                       values=self.part_order, width=24)
        self.part_combo.grid(row=0, column=1, sticky="ew", padx=4, pady=3)
        annotation_frame.columnconfigure(1, weight=1)
        ttk.Label(annotation_frame, text="Shape").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self.shape_var = tk.StringVar(value="circle")
        ttk.Combobox(annotation_frame, textvariable=self.shape_var,
                     values=("circle", "pen"), state="readonly", width=22).grid(row=1, column=1, sticky="ew", padx=4, pady=3)
        self._grid_button(annotation_frame, "Start circle (center → edge)", lambda: self._set_mode("circle"), row=2)
        self._grid_button(annotation_frame, "Start pen", lambda: self._set_mode("pen"), row=3)
        self._grid_button(annotation_frame, "Close polygon", self._close_polygon, row=4)
        self._grid_button(annotation_frame, "Undo point", self._undo_point, row=5)
        self._grid_button(annotation_frame, "Delete selected", self._delete_selected, row=6)

        toggles = ttk.LabelFrame(side, text="Overlay")
        toggles.pack(fill="x", pady=4)
        self.show_names = tk.BooleanVar(value=True)
        self.show_coords = tk.BooleanVar(value=True)
        self.show_centers = tk.BooleanVar(value=True)
        for row, (variable, label) in enumerate(((self.show_names, "part names"), (self.show_coords, "coordinates"), (self.show_centers, "centers"))):
            ttk.Checkbutton(toggles, text=label, variable=variable, command=self._refresh).grid(row=row, column=0, sticky="w", padx=4, pady=2)

        list_frame = ttk.LabelFrame(side, text="Annotations in current image")
        list_frame.pack(fill="both", expand=True, pady=4)
        self.annotation_list = tk.Listbox(list_frame, height=10, exportselection=False)
        self.annotation_list.pack(fill="both", expand=True, padx=4, pady=4)
        self.annotation_list.bind("<<ListboxSelect>>", lambda _event: self._refresh())
        self.help_var = tk.StringVar(value="Load both images, then set board corners for each.")
        ttk.Label(side, textvariable=self.help_var, wraplength=330).pack(anchor="w", pady=(4, 0))

    def _button(self, parent: Any, label: str, command: Any) -> None:
        self.ttk.Button(parent, text=label, command=command).pack(anchor="w", fill="x", padx=4, pady=2)

    def _grid_button(self, parent: Any, label: str, command: Any, *, row: int) -> None:
        self.ttk.Button(parent, text=label, command=command).grid(
            row=row, column=0, columnspan=2, sticky="ew", padx=4, pady=2
        )

    def _labeled_entry(self, parent: Any, label: str, variable: Any, row: int) -> None:
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=3)
        self.ttk.Entry(parent, textvariable=variable, width=12).grid(row=row, column=1, sticky="ew", padx=4, pady=3)
        parent.columnconfigure(1, weight=1)

    def _make_undo_snapshot(self) -> dict[str, Any]:
        """Capture editable geometry and viewport state, never display pixels."""
        state = self.state
        keys = (
            "image", "image_path", "board_width_m", "board_height_m",
            "board_points_source_px", "homography_image_to_board",
            "homography_board_to_image", "rectified", "rectified_size_px",
            "ruler", "reference_rectangle", "annotations", "pixel_operations",
        )
        state_copy = {}
        for key in keys:
            value = state.get(key)
            # Pillow images are replaced, rather than edited in place, by the
            # crop/rotate/rectify operations. Keeping the immutable object
            # reference makes undo restore those views without re-encoding.
            state_copy[key] = value if key in ("image", "rectified") else copy.deepcopy(value)
        return {
            "state_name": self.current_state_name,
            "state": state_copy,
            "view": self.view,
            "mode": self.mode,
            "pending_points": copy.deepcopy(self.pending_points),
            "pending_polygon": copy.deepcopy(self.pending_polygon),
            "zoom_factor": self.zoom_factor,
            "pan_offset": self.pan_offset,
            "next_id": self.next_id,
        }

    def _push_undo(self) -> None:
        snapshot = self._make_undo_snapshot()
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > self.undo_limit:
            del self.undo_stack[:-self.undo_limit]

    def _restore_undo_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        state_name = str(snapshot["state_name"])
        self.current_state_name = state_name
        self.state_var.set(state_name)
        self.states[state_name] = dict(snapshot["state"])
        self.view = str(snapshot["view"])
        self.view_var.set(self.view)
        self.mode = str(snapshot["mode"])
        self.pending_points = copy.deepcopy(snapshot["pending_points"])
        self.pending_polygon = copy.deepcopy(snapshot["pending_polygon"])
        self.zoom_factor = float(snapshot["zoom_factor"])
        self.pan_offset = tuple(snapshot["pan_offset"])
        self.next_id = int(snapshot["next_id"])
        self.width_var.set(f"{self.state.get('board_width_m', DEFAULT_BOARD_WIDTH_M):.4f}")
        self.height_var.set(f"{self.state.get('board_height_m', DEFAULT_BOARD_HEIGHT_M):.4f}")
        self._set_canvas_cursor()
        self._refresh()

    def _undo_last(self) -> None:
        if not self.undo_stack:
            self.status_var.set("Nothing to undo.")
            return
        self._restore_undo_snapshot(self.undo_stack.pop())
        self.status_var.set("Undid the last point, annotation, zoom, or image movement.")

    def _undo_key(self, _event: Any = None) -> str:
        self._undo_last()
        return "break"

    @property
    def state(self) -> dict[str, Any]:
        return self.states[self.current_state_name]

    def _choose_image(self, state_name: str) -> None:
        path = self.filedialog.askopenfilename(title=f"Open {state_name} image", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*")])
        if path:
            self._load_image(Path(path), state_name)

    def _load_image(self, path: Path, state_name: str) -> None:
        try:
            image = self.Image.open(path).convert("RGB")
        except Exception as exc:
            self.messagebox.showerror("Open image", f"Could not open {path}: {exc}")
            return
        state = self.states[state_name]
        state.clear()
        state.update(self._new_state())
        state["image"] = image
        state["image_path"] = str(path.resolve())
        self.current_state_name = state_name
        self.state_var.set(state_name)
        self.view = "source"
        self.view_var.set("source")
        self.pan_offset = (0.0, 0.0)
        self.pending_points.clear()
        self.pending_polygon.clear()
        self._set_mode("idle")
        self._refresh()

    def _change_state(self) -> None:
        self.current_state_name = self.state_var.get()
        self.width_var.set(f"{self.state.get('board_width_m', DEFAULT_BOARD_WIDTH_M):.4f}")
        self.height_var.set(f"{self.state.get('board_height_m', DEFAULT_BOARD_HEIGHT_M):.4f}")
        if self.view == "board" and self.state.get("rectified") is None:
            self.view = "source"
            self.view_var.set("source")
        self.pending_points.clear()
        self.pending_polygon.clear()
        self.pan_offset = (0.0, 0.0)
        self._refresh()

    def _change_view(self) -> None:
        requested = self.view_var.get()
        if requested == "board" and self.state.get("rectified") is None:
            self.view_var.set("source")
            self.status_var.set("Rectify this image before using the board view.")
            return
        self.view = requested
        self.pan_offset = (0.0, 0.0)
        self.zoom_factor = 1.0
        self.pending_points.clear()
        self.pending_polygon.clear()
        self.pan_offset = (0.0, 0.0)
        self._set_mode("idle")
        self._refresh()

    def _set_mode(self, mode: str) -> None:
        if mode in ("circle", "pen") and self.view != "board":
            self.status_var.set("Switch to Rectified board view first.")
            return
        if mode in ("ruler", "reference_rectangle") and self.view == "source" and not self.state.get("homography_image_to_board"):
            self.status_var.set("Set board corners and rectify this image before measuring a source-view reference.")
            return
        if mode in ("board", "crop") and self.view != "source":
            self.view = "source"
            self.view_var.set("source")
            self.pan_offset = (0.0, 0.0)
            self.zoom_factor = 1.0
        self.mode = mode
        self.pending_points.clear()
        self.pending_polygon.clear()
        self._set_canvas_cursor()
        messages = {
            "board": "Click board corners in this order: top-left, top-right, bottom-right, bottom-left.",
            "crop": "Click two opposite crop corners.",
            "ruler": "Click the two ends of the ruler/reference length.",
            "reference_rectangle": "Click measured rectangle corners in TL, TR, BR, BL order.",
            "circle": "Click circle center, then one point on its edge.",
            "pen": "Click polygon vertices. Use Close polygon or click the first vertex.",
            "move": "Drag the image with the left mouse button. Existing points stay fixed to the image.",
            "idle": "Choose a calibration or annotation action.",
        }
        self.status_var.set(messages.get(mode, mode))
        self._refresh()

    def _set_canvas_cursor(self) -> None:
        if hasattr(self, "canvas"):
            self.canvas.configure(cursor="fleur" if self.mode == "move" else "crosshair")

    def _display_source_or_board(self) -> Any:
        return self.state.get("rectified") if self.view == "board" else self.state.get("image")

    def _refresh(self) -> None:
        image = self._display_source_or_board()
        self.canvas.delete("all")
        if image is None:
            self.status_var.set("Open an initial and final image to begin.")
            self.annotation_list.delete(0, "end")
            return
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        image_width, image_height = image.size
        fit_scale = min((canvas_width - 20) / image_width, (canvas_height - 20) / image_height)
        scale = fit_scale * self.zoom_factor
        scale = max(0.05, min(scale, 5.0))
        self.display_scale = scale
        display_size = (max(1, round(image_width * scale)), max(1, round(image_height * scale)))
        self.display_image_size = display_size
        offset = ((canvas_width - display_size[0]) / 2.0 + self.pan_offset[0],
                  (canvas_height - display_size[1]) / 2.0 + self.pan_offset[1])
        self.display_offset = offset
        preview = image.resize(display_size, self._image_resample("LANCZOS"))
        self.tk_image = self.ImageTk.PhotoImage(preview)
        self.canvas.create_image(offset[0], offset[1], image=self.tk_image, anchor="nw", tags="image")
        self._draw_overlays()
        self._refresh_list()
        self._update_help()

    def _image_to_canvas(self, point: Sequence[Any]) -> tuple[float, float]:
        x, y = _point(point)
        return (self.display_offset[0] + x * self.display_scale,
                self.display_offset[1] + y * self.display_scale)

    def _canvas_to_image(self, event: Any) -> tuple[float, float] | None:
        x = (float(event.x) - self.display_offset[0]) / self.display_scale
        y = (float(event.y) - self.display_offset[1]) / self.display_scale
        image = self._display_source_or_board()
        if image is None or not (0 <= x <= image.size[0] - 1 and 0 <= y <= image.size[1] - 1):
            return None
        return (x, y)

    def _draw_overlays(self) -> None:
        draw = self.canvas
        state = self.state
        if self.view == "source":
            for index, point in enumerate(state.get("board_points_source_px") or []):
                x, y = self._image_to_canvas(point)
                draw.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#ffd166", outline="#111", width=2)
                draw.create_text(x + 10, y - 10, text=str(index + 1), fill="#ffd166", anchor="w", font=("TkDefaultFont", 10, "bold"))
            if len(state.get("board_points_source_px") or []) == 4 and state.get("homography_image_to_board"):
                points = [self._image_to_canvas(point) for point in state["board_points_source_px"]]
                draw.create_line(*(coord for point in points for coord in point), fill="#ffd166", width=2, joinstyle="round")
            if self.mode in ("board", "crop") and len(self.pending_points) > 1:
                points = [self._image_to_canvas(point) for point in self.pending_points]
                draw.create_line(*(coord for point in points for coord in point), fill="#00e5ff", width=2)
            if self.mode == "crop" and len(self.pending_points) == 2:
                a, b = [self._image_to_canvas(point) for point in self.pending_points]
                draw.create_rectangle(a[0], a[1], b[0], b[1], outline="#00e5ff", width=2)
            if self.mode in ("ruler", "reference_rectangle") and self.pending_points:
                points = [self._image_to_canvas(point) for point in self.pending_points]
                draw.create_line(*(coord for point in points for coord in point), fill="#9b5de5", width=2)
            return

        for annotation_index, annotation in enumerate(state.get("annotations") or []):
            selected = self._selected_annotation_index() == annotation_index
            color = "#ff4d6d" if selected else "#00e5ff"
            if annotation.get("kind") == "circle":
                cx, cy = self._image_to_canvas(annotation["center_board_px"])
                radius = float(annotation["radius_board_px"]) * self.display_scale
                self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline=color, width=3 if selected else 2)
                if self.show_centers.get():
                    self.canvas.create_line(cx - 8, cy, cx + 8, cy, fill="#ffcf33", width=2)
                    self.canvas.create_line(cx, cy - 8, cx, cy + 8, fill="#ffcf33", width=2)
            else:
                points = [self._image_to_canvas(point) for point in annotation["points_board_px"]]
                self.canvas.create_polygon(*(coord for point in points for coord in point), outline=color, fill="", width=3 if selected else 2)
                if self.show_centers.get():
                    cx, cy = self._image_to_canvas(annotation["center_board_px"])
                    self.canvas.create_line(cx - 8, cy, cx + 8, cy, fill="#ffcf33", width=2)
                    self.canvas.create_line(cx, cy - 8, cx, cy + 8, fill="#ffcf33", width=2)
            if self.show_names.get() or self.show_coords.get():
                cx, cy = self._image_to_canvas(annotation["center_board_px"])
                label = []
                if self.show_names.get():
                    label.append(str(annotation.get("part_name") or "unnamed"))
                if self.show_coords.get() and annotation.get("center_board_m"):
                    x, y = annotation["center_board_m"]
                    label.append(f"({x:+.4f}, {y:+.4f}) m")
                self.canvas.create_text(cx + 10, cy - 10, text="\n".join(label), fill="#fff3b0", anchor="sw", font=("TkDefaultFont", 10, "bold"))

        if self.pending_polygon:
            points = [self._image_to_canvas(point) for point in self.pending_polygon]
            self.canvas.create_line(*(coord for point in points for coord in point), fill="#ff9f1c", width=2)
            for point in points:
                self.canvas.create_oval(point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4, fill="#ff9f1c", outline="#111")
        if len(self.pending_points) == 2 and self.mode == "ruler":
            points = [self._image_to_canvas(point) for point in self.pending_points]
            self.canvas.create_line(*(coord for point in points for coord in point), fill="#ffcf33", width=3)
        if len(self.pending_points) == 4 and self.mode == "reference_rectangle":
            points = [self._image_to_canvas(point) for point in self.pending_points]
            self.canvas.create_line(*(coord for point in points for coord in point), fill="#9b5de5", width=2)

    def _refresh_list(self) -> None:
        self.annotation_list.delete(0, "end")
        for index, annotation in enumerate(self.state.get("annotations") or []):
            name = annotation.get("part_name") or "unnamed"
            kind = annotation.get("kind")
            x, y = annotation.get("center_board_m", (None, None))
            suffix = f"  ({x:+.4f}, {y:+.4f}) m" if x is not None else ""
            self.annotation_list.insert("end", f"{index + 1}. {name} [{kind}]{suffix}")

    def _selected_annotation_index(self) -> int | None:
        selection = self.annotation_list.curselection()
        return int(selection[0]) if selection else None

    def _update_help(self) -> None:
        if self.view == "board":
            ruler = self.state.get("ruler")
            ref = self.state.get("reference_rectangle")
            details = ["Board view: annotate with center→edge circles or closed polygons."]
            if ruler:
                details.append(f"Ruler: {ruler['meters_per_pixel']:.6g} m/px")
            if ref:
                details.append("Measured rectangle scale is active for coordinates.")
            self.help_var.set(" ".join(details))
        else:
            self.help_var.set("Source view: select four board corners, then rectify. Crop/rotate reset that image calibration.")

    def _canvas_press(self, event: Any) -> None:
        if self.mode == "move":
            self._pan_start(event)
            return
        self._canvas_click(event)

    def _canvas_drag(self, event: Any) -> None:
        if self.mode == "move":
            self._pan_move(event)

    def _canvas_release(self, event: Any) -> None:
        if self.mode == "move":
            self._pan_release(event)

    def _canvas_click(self, event: Any) -> None:
        point = self._canvas_to_image(event)
        if point is None:
            return
        if self.mode == "board":
            self._push_undo()
            self.pending_points.append(point)
            if len(self.pending_points) == 4:
                if not _quad_is_valid(self.pending_points):
                    self.status_var.set("Those points do not form a valid ordered quadrilateral; undo and retry.")
                    self.pending_points.clear()
                    return
                self.state["board_points_source_px"] = list(self.pending_points)
                self.pending_points.clear()
                self._rectify_current(record_undo=False)
            self._refresh()
            return
        if self.mode == "crop":
            self._push_undo()
            self.pending_points.append(point)
            if len(self.pending_points) == 2:
                self._apply_crop()
            self._refresh()
            return
        if self.mode == "ruler":
            self._push_undo()
            self.pending_points.append(point)
            if len(self.pending_points) == 2:
                self._set_ruler_from_points()
            self._refresh()
            return
        if self.mode == "reference_rectangle":
            self._push_undo()
            self.pending_points.append(point)
            if len(self.pending_points) == 4:
                self._set_reference_rectangle()
            self._refresh()
            return
        if self.mode == "circle":
            self._push_undo()
            self.pending_points.append(point)
            if len(self.pending_points) == 2:
                self._create_circle()
            self._refresh()
            return
        if self.mode == "pen":
            self._push_undo()
            if self.pending_polygon and distance(point, self.pending_polygon[0]) <= 10.0 / max(self.display_scale, 0.01):
                self._close_polygon(record_undo=False)
            else:
                self.pending_polygon.append(point)
            self._refresh()

    def _canvas_motion(self, event: Any) -> None:
        if self.mode == "pen" and self.pending_polygon:
            point = self._canvas_to_image(event)
            if point:
                self.status_var.set(f"Pen: {len(self.pending_polygon)} vertices; click first vertex or Close polygon. Cursor {point[0]:.1f}, {point[1]:.1f}")

    def _mouse_wheel(self, event: Any) -> None:
        self._zoom(1.1 if event.delta > 0 else 1 / 1.1)

    def _pan_start(self, event: Any) -> None:
        self._push_undo()
        self.pan_anchor = (int(event.x), int(event.y), self.pan_offset[0], self.pan_offset[1])

    def _pan_move(self, event: Any) -> None:
        if self.pan_anchor is None:
            return
        x0, y0, ox, oy = self.pan_anchor
        self.pan_offset = (ox + int(event.x) - x0, oy + int(event.y) - y0)
        self._refresh()

    def _pan_release(self, _event: Any) -> None:
        self.pan_anchor = None

    def _zoom(self, factor: float) -> None:
        # Keep a fit-to-window baseline and apply a user zoom multiplier so
        # precise clicks remain possible on a large board.
        self._push_undo()
        self.zoom_factor = max(0.25, min(5.0, self.zoom_factor * float(factor)))
        self._refresh()

    def _fit_view(self) -> None:
        self._push_undo()
        self.zoom_factor = 1.0
        self.pan_offset = (0.0, 0.0)
        self._refresh()

    def _undo_point(self) -> None:
        self._push_undo()
        if self.pending_polygon:
            self.pending_polygon.pop()
        elif self.pending_points:
            self.pending_points.pop()
        self._refresh()

    def _rectify_current(self, *, record_undo: bool = True) -> None:
        state = self.state
        image = state.get("image")
        points = state.get("board_points_source_px") or self.pending_points
        if image is None:
            self.status_var.set("Open an image first.")
            return
        if len(points) != 4 or not _quad_is_valid(points):
            self.status_var.set("Set four valid board corners in TL, TR, BR, BL order first.")
            return
        if record_undo:
            self._push_undo()
        try:
            width_m = _finite_number(self.width_var.get(), "board width")
            height_m = _finite_number(self.height_var.get(), "board height")
            if width_m <= 0 or height_m <= 0:
                raise GeometryError("board dimensions must be greater than zero")
        except GeometryError as exc:
            self.status_var.set(str(exc))
            return
        state["board_width_m"] = width_m
        state["board_height_m"] = height_m
        aspect = width_m / height_m
        if aspect >= 1.0:
            target_width = DEFAULT_RECTIFIED_LONG_EDGE_PX
            target_height = max(2, round(target_width / aspect))
        else:
            target_height = DEFAULT_RECTIFIED_LONG_EDGE_PX
            target_width = max(2, round(target_height * aspect))
        target = [(0.0, 0.0), (target_width - 1.0, 0.0),
                  (target_width - 1.0, target_height - 1.0), (0.0, target_height - 1.0)]
        try:
            h_image_to_board = solve_homography(points, target)
            h_board_to_image = invert_homography(h_image_to_board)
            # Pillow's perspective filter represents the denominator as
            # ``g*x + h*y + 1``. Normalize the inverse matrix before passing
            # its first eight coefficients to avoid a projective scale error.
            inverse_scale = h_board_to_image[8]
            if abs(inverse_scale) < 1e-12:
                raise GeometryError("inverse board homography cannot be normalized")
            h_board_to_image = tuple(value / inverse_scale for value in h_board_to_image)
            coefficients = tuple(h_board_to_image[:8])
            rectified = image.transform(
                (target_width, target_height), self._perspective_transform(),
                coefficients, resample=self._image_resample("BICUBIC"),
            )
        except Exception as exc:
            self.status_var.set(f"Could not rectify board: {exc}")
            return
        state["board_points_source_px"] = [list(point) for point in points]
        state["homography_image_to_board"] = list(h_image_to_board)
        state["homography_board_to_image"] = list(h_board_to_image)
        state["rectified"] = rectified
        state["rectified_size_px"] = [target_width, target_height]
        state["annotations"] = []
        state["ruler"] = None
        state["reference_rectangle"] = None
        self.pending_points.clear()
        self.view = "board"
        self.view_var.set("board")
        self._set_mode("idle")
        self.status_var.set(f"Board rectified to {target_width} × {target_height} px. Set a ruler or measured rectangle, then annotate.")
        self._refresh()

    def _apply_crop(self) -> None:
        state = self.state
        image = state.get("image")
        if image is None or len(self.pending_points) != 2:
            return
        (x0, y0), (x1, y1) = self.pending_points
        left, right = sorted((round(x0), round(x1)))
        top, bottom = sorted((round(y0), round(y1)))
        if right - left < 20 or bottom - top < 20:
            self.status_var.set("Crop is too small.")
            return
        self._push_undo()
        state["image"] = image.crop((left, top, right + 1, bottom + 1))
        state["pixel_operations"].append({"operation": "crop", "box_xyxy": [left, top, right + 1, bottom + 1]})
        state["board_points_source_px"] = []
        state["homography_image_to_board"] = None
        state["homography_board_to_image"] = None
        state["rectified"] = None
        state["rectified_size_px"] = None
        state["annotations"] = []
        state["ruler"] = None
        state["reference_rectangle"] = None
        self.pending_points.clear()
        self.mode = "idle"
        self.view = "source"
        self.view_var.set("source")
        self.pan_offset = (0.0, 0.0)
        self.status_var.set("Crop applied; set board corners again for this image.")

    def _rotate_current(self) -> None:
        state = self.state
        image = state.get("image")
        if image is None:
            self.status_var.set("Open an image first.")
            return
        self._push_undo()
        state["image"] = image.rotate(-90, expand=True)
        state["pixel_operations"].append({"operation": "rotate_90_clockwise"})
        state["board_points_source_px"] = []
        state["homography_image_to_board"] = None
        state["homography_board_to_image"] = None
        state["rectified"] = None
        state["rectified_size_px"] = None
        state["annotations"] = []
        state["ruler"] = None
        state["reference_rectangle"] = None
        self.view = "source"
        self.view_var.set("source")
        self.pan_offset = (0.0, 0.0)
        self._set_mode("idle")
        self.status_var.set("Rotated 90° clockwise; set board corners again for this image.")

    def _set_ruler_from_points(self) -> None:
        if len(self.pending_points) != 2:
            return
        value = self.simpledialog.askstring("Reference length", "Measured length:", parent=self.root)
        if value is None:
            self.pending_points.clear()
            return
        unit = self.simpledialog.askstring("Reference length", "Unit (mm, cm, or m):", initialvalue="cm", parent=self.root)
        if unit is None:
            self.pending_points.clear()
            return
        try:
            meters = length_to_meters(value, unit)
            board_points = list(self.pending_points)
            if self.view == "source":
                board_points = [
                    apply_homography(self.state["homography_image_to_board"], point)
                    for point in board_points
                ]
            pixels = distance(*board_points)
            if pixels < 1e-6:
                raise GeometryError("reference endpoints are too close")
            self.state["ruler"] = {
                "endpoints_source_or_board_px": [list(point) for point in self.pending_points],
                "endpoints_board_px": [list(point) for point in board_points],
                "length_m": meters,
                "meters_per_pixel": meters / pixels,
                "use_for_coordinates": True,
            }
            self._recompute_annotation_coordinates()
            self.status_var.set(f"Ruler scale set to {meters / pixels:.7g} m/px; it is now used for board dimensions.")
        except GeometryError as exc:
            self.status_var.set(str(exc))
        self.pending_points.clear()
        self.mode = "idle"

    def _set_reference_rectangle(self) -> None:
        if len(self.pending_points) != 4 or not _quad_is_valid(self.pending_points):
            self.status_var.set("Measured rectangle must be a valid ordered quadrilateral.")
            return
        width = self.simpledialog.askstring("Measured rectangle", "Known width:", initialvalue="10", parent=self.root)
        height = self.simpledialog.askstring("Measured rectangle", "Known height:", initialvalue="10", parent=self.root)
        unit = self.simpledialog.askstring("Measured rectangle", "Unit (mm, cm, or m):", initialvalue="cm", parent=self.root)
        if None in (width, height, unit):
            self.pending_points.clear()
            return
        try:
            width_m = length_to_meters(width, unit)
            height_m = length_to_meters(height, unit)
            board_points = list(self.pending_points)
            if self.view == "source":
                board_points = [
                    apply_homography(self.state["homography_image_to_board"], point)
                    for point in board_points
                ]
            top = distance(board_points[0], board_points[1])
            bottom = distance(board_points[3], board_points[2])
            left = distance(board_points[0], board_points[3])
            right = distance(board_points[1], board_points[2])
            px_width = (top + bottom) / 2.0
            px_height = (left + right) / 2.0
            if px_width < 1e-6 or px_height < 1e-6:
                raise GeometryError("measured rectangle is too small")
            self.state["reference_rectangle"] = {
                "corners_source_or_board_px": [list(point) for point in self.pending_points],
                "corners_board_px": [list(point) for point in board_points],
                "width_m": width_m,
                "height_m": height_m,
                "scale_x_m_per_px": width_m / px_width,
                "scale_y_m_per_px": height_m / px_height,
                "use_for_coordinates": True,
            }
            self._recompute_annotation_coordinates()
            self.status_var.set("Measured rectangle recorded. Its independent X/Y scale is now used for coordinates.")
        except GeometryError as exc:
            self.status_var.set(str(exc))
        self.pending_points.clear()
        self.mode = "idle"

    def _create_circle(self) -> None:
        center, edge = self.pending_points
        radius = distance(center, edge)
        if radius < 2.0:
            self.status_var.set("Circle radius is too small.")
            self.pending_points.clear()
            return
        self._append_annotation({
            "id": f"annotation_{self.next_id}",
            "part_name": self.part_var.get().strip() or f"part_{self.next_id}",
            "kind": "circle",
            "closed": True,
            "center_board_px": list(center),
            "radius_board_px": radius,
            "points_board_px": [list(center), list(edge)],
        })
        self.next_id += 1
        self.pending_points.clear()
        self.mode = "idle"

    def _close_polygon(self, *, record_undo: bool = True) -> None:
        if len(self.pending_polygon) < 3:
            if self.pending_polygon:
                self.status_var.set("A polygon needs at least three vertices before it can close.")
            return
        area = abs(polygon_signed_area(self.pending_polygon))
        if area < 1e-6:
            self.status_var.set("Polygon has zero area; add non-collinear vertices.")
            return
        if not polygon_is_simple(self.pending_polygon):
            self.status_var.set("Polygon edges cross; close a simple boundary without self-intersections.")
            return
        if record_undo:
            self._push_undo()
        center = polygon_centroid(self.pending_polygon)
        self._append_annotation({
            "id": f"annotation_{self.next_id}",
            "part_name": self.part_var.get().strip() or f"part_{self.next_id}",
            "kind": "polygon",
            "closed": True,
            "points_board_px": [list(point) for point in self.pending_polygon],
            "center_board_px": list(center),
            "area_board_px2": area,
        })
        self.next_id += 1
        self.pending_polygon.clear()
        self.mode = "idle"
        self.status_var.set("Closed polygon and computed its area-weighted center.")

    def _append_annotation(self, annotation: dict[str, Any]) -> None:
        target = self.state["rectified_size_px"]
        dimensions = _effective_dimensions(self.state, target)
        center = _coordinate_from_board_pixel(annotation["center_board_px"], target, dimensions)
        annotation["center_board_m_from_tl"] = list(center)
        annotation["center_board_m"] = [center[0] - dimensions[0] / 2.0, center[1] - dimensions[1] / 2.0]
        annotation["source_geometry_px"] = [
            list(apply_homography(self.state["homography_board_to_image"], point))
            for point in (annotation.get("points_board_px") or [])
        ]
        self.state["annotations"].append(annotation)
        self.status_var.set(f"Added {annotation['part_name']} at ({annotation['center_board_m'][0]:+.4f}, {annotation['center_board_m'][1]:+.4f}) m.")

    def _recompute_annotation_coordinates(self) -> None:
        target = self.state.get("rectified_size_px")
        if not target:
            return
        dimensions = _effective_dimensions(self.state, target)
        for annotation in self.state.get("annotations") or []:
            center = _coordinate_from_board_pixel(annotation["center_board_px"], target, dimensions)
            annotation["center_board_m_from_tl"] = list(center)
            annotation["center_board_m"] = [center[0] - dimensions[0] / 2.0, center[1] - dimensions[1] / 2.0]

    def _delete_selected(self) -> None:
        index = self._selected_annotation_index()
        if index is None:
            self.status_var.set("Select an annotation in the list first.")
            return
        self._push_undo()
        self.state["annotations"].pop(index)
        self._refresh()

    def _export_dialog(self) -> None:
        for name, state in self.states.items():
            if state.get("rectified") is None:
                self.status_var.set(f"Rectify the {name} image before exporting.")
                self.state_var.set(name)
                self.current_state_name = name
                self.view = "source"
                self.view_var.set("source")
                self._refresh()
                return
        if self.output_path is None:
            chosen = self.filedialog.asksaveasfilename(title="Save annotation project", defaultextension=".json", filetypes=[("JSON", "*.json")])
            if not chosen:
                return
            self.output_path = Path(chosen)
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            states = {}
            for name, raw in self.states.items():
                serial = {key: value for key, value in raw.items() if key not in ("image", "rectified")}
                serial["image_sha256"] = _sha256(Path(raw["image_path"])) if raw.get("image_path") and Path(raw["image_path"]).is_file() else None
                if raw.get("image") is not None:
                    processed = raw["image"].convert("RGB")
                    serial["processed_image_size_px"] = list(processed.size)
                    serial["processed_image_sha256"] = hashlib.sha256(processed.tobytes()).hexdigest()
                    processed_path = self.output_path.with_name(self.output_path.stem + f".{name}.processed.png")
                    processed.save(processed_path)
                    serial["processed_image_path"] = str(processed_path)
                if raw.get("rectified") is not None:
                    rectified_path = self.output_path.with_name(self.output_path.stem + f".{name}.rectified.png")
                    raw["rectified"].save(rectified_path)
                    serial["rectified_image_path"] = str(rectified_path)
                states[name] = serial
            initial_raw = states["initial"]
            initial_target = initial_raw.get("rectified_size_px") or [0, 0]
            export_width_m, export_height_m = _effective_dimensions(
                initial_raw, initial_target
            ) if initial_target[0] and initial_target[1] else (
                float(self.width_var.get()), float(self.height_var.get())
            )
            project_path, task_path, task = export_project(
                self.output_path,
                states=states,
                part_order=self.part_order,
                board_width_m=export_width_m,
                board_height_m=export_height_m,
                source_root=Path.cwd(),
            )
        except Exception as exc:
            self.messagebox.showerror("Export", str(exc))
            return
        warnings = task.get("warnings") or []
        warning_text = "\n\nWarnings:\n" + "\n".join(warnings) if warnings else ""
        self.status_var.set(f"Exported {project_path.name} and {task_path.name}.")
        self.messagebox.showinfo("Export complete", f"Audit project:\n{project_path}\n\nTask coordinates:\n{task_path}{warning_text}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", help="initial-state image")
    parser.add_argument("--final", help="final-state image")
    parser.add_argument("--output", help="default audit JSON output path")
    args = parser.parse_args(argv)
    try:
        import tkinter as tk
    except ImportError:
        print("This desktop UI needs Tk. Install a Tk package (for example python3-tk) and Pillow.", file=sys.stderr)
        return 2
    try:
        root = tk.Tk()
    except Exception as exc:
        print(f"Could not start the desktop UI: {exc}", file=sys.stderr)
        return 2
    try:
        AnnotationApp(root, initial=args.initial, final=args.final, output=args.output)
        root.mainloop()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
