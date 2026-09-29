"""Operator-assisted battery_size1 source localization from a head image.

No part detector is used. No part count, final-layout order or gear assumptions
exist in this tool.

Two localization modes are provided:

homography
  Operator identifies battery_size1 and all four physical board corners in the
  SAME head image. A planar four-corner homography maps the battery pixel into
  normalized board coordinates. Explicit board dimensions convert that to
  metric board offsets, then the completed manual board calibration converts
  those offsets to Vega base-frame XY.

board-offset
  Operator supplies battery_size1 offset from manually calibrated board center
  directly in millimetres along board +X/+Y. The selected head pixel is still
  recorded for identity/audit, but is not used geometrically. This fallback
  does not require the still-unknown second board dimension.

Neither mode commands robot motion.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.battery_size1_source import (
    MEASURED_BOARD_WIDTH_MM,
    PART_NAME,
    base_xy_from_board_offset,
    dimensions_from_measured_width,
    file_sha256,
    homography_board_fraction,
    load_manual_board_calibration,
    metric_offset_from_fraction,
)
from steadyhand.cameras.vega import VegaHeadCamera
from steadyhand.config import load_bundle

ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _fresh_output(value, prefix):
    if value:
        out = _resolve(value)
    else:
        out = ROOT / "runs" / datetime.now(timezone.utc).strftime(
            f"{prefix}_%Y%m%dT%H%M%S_%fZ"
        )
    out.mkdir(parents=True, exist_ok=False)
    return out


def _load_rgb(path):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read head image {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _validate_pixel(uv, image_shape, name):
    u, v = (float(x) for x in uv)
    if not all(math.isfinite(x) for x in (u, v)):
        raise ValueError(f"{name} must contain finite pixels")
    h, w = image_shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        raise ValueError(f"{name} {(u, v)} lies outside image {w}x{h}")
    return (u, v)


def _copy_source_image(source, output):
    target = output / "head_source.png"
    rgb = _load_rgb(source)
    import cv2
    if not cv2.imwrite(str(target), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {target}")
    return rgb, target


def _manual(args, cfg):
    return load_manual_board_calibration(
        _resolve(args.manual_calibration),
        cfg,
        max_age_minutes=float(args.max_calibration_age_min),
    )


def _base_record(*, args, cfg, manual, image_path, copied_image, battery_uv, method):
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "part": PART_NAME,
        "method": method,
        "robot_name": cfg["robot_name"],
        "base_frame": cfg["kinematics"]["base_frame"],
        "manual_board_calibration": {
            "path": manual["path"],
            "sha256": manual["sha256"],
            "generated_at_utc": manual["generated_at_utc"],
            "age_minutes": manual["age_minutes"],
            "permanent_fallback": manual["is_permanent_fallback"],
            "operator_confirmed_board_unchanged": True,
        },
        "source_image": {
            "input_path": str(image_path),
            "copied_path": str(copied_image),
            "sha256": file_sha256(copied_image),
        },
        "battery_size1": {
            "operator_selected_head_pixel_uv": [float(v) for v in battery_uv],
            "identity_source": "operator_explicit",
        },
    }


def _write_overlay(rgb, output, *, battery_uv, corners=None):
    import cv2
    import numpy as np

    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    battery = tuple(int(round(v)) for v in battery_uv)
    cv2.drawMarker(bgr, battery, (0, 0, 255), cv2.MARKER_CROSS, 36, 3)
    cv2.putText(
        bgr,
        "battery_size1",
        (battery[0] + 12, max(20, battery[1] - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    if corners:
        order = ("xm_ym", "xp_ym", "xp_yp", "xm_yp")
        pts = []
        for name in order:
            point = tuple(int(round(v)) for v in corners[name])
            pts.append(point)
            cv2.circle(bgr, point, 9, (255, 0, 0), 2)
            cv2.putText(
                bgr, name, (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA
            )
        cv2.polylines(
            bgr,
            [np.asarray(pts, dtype=np.int32)],
            True,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    path = output / "head_selection_overlay.png"
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"failed to write {path}")
    return path


def _print_result(record, output):
    x, y = record["coarse_base_xy_m"]
    print("", flush=True)
    print("BATTERY_SIZE1 COARSE BASE XY =", f"{x:.6f}", f"{y:.6f}", flush=True)
    print("CENTER ARGUMENT =", "--coarse-xy", f"{x:.6f}", f"{y:.6f}", flush=True)
    print("LOCALIZATION FILE =", (output / "localization.json").resolve(), flush=True)
    print(
        "NOTE: move the right TCP to a safe low hover at this XY before "
        "vega_battery_size1_center.py; --coarse-xy is a gate, not a navigation command.",
        flush=True,
    )


def capture(args):
    import cv2
    import numpy as np

    cfg = load_bundle("vega")["robot"]
    env_name = os.environ.get("ROBOT_NAME")
    if env_name and env_name != cfg["robot_name"]:
        raise ValueError("ROBOT_NAME disagrees with configured competition robot")
    os.environ.setdefault("ROBOT_NAME", cfg["robot_name"])

    out = _fresh_output(args.output, "battery_size1_head")
    camera = VegaHeadCamera()
    try:
        camera.connect()
        frame = camera.read(include_depth=False, timeout_s=float(args.timeout_s))
        rgb = np.asarray(frame.left_rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"head RGB expected uint8 HxWx3, got {rgb.shape}/{rgb.dtype}")
        mean, std = float(rgb.mean()), float(rgb.std())
        if mean < 4.0 or std < 2.0:
            raise RuntimeError(
                f"head image is black/invalid (mean={mean:.2f}, std={std:.2f})"
            )
        image_path = out / "head_source.png"
        if not cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"failed to write {image_path}")
        metadata = {
            "robot_name": cfg["robot_name"],
            "camera": "head_left_rectified",
            "shape": list(rgb.shape),
            "left_timestamp_ns": frame.left_timestamp_ns,
            "mean": mean,
            "std": std,
            "sha256": file_sha256(image_path),
        }
        (out / "capture.json").write_text(
            json.dumps(metadata, indent=2, default=int) + "\n",
            encoding="utf-8",
        )
        print("HEAD IMAGE =", image_path.resolve(), flush=True)
        print("CAPTURE =", (out / "capture.json").resolve(), flush=True)
        return 0
    finally:
        camera.close()


def homography(args):
    cfg = load_bundle("vega")["robot"]
    if not args.confirm_board_unchanged_since_calibration:
        raise SystemExit(
            "homography localization requires "
            "--confirm-board-unchanged-since-calibration"
        )
    manual = _manual(args, cfg)
    source = _resolve(args.image)
    output = _fresh_output(args.output, "battery_size1_source")
    rgb, copied = _copy_source_image(source, output)
    battery_uv = _validate_pixel(args.battery_pixel, rgb.shape, "battery pixel")

    corners = {
        "xm_ym": _validate_pixel(args.corner_xm_ym, rgb.shape, "corner xm_ym"),
        "xp_ym": _validate_pixel(args.corner_xp_ym, rgb.shape, "corner xp_ym"),
        "xp_yp": _validate_pixel(args.corner_xp_yp, rgb.shape, "corner xp_yp"),
        "xm_yp": _validate_pixel(args.corner_xm_yp, rgb.shape, "corner xm_yp"),
    }
    fraction, H = homography_board_fraction(battery_uv, corners)
    board_width_mm = MEASURED_BOARD_WIDTH_MM
    board_x_mm, board_y_mm = dimensions_from_measured_width(
        width_axis=args.width_axis,
        other_dimension_mm=args.other_dimension_mm,
        measured_width_mm=board_width_mm,
    )
    offset = metric_offset_from_fraction(
        fraction,
        board_x_mm=board_x_mm,
        board_y_mm=board_y_mm,
    )
    base_xy = base_xy_from_board_offset(manual, offset)

    record = _base_record(
        args=args,
        cfg=cfg,
        manual=manual,
        image_path=source,
        copied_image=copied,
        battery_uv=battery_uv,
        method="explicit_four_corner_homography",
    )
    record["board_geometry"] = {
        "measured_width_mm": float(board_width_mm),
        "measured_width_source": "fixed_competition_board_specification",
        "width_axis": args.width_axis,
        "other_dimension_mm": float(args.other_dimension_mm),
        "other_dimension_source": "operator_supplied",
        "board_x_dimension_mm": board_x_mm,
        "board_y_dimension_mm": board_y_mm,
        "no_400mm_assumption": True,
        "board_size_contract": "386_mm_span",
    }
    record["homography"] = {
        "corner_semantics": {
            "xm_ym": "board -X,-Y",
            "xp_ym": "board +X,-Y",
            "xp_yp": "board +X,+Y",
            "xm_yp": "board -X,+Y",
        },
        "corners_image_uv": {key: list(value) for key, value in corners.items()},
        "image_to_board_fraction_matrix": [
            [float(value) for value in row] for row in H.tolist()
        ],
    }
    record["board_coordinates"] = {
        "fraction_xy_from_minus_corner": list(fraction),
        "offset_from_center_m": list(offset),
        "offset_from_center_mm": [1000.0 * v for v in offset],
    }
    record["coarse_base_xy_m"] = list(base_xy)
    overlay = _write_overlay(rgb, output, battery_uv=battery_uv, corners=corners)
    record["selection_overlay"] = str(overlay)
    (output / "localization.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    _print_result(record, output)
    return 0


def board_offset(args):
    cfg = load_bundle("vega")["robot"]
    if not args.confirm_board_unchanged_since_calibration:
        raise SystemExit(
            "board-offset localization requires "
            "--confirm-board-unchanged-since-calibration"
        )
    manual = _manual(args, cfg)
    source = _resolve(args.image)
    output = _fresh_output(args.output, "battery_size1_source")
    rgb, copied = _copy_source_image(source, output)
    battery_uv = _validate_pixel(args.battery_pixel, rgb.shape, "battery pixel")
    dx_mm, dy_mm = (float(v) for v in args.board_offset_mm)
    if not all(math.isfinite(v) for v in (dx_mm, dy_mm)):
        raise ValueError("board offset must contain two finite millimetre values")
    offset = (dx_mm / 1000.0, dy_mm / 1000.0)
    base_xy = base_xy_from_board_offset(manual, offset)

    record = _base_record(
        args=args,
        cfg=cfg,
        manual=manual,
        image_path=source,
        copied_image=copied,
        battery_uv=battery_uv,
        method="operator_board_offset",
    )
    record["board_geometry"] = {
        "measured_width_mm": MEASURED_BOARD_WIDTH_MM,
        "measured_width_source": "fixed_competition_board_specification",
        "width_axis": None,
        "other_dimension_mm": None,
        "note": (
            "Board dimensions are not used in operator_board_offset mode. "
            "The second dimension remains intentionally unspecified."
        ),
        "no_400mm_assumption": True,
        "board_size_contract": "386_mm_span",
    }
    record["board_coordinates"] = {
        "offset_from_center_m": list(offset),
        "offset_from_center_mm": [dx_mm, dy_mm],
        "source": "operator_explicit",
    }
    record["coarse_base_xy_m"] = list(base_xy)
    overlay = _write_overlay(rgb, output, battery_uv=battery_uv)
    record["selection_overlay"] = str(overlay)
    (output / "localization.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    _print_result(record, output)
    return 0


def add_common_localize(p):
    p.add_argument("--manual-calibration", default="calibration/vega_board_manual.json")
    p.add_argument("--max-calibration-age-min", type=float, default=720.0)
    p.add_argument("--image", required=True)
    p.add_argument("--battery-pixel", nargs=2, type=float, required=True, metavar=("U", "V"))
    p.add_argument("--output")
    p.add_argument("--confirm-board-unchanged-since-calibration", action="store_true")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture")
    cap.add_argument("--output")
    cap.add_argument("--timeout-s", type=float, default=15.0)

    hom = sub.add_parser("homography")
    add_common_localize(hom)
    hom.add_argument("--corner-xm-ym", nargs=2, type=float, required=True, metavar=("U", "V"))
    hom.add_argument("--corner-xp-ym", nargs=2, type=float, required=True, metavar=("U", "V"))
    hom.add_argument("--corner-xp-yp", nargs=2, type=float, required=True, metavar=("U", "V"))
    hom.add_argument("--corner-xm-yp", nargs=2, type=float, required=True, metavar=("U", "V"))
    hom.add_argument(
        "--width-axis",
        choices=("x", "y"),
        required=True,
        help="which manually calibrated board axis the 386 mm span occupies",
    )
    hom.add_argument(
        "--other-dimension-mm",
        type=float,
        required=True,
        help="measured second board dimension; intentionally has no default",
    )

    offset = sub.add_parser("board-offset")
    add_common_localize(offset)
    offset.add_argument(
        "--board-offset-mm",
        nargs=2,
        type=float,
        required=True,
        metavar=("DX", "DY"),
        help="battery center offset from board center along calibrated +X,+Y",
    )

    args = p.parse_args(argv)
    if args.command == "capture":
        if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
            p.error("--timeout-s must be finite and > 0")
        return capture(args)
    if not math.isfinite(args.max_calibration_age_min) or not 15 <= args.max_calibration_age_min <= 1440:
        p.error("--max-calibration-age-min must be 15..1440")
    if args.command == "homography":
        return homography(args)
    if args.command == "board-offset":
        return board_offset(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
