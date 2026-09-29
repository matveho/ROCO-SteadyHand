"""Detect the task board and any visible dark part-like regions in a saved head image.

OFFLINE tool: never imports dexcontrol and never contacts the robot.

Input may be:
  - a snapshot directory containing head_left_rgb.npy
  - a .npy RGB array
  - a PNG/JPEG image

Outputs:
  board_detection.json
  board_overlay.png
  board_rectified.png

The task board is perspective-rectified before dark-component segmentation.
Part count and arrangement are intentionally not assumed: initial, intermediate,
partial, and final task states are all accepted.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.vision.board import detect_white_board_corners
from steadyhand.board_geometry import BOARD_SIZE_M
from steadyhand.vision.task_parts import (
    detect_dark_part_boxes,
    label_final_layout,
    project_part_boxes_to_image,
    rectify_board,
)


def _load_rgb(path):
    import cv2
    import numpy as np

    path = Path(path)
    if path.is_dir():
        path = path / "head_left_rgb.npy"
    if path.suffix.lower() == ".npy":
        rgb = np.load(path)
    elif path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise SystemExit(f"could not read {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        raise SystemExit("input must be snapshot dir, .npy, PNG, or JPEG")
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise SystemExit(f"expected HxWx3 RGB array, got {rgb.shape}")
    return rgb[..., :3].astype(np.uint8, copy=False), path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("--output-dir", help="defaults beside input image")
    p.add_argument("--min-value", type=int, default=150)
    p.add_argument("--max-chroma", type=int, default=65)
    p.add_argument("--dark-threshold", type=int, default=140)
    p.add_argument("--layout", choices=("unlabeled", "final"), default="unlabeled")
    args = p.parse_args(argv)

    import cv2
    import numpy as np

    rgb, source = _load_rgb(args.input)
    corners = detect_white_board_corners(
        rgb,
        min_value=args.min_value,
        max_chroma=args.max_chroma,
    )
    rectified, H = rectify_board(rgb, corners)
    parts = detect_dark_part_boxes(
        rectified,
        dark_threshold=args.dark_threshold,
    )
    parts = project_part_boxes_to_image(parts, H)
    if args.layout == "final":
        parts = label_final_layout(parts)

    output = Path(args.output_dir) if args.output_dir else source.parent
    output.mkdir(parents=True, exist_ok=True)

    result = {
        "source": str(source),
        "image_shape": list(rgb.shape),
        "board_size_m": BOARD_SIZE_M,
        "parameters": {
            "min_value": args.min_value,
            "max_chroma": args.max_chroma,
            "dark_threshold": args.dark_threshold,
        },
        "corners_px": {
            label: [int(x), int(y)]
            for label, (x, y) in zip(("tl", "tr", "br", "bl"), corners)
        },
        "layout": args.layout,
        "parts": parts,
    }
    (output / "board_detection.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    overlay = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    cv2.polylines(
        overlay,
        [np.asarray(corners, dtype=np.int32)],
        True, (0, 255, 0), 3,
    )
    for part in parts:
        quad = np.rint(part["quad_image_px"]).astype(np.int32)
        cv2.polylines(overlay, [quad], True, (0, 0, 255), 2)
        x, y = (int(v) for v in quad[0])
        label = part.get("name") or str(part["index"])
        cv2.putText(
            overlay, label, (x, max(18, y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output / "board_overlay.png"), overlay)

    rect_bgr = cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR)
    for part in parts:
        x0, y0, x1, y1 = part["box_board_px"]
        cv2.rectangle(rect_bgr, (x0, y0), (x1, y1), (0, 0, 255), 3)
        label = part.get("name") or str(part["index"])
        cv2.putText(
            rect_bgr, label, (x0, max(18, y0 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output / "board_rectified.png"), rect_bgr)

    print("BOARD PIXELS =", result["corners_px"])
    print("PARTS =", len(parts))
    for part in parts:
        print(
            f"  {part.get('name') or part['index']}: board_mm={tuple(round(v, 1) for v in part['center_board_mm_from_tl'])} "
            f"box={part['box_board_px']}"
        )
    print("WROTE", output / "board_detection.json")
    print("WROTE", output / "board_overlay.png")
    print("WROTE", output / "board_rectified.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
