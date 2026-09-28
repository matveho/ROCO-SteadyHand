"""Run coarse task-board corner detection against a saved head RGB image.

This is an OFFLINE development tool: it never imports dexcontrol and never
contacts the robot.  Give it either a snapshot directory containing
head_left_rgb.npy or the .npy file itself.  It writes:
  - board_detection.json
  - board_overlay.ppm

PPM is used deliberately so the tool only needs NumPy.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.vision.board import detect_white_board_corners


def _load_rgb(path):
    import numpy as np

    path = Path(path)
    if path.is_dir():
        path = path / "head_left_rgb.npy"
    if path.suffix.lower() != ".npy":
        raise SystemExit("input must be a snapshot directory or head_left_rgb.npy")
    rgb = np.load(path)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise SystemExit(f"expected HxWx3 RGB array, got {rgb.shape}")
    return rgb[..., :3].astype(np.uint8, copy=False), path


def _line(image, p0, p1, value=(255, 0, 0), thickness=3):
    import numpy as np

    x0, y0 = p0
    x1, y1 = p1
    n = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
    xs = np.rint(np.linspace(x0, x1, n)).astype(int)
    ys = np.rint(np.linspace(y0, y1, n)).astype(int)
    h, w = image.shape[:2]
    radius = max(0, int(thickness) // 2)
    for y, x in zip(ys, xs):
        ya, yb = max(0, y-radius), min(h, y+radius+1)
        xa, xb = max(0, x-radius), min(w, x+radius+1)
        image[ya:yb, xa:xb, :3] = value


def _cross(image, p, value=(0, 255, 0), radius=9, thickness=3):
    x, y = p
    _line(image, (x-radius, y), (x+radius, y), value, thickness)
    _line(image, (x, y-radius), (x, y+radius), value, thickness)


def _write_ppm(path, rgb):
    h, w = rgb.shape[:2]
    with Path(path).open("wb") as stream:
        stream.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        stream.write(rgb[..., :3].tobytes(order="C"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="snapshot directory or head_left_rgb.npy")
    p.add_argument("--output-dir", help="defaults beside input image")
    p.add_argument("--min-value", type=int, default=150)
    p.add_argument("--max-chroma", type=int, default=65)
    args = p.parse_args(argv)

    rgb, source = _load_rgb(args.input)
    corners = detect_white_board_corners(
        rgb,
        min_value=args.min_value,
        max_chroma=args.max_chroma,
    )
    labels = ("tl", "tr", "br", "bl")

    output = Path(args.output_dir) if args.output_dir else source.parent
    output.mkdir(parents=True, exist_ok=True)

    result = {
        "source": str(source),
        "image_shape": list(rgb.shape),
        "parameters": {
            "min_value": args.min_value,
            "max_chroma": args.max_chroma,
        },
        "corners_px": {
            label: [int(x), int(y)]
            for label, (x, y) in zip(labels, corners)
        },
    }
    (output / "board_detection.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    overlay = rgb.copy()
    for i in range(4):
        _line(overlay, corners[i], corners[(i+1) % 4])
    for corner in corners:
        _cross(overlay, corner)
    _write_ppm(output / "board_overlay.ppm", overlay)

    print("BOARD PIXELS =", result["corners_px"])
    print("WROTE", output / "board_detection.json")
    print("WROTE", output / "board_overlay.ppm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
