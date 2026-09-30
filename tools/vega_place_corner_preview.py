"""Inspect board-corner placement vision on saved images; never connects a robot."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from steadyhand.vision.board_corners import (
    CornerVisualError, draw_corners, make_reference, match_corners,
)


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--reference", type=Path, help="earlier image of this same placement hover")
    parser.add_argument("--corner", action="append", nargs=2, type=float, metavar=("U", "V"),
                        help="optional visible outer corner in the reference image; repeat up to four times")
    parser.add_argument("--output", type=Path, default=Path("runs/place_corner_preview"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        image = read_image(args.image)
        reference_image = read_image(args.reference) if args.reference else image
        reference = make_reference(reference_image, args.corner)
        matches = match_corners(image, reference)
        overlay = args.output / "BOARD_CORNERS_REVIEW.png"
        if not cv2.imwrite(str(overlay), cv2.cvtColor(draw_corners(image, reference, matches), cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not write {overlay}")
        report = {"mode": "OFFLINE_IMAGES_ONLY_NO_ROBOT", "image": str(args.image),
                  "reference_image": str(args.reference or args.image),
                  "reference": reference, "matched_corners": matches}
        (args.output / "corner_detection.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"MATCHED {len(matches)} / {len(reference['corners'])} corners")
        for key, value in matches.items():
            print(f"{key}: {value['uv'][0]:.1f} {value['uv'][1]:.1f}  score={value['score']:.3f}")
        print(f"REVIEW: {overlay.resolve()}")
        print("Green crosses = reference; orange circles = current. No calibration files changed.")
        return 0
    except (CornerVisualError, ValueError, OSError) as exc:
        print(f"CORNERS NOT VERIFIED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
