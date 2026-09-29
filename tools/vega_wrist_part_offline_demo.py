"""Offline confidence/occlusion check for the confirmed board-part labels.

This does not claim to replace the live wrist camera.  It exercises the same
saved-patch ``TemplateTracker`` used after an onsite wrist profile is taught,
using the supplied overhead board photograph.  It writes clean and simulated
gripper-occluded overlays plus a machine-readable result report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.vision.wrist_servo import TemplateTracker


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = Path("/home/sharpa/Downloads/9d0c4980-d81a-4e87-a49b-517513700eed.jpeg")

# Coordinates are in the original 2560x1920 attachment.  They are deliberately
# kept here as review fixtures, not as robot task coordinates.
BOXES = {
    "gear_60teeth": (1495, 205, 1779, 498),
    "gear_20teeth": (1334, 428, 1432, 535),
    "usb_a": (1861, 317, 1951, 440),
    "hdmi": (614, 551, 694, 653),
    "pin": (1034, 666, 1109, 743),
    "bolt_8mm": (1659, 661, 1734, 732),
    "rod_16mm": (1449, 899, 1548, 998),
    "battery_size1": (1531, 1217, 1682, 1502),
    "battery_size5": (1671, 1222, 1740, 1443),
}


def _center(box):
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _template(rgb, center, radius=20):
    u, v = (int(round(value)) for value in center)
    if not (radius <= u < rgb.shape[1] - radius and radius <= v < rgb.shape[0] - radius):
        raise ValueError("fixture center is too close to image edge")
    return rgb[v - radius:v + radius + 1, u - radius:u + radius + 1].copy()


def _occlude(rgb, box):
    """Simulate a dark jaw/fixture over the right side while leaving the center."""
    import cv2

    image = rgb.copy()
    x0, y0, x1, y1 = box
    start = int(round(x0 + 0.68 * (x1 - x0)))
    cv2.rectangle(image, (start, y0), (x1, y1), (24, 24, 24), thickness=-1)
    return image


def _draw(path, rgb, results, title):
    import cv2
    import numpy as np

    image = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    cv2.putText(image, title, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (255, 255, 255), 3, cv2.LINE_AA)
    for part, record in results.items():
        x0, y0, x1, y1 = BOXES[part]
        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 200, 0), 3)
        color = (0, 210, 0) if record["ok"] else (0, 0, 255)
        if record.get("detected_uv") is not None:
            u, v = (int(round(value)) for value in record["detected_uv"])
            cv2.drawMarker(image, (u, v), color, cv2.MARKER_CROSS, 28, 4)
        label = f"{part}: {record['status']}"
        if record.get("score") is not None:
            label += f" {record['score']:.2f}"
        cv2.putText(image, label, (x0, max(25, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(path), image)


def _run(rgb, *, occluded):
    import numpy as np

    frame = _occlude(rgb, BOXES[next(iter(BOXES))]) if False else rgb
    results = {}
    for part, box in BOXES.items():
        center = _center(box)
        template = _template(rgb, center)
        candidate = _occlude(rgb, box) if occluded else rgb
        record = {"expected_uv": list(center), "detected_uv": None,
                  "score": None, "status": "error", "ok": False}
        try:
            tracker = TemplateTracker.from_saved_template(
                candidate, template, (20, 20), search_radius=260,
                min_score=0.62, min_margin=0.03,
            )
            uv, score = tracker.uv, None
            # Global initialization stores the match but not its score. A second
            # local locate returns the production score without moving anything.
            uv, score = tracker.locate(candidate)
            record.update(detected_uv=list(uv), score=float(score),
                          status="ok", ok=True)
        except Exception as exc:
            record.update(status=f"STOP: {type(exc).__name__}: {exc}")
        results[part] = record
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--output", default="runs/part_identity_review_20260928")
    args = parser.parse_args(argv)

    import cv2
    import numpy as np

    image_path = Path(args.image).expanduser()
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        parser.error(f"could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    clean = _run(rgb, occluded=False)
    occluded = _run(rgb, occluded=True)
    _draw(output / "wrist_detector_clean.png", rgb, clean, "Wrist template detector: clean board fixture")
    _draw(output / "wrist_detector_occluded.png", _occlude(rgb, BOXES["gear_60teeth"]), occluded,
          "Wrist template detector: simulated right-side jaw occlusion")
    report = {
        "image": str(image_path),
        "template_tracker": {"min_score": 0.62, "min_margin": 0.03,
                              "template_radius_px": 20},
        "clean": clean,
        "occluded": occluded,
    }
    (output / "wrist_detector_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"WROTE {output / 'wrist_detector_clean.png'}")
    print(f"WROTE {output / 'wrist_detector_occluded.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
