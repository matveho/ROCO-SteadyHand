"""Write a one-part direct-TCP runtime target file for Vega bring-up."""
import argparse
import json
import math
from pathlib import Path


def _vec(values, n, label):
    if len(values) != n or not all(math.isfinite(float(v)) for v in values):
        raise SystemExit(f"{label} requires {n} finite numbers")
    return [float(v) for v in values]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--part", default="battery_size1")
    p.add_argument("--pick", nargs=3, type=float, required=True, metavar=("X","Y","Z"))
    p.add_argument("--place", nargs=3, type=float, required=True, metavar=("X","Y","Z"))
    p.add_argument("--quat", nargs=4, type=float, required=True, metavar=("W","X","Y","Z"))
    p.add_argument("--min-z", type=float, default=0.4560002716867571)
    args = p.parse_args(argv)

    pick = _vec(args.pick, 3, "--pick")
    place = _vec(args.place, 3, "--place")
    quat = _vec(args.quat, 4, "--quat")
    norm = math.sqrt(sum(v*v for v in quat))
    if abs(norm - 1.0) > 0.001:
        raise SystemExit("--quat must be unit length")
    if pick[2] < args.min_z or place[2] < args.min_z:
        raise SystemExit(f"pick/place z must be >= hard floor {args.min_z:.9f} m")

    value = {
        "schema_version": 1,
        "robot_id": "vega",
        "pose_frame": "robot_base",
        "pose_type": "tcp",
        "generated_at": None,
        "base_frame": "vega_1u_base_link",
        "position_units": "m",
        "quaternion_order": "wxyz",
        "parts": {
            args.part: {
                "pick_pose": {"position_m": pick, "quaternion_wxyz": quat},
                "place_pose": {"position_m": place, "quaternion_wxyz": quat},
            }
        },
    }
    path = Path(args.output)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    raise SystemExit(main())
