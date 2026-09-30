"""Small bounded Cartesian jog for Vega bring-up.

Moves only the configured working-arm TCP, preserving its current orientation.
Use repeated small commands while visually aligning a taught pick/place pose.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.vega import VegaAdapter
from steadyhand.config import load_bundle
from steadyhand.models import Pose


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--dy", type=float, default=0.0)
    p.add_argument("--dz", type=float, default=0.0)
    p.add_argument("--speed-scale", type=float, default=0.10)
    p.add_argument('--confirm-head-motion', action="store_true", default=True, help=argparse.SUPPRESS)
    p.add_argument('--confirm-physical-motion', action="store_true", default=True, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    delta = (float(args.dx), float(args.dy), float(args.dz))
    if not all(math.isfinite(v) for v in delta):
        raise SystemExit("Jog delta must be finite")
    distance = math.sqrt(sum(v * v for v in delta))
    if distance <= 0 or distance > 0.05:
        raise SystemExit("Each jog must be >0 and <=0.05 m total")
    if not 0 < args.speed_scale <= 0.25:
        raise SystemExit("--speed-scale must be in (0, 0.25] for bring-up")

    bundle = load_bundle("vega")
    cfg = bundle["robot"]
    cfg["allow_robot_init_head_motion"] = True

    robot = VegaAdapter(cfg)
    robot.prepare()
    try:
        robot.connect()
        before = robot.get_tcp_pose()
        target = Pose(
            tuple(a + b for a, b in zip(before.position_m, delta)),
            before.quaternion_wxyz,
        )

        print("before =", before.position_m)
        print("target =", target.position_m)
        robot.move_tcp(target, speed_scale=args.speed_scale)
        after = robot.get_tcp_pose()
        print("after  =", after.position_m)
    finally:
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
