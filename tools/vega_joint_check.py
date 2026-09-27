"""Gated Vega connection/state check and one reversible tiny joint move.

This never touches CAN. Robot() may move the head. Run first with
--connect-only, then use a small --delta-rad with an engineer and e-stop ready.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.vega import VegaAdapter
from tools.vega_run_part import (
    add_hardware_arguments,
    close_report,
    configured_bundle,
    stop_report,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect-only", action="store_true")
    parser.add_argument("--joint-index", type=int, choices=range(1, 8))
    parser.add_argument("--delta-rad", type=float)
    add_hardware_arguments(parser)
    args = parser.parse_args(argv)
    if not args.connect_only and (args.joint_index is None or args.delta_rad is None):
        parser.error("a motion check requires --joint-index and --delta-rad")
    if args.connect_only and (args.joint_index is not None or args.delta_rad is not None):
        parser.error("--connect-only cannot be combined with a joint target")

    bundle = configured_bundle(args)
    robot = VegaAdapter(bundle["robot"])
    robot.prepare()
    if args.check_only:
        print("Local config and IK passed; Robot was not constructed.")
        return 0

    moved = False
    initial = None
    try:
        robot.connect()
        observation = robot.observe()
        initial = list(observation.joint_positions)
        print("joint_names =", list(robot._joint_names))
        print("joint_positions_rad =", initial)
        print("wrench_raw =", observation.extras.get("wrench"))
        if args.connect_only:
            return 0
        if args.delta_rad == 0 or abs(args.delta_rad) > 0.02:
            raise ValueError("First motion delta must be nonzero and at most 0.02 rad")
        target = initial[:]
        target[args.joint_index - 1] += args.delta_rad
        input(
            f"Ready for joint {args.joint_index} delta {args.delta_rad:+.6f} rad. "
            "Press Enter to command; Ctrl-C aborts: "
        )
        robot.move_joints(target, speed_scale=args.speed_scale)
        moved = True
        input("Motion reached. Press Enter to return to the initial joints; Ctrl-C stops: ")
        robot.move_joints(initial, speed_scale=args.speed_scale)
        moved = False
        print("Tiny motion and return completed with fresh-state readback.")
        return 0
    except BaseException as exc:
        stop_report(robot)
        print(f"JOINT CHECK ABORTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        if moved:
            print("Robot may not be at its initial pose; do not auto-return after a fault.", file=sys.stderr)
        return 130 if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 1
    finally:
        close_report(robot)


if __name__ == "__main__":
    raise SystemExit(main())
