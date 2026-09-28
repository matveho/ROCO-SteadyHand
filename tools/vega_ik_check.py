"""Validate Vega URDF/frame/joint mapping and IK without robot access.

Example:
/usr/bin/python3 tools/vega_ik_check.py \
  --urdf ~/Downloads/Dexmate/vega_1u_gripper.urdf \
  --base-frame <verified-fixed-urdf-frame> \
  --ee-frame R_ee \
  --arm right \
  --fixed Lift=0.0 --fixed torso_flip=0.0 \
  --q 0 1.2 1.4 -1.57 -1.57 1 -0.35 \
  --dz 0.01

If a fixed value is missing for a movable joint in the EE kinematic chain, the
tool fails and names that joint instead of silently assuming a value.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.config import load_bundle
from steadyhand.kinematics import PinocchioArmKinematics
from steadyhand.models import Pose


def fixed_pair(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected JOINT=VALUE")
    name, value = text.split("=", 1)
    return name, float(value)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", required=True)
    p.add_argument("--ee-frame", required=True)
    p.add_argument("--base-frame", required=True)
    p.add_argument("--arm", choices=("right",), default="right")
    p.add_argument("--fixed", action="append", type=fixed_pair, default=[])
    p.add_argument("--q", nargs=7, type=float, required=True)
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--dy", type=float, default=0.0)
    p.add_argument("--dz", type=float, default=0.01)
    args = p.parse_args(argv)

    cfg = load_bundle("vega")["robot"]["kinematics"]
    cfg = dict(cfg)
    if len(dict(args.fixed)) != len(args.fixed):
        raise ValueError("Duplicate --fixed joint")
    cfg["fixed_joint_values"] = dict(args.fixed)
    cfg["base_frame"] = args.base_frame
    robot_cfg = load_bundle("vega")["robot"]
    cfg["joint_limits_rad"] = robot_cfg["arm_joint_limits_rad"][args.arm]
    names = cfg["right_arm_joint_names"]

    kin = PinocchioArmKinematics(
        args.urdf,
        args.ee_frame,
        names,
        cfg,
    )
    current = kin.forward(args.q)
    x, y, z = current.position_m
    target = Pose(
        (x + args.dx, y + args.dy, z + args.dz),
        current.quaternion_wxyz,
    )
    solved = kin.solve(target, args.q)

    print("current_tcp =", current)
    print("target_tcp  =", target)
    print("solution_q  =", [round(x, 6) for x in solved])
    print(
        "max_delta_rad =",
        round(max(abs(a-b) for a, b in zip(solved, args.q)), 6),
    )
    check = kin.forward(solved)
    print("solution_tcp =", check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
