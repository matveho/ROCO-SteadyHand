"""First real Vega pick/place bring-up using a taught current TCP pose.

This intentionally bypasses uncalibrated object->TCP perception for the first
physical end-to-end test. The operator places battery_size1 under the already
open right jaw at the current TCP pose. The robot then:
  hover -> descend -> current-limited grip -> lift -> base-Y transfer -> place
  -> open -> retract.

Normal competition execution should return to runtime object poses plus a
calibrated T_part_tcp after this bring-up succeeds.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.vega import VegaAdapter
from steadyhand.config import load_bundle
from steadyhand.executor import execute_part, validate_execution
from steadyhand.models import PartGoal, Pose
from steadyhand.skill_config import load_vega_skills, skill_for_part


def verify(stage, name):
    prompts = {
        "grasp": "grasp looks secure",
        "lift": "battery stayed in the jaws after lift",
        "placement": "battery was released stably at the destination",
    }
    return input(f"{name}: {prompts[stage]}? type yes: ").strip().lower() == "yes"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transfer-y", type=float, default=-0.08,
                   help="signed metres in base-Y; default -0.08 moves right arm outward 8 cm")
    p.add_argument("--speed-scale", type=float, default=0.45)
    args = p.parse_args(argv)

    bundle = load_bundle("vega")
    cfg = bundle["robot"]
    cfg["allow_robot_init_head_motion"] = True
    cfg["auto_clear_software_estop_on_connect"] = True

    skills_cfg = load_vega_skills()
    safety = dict(skills_cfg.get("safety") or {})
    skill = skill_for_part(skills_cfg, "battery_size1")
    skill["runtime_pose_type"] = "tcp"
    if skill.get("grip_current_a") is None:
        skill["grip_current_a"] = cfg["gripper"]["grip_current_a"]

    robot = VegaAdapter(cfg)
    robot.prepare()

    try:
        print("Connecting robot...")
        robot.connect()

        print("Homing configured right gripper, then opening...")
        robot.connect_gripper()
        robot.open_gripper("battery_size1")

        pick = robot.get_tcp_pose()
        place = Pose(
            (
                pick.position_m[0],
                pick.position_m[1] + float(args.transfer_y),
                pick.position_m[2],
            ),
            pick.quaternion_wxyz,
        )
        goal = PartGoal(
            name="battery_size1",
            release_mode="open",
            pick_pose=pick,
            place_pose=place,
            verification_method=None,
        )

        validate_execution(goal, skill, safety, args.speed_scale)

        print("PICK TCP =", pick.position_m)
        print("PLACE TCP =", place.position_m)
        print()
        answer = input(
            "Place battery_size1 resting on the table and centered between the "
            "OPEN right jaws at the current pose. Clear the 8 cm -Y destination "
            "and the vertical path. Type yes to run the full pick/place: "
        ).strip().lower()
        if answer != "yes":
            print("Not started.")
            return 2

        def event(phase, state, details):
            if phase != "tcp_waypoint":
                print(f"[{phase}] {state}")

        result = execute_part(
            robot,
            goal,
            skill,
            safety=safety,
            speed_scale=args.speed_scale,
            event=event,
            verify=verify,
        )
        print("RESULT =", result)
        return 0
    finally:
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
