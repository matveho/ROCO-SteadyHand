"""Run one physical Vega board part after onsite bring-up.

This script is intentionally explicit. It will not run unless:
- the user acknowledges Robot()'s automatic head motion;
- the user acknowledges physical arm/gripper motion;
- required robot/IK/timing fields are supplied;
- a runtime target file contains this part's current real-world poses.

Example after bring-up:

/usr/bin/python3 tools/vega_run_part.py \
  --targets runs/.../targets.json \
  --part battery_size1 \
  --operator Matvey \
  --working-arm left \
  --robot-name 'dm/...' \
  --urdf ~/Downloads/Dexmate/vega_1u_gripper.urdf \
  --ee-frame L_ee \
  --step-wait 0.5 \
  --gripper-scope both \
  --confirm-head-motion \
  --confirm-physical-motion
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.vega import VegaAdapter
from steadyhand.config import load_bundle
from steadyhand.executor import execute_part
from steadyhand.sessions import (
    append_event,
    append_trial,
    create_session,
    mark_hardware_connected,
)
from steadyhand.skill_config import load_vega_skills, skill_for_part
from steadyhand.targets import load_runtime_targets, require_goal


def parser():
    p = argparse.ArgumentParser(description="Gated single-part physical Vega runner")
    p.add_argument("--targets", required=True)
    p.add_argument("--part", required=True)
    p.add_argument("--operator", required=True)
    p.add_argument("--working-arm", choices=("left", "right"), required=True)
    p.add_argument("--robot-name")
    p.add_argument("--urdf", required=True)
    p.add_argument("--ee-frame", required=True)
    p.add_argument("--step-wait", type=float, required=True)
    p.add_argument("--speed-scale", type=float, default=1.0)
    p.add_argument("--gripper-scope", choices=("both",), required=True)
    p.add_argument("--grip-current", type=float)
    p.add_argument("--force-delta-limit", type=float)
    p.add_argument("--allow-snap-without-force-guard", action="store_true")
    p.add_argument("--confirm-head-motion", action="store_true")
    p.add_argument("--confirm-physical-motion", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.confirm_head_motion:
        raise SystemExit(
            "Refusing to construct Robot(): pass --confirm-head-motion only "
            "after clearing the head workspace and verifying the e-stop."
        )
    if not args.confirm_physical_motion:
        raise SystemExit(
            "Refusing physical commands: pass --confirm-physical-motion only "
            "with the workspace clear and physical e-stop ready."
        )

    bundle = load_bundle("vega")
    robot_cfg = bundle["robot"]
    robot_cfg["allow_robot_init_head_motion"] = True
    robot_cfg["working_arm"] = args.working_arm
    robot_cfg["urdf_path"] = str(Path(args.urdf).expanduser())
    robot_cfg["kinematics"]["ee_frame"] = args.ee_frame
    robot_cfg["motion"]["step_wait_time_s"] = args.step_wait
    robot_cfg["gripper"]["scope"] = args.gripper_scope
    if args.robot_name:
        robot_cfg["robot_name"] = args.robot_name
    if args.grip_current is not None:
        robot_cfg["gripper"]["grip_current_a"] = args.grip_current

    _, goals = load_runtime_targets(args.targets, bundle["tasks"])
    goal = require_goal(goals, args.part)

    skills = load_vega_skills()
    skill = skill_for_part(skills, args.part)
    if args.grip_current is not None:
        skill["grip_current_a"] = args.grip_current
    if args.allow_snap_without_force_guard:
        skill["allow_snap_without_force_guard"] = True

    safety = dict(skills.get("safety") or {})
    if args.force_delta_limit is not None:
        safety["force_delta_limit"] = args.force_delta_limit

    folder = create_session(
        "vega",
        args.operator,
        "physical_run",
        bundle,
    )
    Path(folder, "runtime_targets.json").write_text(
        Path(args.targets).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    Path(folder, "effective_skill.json").write_text(
        json.dumps({"part": args.part, "skill": skill, "safety": safety}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    def log(phase, event, details):
        append_event(
            folder,
            mode="physical_run",
            robot="vega",
            part=args.part,
            phase=phase,
            event=event,
            details=details,
        )
        print(f"[{phase}] {event}")

    robot = VegaAdapter(robot_cfg)
    started = time.monotonic()
    result = None
    failure = ""
    try:
        print(f"Session: {folder}")
        print("Connecting dexcontrol; Robot() may move the head now.")
        robot.connect()
        mark_hardware_connected(folder)

        print("Connecting/homing CAN grippers. Keep hands clear.")
        robot.connect_gripper()

        result = execute_part(
            robot,
            goal,
            skill,
            safety=safety,
            speed_scale=args.speed_scale,
            event=log,
        )
        print(result)
        return 0
    except KeyboardInterrupt:
        failure = "KeyboardInterrupt"
        print("Interrupted: activating software e-stop.", file=sys.stderr)
        try:
            robot.stop()
        finally:
            return 130
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        print(f"FAILED: {failure}", file=sys.stderr)
        try:
            robot.stop()
        except Exception as stop_exc:
            print(f"Software stop also failed: {stop_exc}", file=sys.stderr)
        return 1
    finally:
        elapsed = time.monotonic() - started
        append_trial(
            folder,
            trial_id=1,
            part=args.part,
            grasp_success=None if result is None else result.grasp_verified,
            placement_success=None if result is None else result.placement_verified,
            elapsed_s=round(elapsed, 3),
            failure_stage=failure,
            notes=(
                ""
                if result is None or result.note is None
                else result.note
            ),
        )
        try:
            robot.close()
        except Exception as close_exc:
            print(f"Shutdown warning: {close_exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
