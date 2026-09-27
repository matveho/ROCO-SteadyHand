"""Run a physical Vega multi-part sequence with one robot/gripper session.

Use this only after vega_run_part.py has validated the individual skills.
The default order follows configs/task_board.json.

A validated --return-home-q can be supplied to reproduce the submitted
policy's consistent IK seed between parts.
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


def main(argv=None):
    p = argparse.ArgumentParser(description="Gated physical Vega sequence runner")
    p.add_argument("--targets", required=True)
    p.add_argument("--operator", required=True)
    p.add_argument("--parts", help="comma-separated subset; default task order")
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
    p.add_argument("--return-home-q", nargs=7, type=float)
    p.add_argument("--confirm-head-motion", action="store_true")
    p.add_argument("--confirm-physical-motion", action="store_true")
    args = p.parse_args(argv)

    if not args.confirm_head_motion or not args.confirm_physical_motion:
        raise SystemExit(
            "Refusing hardware run. Both --confirm-head-motion and "
            "--confirm-physical-motion are required."
        )

    bundle = load_bundle("vega")
    order = list(bundle["tasks"]["part_order"])
    if args.parts:
        requested = [x.strip() for x in args.parts.split(",") if x.strip()]
        unknown = [x for x in requested if x not in order]
        if unknown:
            raise SystemExit("Unknown parts: " + ", ".join(unknown))
        order = requested

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
    for name in order:
        require_goal(goals, name)

    skills_cfg = load_vega_skills()
    safety = dict(skills_cfg.get("safety") or {})
    if args.force_delta_limit is not None:
        safety["force_delta_limit"] = args.force_delta_limit

    folder = create_session("vega", args.operator, "physical_sequence", bundle)
    Path(folder, "runtime_targets.json").write_text(
        Path(args.targets).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    Path(folder, "sequence.json").write_text(
        json.dumps(
            {
                "parts": order,
                "return_home_q": args.return_home_q,
                "speed_scale": args.speed_scale,
                "safety": safety,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    robot = VegaAdapter(robot_cfg)
    try:
        print(f"Session: {folder}")
        robot.connect()
        mark_hardware_connected(folder)
        robot.connect_gripper()

        for trial_id, name in enumerate(order, 1):
            goal = goals[name]
            skill = skill_for_part(skills_cfg, name)
            if args.grip_current is not None:
                skill["grip_current_a"] = args.grip_current
            if args.allow_snap_without_force_guard:
                skill["allow_snap_without_force_guard"] = True

            def log(phase, event, details, part=name):
                append_event(
                    folder,
                    mode="physical_sequence",
                    robot="vega",
                    part=part,
                    phase=phase,
                    event=event,
                    details=details,
                )
                print(f"[{part}:{phase}] {event}")

            started = time.monotonic()
            result = None
            failure = ""
            try:
                result = execute_part(
                    robot,
                    goal,
                    skill,
                    safety=safety,
                    speed_scale=args.speed_scale,
                    event=log,
                )
                if args.return_home_q and trial_id < len(order):
                    log("return_home", "started", {})
                    robot.move_joints(
                        args.return_home_q,
                        speed_scale=args.speed_scale,
                    )
                    log("return_home", "completed", {})
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
                append_trial(
                    folder,
                    trial_id=trial_id,
                    part=name,
                    elapsed_s=round(time.monotonic() - started, 3),
                    failure_stage=failure,
                )
                print(f"{name} FAILED: {failure}", file=sys.stderr)
                print("Aborting sequence and activating software e-stop.", file=sys.stderr)
                robot.stop()
                return 1

            append_trial(
                folder,
                trial_id=trial_id,
                part=name,
                grasp_success=result.grasp_verified,
                placement_success=result.placement_verified,
                elapsed_s=round(time.monotonic() - started, 3),
                notes=result.note or "",
            )

        return 0
    except KeyboardInterrupt:
        print("Interrupted: activating software e-stop.", file=sys.stderr)
        try:
            robot.stop()
        finally:
            return 130
    finally:
        try:
            robot.close()
        except Exception as exc:
            print(f"Shutdown warning: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
