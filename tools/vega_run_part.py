"""One-part runner. --check-only validates local inputs without Robot or CAN."""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steadyhand.adapters.vega import VegaAdapter
from steadyhand.config import load_bundle, read_json, validate_bundle
from steadyhand.executor import execute_part, validate_execution
from steadyhand.grippers.vega import VegaCanGripper
from steadyhand.sessions import append_event, append_trial, create_session, mark_hardware_connected, write_json
from steadyhand.skill_config import load_vega_skills, skill_for_part
from steadyhand.targets import load_runtime_targets, require_goal


def fixed_pair(text):
    try:
        name, value = text.split("=", 1)
        if not name:
            raise ValueError()
        return name, float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected JOINT=VALUE in URDF units") from exc


def add_hardware_arguments(p):
    p.add_argument("--robot-config", help="onsite copy of configs/robots/vega.json")
    p.add_argument("--skills", help="onsite copy of configs/skills/vega.json")
    p.add_argument("--working-arm", choices=("left", "right"))
    p.add_argument("--robot-name")
    p.add_argument("--urdf")
    p.add_argument("--ee-frame")
    p.add_argument("--base-frame")
    p.add_argument("--fixed", action="append", type=fixed_pair, default=[])
    p.add_argument("--step-wait", type=float)
    p.add_argument("--max-joint-speed", type=float)
    p.add_argument("--control-hz", type=float)
    p.add_argument("--joint-tolerance", type=float)
    p.add_argument("--joint-timeout", type=float)
    p.add_argument("--speed-scale", type=float, default=0.15)
    p.add_argument("--gripper-scope", choices=("right",))
    p.add_argument("--grip-current", type=float)
    p.add_argument("--force-delta-limit", type=float)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--confirm-head-motion", action="store_true")
    p.add_argument("--confirm-physical-motion", action="store_true")


def configured_bundle(args):
    bundle = load_bundle("vega")
    if args.robot_config:
        bundle["robot"] = read_json(args.robot_config)
        validate_bundle(bundle)
        if bundle["robot"]["robot_id"] != "vega":
            raise ValueError("--robot-config must describe Vega")
    cfg = bundle["robot"]
    for arg, key in (("working_arm", "working_arm"), ("robot_name", "robot_name"),
                     ("max_joint_speed", "max_joint_speed_rad_s")):
        if getattr(args, arg) is not None:
            cfg[key] = getattr(args, arg)
    if args.urdf:
        cfg["urdf_path"] = str(Path(args.urdf).expanduser())
    kin = cfg.setdefault("kinematics", {})
    for key in ("ee_frame", "base_frame"):
        if getattr(args, key) is not None:
            kin[key] = getattr(args, key)
    if len(dict(args.fixed)) != len(args.fixed):
        raise ValueError("Duplicate --fixed joint")
    kin.setdefault("fixed_joint_values", {}).update(dict(args.fixed))
    motion = cfg.setdefault("motion", {})
    for arg, key in (("step_wait", "step_wait_time_s"), ("control_hz", "control_hz"),
                     ("joint_tolerance", "joint_reached_tolerance_rad"),
                     ("joint_timeout", "joint_timeout_s")):
        if getattr(args, arg) is not None:
            motion[key] = getattr(args, arg)
    gripper = cfg.setdefault("gripper", {})
    if args.gripper_scope:
        gripper["scope"] = args.gripper_scope
    elif not gripper.get("scope") and cfg.get("working_arm") == "right":
        gripper["scope"] = cfg["working_arm"]
    if args.grip_current is not None:
        gripper["grip_current_a"] = args.grip_current
    if not args.check_only:
        if not args.confirm_head_motion or not args.confirm_physical_motion:
            raise ValueError("Hardware run requires --confirm-head-motion and --confirm-physical-motion")
        if not sys.stdin.isatty():
            raise ValueError("Physical runner needs an interactive terminal for grasp/placement verification")
    cfg["allow_robot_init_head_motion"] = bool(args.confirm_head_motion)
    return bundle


def prepare_inputs(args, bundle, order):
    cfg = bundle["robot"]
    if not cfg["kinematics"].get("base_frame"):
        raise ValueError("Configure kinematics.base_frame or --base-frame from the actual URDF")
    targets, goals = load_runtime_targets(args.targets, bundle["tasks"],
                                         expected_base_frame=cfg["kinematics"]["base_frame"])
    skills = load_vega_skills(args.skills)
    safety = dict(skills.get("safety") or {})
    if args.force_delta_limit is not None:
        safety["force_delta_limit"] = args.force_delta_limit
    effective = {}
    for name in order:
        goal = require_goal(goals, name)
        skill = skill_for_part(skills, name)
        skill["runtime_pose_type"] = targets.get("pose_type", "object")
        if args.grip_current is not None or skill.get("grip_current_a") is None:
            skill["grip_current_a"] = cfg["gripper"].get("grip_current_a")
        validate_execution(goal, skill, safety, args.speed_scale)
        effective[name] = skill
    if cfg["gripper"].get("scope") != "right":
        raise ValueError(
            "Physical part execution requires one gripper scope matching the "
            "working arm; the official driver has no top-level both-grip"
        )
    VegaCanGripper(cfg["gripper"]).validate_config()
    robot = VegaAdapter(cfg)
    robot.prepare()
    return robot, targets, goals, effective, safety


def operator_verify(stage, name):
    prompts = {"grasp": "Jaws hold the intended part without crushing it",
               "lift": "Lifted part remains securely held and the path is clear",
               "insertion": "Part is physically seated correctly; release is safe",
               "placement": "Released part is correctly placed and stable"}
    return input(f"{name}: {prompts[stage]}? Type yes to continue: ").strip().lower() == "yes"


def stop_report(robot):
    try:
        robot.stop()
    except BaseException as exc:
        print(f"SOFTWARE STOP FAILED; USE PHYSICAL E-STOP: {exc}", file=sys.stderr)


def close_report(robot):
    try:
        robot.close()
    except BaseException as exc:
        print(f"Shutdown warning; verify robot stopped: {exc}", file=sys.stderr)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", required=True)
    p.add_argument("--part", required=True)
    p.add_argument("--operator", required=True)
    add_hardware_arguments(p)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    bundle = configured_bundle(args)
    robot, targets, goals, skills, safety = prepare_inputs(args, bundle, [args.part])
    if args.check_only:
        print("Local configuration/URDF validation passed. No Robot or CAN constructed; no path/collision proof.")
        return 0
    folder = create_session("vega", args.operator, "physical_run", bundle)
    write_json(folder / "runtime_targets.json", targets)
    write_json(folder / "effective_skill.json", {"part": args.part, "skill": skills[args.part], "safety": safety})
    def log(phase, event, details):
        append_event(folder, mode="physical_run", robot="vega", part=args.part,
                     phase=phase, event=event, details=details)
        print(f"[{phase}] {event}")
    started, result, failure = time.monotonic(), None, ""
    try:
        print(
            f"Session: {folder}\nConnecting Robot: head may move. "
            f"CAN homes the {bundle['robot']['gripper']['scope']} jaw next."
        )
        robot.connect()
        mark_hardware_connected(folder)
        robot.connect_gripper()
        result = execute_part(robot, goals[args.part], skills[args.part], safety=safety,
                              speed_scale=args.speed_scale, event=log, verify=operator_verify)
        print(result)
        return 0
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        stop_report(robot)
        print(f"FAILED: {failure}", file=sys.stderr)
        return 130 if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 1
    finally:
        close_report(robot)  # cleanup before potentially failing disk I/O
        append_trial(folder, trial_id=1, part=args.part,
                     grasp_success=None if result is None else result.grasp_verified,
                     placement_success=None if result is None else result.placement_verified,
                     elapsed_s=round(time.monotonic() - started, 3), failure_stage=failure,
                     notes="" if result is None else result.note or "")


if __name__ == "__main__":
    raise SystemExit(main())
