"""Validated physical sequence. First bring-up defaults to battery_size1."""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steadyhand.executor import execute_part
from steadyhand.sessions import append_event, append_trial, create_session, mark_hardware_connected, write_json
from tools.vega_run_part import (add_hardware_arguments, configured_bundle,
    prepare_inputs, operator_verify, stop_report, close_report)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", required=True)
    p.add_argument("--operator", required=True)
    p.add_argument("--parts", default="battery_size1", help="explicit comma-separated order")
    p.add_argument("--return-home-q", nargs=7, type=float)
    add_hardware_arguments(p)
    args = p.parse_args(argv)
    bundle = configured_bundle(args)
    order = [x.strip() for x in args.parts.split(",") if x.strip()]
    if not order or len(set(order)) != len(order) or any(x not in bundle["tasks"]["parts"] for x in order):
        raise ValueError("--parts must be a nonempty sequence of distinct known parts")
    if len(order) > 1 and args.return_home_q is None:
        raise ValueError("Multi-part execution needs a physically validated --return-home-q and clear joint path")
    robot, targets, goals, skills, safety = prepare_inputs(args, bundle, order)
    if args.return_home_q is not None:
        robot._kinematics.forward(args.return_home_q)
    if args.check_only:
        print("Local sequence inputs validated; no hardware contacted and no path/collision proof.")
        return 0
    folder = create_session("vega", args.operator, "physical_sequence", bundle)
    write_json(folder / "runtime_targets.json", targets)
    write_json(folder / "sequence.json", {"parts": order, "skills": skills, "safety": safety,
        "return_home_q": args.return_home_q, "speed_scale": args.speed_scale})
    try:
        robot.connect()
        mark_hardware_connected(folder)
        robot.connect_gripper()
        for trial_id, name in enumerate(order, 1):
            def log(phase, event, details):
                append_event(folder, mode="physical_sequence", robot="vega", part=name,
                             phase=phase, event=event, details=details)
                print(f"[{name}:{phase}] {event}")
            started, result, failure = time.monotonic(), None, ""
            try:
                # Submitted policy homes before each part after the first.
                if trial_id > 1:
                    log("return_home", "started", {})
                    robot.move_joints(args.return_home_q, speed_scale=args.speed_scale)
                    log("return_home", "completed", {})
                result = execute_part(robot, goals[name], skills[name], safety=safety,
                                      speed_scale=args.speed_scale, event=log, verify=operator_verify)
            except BaseException as exc:
                failure = f"{type(exc).__name__}: {exc}"
                stop_report(robot)
                raise
            finally:
                append_trial(folder, trial_id=trial_id, part=name,
                    grasp_success=None if result is None else result.grasp_verified,
                    placement_success=None if result is None else result.placement_verified,
                    elapsed_s=round(time.monotonic() - started, 3), failure_stage=failure,
                    notes="" if result is None else result.note or "")
        return 0
    except BaseException as exc:
        stop_report(robot)
        print(f"SEQUENCE ABORTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 130 if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 1
    finally:
        close_report(robot)


if __name__ == "__main__":
    raise SystemExit(main())
