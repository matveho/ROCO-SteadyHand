"""Standalone Vega joint/motion trace for one small free-space Cartesian move.

This tool is intentionally separate from normal task execution. It connects to
Vega, solves one Cartesian target, sends exactly one tracked joint target through
DexControl's ``move_to_joint_pos()``, then records plugin completion and measured
joint feedback as independent facts.

Default motion is +5 mm in base Z from the current TCP while preserving the
current TCP orientation. No gripper is opened, homed, closed, or otherwise
connected by this tool. Targets below the configured task floor plus a small
free-space clearance are rejected.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DELTA_M = (0.0, 0.0, 0.005)
DEFAULT_POLL_SECONDS = 4.0
DEFAULT_POLL_HZ = 20.0
DEFAULT_MAX_TRANSLATION_M = 0.03
MIN_FREE_SPACE_CLEARANCE_M = 0.03


@dataclass(frozen=True)
class JointSample:
    elapsed_s: float
    timestamp_ns: int
    positions_rad: tuple[float, ...]
    velocities_rad_s: tuple[float, ...]


def _finite_vector(values, length, name):
    result = tuple(float(v) for v in values)
    if len(result) != length or not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _max_joint_error(target, measured):
    return max(abs(float(a) - float(b)) for a, b in zip(target, measured))


def summarize_post_motion(
    *,
    plugin_state,
    pre_motion_timestamp_ns,
    target_joints_rad,
    tolerance_rad,
    samples,
):
    """Summarize plugin state, feedback freshness, and endpoint convergence.

    These predicates are intentionally independent. In particular, a stale
    cached sample can numerically match the target while ``fresh_feedback`` is
    still false.
    """
    target = _finite_vector(target_joints_rad, 7, "target joints")
    tolerance = float(tolerance_rad)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance_rad must be finite and positive")
    pre_stamp = int(pre_motion_timestamp_ns)
    if pre_stamp <= 0:
        raise ValueError("pre_motion_timestamp_ns must be positive")

    normalized = []
    for sample in samples:
        if not isinstance(sample, JointSample):
            sample = JointSample(**sample)
        q = _finite_vector(sample.positions_rad, 7, "sample positions")
        dq = _finite_vector(sample.velocities_rad_s, 7, "sample velocities")
        stamp = int(sample.timestamp_ns)
        if stamp <= 0:
            raise ValueError("sample timestamp_ns must be positive")
        elapsed = float(sample.elapsed_s)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("sample elapsed_s must be finite and non-negative")
        normalized.append(JointSample(elapsed, stamp, q, dq))

    plugin_finished = str(plugin_state) == "finished"
    fresh_feedback = any(s.timestamp_ns > pre_stamp for s in normalized)

    errors = [_max_joint_error(target, s.positions_rad) for s in normalized]
    reached = [err <= tolerance for err in errors]
    target_reached = any(reached)
    first_reached_elapsed_s = next(
        (s.elapsed_s for s, ok in zip(normalized, reached) if ok),
        None,
    )

    timestamp_progressions = 0
    previous = None
    for sample in normalized:
        if previous is not None and sample.timestamp_ns > previous:
            timestamp_progressions += 1
        previous = sample.timestamp_ns

    return {
        "plugin_finished": plugin_finished,
        "fresh_feedback": fresh_feedback,
        "target_reached": target_reached,
        "first_reached_elapsed_s": first_reached_elapsed_s,
        "sample_count": len(normalized),
        "fresh_sample_count": sum(s.timestamp_ns > pre_stamp for s in normalized),
        "timestamp_progressions": timestamp_progressions,
        "initial_max_joint_error_rad": errors[0] if errors else None,
        "final_max_joint_error_rad": errors[-1] if errors else None,
        "minimum_max_joint_error_rad": min(errors) if errors else None,
    }


def _joint_limit_report(names, limits, target):
    rows = []
    for name, (lo, hi), value in zip(names, limits, target):
        lo = float(lo)
        hi = float(hi)
        value = float(value)
        rows.append(
            {
                "name": str(name),
                "target_rad": value,
                "to_lower_limit_rad": value - lo,
                "to_upper_limit_rad": hi - value,
                "nearest_limit_rad": min(value - lo, hi - value),
            }
        )
    return rows


def _print_vector(label, values, digits=6):
    print(
        f"{label} = (" + ", ".join(f"{float(v):.{digits}f}" for v in values) + ")",
        flush=True,
    )


def _read_sample(robot, elapsed_s):
    q = tuple(float(v) for v in robot._read_joint_positions())
    stamp = int(robot._state_timestamp())
    velocities = _finite_vector(
        robot._arm.get_joint_vel(), 7, "Vega joint velocity"
    )
    return JointSample(
        elapsed_s=float(elapsed_s),
        timestamp_ns=stamp,
        positions_rad=q,
        velocities_rad_s=velocities,
    )


def _sample_record(sample, target, pre_stamp, tolerance, previous_stamp=None):
    error = tuple(
        float(measured) - float(commanded)
        for measured, commanded in zip(sample.positions_rad, target)
    )
    max_error = max(abs(v) for v in error)
    max_velocity = max(abs(v) for v in sample.velocities_rad_s)
    return {
        **asdict(sample),
        "fresh_from_pre_motion": sample.timestamp_ns > pre_stamp,
        "timestamp_advanced_from_previous": (
            None if previous_stamp is None else sample.timestamp_ns > previous_stamp
        ),
        "joint_error_rad": error,
        "max_joint_error_rad": max_error,
        "max_abs_joint_velocity_rad_s": max_velocity,
        "target_reached": max_error <= tolerance,
    }


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="absolute TCP position in robot base metres; preserves current orientation",
    )
    group.add_argument(
        "--delta",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        help="relative TCP translation in robot base metres; preserves current orientation",
    )
    parser.add_argument("--speed-scale", type=float, default=0.45)
    parser.add_argument(
        "--max-translation-m",
        type=float,
        default=DEFAULT_MAX_TRANSLATION_M,
        help="refuse a Cartesian displacement larger than this; raise explicitly for a known reproduction",
    )
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--poll-hz", type=float, default=DEFAULT_POLL_HZ)
    parser.add_argument(
        "--wait-timeout-s",
        type=float,
        default=None,
        help="MotionHandle.wait timeout; default uses configured Vega joint timeout",
    )
    parser.add_argument("--output", help="JSON trace path; default runs/vega_motion_trace_<UTC>.json")
    parser.add_argument("--confirm-physical-motion", action="store_true")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.confirm_physical_motion:
        parser.error("--confirm-physical-motion is required")
    if not 0 < float(args.speed_scale) <= 1.0:
        parser.error("--speed-scale must be in (0, 1]")
    if not 0 < float(args.max_translation_m) <= 0.30:
        parser.error("--max-translation-m must be in (0, 0.30]")
    if not 3.0 <= float(args.poll_seconds) <= 5.0:
        parser.error("--poll-seconds must be 3..5 seconds")
    if not 10.0 <= float(args.poll_hz) <= 50.0:
        parser.error("--poll-hz must be 10..50 Hz")
    if args.wait_timeout_s is not None and float(args.wait_timeout_s) <= 0:
        parser.error("--wait-timeout-s must be positive")

    # Delay hardware/project imports until all CLI-only validation has passed.
    from steadyhand.adapters.vega import VegaAdapter
    from steadyhand.config import load_bundle
    from steadyhand.geometry import pose_distance
    from steadyhand.models import Pose
    from steadyhand.skill_config import load_vega_skills

    cfg = load_bundle("vega")["robot"]
    cfg["allow_robot_init_head_motion"] = True
    cfg["auto_clear_software_estop_on_connect"] = True
    floor_m = float(load_vega_skills()["safety"]["min_tcp_z_m"])
    tolerance = float(cfg["motion"]["joint_reached_tolerance_rad"])
    wait_timeout = (
        float(cfg["motion"]["joint_timeout_s"])
        if args.wait_timeout_s is None
        else float(args.wait_timeout_s)
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (
        Path(args.output).expanduser()
        if args.output
        else ROOT / "runs" / f"vega_motion_trace_{stamp}.json"
    )

    trace = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "tools/vega_motion_trace.py",
        "robot_name": cfg.get("robot_name"),
        "working_arm": cfg.get("working_arm"),
        "ee_frame": cfg.get("kinematics", {}).get("ee_frame"),
        "floor_m": floor_m,
        "joint_tolerance_rad": tolerance,
        "samples": [],
    }

    robot = VegaAdapter(cfg)
    failure = None
    software_stop_asserted = False
    try:
        robot.connect()
        estop = robot._read_estop_status()
        print("E-STOP STATUS =", estop, flush=True)
        if estop["button_pressed"]:
            raise RuntimeError("Physical e-stop is active; refusing diagnostic motion")
        if estop["software_estop_enabled"]:
            raise RuntimeError("Software e-stop remained active after opt-in clear")

        current_q = tuple(float(v) for v in robot._read_joint_positions())
        pre_stamp = int(robot._state_timestamp())
        current_tcp = robot._kinematics.forward(current_q)

        delta = DEFAULT_DELTA_M if args.target is None and args.delta is None else args.delta
        if args.target is not None:
            requested_xyz = _finite_vector(args.target, 3, "TCP target")
        else:
            delta = _finite_vector(delta, 3, "TCP delta")
            requested_xyz = tuple(
                float(p) + float(d) for p, d in zip(current_tcp.position_m, delta)
            )

        displacement = math.sqrt(
            sum((float(a) - float(b)) ** 2 for a, b in zip(requested_xyz, current_tcp.position_m))
        )
        if displacement > float(args.max_translation_m) + 1e-12:
            raise RuntimeError(
                f"Requested TCP displacement {displacement:.4f} m exceeds "
                f"--max-translation-m={float(args.max_translation_m):.4f}; "
                "raise the gate only for a known free-space reproduction"
            )
        minimum_z = floor_m + MIN_FREE_SPACE_CLEARANCE_M
        if float(requested_xyz[2]) < minimum_z:
            raise RuntimeError(
                f"Requested TCP z={float(requested_xyz[2]):.4f} m is below "
                f"diagnostic free-space floor guard {minimum_z:.4f} m"
            )

        requested_tcp = Pose(
            position_m=requested_xyz,
            quaternion_wxyz=tuple(float(v) for v in current_tcp.quaternion_wxyz),
        )
        target_q = tuple(float(v) for v in robot._kinematics.solve(requested_tcp, current_q))
        robot._check_joint_limits(target_q)
        joint_delta = tuple(b - a for a, b in zip(current_q, target_q))
        worst_joint_delta = max(abs(v) for v in joint_delta)
        max_total = float(cfg["motion"]["max_total_delta_rad"])
        if worst_joint_delta > max_total:
            raise RuntimeError(
                f"Solved target requires {worst_joint_delta:.4f} rad max joint delta, "
                f"above configured max_total_delta_rad={max_total:.4f}"
            )

        solved_tcp = robot._kinematics.forward(target_q)
        position_residual_m, orientation_residual_rad = pose_distance(solved_tcp, requested_tcp)
        limit_rows = _joint_limit_report(robot._joint_names, robot._joint_limits, target_q)

        print("MOTION TRACE PRECHECK", flush=True)
        _print_vector("CURRENT TCP XYZ m", current_tcp.position_m)
        _print_vector("CURRENT TCP WXYZ", current_tcp.quaternion_wxyz)
        _print_vector("CURRENT JOINTS rad", current_q)
        print(f"CURRENT JOINT TIMESTAMP ns = {pre_stamp}", flush=True)
        _print_vector("REQUESTED TCP XYZ m", requested_tcp.position_m)
        _print_vector("SOLVED TARGET JOINTS rad", target_q)
        _print_vector("PER-JOINT TARGET DELTA rad", joint_delta)
        print(f"MAX JOINT DELTA rad = {worst_joint_delta:.6f}", flush=True)
        print(
            f"IK RESIDUAL position_m={position_residual_m:.6f} "
            f"orientation_rad={orientation_residual_rad:.6f}",
            flush=True,
        )
        for row in limit_rows:
            print(
                "JOINT LIMIT "
                f"{row['name']}: target={row['target_rad']:.6f} "
                f"lower_margin={row['to_lower_limit_rad']:.6f} "
                f"upper_margin={row['to_upper_limit_rad']:.6f} "
                f"nearest={row['nearest_limit_rad']:.6f}",
                flush=True,
            )

        trace["pre_motion"] = {
            "current_tcp": asdict(current_tcp) if hasattr(current_tcp, "__dataclass_fields__") else {
                "position_m": list(current_tcp.position_m),
                "quaternion_wxyz": list(current_tcp.quaternion_wxyz),
            },
            "current_joints_rad": current_q,
            "current_joint_timestamp_ns": pre_stamp,
            "requested_tcp": {
                "position_m": requested_tcp.position_m,
                "quaternion_wxyz": requested_tcp.quaternion_wxyz,
            },
            "solved_target_joints_rad": target_q,
            "joint_delta_rad": joint_delta,
            "max_joint_delta_rad": worst_joint_delta,
            "joint_limits": limit_rows,
            "ik_residual_position_m": position_residual_m,
            "ik_residual_orientation_rad": orientation_residual_rad,
        }

        handle = robot._arm.move_to_joint_pos(
            target_q,
            relative=False,
            velocity_scale=float(args.speed_scale),
        )
        robot._active_motion_handle = handle
        initial_state = getattr(handle, "state", None)
        motion_id = getattr(handle, "motion_id", None)
        initial_message = getattr(handle, "message", "")
        print(
            f"MOTION HANDLE INITIAL state={initial_state!r} motion_id={motion_id!r} "
            f"message={initial_message!r}",
            flush=True,
        )

        wait_started = time.monotonic()
        terminal_state = None
        wait_error = None
        try:
            terminal_state = handle.wait(timeout=wait_timeout)
        except BaseException as exc:
            wait_error = exc
            terminal_state = getattr(handle, "state", None)
        wait_elapsed = time.monotonic() - wait_started
        terminal_message = getattr(handle, "message", "")
        print(
            f"MOTION HANDLE TERMINAL state={terminal_state!r} "
            f"message={terminal_message!r} elapsed_s={wait_elapsed:.6f}",
            flush=True,
        )
        trace["motion_handle"] = {
            "motion_id": motion_id,
            "initial_state": initial_state,
            "initial_message": initial_message,
            "wait_elapsed_s": wait_elapsed,
            "terminal_state": terminal_state,
            "terminal_message": terminal_message,
            "wait_error": None if wait_error is None else repr(wait_error),
        }
        if wait_error is not None:
            raise wait_error
        robot._active_motion_handle = None

        poll_started = time.monotonic()
        previous_stamp = None
        samples = []
        next_due = poll_started
        deadline = poll_started + float(args.poll_seconds)
        while True:
            now = time.monotonic()
            if now < next_due:
                time.sleep(next_due - now)
                now = time.monotonic()
            sample = _read_sample(robot, now - poll_started)
            samples.append(sample)
            record = _sample_record(
                sample,
                target_q,
                pre_stamp,
                tolerance,
                previous_stamp=previous_stamp,
            )
            trace["samples"].append(record)
            print(
                "SAMPLE "
                f"t={sample.elapsed_s:.3f}s "
                f"timestamp_ns={sample.timestamp_ns} "
                f"advanced={record['timestamp_advanced_from_previous']} "
                f"fresh={record['fresh_from_pre_motion']} "
                f"max_error_rad={record['max_joint_error_rad']:.6f} "
                f"max_velocity_rad_s={record['max_abs_joint_velocity_rad_s']:.6f} "
                f"reached={record['target_reached']}",
                flush=True,
            )
            if len(samples) == 1:
                _print_vector("MEASURED JOINTS AFTER HANDLE rad", sample.positions_rad)
                _print_vector("MEASURED JOINT VELOCITIES AFTER HANDLE rad/s", sample.velocities_rad_s)
                print(f"MEASURED JOINT TIMESTAMP AFTER HANDLE ns = {sample.timestamp_ns}", flush=True)
                _print_vector(
                    "PER-JOINT TARGET ERROR AFTER HANDLE rad",
                    tuple(m - t for m, t in zip(sample.positions_rad, target_q)),
                )
            previous_stamp = sample.timestamp_ns
            if now >= deadline:
                break
            next_due += 1.0 / float(args.poll_hz)
            if next_due > deadline:
                next_due = deadline

        summary = summarize_post_motion(
            plugin_state=terminal_state,
            pre_motion_timestamp_ns=pre_stamp,
            target_joints_rad=target_q,
            tolerance_rad=tolerance,
            samples=samples,
        )
        trace["summary"] = summary
        print("MOTION TRACE SUMMARY", flush=True)
        print(f"plugin_finished = {summary['plugin_finished']}", flush=True)
        print(f"fresh_feedback = {summary['fresh_feedback']}", flush=True)
        print(f"target_reached = {summary['target_reached']}", flush=True)
        print(
            f"first_reached_elapsed_s = {summary['first_reached_elapsed_s']!r}",
            flush=True,
        )

        if not (
            summary["plugin_finished"]
            and summary["fresh_feedback"]
            and summary["target_reached"]
        ):
            print(
                "DIAGNOSTIC RESULT INCOMPLETE/FAILED; asserting software e-stop",
                file=sys.stderr,
                flush=True,
            )
            robot.stop()
            software_stop_asserted = True
            return 2
        return 0
    except BaseException as exc:
        failure = exc
        trace["failure"] = repr(exc)
        try:
            robot.stop()
            software_stop_asserted = True
        except BaseException as stop_exc:
            trace["stop_failure"] = repr(stop_exc)
            print(f"STOP FAILED: {stop_exc}; use physical e-stop", file=sys.stderr, flush=True)
        raise
    finally:
        trace["software_stop_asserted"] = software_stop_asserted
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(trace, indent=2, default=list) + "\n", encoding="utf-8")
        print("TRACE JSON =", output.resolve(), flush=True)
        try:
            robot.close()
        except BaseException as close_exc:
            if failure is None:
                raise
            print(f"SHUTDOWN WARNING: {close_exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
