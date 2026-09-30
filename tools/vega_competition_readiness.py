"""Produce a no-hardware readiness report for the Vega competition path.

This command validates the files and gates consumed by the competition menu,
then runs the repository's offline tests.  It never connects to a robot,
camera, gripper, or head and never edits calibration or task-coordinate data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.board_calibration import load_board_calibration
from steadyhand.board_geometry import (
    validate_task_board_geometry,
    validate_task_coordinate_extent,
)
from steadyhand.config import load_bundle
from steadyhand.wrist_part_profiles import PART_NAMES, load_profiles
from steadyhand.wrist_part_profiles import file_sha256
from steadyhand.execution_offsets import load_offsets
from tools.vega_competition_pipeline import (
    FALLBACK_CALIBRATION,
    ROOT,
    TASK_COORDINATES,
    _load_competition_plan,
    _profile_ready_for_action,
    _load_competition_actions,
    _load_runtime,
    _task_targets,
)


def _git_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _unchanged_paths():
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--", "calibration", "configs/task_coordinates.json"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line for line in result.stdout.splitlines() if line]


def _run_regression_suite():
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"]
    try:
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": " ".join(command), "passed": False,
            "returncode": None, "output": str(exc),
        }
    output = (result.stdout + "\n" + result.stderr).strip()
    return {
        "command": " ".join(command), "passed": result.returncode == 0,
        "returncode": result.returncode, "output": output[-4000:],
    }


def _pickup_route_audit(runtime, profiles):
    """Real URDF IK, no Robot()/CAN/camera creation or motion commands."""
    from copy import deepcopy
    from steadyhand.adapters.vega import VegaAdapter
    from steadyhand.board_relative import resolve_profile_target
    from steadyhand.executor import preflight_tcp_segmented
    from steadyhand.models import Pose
    from steadyhand.vega_presets import configured_right_preset
    from tools.vega_wrist_part_calibrate import PartSession, MOTION_STEPS
    cfg = deepcopy(runtime[0]["robot"])
    cfg["kinematics"]["near_target_fallback"] = True
    robot = VegaAdapter(cfg)
    robot.prepare()  # local URDF only; deliberately never connect()
    seed, _ = configured_right_preset(cfg, "right_ready")
    start = robot._kinematics.forward(seed)
    outcomes = []
    # Sample the reference pose and four axial ±10 mm board shifts.
    for dx, dy in ((0., 0.), (.01, 0.), (-.01, 0.), (0., .01), (0., -.01)):
        live = list(runtime)
        center, ux, uy, plane = runtime[2]
        live[2] = ((center[0] + dx, center[1] + dy), ux, uy, deepcopy(plane))
        for part, profile in profiles["parts"].items():
            s = PartSession.__new__(PartSession)
            s.runtime, s.part = tuple(live), part
            s.targets = _task_targets(s.runtime, runtime[1], .100, use_profiles=False)
            nominal = s.targets[f"task.{part}.pick"]
            xy, quat = resolve_profile_target(profile, s.runtime[2], runtime[3], nominal,
                                              action="pick", no_cv=True)
            hover = Pose((*xy, s.surface(*xy) + .100), quat)
            x, y = hover.position_m[:2]
            grasp = Pose((x, y, s.surface(x, y) + profile["grasp_clearance_m"]), hover.quaternion_wxyz)
            row = {"part": part, "board_shift_mm": [dx * 1000, dy * 1000],
                   "hover_tcp_m": list(hover.position_m), "grasp_tcp_m": list(grasp.position_m)}
            try:
                q = seed
                previous = start
                for stage, target in (("approach", hover), ("descent", grasp), ("lift", hover)):
                    q = preflight_tcp_segmented(robot._kinematics, q, previous, target, **MOTION_STEPS)
                    previous = target
                row.update(passed=True, final_joints_rad=list(q))
            except Exception as exc:
                row.update(passed=False, stage=stage, error=f"{type(exc).__name__}: {exc}")
            outcomes.append(row)
    return {"scope": "saved pickup approach/descent/lift, reference board and axial ±10 mm shifts; no collision/physical validation",
            "passed": all(row["passed"] for row in outcomes), "routes": outcomes}


def build_report(*, run_tests=True, run_ik=False):
    """Validate static readiness contracts and return a JSON-serializable report."""
    report = {
        "schema_version": 1,
        "report_kind": "vega_competition_onsite_readiness",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "hardware_accessed": False,
        "calibration_or_task_files_modified": _unchanged_paths(),
        "checks": {},
        "profile_readiness": {},
        "onsite_commands": [
            "python3 tools/vega_competition_pipeline.py --check-only --competition-run",
            "python3 tools/vega_competition_pipeline.py --test-positions task.battery_size1.pick task.gear_60teeth.pick --clearance-mm 100 --confirm-physical-motion",
            "python3 tools/vega_competition_pipeline.py --task-test battery_size1.pick_place --confirm-head-motion --confirm-physical-motion",
            "python3 tools/vega_competition_pipeline.py --competition-run --confirm-physical-motion",
        ],
        "physical_prerequisites": [
            "Inspect fresh board registration and projected hovers; board translation alone does not require full recalibration.",
            "Verify existing pickups first. Re-teach only a failed part; preserve working pickup/place profiles.",
            "Confirm grasp and release physically; inspect every failed or possible-held result.",
            "Collision clearance and gripper reliability still require onsite observation.",
        ],
    }
    checks = report["checks"]
    try:
        task_data = json.loads(TASK_COORDINATES.read_text(encoding="utf-8"))
        validate_task_board_geometry(task_data)
        validate_task_coordinate_extent(task_data)
        checks["task_coordinates"] = {
            "passed": True,
            "board_width_m": task_data.get("board_width_m"),
            "official_parts": list(task_data.get("official_order") or []),
        }
    except Exception as exc:
        checks["task_coordinates"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        cfg = load_bundle("vega")["robot"]
        calibration = load_board_calibration(
            ROOT / "calibration" / "vega_board_manual.json", cfg,
            fallback_path=FALLBACK_CALIBRATION,
        )
        checks["calibration"] = {
            "passed": calibration.get("schema_version") == 2,
            "path": calibration.get("path"),
            "schema_version": calibration.get("schema_version"),
            "permanent_fallback": calibration.get("is_permanent_fallback"),
            "sha256": calibration.get("sha256"),
            "axis_angle_error_deg": calibration.get("axis_angle_error_deg"),
        }
        profiles = load_profiles(ROOT / "calibration" / "wrist_part_profiles.json", cfg)
        plan = _load_competition_plan()
        for part in PART_NAMES:
            profile = (profiles.get("parts") or {}).get(part)
            pick_ready, pick_reason = _profile_ready_for_action(profiles, part, "pick")
            place_ready, place_reason = _profile_ready_for_action(profiles, part, "pick_place")
            report["profile_readiness"][part] = {
                "profile_present": isinstance(profile, dict),
                "pick": {"ready": pick_ready, "reason": pick_reason},
                "pick_place": {"ready": place_ready, "reason": place_reason},
            }
        eligible = [
            f"{part}.{plan['default_action']}" for part in plan["pick_priority"]
            if report["profile_readiness"][part][plan["default_action"]]["ready"]
        ]
        checks["competition_plan"] = {
            "passed": True,
            "default_action": plan["default_action"],
            "priority": plan["pick_priority"],
            "max_retries_per_part": plan["max_retries_per_part"],
            "eligible_actions_now": eligible,
        }
        settings = _load_competition_actions()
        offsets = load_offsets()
        checks["execution_settings"] = {"passed": True, "competition": settings,
                                         "offsets": offsets.as_dict()}
        runtime = _load_runtime()
        targets = _task_targets(runtime, runtime[1], .100)
        report["reference_hovers_m"] = {name: list(pose.position_m) for name, pose in targets.items()}
        for part, profile in profiles["parts"].items():
            info = report["profile_readiness"][part]
            info.update(grasp_clearance_mm=profile["grasp_clearance_m"] * 1000 if profile.get("grasp_clearance_m") is not None else None,
                        jaw_open_fraction=profile.get("gripper_open_fraction"),
                        board_reference=(profile.get("pickup_board", {}).get("migration", {}).get("confidence")
                                         or ("taught" if profile.get("pickup_board") else "legacy reference assumed")))
            try:
                path = Path(profile["template"]["path"])
                if not path.is_absolute():
                    path = ROOT / path
                info["template_integrity_ok"] = file_sha256(path) == profile["template"]["sha256"]
            except OSError:
                info["template_integrity_ok"] = False
        if run_ik:
            try:
                report["pickup_ik_audit"] = _pickup_route_audit(runtime, profiles)
            except Exception as exc:
                report["pickup_ik_audit"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
            audit = report["pickup_ik_audit"]
            checks["pickup_ik_routes"] = {"passed": audit["passed"],
                "routes_checked": len(audit.get("routes", [])),
                "failed": sum(not row["passed"] for row in audit.get("routes", []))}
    except Exception as exc:
        checks["calibration"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        checks["competition_plan"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

    if report["calibration_or_task_files_modified"] is not None:
        checks["protected_inputs_unchanged"] = {
            "passed": not report["calibration_or_task_files_modified"],
            "modified_paths": report["calibration_or_task_files_modified"],
            "required_for_readiness": False,
            "note": "Onsite teaching normally modifies calibration. Preserve these files; validated edits are not a readiness failure.",
        }
    else:
        checks["protected_inputs_unchanged"] = {
            "passed": False, "error": "git diff could not be inspected",
            "required_for_readiness": False,
        }
    report["tests"] = _run_regression_suite() if run_tests else {"skipped": True}
    report["passed"] = all(
        value.get("passed", False) for value in checks.values()
        if value.get("required_for_readiness", True)
    ) and (not run_tests or report["tests"].get("passed", False))
    eligible = (checks.get("competition_plan") or {}).get("eligible_actions_now") or []
    report["competition_actions_eligible"] = bool(eligible)
    report["ready_for_onsite_validation"] = bool(report["passed"])
    report["competition_run_ready"] = bool(report["passed"] and eligible)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="runs/onsite_readiness_report.json",
        help="report path relative to the repository (default: runs/onsite_readiness_report.json)",
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--ik", action="store_true", help="also dry-run all pickup routes and ±10 mm board shifts using local Pinocchio")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(run_tests=not args.skip_tests, run_ik=args.ik)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"READINESS REPORT = {output}")
    print(f"HARDWARE ACCESSED = {report['hardware_accessed']}")
    print(f"READY FOR ONSITE VALIDATION = {report['ready_for_onsite_validation']}")
    print(f"VERIFIED COMPETITION ACTIONS AVAILABLE = {report['competition_actions_eligible']}")
    for name, check in report["checks"].items():
        status = "PASS" if check.get("passed") else ("WARN" if not check.get("required_for_readiness", True) else "FAIL")
        print(f"  {status} {name}")
    if report["tests"].get("skipped"):
        print("REGRESSION TESTS SKIPPED; this result covers requested static/IK checks only.")
    for route in report.get("pickup_ik_audit", {}).get("routes", []):
        if not route["passed"]:
            print(f"  IK FAIL {route['part']} board shift {route['board_shift_mm']} mm "
                  f"at {route['stage']}: {route['error']}")
    if not report["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
