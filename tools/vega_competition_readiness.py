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
from tools.vega_competition_pipeline import (
    COMPETITION_PLAN,
    FALLBACK_CALIBRATION,
    ROOT,
    TASK_COORDINATES,
    _load_competition_plan,
    _profile_ready_for_action,
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


def build_report(*, run_tests=True):
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
            "python3 tools/vega_competition_pipeline.py --check-only --competition-plan priority_pick_place",
            "python3 tools/vega_competition_pipeline.py --recalibrate --confirm-head-motion --confirm-physical-motion",
            "python3 tools/vega_competition_pipeline.py --check-only --competition-plan priority_pick_place",
            "python3 tools/vega_competition_pipeline.py --wrist-calibrate battery_size1 --confirm-head-motion --confirm-physical-motion",
            "python3 tools/vega_competition_pipeline.py --task-test battery_size1.pick_place --confirm-head-motion --confirm-physical-motion",
            "python3 tools/vega_competition_pipeline.py --competition-plan priority_pick_place --confirm-head-motion --confirm-physical-motion",
        ],
        "physical_prerequisites": [
            "Recalibrate after the board is placed onsite.",
            "Teach and verify each wrist_a profile before its action becomes eligible.",
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
    except Exception as exc:
        checks["calibration"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        checks["competition_plan"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

    if report["calibration_or_task_files_modified"] is not None:
        checks["protected_inputs_unchanged"] = {
            "passed": not report["calibration_or_task_files_modified"],
            "modified_paths": report["calibration_or_task_files_modified"],
        }
    else:
        checks["protected_inputs_unchanged"] = {
            "passed": False, "error": "git diff could not be inspected",
        }
    report["tests"] = _run_regression_suite() if run_tests else {"skipped": True}
    report["passed"] = all(
        value.get("passed", False) for value in checks.values()
    ) and report["tests"].get("passed", False)
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
    args = parser.parse_args(argv)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(run_tests=not args.skip_tests)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"READINESS REPORT = {output}")
    print(f"HARDWARE ACCESSED = {report['hardware_accessed']}")
    print(f"READY FOR ONSITE VALIDATION = {report['ready_for_onsite_validation']}")
    print(f"VERIFIED COMPETITION ACTIONS AVAILABLE = {report['competition_actions_eligible']}")
    for name, check in report["checks"].items():
        print(f"  {'PASS' if check.get('passed') else 'FAIL'} {name}")
    if not report["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
