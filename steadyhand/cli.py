"""Small offline commands for preparing an onsite session."""

import argparse
import platform
import sys

from .config import (ROBOTS, WORKSPACE, load_bundle, missing_motion_setup, missing_perception_setup, missing_setup)
from .runner import PHASES, dry_run
from .sessions import create_session


def main(argv=None):
    parser = argparse.ArgumentParser(description="SteadyHand onsite preparation (offline only)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect local Python and configuration; no robot access")
    check = commands.add_parser("check-config", help="Check templates and list missing setup")
    check.add_argument("--robot", choices=ROBOTS, required=True)
    check.add_argument("--require-ready", action="store_true", help="Fail if setup fields are missing")
    session = commands.add_parser("new-session", help="Create a folder for physical trial notes")
    session.add_argument("--robot", choices=ROBOTS, required=True)
    session.add_argument("--operator", required=True)
    dry = commands.add_parser("dry-run", help="Exercise a mock sequence and save event logs")
    dry.add_argument("--robot", choices=ROBOTS, required=True)
    dry.add_argument("--part", required=True)
    dry.add_argument("--operator", default="offline")
    dry.add_argument("--fail-at", choices=PHASES)
    args = parser.parse_args(argv)
    try:
        if sys.version_info < (3, 10):
            raise ValueError("Python 3.10 or newer is required")
        if args.command == "doctor":
            print(f"Python {platform.python_version()} on {platform.system()}")
            print(f"Workspace: {WORKSPACE}")
            print("Offline doctor dependencies: standard library only")
            for robot in ROBOTS:
                bundle = load_bundle(robot)
                motion_missing = missing_motion_setup(bundle)
                perception_missing = missing_perception_setup(bundle)
                print(
                    f"{robot}: templates valid; "
                    f"{len(motion_missing)} motion fields missing; "
                    f"{len(perception_missing)} perception/calibration fields missing"
                )
            print("Vega: live joint/camera/gripper/IK path is implemented but configuration-gated; no connection attempted")
            print("Sharpa: full-body live adapter still awaits the onsite North interface")
            return 0
        bundle = load_bundle(args.robot)
        if args.command == "check-config":
            motion_missing = missing_motion_setup(bundle)
            perception_missing = missing_perception_setup(bundle)
            missing = motion_missing + perception_missing
            print(f"{args.robot}: configuration structure is valid")
            print("  Motion:")
            for item in motion_missing:
                print(f"    MISSING {item}")
            print("  Perception/calibration:")
            for item in perception_missing:
                print(f"    MISSING {item}")
            if args.robot == "vega":
                print("  Per-attempt part poses are supplied separately via runtime_targets.json.")
            print("Configuration checks do not establish hardware readiness.")
            return 2 if missing and args.require_ready else 0
        if args.command == "new-session":
            folder = create_session(args.robot, args.operator, "physical_notes", bundle)
            print(f"Session created: {folder}")
            print("Record physical attempts in trials.csv. No robot was contacted.")
            return 0
        if args.part not in bundle["tasks"]["parts"]:
            raise ValueError(f"Unknown part {args.part!r}; choose from {', '.join(bundle['tasks']['part_order'])}")
        folder = create_session(args.robot, args.operator, "dry_run", bundle)
        result = dry_run(folder, args.robot, args.part, args.fail_at)
        print(f"Dry-run logs: {folder}")
        print(result["note"])
        if result["failed_phase"]:
            print(f"Injected failure at {result['failed_phase']}; mock sequence stopped.")
            return 1
        print("Mock sequence completed.")
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
