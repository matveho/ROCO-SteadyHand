"""Read-only robot evidence collector; stdlib only, runnable without deployment.

Copies current inputs, original images, and recent run records into a ZIP. The
only subprocesses are Git inspection and explicitly no-motion diagnostics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

PREFIX = "competition_status/latest/"
FILE_LIMIT = 40 * 1024 * 1024  # each published Git blob stays below host limits


def digest(data):
    return hashlib.sha256(data).hexdigest()


def mapping(value):
    return value if isinstance(value, dict) else {}


def command(root, args, *, timeout=45):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.setdefault("ROBOT_NAME", "dm/vgfcb66075ea-1u")
    try:
        result = subprocess.run(args, cwd=root, env=env, capture_output=True,
                                text=True, errors="replace", timeout=timeout)
        return {"command": args, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": args, "returncode": None, "error": str(exc)}


def files_under(directory):
    if not directory.is_dir() or directory.is_symlink():
        return
    for current, dirs, names in os.walk(directory, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "__pycache__")
                         and not (Path(current) / d).is_symlink())
        for name in sorted(names):
            path = Path(current) / name
            if not path.is_symlink() and path.is_file():
                yield path


def references(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)
    elif isinstance(value, str) and (value.startswith(("runs/", "calibration/", "/"))):
        yield value


def discover_runs(root):
    """Include interrupted teaching sessions which have no final summary yet."""
    folders = {}
    for path in files_under(root / "runs"):
        if path.name not in ("run_summary.json", "events.jsonl"):
            continue
        # Avoid recursively reporting earlier exports/backups.
        rel = path.relative_to(root / "runs")
        if any(p.startswith(("robot_report_", "robot_pull_", "board_profile_migrations")) for p in rel.parts):
            continue
        try:
            folders[path.parent] = max(folders.get(path.parent, 0), path.stat().st_mtime_ns)
        except OSError:
            continue
    return sorted(folders, key=lambda p: (folders[p], str(p)), reverse=True)


def collect(root, output, *, recent_runs=8, max_mb=256, include_runs=(), diagnostics=True):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not (root / "tools/vega_competition_pipeline.py").is_file():
        raise ValueError(f"Not the live Vega repository: {root}")
    if output.suffix.lower() != ".zip" or (output.is_relative_to(root) and not output.is_relative_to(root / "runs")):
        raise ValueError("Output must be a ZIP outside the repository or under runs/")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_root": str(root), "python": sys.executable,
              "collector_sha256": digest(Path(__file__).read_bytes()),
              "hardware_accessed": False, "physical_readiness": "requires operator confirmation",
              "warnings": [], "omitted_files": [], "missing_references": [],
              "diagnostics": {}, "profiles": {}, "recent_runs": [], "latest_by_part": {}}
    manifest, saved, input_hashes = {}, {}, {}
    total = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        def add_bytes(name, data):
            nonlocal total
            if name in manifest:
                return
            archive.writestr(PREFIX + name, data)
            manifest[name] = {"bytes": len(data), "sha256": digest(data)}
            total += len(data)

        def add(path, *, required=False):
            try:
                relative = path.relative_to(root).as_posix()
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    raise ValueError("symlink/outside repository")
                if relative in manifest:
                    return saved.get(relative)
                before = path.stat()
                if before.st_size > FILE_LIMIT or total + before.st_size > max_mb * 1024 * 1024:
                    raise ValueError("file/total size budget exceeded")
                data = path.read_bytes()
                if path.stat().st_mtime_ns != before.st_mtime_ns:
                    report["warnings"].append(f"Changed during capture: {relative}")
                add_bytes(relative, data)
                if path.suffix == ".json":
                    try:
                        saved[relative] = json.loads(data)
                    except ValueError:
                        report["warnings"].append(f"Invalid JSON preserved: {relative}")
                if required:
                    input_hashes[relative] = digest(data)
                return saved.get(relative)
            except (OSError, ValueError) as exc:
                report["omitted_files"].append({"path": str(path), "reason": str(exc), "required_input": required})
                return None

        # Preserve robot inputs as evidence, never install them over laptop inputs.
        for directory in ("calibration", "configs", "poses"):
            for path in files_under(root / directory):
                add(path, required=True)
        for name in ("competition_offsets.json", "competition_actions.json"):
            if (root / name).is_file():
                add(root / name, required=True)
        for path in sorted((root / "runs").glob("competition_progress*.json")):
            add(path)
        for relative in ("calibration/wrist_part_profiles.json", "configs/competition_actions.json"):
            if relative not in saved:
                report["warnings"].append(f"Missing or invalid required input: {relative}")
        profiles = saved.get("calibration/wrist_part_profiles.json") or {}
        config = saved.get("configs/competition_actions.json") or {}
        profiles, config = mapping(profiles), mapping(config)
        for part, profile in mapping(profiles.get("parts")).items():
            if not isinstance(profile, dict):
                continue
            cv = mapping(profile.get("place_cv"))
            part_config = mapping(mapping(config.get("parts")).get(part))
            report["profiles"][part] = {
                "grasp_verified": profile.get("grasp_verified"),
                "place_verified": profile.get("place_verified"),
                "place_cv_reference_present": bool(cv), "place_cv_enabled": cv.get("enabled", False),
                "grasp_clearance_m": profile.get("grasp_clearance_m"), "place": profile.get("place"),
                "configured": part_config, "pickup_board": profile.get("pickup_board"),
                "placement_board": profile.get("placement_board")}

        run_dirs = discover_runs(root)
        full_runs = []
        for raw in include_runs:
            path = Path(raw)
            path = path if path.is_absolute() else root / path
            path = path.resolve()
            if not path.is_relative_to(root / "runs") or not path.is_dir():
                raise ValueError(f"Requested run must exist under {root / 'runs'}: {raw}")
            if path not in full_runs:
                full_runs.append(path)
        full_runs += [p for p in run_dirs[:recent_runs] if p not in full_runs]
        # Copy latest metadata for every part, plus recent runs. Images are
        # unmodified originals, with explicit omissions if the budget is reached.
        latest_keys = set()
        for directory in run_dirs:
            try:
                summary = json.loads((directory / "run_summary.json").read_text())
                if not isinstance(summary, dict):
                    summary = {}
            except (OSError, ValueError):
                summary = {}
            part, action = summary.get("part"), summary.get("action")
            part = part if isinstance(part, str) else None
            action = action if isinstance(action, str) else None
            key = (part, action)
            latest = bool(part and key not in latest_keys)
            if latest:
                latest_keys.add(key)
                report["latest_by_part"].setdefault(part, {})[str(action)] = summary
            if not latest and directory not in full_runs:
                continue
            relative = directory.relative_to(root).as_posix()
            report["recent_runs"].append({"path": relative, "full_evidence_requested": directory in full_runs,
                "part": part, "action": action, "status": summary.get("status", "no_final_summary"),
                "last_error": summary.get("last_error"), "holding_may_be_true": summary.get("holding_may_be_true"),
                "board_reference": summary.get("board_reference")})
            for filename in ("run_summary.json", "events.jsonl"):
                if (directory / filename).is_file():
                    add(directory / filename)
        for directory in full_runs:
            for path in files_under(directory):
                add(path)
        # Current profiles can reference images outside the recent-run window.
        for raw in sorted(set(references(profiles))):
            path = Path(raw)
            path = path if path.is_absolute() else root / path
            if not path.resolve().is_relative_to(root):
                report["missing_references"].append(raw)
            elif path.is_file():
                add(path)
            else:
                report["missing_references"].append(raw)

        for name, args in {
            "git_revision": ["git", "rev-parse", "HEAD"],
            "git_status": ["git", "status", "--porcelain=v1"],
            "code_diff": ["git", "diff", "HEAD", "--", "tools", "steadyhand"],
        }.items():
            result = command(root, args)
            report[name] = result.get("stdout", "").strip() if name != "code_diff" else "diagnostics/code_diff.json"
            add_bytes(f"diagnostics/{name}.json", json.dumps(result, indent=2).encode())
        if diagnostics:
            with tempfile.TemporaryDirectory(prefix="vega-status-") as temporary:
                readiness = Path(temporary) / "readiness.json"
                commands = {
                    "preflight": [sys.executable, "tools/vega_preflight.py", "--json"],
                    "competition_check": [sys.executable, "tools/vega_competition_pipeline.py", "--check-only", "--competition-run"],
                    "readiness": [sys.executable, "tools/vega_competition_readiness.py", "--skip-tests", "--output", str(readiness)],
                }
                for name, args in commands.items():
                    result = command(root, args)
                    report["diagnostics"][name] = {"returncode": result["returncode"], "error": result.get("error")}
                    add_bytes(f"diagnostics/{name}.json", json.dumps(result, indent=2).encode())
                if readiness.is_file():
                    add_bytes("readiness.json", readiness.read_bytes())
        else:
            report["warnings"].append("No-motion diagnostic commands were skipped")
        # Flag snapshot races instead of presenting mismatched calibration as coherent.
        for relative, before in input_hashes.items():
            try:
                if digest((root / relative).read_bytes()) != before:
                    report["warnings"].append(f"Input changed during collection: {relative}")
            except OSError:
                report["warnings"].append(f"Input disappeared during collection: {relative}")
        report["full_run_folders"] = [p.relative_to(root).as_posix() for p in full_runs]
        report["included_file_count"] = len(manifest)
        report["included_bytes"] = total
        report["evidence_complete_within_selection"] = not (report["omitted_files"] or report["missing_references"] or report["warnings"])
        add_bytes("report.json", (json.dumps(report, indent=2) + "\n").encode())
        archive.writestr(PREFIX + "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--recent-runs", type=int, default=8)
    parser.add_argument("--max-mb", type=int, default=256)
    parser.add_argument("--include-run", action="append", default=[])
    parser.add_argument("--skip-diagnostics", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.recent_runs <= 100 or not 16 <= args.max_mb <= 2048:
        parser.error("recent-runs must be 1..100; max-mb must be 16..2048")
    report = collect(args.root, args.output, recent_runs=args.recent_runs, max_mb=args.max_mb,
                     include_runs=args.include_run, diagnostics=not args.skip_diagnostics)
    print(f"Robot revision: {report['git_revision']}")
    print(f"Collected {report['included_file_count']} files; {len(report['full_run_folders'])} recent run folders")
    print(f"Warnings: {len(report['warnings'])}; omitted: {len(report['omitted_files'])}; missing references: {len(report['missing_references'])}")
    print("BUNDLE_SHA256=" + digest(Path(args.output).read_bytes()))
    return 0  # failing checks belong in the report and must still be offloaded


if __name__ == "__main__":
    raise SystemExit(main())
