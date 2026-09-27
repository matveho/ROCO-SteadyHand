"""One directory per session, with configuration snapshots and append-only events."""

import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import WORKSPACE


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def create_session(robot, operator, mode, bundle, runs_dir=None):
    root = Path(runs_dir) if runs_dir is not None else WORKSPACE / "runs"
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = root / f"{name}_{robot}_{mode}_{uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    submission = WORKSPACE / "policy.py"
    digest = hashlib.sha256(submission.read_bytes()).hexdigest() if submission.exists() else None
    write_json(folder / "session.json", {
        "schema_version": 1, "created_at_utc": timestamp(), "robot": robot,
        "operator": operator, "mode": mode, "python": platform.python_version(),
        "platform": platform.system(), "submission_sha256": digest,
        "hardware_connected_by_toolkit": False,
    })
    write_json(folder / "config_snapshot.json", bundle)
    with (folder / "trials.csv").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow([
            "trial_id", "part", "grasp_success", "placement_success", "elapsed_s",
            "failure_stage", "video_path", "notes",
        ])
    (folder / "notes.md").write_text(
        "# Session notes\n\n"
        f"Robot: {robot}\n\nMode: {mode}\n\n"
        "## Setup\n\nBoard/part layout, active software revision, and hardware condition:\n\n"
        "## Attempts\n\nRecord physical outcomes in trials.csv; use blank for unknown.\n\n"
        "## Next action\n\n",
        encoding="utf-8",
    )
    return folder


def append_event(folder, **event):
    record = {"timestamp_utc": timestamp(), **event}
    with (Path(folder) / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def mark_hardware_connected(folder):
    path = Path(folder) / "session.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["hardware_connected_by_toolkit"] = True
    value["hardware_connected_at_utc"] = timestamp()
    write_json(path, value)


def append_trial(
    folder,
    *,
    trial_id,
    part,
    grasp_success=None,
    placement_success=None,
    elapsed_s=None,
    failure_stage="",
    video_path="",
    notes="",
):
    def cell(value):
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        return value

    with (Path(folder) / "trials.csv").open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow([
            trial_id,
            part,
            cell(grasp_success),
            cell(placement_success),
            cell(elapsed_s),
            failure_stage,
            video_path,
            notes,
        ])
