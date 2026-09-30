"""Atomic competition checkpoints. No robot connection or movement here."""
import json
import os
import time
from pathlib import Path


class CompetitionProgress:
    def __init__(self, path, mode, actions, *, recovered_empty=False, new_run=False):
        self.path = Path(path)
        keys = [f"{part}.{action}" for part, action in actions]
        if len(keys) != len(set(keys)):
            raise ValueError("Competition progress requires unique part/actions")
        old = json.loads(self.path.read_text()) if self.path.exists() else None
        if old and old.get("status") != "finished":
            self.data = old
            # A process may have stopped after its child saved a terminal
            # summary but before the outer loop recorded the result.
            for entry in old["actions"].values():
                if entry["status"] == "running":
                    self._recover_artifact(entry)
            if all(e["status"] in ("completed", "skipped") for e in old["actions"].values()):
                old["status"] = "finished"
            blocked = any(e["status"] in ("running", "blocked") for e in old["actions"].values())
            if blocked and not recovered_empty:
                self.save()
                raise ValueError("Previous run stopped with unknown motion/holding state. "
                    "After inspecting the robot, recovering any held part, and verifying the gripper is empty, "
                    "rerun with --resume-after-inspection. Completed actions remain saved.")
            if not new_run:
                if old["mode"] != mode or list(old["actions"]) != keys:
                    raise ValueError("Unfinished competition plan differs. Resume the original menu/config, "
                                     "or use --new-competition-run for a deliberately new plan.")
                for entry in old["actions"].values():
                    if entry["status"] in ("running", "blocked"):
                        entry["status"] = "pending"
                        entry["operator_recovered_empty_at_ns"] = time.time_ns()
                self.save()
                print(f"RESUMING COMPETITION: {self.path}", flush=True)
                return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if old:
            archive = self.path.with_name(f"competition_progress_{time.time_ns()}.json")
            archive.write_text(json.dumps(old, indent=2) + "\n")
        self.data = {"schema_version": 1, "mode": mode, "status": "active",
                     "actions": {key: {"status": "pending", "attempts": 0} for key in keys}}
        self.save()

    @staticmethod
    def _recover_artifact(entry):
        try:
            summary = json.loads((Path(entry["output"]) / "run_summary.json").read_text())
        except (OSError, ValueError, KeyError):
            return
        entry["summary_status"] = summary.get("status")
        if summary.get("holding_may_be_true") is not False:
            return
        if summary.get("status") in ("completed", "pick_complete_returned", "pick_complete_place_blocked_returned"):
            entry["status"] = "completed"
        elif summary.get("automatic_continuation_safe") is True:
            entry["status"] = "pending"

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at_ns"] = time.time_ns()
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            stream.write(json.dumps(self.data, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)

    def entry(self, part, action):
        return self.data["actions"][f"{part}.{action}"]

    def begin_attempt(self, part, action, output):
        entry = self.entry(part, action)
        entry.update(status="running", output=str(output), attempts=entry["attempts"] + 1)
        self.save()  # durable before the child can issue any motion

    def finish(self, part, action, result, summary=None):
        entry = self.entry(part, action)
        entry["status"] = {0: "completed", -1: "skipped"}.get(result, "blocked")
        entry["summary_status"] = (summary or {}).get("status")
        entry["last_error"] = (summary or {}).get("last_error")
        if all(e["status"] in ("completed", "skipped") for e in self.data["actions"].values()):
            self.data["status"] = "finished"
        self.save()

    def retry_pending(self, part, action, summary):
        entry = self.entry(part, action)
        entry.update(status="pending", summary_status=summary.get("status"), last_error=summary.get("last_error"))
        self.save()
