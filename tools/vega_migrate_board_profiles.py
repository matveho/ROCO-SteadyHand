"""Add board references without deleting any legacy calibration fields.

Default is a read-only report. --apply takes a full backup first. Historical
teaching scenes are preferred; unavailable evidence is explicitly marked.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.board_calibration import load_board_calibration
from steadyhand.board_relative import (legacy_pickup_record, legacy_placement_record,
    record_target, reference_from_calibration)
from steadyhand.config import load_bundle
from steadyhand.models import Pose
from steadyhand.vega_presets import configured_right_preset
from steadyhand.wrist_part_profiles import load_profiles, validate_profiles

ROOT = Path(__file__).resolve().parents[1]


def teaching_evidence(root, part, profile, reference, cfg):
    """Use only the handoff that saved this exact template, not a later trial."""
    matches = []
    for path in (root / "runs").rglob(f"{part}_handoff.json"):
        try:
            handoff = json.loads(path.read_text())
            if (handoff.get("template", {}).get("sha256") == profile["template"]["sha256"]
                and handoff.get("created_at_utc") == profile.get("created_at_utc")):
                matches.append(path)
        except (OSError, ValueError, KeyError):
            continue
    if len(matches) != 1:
        return None, None, "matching teaching handoff absent or ambiguous"
    path = matches[0]
    scenes = sorted(path.parent.glob("board_[0-9]*.json"))
    if len(scenes) > 1:
        # Filename clock is time.time_ns(), the same clock used by the handoff.
        try:
            cutoff = int(datetime.fromisoformat(profile["created_at_utc"]).timestamp() * 1e9)
            scenes = [p for p in scenes if int(p.stem.split("_")[1]) <= cutoff]
            scenes = scenes[-1:]
        except (KeyError, ValueError):
            scenes = []
    if len(scenes) != 1:
        return None, None, "teaching board scene absent or ambiguous"
    scene = json.loads(scenes[0].read_text())
    if scene.get("registered_board_reference"):
        taught_ref = scene["registered_board_reference"]
    else:
        # Re-express the old observation using the SAME differential model
        # as today's runtime. Reusing the old biased camera-center correction
        # here would introduce a jump even when the board has not moved.
        from steadyhand.board_relative import register
        cal_board = cfg.get("_reference_camera_board") or {}
        taught_ref = register(reference, cal_board, scene["board"])
        taught_ref["registration"]["teaching_evidence"] = str(scenes[0])
    events_path = path.parent / "events.jsonl"
    events = []
    if events_path.exists():
        for line in events_path.read_text().splitlines():
            try:
                event = json.loads(line)
                if event.get("part") == part:
                    events.append(event)
            except ValueError:
                continue
    successful_hover = None
    anchor = None
    for event in events:
        if event.get("event") == "grasp_approach_readback":
            anchor = event.get("anchor_xy_m")
        if event.get("event") == "grip_result" and (event.get("result") or {}).get("gripped") is True:
            successful_hover = anchor
        if event.get("event") == "successful_pickup_pose":
            successful_hover = event.get("hover_tcp", [])[:2]
    return taught_ref, successful_hover, str(path)


def placement_reference(root, part, profile, reference, camera_board):
    import math
    from steadyhand.board_relative import register
    found = []
    expected = profile.get("place_release_tcp_m")
    for events in (root / "runs").rglob("events.jsonl"):
        try:
            matching = False
            for line in events.read_text().splitlines():
                event = json.loads(line)
                if event.get("part") == part and event.get("event") == "drop_release":
                    if math.dist(event.get("release_tcp", []), expected) < 1e-7:
                        matching = True
            if not matching:
                continue
            scenes = sorted(events.parent.glob("board_[0-9]*.json"))
            if len(scenes) != 1:
                continue
            scene = json.loads(scenes[0].read_text())
            frame = scene.get("registered_board_reference") or register(reference, camera_board, scene["board"])
            found.append((frame, str(events)))
        except (OSError, KeyError, ValueError, TypeError):
            continue
    return found[0] if len(found) == 1 else (None, "placement scene absent or ambiguous")


def migrate(root, profiles, calibration, ready, cfg):
    result = deepcopy(profiles)
    reference = reference_from_calibration(calibration)
    report = []
    for part, profile in result["parts"].items():
        item = {"part": part, "pickup": "already board-relative", "placement": "unchanged"}
        if "pickup_board" not in profile:
            matching_hash = profile.get("calibration_sha256") == calibration["sha256"]
            taught_ref, grasp_xy, evidence = (None, None, "board hash differs; reference assumed")
            if matching_hash:
                evidence_cfg = dict(cfg, _reference_camera_board=calibration["raw"].get("camera_board_read"))
                try:
                    taught_ref, grasp_xy, evidence = teaching_evidence(root, part, profile, reference, evidence_cfg)
                except (KeyError, TypeError, ValueError, OSError) as exc:
                    evidence = f"teaching evidence rejected: {exc}"
            record = legacy_pickup_record(profile, taught_ref or reference, ready.quaternion_wxyz)
            if taught_ref:
                record["migration"] = {"confidence": "reconstructed_teaching_frame",
                    "source": evidence, "warning": "Teaching observation reconstructed; verify hover"}
            if grasp_xy and len(grasp_xy) == 2:
                quat = record["grasp"]["tcp_pose"]["quaternion_wxyz"]
                pose = Pose((*grasp_xy, 0.), tuple(quat))
                record["grasp"] = record_target(taught_ref, pose, source="successful_grip_log_hover_anchor")
                record["grasp"]["tcp_pose"]["position_m"][2] = None
                record["grasp"]["z_not_recorded"] = True
            profile["pickup_board"] = record
            item.update(pickup=record["migration"]["confidence"], evidence=evidence,
                        successful_grasp_recovered=bool(grasp_xy))
        # Placement has its own teaching observation, never the pickup's.
        if profile.get("place") and "placement_board" not in profile:
            position = profile.get("place_release_tcp_m")
            if position:
                taught_ref, evidence = placement_reference(root, part, profile, reference,
                    calibration["raw"].get("camera_board_read"))
                record = legacy_placement_record(profile, taught_ref or reference, ready.quaternion_wxyz)
                if taught_ref:
                    record["migration"] = {"confidence": "reconstructed_placement_frame", "source": evidence}
                profile["placement_board"] = record
                item["placement"] = record["migration"]["confidence"]
                item["placement_evidence"] = evidence
            else:
                item["placement"] = "legacy board offsets retained; no measured release saved"
        report.append(item)
    return result, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    cfg = load_bundle("vega")["robot"]
    path = root / "calibration/wrist_part_profiles.json"
    profiles = load_profiles(path, cfg)
    calibration = load_board_calibration(root / "calibration/vega_board_manual.json", cfg,
        fallback_path=root / "calibration/vega_board_manual_fallback.json")
    _, ready = configured_right_preset(cfg, "right_ready")
    result, report = migrate(root, profiles, calibration, ready, cfg)
    validate_profiles(result, cfg)
    print(json.dumps(report, indent=2))
    if args.apply and result != profiles:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        backup = root / "runs/board_profile_migrations" / stamp
        backup.mkdir(parents=True)
        shutil.copytree(root / "calibration", backup / "calibration")
        (backup / "migration_report.json").write_text(json.dumps(report, indent=2) + "\n")
        pending = path.with_suffix(".json.tmp")
        pending.write_text(json.dumps(result, indent=2) + "\n")
        pending.replace(path)
        print(f"SAVED: {path}\nFULL CALIBRATION BACKUP: {backup}")
    else:
        print("NO FILES CHANGED" + ("; use --apply after reviewing the report" if not args.apply else ""))
    print("Verify projected pickup/place hovers in menu 2 before descending. No depth/template recalibration needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
