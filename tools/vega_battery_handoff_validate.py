"""Validate saved battery_size1 stage artifacts before a physical handoff.

This is a read-only provenance gate.  It does not connect to the robot or
claim that a saved pose is still current; operators must still run each
physical stage's board-unchanged and live-pose checks onsite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.battery_size1 import load_alignment_result, load_calibration, require_right_battery_config
from steadyhand.battery_size1_hover import load_source_localization
from steadyhand.battery_size1_source import load_manual_board_calibration
from steadyhand.config import load_bundle
from steadyhand.skill_config import load_vega_skills

ROOT = Path(__file__).resolve().parents[1]


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def validate_handoff(*, manual_path, source_path=None, coarse_path=None,
                     alignment_path=None, grasp_path=None, pick_path=None):
    cfg = load_bundle("vega")["robot"]
    floor = float((load_vega_skills().get("safety") or {}).get("min_tcp_z_m", 0.456))
    reasons = []
    stages = {}
    try:
        manual = load_manual_board_calibration(manual_path, cfg)
        stages["board_calibration"] = {"status": "pass", "sha256": manual["sha256"], "permanent_fallback": manual["is_permanent_fallback"]}
    except Exception as exc:
        manual = None
        stages["board_calibration"] = {"status": "blocked", "reason": str(exc)}
        reasons.append("board calibration: " + str(exc))

    localization = None
    if source_path is None:
        reasons.append("missing source localization artifact")
        stages["source_localization"] = {"status": "blocked", "reason": "artifact not supplied"}
    elif manual is None:
        stages["source_localization"] = {"status": "blocked", "reason": "current board calibration is invalid"}
        reasons.append("source localization cannot be checked without a valid board calibration")
    else:
        try:
            localization = load_source_localization(source_path, cfg, manual)
            stages["source_localization"] = {"status": "pass", "sha256": localization["sha256"], "coarse_base_xy_m": list(localization["coarse_base_xy_m"])}
        except Exception as exc:
            stages["source_localization"] = {"status": "blocked", "reason": str(exc)}
            reasons.append("source localization: " + str(exc))

    coarse = _artifact_result(coarse_path, "coarse_hover", reasons, stages)
    if coarse is not None and coarse.get("status") != "reached":
        reasons.append(f"coarse hover status is {coarse.get('status')!r}")
    if coarse is not None and localization is not None:
        expected = localization["sha256"]
        actual = ((coarse.get("localization") or {}).get("sha256"))
        if actual != expected:
            reasons.append("coarse hover was produced from a different source localization")

    calibration = None
    if grasp_path is None:
        reasons.append("missing completed grasp calibration artifact")
        stages["grasp_calibration"] = {"status": "blocked", "reason": "artifact not supplied"}
    else:
        try:
            require_right_battery_config(cfg)
            calibration = load_calibration(grasp_path, cfg, floor_m=floor)
            stages["grasp_calibration"] = {"status": "pass", "path": str(grasp_path)}
        except Exception as exc:
            stages["grasp_calibration"] = {"status": "blocked", "reason": str(exc)}
            reasons.append("grasp calibration: " + str(exc))

    if alignment_path is None:
        reasons.append("missing converged wrist alignment artifact")
        stages["wrist_alignment"] = {"status": "blocked", "reason": "artifact not supplied"}
    elif calibration is None:
        stages["wrist_alignment"] = {"status": "blocked", "reason": "grasp calibration is invalid"}
        reasons.append("wrist alignment cannot be checked without grasp calibration")
    else:
        try:
            alignment, _tcp = load_alignment_result(alignment_path, grasp_path, calibration)
            stages["wrist_alignment"] = {"status": "pass", "status_value": alignment.get("status")}
        except Exception as exc:
            stages["wrist_alignment"] = {"status": "blocked", "reason": str(exc)}
            reasons.append("wrist alignment: " + str(exc))

    if pick_path is None:
        stages["pick"] = {"status": "not_yet_run"}
    else:
        try:
            pick = _json(pick_path)
            status = pick.get("status")
            stages["pick"] = {"status": "pass" if status == "lift_retention_verified" else "blocked", "status_value": status}
            if status != "lift_retention_verified":
                reasons.append(f"pick artifact status is {status!r}")
        except Exception as exc:
            stages["pick"] = {"status": "blocked", "reason": str(exc)}
            reasons.append("pick artifact: " + str(exc))

    return {
        "schema_version": 1,
        "report_kind": "vega_battery_handoff_validation",
        "part": "battery_size1",
        "status": "READY FOR NEXT STAGE" if not reasons else "BLOCKED",
        "reasons": reasons,
        "stages": stages,
    }


def _artifact_result(path, name, reasons, stages):
    if path is None:
        stages[name] = {"status": "blocked", "reason": "artifact not supplied"}
        reasons.append(f"missing {name} artifact")
        return None
    try:
        value = _json(path)
        stages[name] = {"status": "pass", "path": str(path), "result_status": value.get("status")}
        return value
    except Exception as exc:
        stages[name] = {"status": "blocked", "reason": str(exc)}
        reasons.append(f"{name}: {exc}")
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-calibration", default="calibration/vega_board_manual.json")
    parser.add_argument("--source-localization")
    parser.add_argument("--coarse-hover", help="coarse-hover result.json")
    parser.add_argument("--wrist-alignment", help="wrist-center result.json")
    parser.add_argument("--grasp-calibration", help="completed battery_size1_grasp.json")
    parser.add_argument("--pick-result", help="battery pick result.json")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    def resolve(value):
        if value is None: return None
        path = Path(value)
        return path if path.is_absolute() else ROOT / path
    report = validate_handoff(
        manual_path=resolve(args.manual_calibration),
        source_path=resolve(args.source_localization),
        coarse_path=resolve(args.coarse_hover),
        alignment_path=resolve(args.wrist_alignment),
        grasp_path=resolve(args.grasp_calibration),
        pick_path=resolve(args.pick_result),
    )
    print(json.dumps(report, indent=2))
    if args.output:
        output = resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "READY FOR NEXT STAGE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
