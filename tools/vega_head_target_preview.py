"""Preview a head-camera target without gripping or releasing anything.

This is the offsite safety path: it registers the board, chooses one part from
the fresh head image (falling back to the reviewed task coordinate), moves only
to a slow 40 mm hover, and optionally runs the bounded wrist visual servo.  No
gripper API is called by this tool.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.vega import VegaAdapter
from steadyhand.cameras.vega import VegaWristCameras
from steadyhand.config import load_bundle
from steadyhand.executor import move_tcp_segmented
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills
from steadyhand.vega_presets import configured_right_preset
from steadyhand.wrist_part_profiles import PART_NAMES, load_profiles
from steadyhand.vision.wrist_servo import TemplateTracker, run_xy_servo
from tools.vega_competition_pipeline import (
    ROOT, _capture_downward_head_frame, _load_runtime, _runtime_from_board_scene,
    _task_targets,
)
from tools.vega_wrist_part_calibrate import WristAOnlyCapture, _load_template, _yaw_pose


def _pause(label, remote_safe):
    if not remote_safe:
        return
    answer = input(f"PREVIEW CHECKPOINT {label}: press Enter to continue, abort to stop: ").strip().lower()
    if answer in ("abort", "stop", "q", "quit"):
        raise KeyboardInterrupt()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", required=True, choices=PART_NAMES)
    parser.add_argument("--center", action="store_true", help="also run bounded wrist centering when a profile has a template")
    parser.add_argument("--hover-clearance-mm", type=float, default=40.0)
    parser.add_argument("--remote-safe", action="store_true")
    parser.add_argument("--confirm-head-motion", action="store_true")
    parser.add_argument("--confirm-physical-motion", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_head_motion or not args.confirm_physical_motion:
        parser.error("requires --confirm-head-motion and --confirm-physical-motion")
    if not 20.0 <= args.hover_clearance_mm <= 100.0:
        parser.error("--hover-clearance-mm must be 20..100")

    bundle, task_data, calibrated, ready_pose = _load_runtime()
    cfg = bundle["robot"]
    cfg["allow_robot_init_head_motion"] = True
    cfg["auto_clear_software_estop_on_connect"] = True
    cfg.setdefault("motion", {})["joint_reached_tolerance_rad"] = max(
        float(cfg["motion"].get("joint_reached_tolerance_rad", .005)), .020
    )
    profiles = load_profiles(ROOT / "calibration" / "wrist_part_profiles.json", cfg)
    output = ROOT / "runs" / "head_target_previews" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    robot = VegaAdapter(cfg)
    cameras = None
    capture = None
    try:
        robot.connect()
        floor = float(load_vega_skills()["safety"]["min_tcp_z_m"])
        _pause("before board image", args.remote_safe)
        scene = _capture_downward_head_frame(robot, floor_m=floor, bundle=bundle)
        runtime = _runtime_from_board_scene((bundle, task_data, calibrated, ready_pose), scene)
        targets = _task_targets(runtime, task_data, .100)
        observed_xy = None
        try:
            from tools.vega_head_fallback import match_expected_parts
            observations = match_expected_parts(scene, runtime, targets, task_data=task_data)
            value = observations.get(args.part, {}).get("selected_xy_m")
            if isinstance(value, list) and len(value) == 2:
                observed_xy = (float(value[0]), float(value[1]))
        except Exception as exc:
            print(f"HEAD PART ASSOCIATION FALLBACK: {exc}", flush=True)
        target = targets[f"task.{args.part}.pick"]
        if observed_xy is not None:
            x, y = observed_xy
            z = runtime[2][3].get("coefficients")
            from tools.vega_task_coordinate_reachability import calibrated_surface_z
            target = Pose((x, y, calibrated_surface_z(x, y, runtime[2][3]) + .100), target.quaternion_wxyz)
        hover = Pose((target.position_m[0], target.position_m[1], target.position_m[2] - .060), target.quaternion_wxyz)
        _pause("head target before move", args.remote_safe)
        ready_q, _ = configured_right_preset(cfg, "right_ready")
        robot.move_joints(ready_q, speed_scale=.16 if args.remote_safe else .28)
        move_tcp_segmented(robot, hover, speed_scale=.16 if args.remote_safe else .28,
                           max_translation_step_m=.008 if args.remote_safe else .020,
                           max_orientation_step_rad=.08, min_tcp_z_m=None)
        measured = robot.get_tcp_pose()
        result = {
            "part": args.part,
            "source": "head_detection" if observed_xy is not None else "reviewed_task_coordinate",
            "target_hover_m": list(hover.position_m),
            "measured_tcp_m": list(measured.position_m),
            "scene": scene,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (output / "preview.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        print("HEAD TARGET PREVIEW REACHED (NO GRIPPER MOTION)", json.dumps(result, indent=2, default=str), flush=True)
        _pause("coarse hover", args.remote_safe)
        if args.center:
            profile = (profiles.get("parts") or {}).get(args.part)
            if not profile or not profile.get("template") or profile.get("goal_uv") is None:
                print("WRIST CENTER SKIPPED: no saved wrist profile/template for this part", flush=True)
            else:
                cameras = VegaWristCameras()
                cameras.connect()
                capture = WristAOnlyCapture(cameras, output, settle_s=.20, warmup_attempts=8)
                rgb = capture()
                template = _load_template(profile)
                goal = tuple(float(v) for v in profile["goal_uv"])
                result["wrist_center"] = run_xy_servo(
                    robot, capture, floor_m=floor, goal_uv=goal,
                    max_radius_m=.030, probe_m=.006, gain=.30,
                    max_step_m=.006, tolerance_px=14., max_iterations=6,
                    speed_scale=.45,
                    tracker_factory=lambda image, _uv: TemplateTracker.from_saved_template(
                        image, template, template_uv=profile["template"].get("template_uv"),
                        initial_uv=profile.get("feature_uv"), search_radius=110,
                    ),
                )
                (output / "preview.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
                print("WRIST CENTER PREVIEW COMPLETE (NO GRIPPER MOTION)", json.dumps(result, indent=2, default=str), flush=True)
        _pause("finished preview", args.remote_safe)
        return 0
    finally:
        if cameras is not None:
            cameras.close()
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
