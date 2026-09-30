"""Supervised head target + optional wrist XY preview. Never commands a jaw.

At the default 40 mm clearance, a fresh feature/goal annotation is used for the
wrist preview: 100 mm pickup pixels cannot be transferred to a different height.
No annotations from this preview overwrite a pickup or placement profile.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.vega import VegaAdapter
from steadyhand.cameras.vega import VegaWristCameras
from steadyhand.executor import move_tcp_segmented
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills
from steadyhand.vega_presets import configured_right_preset
from steadyhand.vision.wrist_servo import TemplateTracker, run_xy_servo
from steadyhand.vision.wrist_review import select_pixel
from tools.vega_competition_pipeline import (
    ROOT, _capture_downward_head_frame, _load_runtime, _runtime_from_board_scene, _task_targets,
)
from tools.vega_task_coordinate_reachability import calibrated_surface_z
from tools.vega_wrist_fine_center import WristAOnlyCapture
from steadyhand.wrist_part_profiles import PART_NAMES


def preview_target(runtime, part, observation, clearance_m):
    if observation.get("selection") != "head_detection":
        raise ValueError("No unique nearby head-camera detection; no target motion authorized")
    x, y = observation["selected_xy_m"]
    return Pose((x, y, calibrated_surface_z(x, y, runtime[2][3]) + clearance_m),
                runtime[3].quaternion_wxyz)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", required=True, choices=PART_NAMES)
    parser.add_argument("--center", action="store_true", help="annotate and test wrist centering at this preview height")
    parser.add_argument("--feature", type=float, nargs=2, metavar=("U", "V"))
    parser.add_argument("--goal-pixel", type=float, nargs=2, metavar=("U", "V"))
    parser.add_argument("--hover-clearance-mm", type=float, default=40.0)
    parser.add_argument("--remote-safe", action="store_true")
    parser.add_argument("--confirm-head-motion", action="store_true")
    parser.add_argument("--confirm-physical-motion", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if not args.confirm_head_motion or not args.confirm_physical_motion:
        parser.error("requires --confirm-head-motion and --confirm-physical-motion")
    if not 20.0 <= args.hover_clearance_mm <= 100.0:
        parser.error("--hover-clearance-mm must be 20..100")
    runtime = _load_runtime()
    bundle, task_data, _, _ = runtime
    cfg = bundle["robot"]
    cfg["allow_robot_init_head_motion"] = True
    cfg["auto_clear_software_estop_on_connect"] = True
    cfg.setdefault("motion", {})["joint_reached_tolerance_rad"] = max(
        float(cfg["motion"].get("joint_reached_tolerance_rad", .005)), .020)
    output = Path(args.output) if args.output else ROOT / "runs" / "head_target_previews" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    robot = VegaAdapter(cfg)
    cameras = VegaWristCameras()
    capture = WristAOnlyCapture(cameras, output, settle_s=.20, warmup_attempts=8)
    result = {"part": args.part, "mode": "preview_no_gripper", "status": "starting",
              "created_at_utc": datetime.now(timezone.utc).isoformat()}

    def persist():
        (output / "preview.json").write_text(json.dumps(result, indent=2, default=str) + "\n")

    def event(kind, fields):
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": kind, **fields}, default=str) + "\n")

    def image(label):
        rgb = capture()
        raw = output / f"{capture.index - 1:03d}_wrist_a.png"
        live = ROOT / "runs" / "wrist_live"
        live.mkdir(parents=True, exist_ok=True)
        temporary = live / ".preview.png.tmp"
        shutil.copyfile(raw, temporary)
        temporary.replace(live / "latest_wrist_a.png")
        (live / "latest_wrist_a.json").write_text(json.dumps({
            "part": args.part, "stage": label, "source_run": str(raw),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }) + "\n")
        print(f"{label.upper()} IMAGE: {raw}", flush=True)
        return rgb, raw

    def checkpoint(label):
        pose = robot.get_tcp_pose()
        record = {"label": label, "tcp": pose.position_m, "quaternion_wxyz": pose.quaternion_wxyz,
                  "joints": [float(v) for v in robot._read_joint_positions()],
                  "timestamp": datetime.now(timezone.utc).isoformat()}
        try:
            _, raw = image(label)
            record["wrist_image"] = str(raw)
        except Exception as exc:
            record["image_error"] = str(exc)
            print(f"CHECKPOINT IMAGE UNAVAILABLE: {exc}", flush=True)
        event("checkpoint", record)
        while True:
            answer = input(f"PREVIEW {label}: Enter to continue / abort: ").strip().lower()
            if not answer:
                event("decision", {"label": label, "decision": "continue"})
                return
            if answer in ("abort", "stop", "q", "exit"):
                event("decision", {"label": label, "decision": "abort"})
                raise KeyboardInterrupt()

    def move(target):
        move_tcp_segmented(robot, target, speed_scale=.16, max_translation_step_m=.008,
                           max_orientation_step_rad=.08,
                           after_waypoint=(lambda: checkpoint("waypoint")) if args.remote_safe else None)

    persist()
    try:
        robot.connect()
        cameras.connect()
        floor = float(load_vega_skills()["safety"]["min_tcp_z_m"])
        scene = _capture_downward_head_frame(robot, floor_m=floor, bundle=bundle,
            speed_scale=.16, checkpoint=checkpoint, output=output)
        runtime = _runtime_from_board_scene(runtime, scene)
        # Preserve the same validated physical axes as teaching/competition.
        physical_task_data = dict(task_data, task_coordinate_mirror_y=False)
        targets = _task_targets(runtime, physical_task_data, .100)
        from tools.vega_head_fallback import match_expected_parts
        observation = match_expected_parts(scene, runtime, targets)[args.part]
        result.update({"scene": scene, "head_observation": observation})
        persist()
        hover = preview_target(runtime, args.part, observation, args.hover_clearance_mm / 1000.)
        result["target_hover_m"] = list(hover.position_m)
        persist()
        print("HEAD TARGET", json.dumps(result, default=str), flush=True)
        checkpoint("before_ready")
        ready_q, _ = configured_right_preset(cfg, "right_ready")
        robot.move_joints(ready_q, speed_scale=.16)
        checkpoint("before_coarse_hover")
        high = preview_target(runtime, args.part, observation, .100)
        move(high)
        checkpoint("before_preview_height")
        move(hover)
        checkpoint("coarse_hover")
        if args.center:
            rgb, raw = image("wrist preview reference")
            feature = args.feature or select_pixel(rgb, raw, "Select a distinctive feature on the selected part", use_viewer=False)
            goal = args.goal_pixel or select_pixel(rgb, raw, "Select desired feature pixel at THIS preview height (not a saved 100 mm pixel)", use_viewer=False)
            result["preview_annotation"] = {"feature_uv": feature, "goal_uv": goal, "image": str(raw)}
            persist()
            checkpoint("before_wrist_centering")
            result["wrist_center"] = run_xy_servo(
                robot, lambda: image("servo")[0], floor_m=floor, feature_uv=feature, goal_uv=goal,
                max_radius_m=.060, probe_m=.008, gain=.45, max_step_m=.008,
                tolerance_px=8., max_iterations=12, speed_scale=.16,
                checkpoint=checkpoint, event=event,
                surface_z=lambda x, y: calibrated_surface_z(x, y, runtime[2][3]),
                reference_quaternion_wxyz=hover.quaternion_wxyz,
            )
            checkpoint("after_wrist_centering")
        result["status"] = "completed_no_gripper_motion"
        return 0
    except (KeyboardInterrupt, EOFError):
        result["status"] = "aborted"
        return 1
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(f"PREVIEW STOPPED: {exc}; no grip or release commanded", flush=True)
        return 2
    finally:
        try:
            result["measured_tcp_m"] = list(robot.get_tcp_pose().position_m)
        except Exception as exc:
            result["state_error"] = str(exc)
        persist()
        try:
            cameras.close()
        finally:
            robot.close()
        print(f"PREVIEW ARTIFACTS: {output}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
