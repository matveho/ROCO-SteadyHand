"""Supervised head target + optional wrist XY preview. Never commands a jaw.

At the default 40 mm clearance, a fresh feature/goal annotation is used for the
wrist preview: 100 mm pickup pixels cannot be transferred to a different height.
No annotations from this preview overwrite a pickup or placement profile.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
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
from steadyhand.remote_motion import needs_low_clearance_confirmation
from steadyhand.board_geometry import BOARD_SIZE_M, board_relative_task_xy


def preview_target(runtime, part, observation, clearance_m):
    if observation.get("selection") not in ("head_detection", "operator_head_pixel"):
        raise ValueError("A fresh head detection or reviewed head-image pixel is required")
    x, y = observation["selected_xy_m"]
    if not all(math.isfinite(float(v)) for v in (x, y, clearance_m)):
        raise ValueError("Head target coordinates must be finite")
    return Pose((x, y, calibrated_surface_z(x, y, runtime[2][3]) + clearance_m),
                runtime[3].quaternion_wxyz)


def head_pixel_target(scene, runtime, task_data, expected, part, pixel):
    """A reviewed pixel uses the 386 mm board and calibrated axes, not raw FK."""
    import cv2
    import numpy as np

    corners = np.asarray([scene["board"]["corners_px"][k] for k in ("tl", "tr", "br", "bl")],
                         dtype=np.float32)
    uv = np.asarray(pixel, dtype=float)
    if uv.shape != (2,) or not np.all(np.isfinite(uv)):
        raise ValueError("Pixel must contain two finite numbers")
    if cv2.pointPolygonTest(corners, tuple(float(v) for v in uv), False) < 0:
        raise ValueError("Selected pixel is outside the detected board outline")
    # Same homography convention as scene perception, without re-detecting.
    H = cv2.getPerspectiveTransform(corners, np.asarray(
        ((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32))
    fraction = cv2.perspectiveTransform(uv.astype(np.float32).reshape(1, 1, 2), H)[0, 0]
    observed = (fraction - .5) * BOARD_SIZE_M
    reference = board_relative_task_xy(
        task_data["parts"][part]["pick"][:2], task_data["source_board_center_xy_m"],
        rotation_deg=task_data.get("task_coordinate_rotation_deg", 0.),
        mirror_x=task_data.get("task_coordinate_mirror_x", False),
        mirror_y=task_data.get("task_coordinate_mirror_y", False),
    )
    delta = observed - reference
    if float(np.linalg.norm(delta)) > .100:
        raise ValueError("Selected point is over 100 mm from this part's task position; inspect board outline/orientation")
    _, ux, uy, _ = runtime[2]
    xy = [float(expected.position_m[i] + delta[0] * ux[i] + delta[1] * uy[i]) for i in range(2)]
    return {"selection": "operator_head_pixel", "selected_xy_m": xy,
            "head_pixel_uv": list(float(v) for v in uv),
            "observed_board_xy_m": observed.tolist(), "operator_reviewed_board_outline": True}


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

    def surface(x, y):
        return calibrated_surface_z(x, y, runtime[2][3])

    def checkpoint(label, target=None):
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
        if label == "before_head_down" or not needs_low_clearance_confirmation(pose, target, surface):
            event("decision", {"label": label, "decision": "auto_continue"})
            return
        if target is not None:
            print("NEXT ARM TARGET =", target.position_m, flush=True)
        while True:
            answer = input(f"PREVIEW {label}: Enter to continue / abort: ").strip().lower()
            if not answer:
                event("decision", {"label": label, "decision": "continue"})
                return
            if answer in ("abort", "stop", "q", "exit"):
                event("decision", {"label": label, "decision": "abort"})
                raise KeyboardInterrupt()

    def waypoint_guard(target):
        if needs_low_clearance_confirmation(robot.get_tcp_pose(), target, surface):
            checkpoint("before_low_waypoint", target)

    def move(target):
        move_tcp_segmented(robot, target, speed_scale=.16, max_translation_step_m=.008,
                           max_orientation_step_rad=.08,
                           waypoint_guard=waypoint_guard,
                           after_waypoint=(lambda: checkpoint("waypoint")) if args.remote_safe else None)

    persist()
    try:
        robot.connect()
        cameras.connect()
        floor = float(load_vega_skills()["safety"]["min_tcp_z_m"])
        scene = _capture_downward_head_frame(robot, floor_m=floor, bundle=bundle,
            speed_scale=.16, checkpoint=checkpoint, output=output)
        updated = _runtime_from_board_scene(runtime, scene)
        fresh_registration = updated is not runtime
        runtime = updated
        result["fresh_board_registration_accepted"] = fresh_registration
        # Preserve the same validated physical axes as teaching/competition.
        physical_task_data = dict(task_data, task_coordinate_mirror_y=False)
        targets = _task_targets(runtime, physical_task_data, .100)
        from tools.vega_head_fallback import match_expected_parts
        observation = match_expected_parts(scene, runtime, targets, task_data=physical_task_data)[args.part]
        result.update({"scene": scene, "head_observation": observation})
        persist()
        # Keep the capture and overlay accessible through the existing Windows
        # image-transfer helper.  No desktop/display is needed on the Jetson.
        import cv2
        from steadyhand.vision.scene import render_scene_overlay
        rgb = cv2.cvtColor(cv2.imread(scene["head_image_path"]), cv2.COLOR_BGR2RGB)
        overlay = render_scene_overlay(rgb, scene)
        overlay_path = output / "HEAD_TARGET_REVIEW.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        live = ROOT / "runs" / "wrist_live"
        live.mkdir(parents=True, exist_ok=True)
        temporary = live / ".head_preview.png.tmp"
        shutil.copyfile(overlay_path, temporary)
        temporary.replace(live / "latest_wrist_a.png")
        (live / "latest_wrist_a.json").write_text(json.dumps({
            "part": args.part, "stage": "HEAD_TARGET_REVIEW", "camera": "head_camera",
            "source_run": str(overlay_path), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }) + "\n")
        print("HEAD ASSOCIATION:", json.dumps(observation), flush=True)
        print("HEAD TARGET REVIEW IMAGE:", overlay_path, flush=True)
        if not fresh_registration or observation.get("selection") != "head_detection":
            print("Target review needed: " + ("board geometry changed" if not fresh_registration else
                  str(observation.get("rejection_reason"))) + ". No target approach has started.", flush=True)
            print("Use the laptop image helper. Green outline must match the board. "
                  "Select the part's center: detection NUMBER or pixel U V. "
                  "Type abort if the outline is wrong; no recalibration files will be changed.", flush=True)
            while True:
                answer = input("HEAD TARGET> ").strip().lower()
                if answer in ("abort", "q", "exit", "stop"):
                    raise KeyboardInterrupt()
                try:
                    values = answer.split()
                    if len(values) == 1:
                        chosen = next((item for item in scene["parts"] if item["index"] == int(values[0])), None)
                        if chosen is None:
                            raise ValueError("No detection with that number")
                        pixel = chosen["center_image_px"]
                    elif len(values) == 2:
                        pixel = [float(v) for v in values]
                    else:
                        raise ValueError("Enter a detection number or two pixel coordinates")
                    observation = head_pixel_target(scene, runtime, physical_task_data,
                        targets[f"task.{args.part}.pick"], args.part, pixel)
                    break
                except (ValueError, TypeError) as exc:
                    print("TARGET NOT ACCEPTED:", exc, flush=True)
            result["head_observation"] = observation
            persist()
        hover = preview_target(runtime, args.part, observation, args.hover_clearance_mm / 1000.)
        result["target_hover_m"] = list(hover.position_m)
        persist()
        print("HEAD TARGET", json.dumps({"part": args.part, "source": observation["selection"],
              "xyz_m": list(hover.position_m)}), flush=True)
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
                waypoint_guard=waypoint_guard,
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
