"""Supervised head target + optional wrist XY preview. Never commands a jaw.

The default 100 mm clearance keeps more of the part visible above the jaws.
At 40 mm, 100 mm pickup pixels cannot be transferred to the different height.
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
from steadyhand.executor import move_tcp_segmented, preflight_tcp_segmented
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills
from steadyhand.vega_presets import configured_right_preset
from steadyhand.vision.wrist_servo import ServoWaypointError, run_xy_servo
from steadyhand.operator_input import clean_choice
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
    parser.add_argument("--hover-clearance-mm", type=float, default=100.0)
    parser.add_argument("--remote-safe", action="store_true")
    parser.add_argument("--speed-scale", type=float, default=.38)
    parser.add_argument("--confirm-head-motion", action="store_true")
    parser.add_argument("--confirm-physical-motion", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if not args.confirm_physical_motion:
        parser.error("requires --confirm-physical-motion")
    if not 20.0 <= args.hover_clearance_mm <= 100.0:
        parser.error("--hover-clearance-mm must be 20..100")
    if not .10 <= args.speed_scale <= .70:
        parser.error("--speed-scale must be .10..70")
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
        if kind == "motion":
            print(f"WRIST {fields['label']}: measured error "
                  f"{fields['position_error_m'] * 1000:.2f} mm", flush=True)
        elif kind == "error":
            print(f"WRIST centering step {fields['iteration']}: "
                  f"{fields['error_px']:.1f} pixels from goal", flush=True)

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

    def checkpoint(label, target=None, *, capture_image=False):
        pose = robot.get_tcp_pose()
        record = {"label": label, "tcp": pose.position_m, "quaternion_wxyz": pose.quaternion_wxyz,
                  "joints": [float(v) for v in robot._read_joint_positions()],
                  "timestamp": datetime.now(timezone.utc).isoformat()}
        if capture_image:
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
        before = robot.get_tcp_pose()
        steps = dict(max_translation_step_m=.020, max_orientation_step_rad=.08)
        preflight_tcp_segmented(robot._kinematics, robot._read_joint_positions(),
                                before, target, **steps)
        start_clearance = before.position_m[2] - surface(*before.position_m[:2])
        end_clearance = target.position_m[2] - surface(*target.position_m[:2])
        if end_clearance < start_clearance - .0005:
            checkpoint("before_lowering", target, capture_image=True)
        move_tcp_segmented(robot, target, speed_scale=args.speed_scale, **steps,
                           waypoint_guard=waypoint_guard)

    persist()
    try:
        print(f"PREVIEW {args.part}: {args.hover_clearance_mm:g} mm hover, "
              f"wrist centering {'ON' if args.center else 'OFF'}. Jaws never move; profiles are not changed.", flush=True)
        print("[1/5] Clear the arm, look down, capture the board.", flush=True)
        robot.connect()
        cameras.connect()
        floor = float(load_vega_skills()["safety"]["min_tcp_z_m"])
        scene = _capture_downward_head_frame(robot, floor_m=floor, bundle=bundle,
            checkpoint=checkpoint, output=output)
        updated = _runtime_from_board_scene(runtime, scene)
        fresh_registration = updated[2][3].get("registration", {}).get("status") == "fresh"
        if "registration" not in updated[2][3]:
            fresh_registration = updated is not runtime  # compatibility with old saved fixtures
        if not fresh_registration and "registration" in updated[2][3]:
            print("Board fit rejected; capturing one more head image before review.", flush=True)
            scene = _capture_downward_head_frame(robot, floor_m=floor, bundle=bundle,
                checkpoint=checkpoint, output=output)
            updated = _runtime_from_board_scene(runtime, scene)
            fresh_registration = updated[2][3].get("registration", {}).get("status") == "fresh"
        runtime = updated
        result["fresh_board_registration_accepted"] = fresh_registration
        # Preserve the same validated physical axes as teaching/competition.
        physical_task_data = dict(task_data, task_coordinate_mirror_y=False)
        targets = _task_targets(runtime, physical_task_data, .100, use_profiles=False)
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
        chosen = next((p for p in scene["parts"] if p["index"] == observation.get("detection_index")), None)
        if observation.get("selection") == "head_detection" and chosen is not None:
            uv = tuple(int(round(v)) for v in chosen["center_image_px"])
            cv2.circle(overlay, uv, 30, (255, 0, 255), 4)
            cv2.putText(overlay, f"{args.part}  #{chosen['index']}", (max(0, uv[0]-80), max(30, uv[1]-38)),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 255), 2, cv2.LINE_AA)
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
        event("head_association", observation)
        print("[2/5] Head target association (full diagnostics saved in preview.json).", flush=True)
        if observation.get("selection") == "head_detection":
            expected = observation["expected_xy_m"]
            selected = observation["selected_xy_m"]
            print(f"Selected image detection #{observation['detection_index']} for {args.part}; "
                  f"correction from task map: forward {(selected[0]-expected[0])*1000:+.1f} mm, "
                  f"right {-(selected[1]-expected[1])*1000:+.1f} mm.", flush=True)
        print("HEAD TARGET REVIEW IMAGE:", overlay_path, flush=True)
        if not fresh_registration or observation.get("selection") != "head_detection":
            print("Target review needed: " + ("board geometry changed" if not fresh_registration else
                  str(observation.get("rejection_reason"))) + ". No target approach has started.", flush=True)
            print("This is an image review, not recalibration. Open the latest laptop image: "
                  "check that the GREEN outline matches all four board corners.\n"
                  "Enter the NUMBER printed next to your part in that image (not its menu number), "
                  "or its center pixel U V. 'show' resends the image; 'abort' cancels.\n"
                  "If the outline is wrong, abort. No approach starts until you choose a target.", flush=True)
            while True:
                answer = clean_choice(input("HEAD TARGET [image number / U V / show / abort]> ")).lower()
                if answer in ("abort", "q", "exit", "stop"):
                    raise KeyboardInterrupt()
                if answer == "show":
                    shutil.copyfile(overlay_path, temporary)
                    temporary.replace(live / "latest_wrist_a.png")
                    print("Head review image resent to the laptop helper.", flush=True)
                    continue
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
        print("[3/5] Move through RIGHT_READY to the head target at 100 mm clearance.", flush=True)
        checkpoint("before_ready")
        ready_q, _ = configured_right_preset(cfg, "right_ready")
        robot.move_joints(ready_q, speed_scale=args.speed_scale)
        checkpoint("before_coarse_hover")
        high = preview_target(runtime, args.part, observation, .100)
        move(high)
        print(f"[4/5] Preview at {args.hover_clearance_mm:g} mm clearance.", flush=True)
        if args.hover_clearance_mm != 100.0:
            checkpoint("before_preview_height")
            move(hover)
        checkpoint("coarse_hover")
        result["head_target_reached"] = True
        persist()
        if args.center:
            print("[5/5] Wrist XY centering only: select a feature ON the part, then where that SAME "
                  "feature should appear in a correctly aligned grasp at this height. "
                  "The image center is not automatically the grasp point. Type abort to stop.", flush=True)
            rgb, raw = image("wrist preview reference")
            feature = args.feature or select_pixel(rgb, raw, "Select a distinctive feature on the selected part", use_viewer=False)
            goal = args.goal_pixel or select_pixel(rgb, raw, "Select desired feature pixel at THIS preview height (not a saved 100 mm pixel)", use_viewer=False)
            result["preview_annotation"] = {"feature_uv": feature, "goal_uv": goal, "image": str(raw)}
            persist()
            checkpoint("before_wrist_centering")
            result["wrist_center"] = run_xy_servo(
                robot, lambda: image("servo")[0], floor_m=floor, feature_uv=feature, goal_uv=goal,
                max_radius_m=.060, probe_m=.008, gain=.45, max_step_m=.008,
                tolerance_px=8., max_iterations=12, speed_scale=.45,
                checkpoint=checkpoint, event=event,
                waypoint_guard=waypoint_guard,
                surface_z=lambda x, y: calibrated_surface_z(x, y, runtime[2][3]),
                reference_quaternion_wxyz=hover.quaternion_wxyz,
            )
            checkpoint("after_wrist_centering")
        else:
            image("head target reached")
            print("[5/5] Head target reached. Wrist centering was OFF; inspect the latest wrist image.", flush=True)
        result["status"] = "completed_no_gripper_motion"
        print("PREVIEW COMPLETE: no pickup/release was attempted.", flush=True)
        return 0
    except ServoWaypointError as exc:
        result.update(status="head_target_reached_wrist_not_verified", error=str(exc))
        try:
            image("wrist_center_stopped")
        except Exception as image_exc:
            event("stopped_image_unavailable", {"error": str(image_exc)})
        print(f"HEAD APPROACH COMPLETED; WRIST CENTERING NOT VERIFIED: {exc}. "
              "No further movement or jaw command will be issued. "
              "Use the latest wrist image and send preview.json/events.jsonl; do not recalibrate the board for this error.", flush=True)
        return 2
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
        print("Closing the preview connections. The following sensor/robot shutdown lines are session cleanup.", flush=True)
        try:
            cameras.close()
        finally:
            robot.close()
        print(f"PREVIEW ARTIFACTS: {output}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
