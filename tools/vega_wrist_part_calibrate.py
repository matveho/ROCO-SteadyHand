"""Teach or test a right-wrist visual grasp profile for one competition part.

The tool reuses the live path used by battery centering: fresh board image,
measured RIGHT_READY, coarse TCP hover, fresh physical-right ``wrist_a`` RGB,
bounded measured-Jacobian XY servo, and explicit software gripper gating.
It never operates the claw by hand.  ``grab`` is the only command that
descends or connects the gripper.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.vega import VegaAdapter
from steadyhand.board_calibration import load_board_calibration
from steadyhand.board_geometry import validate_task_coordinate_extent
from steadyhand.cameras.vega import VegaWristCameras
from steadyhand.config import load_bundle
from steadyhand.executor import move_tcp_segmented, preflight_tcp_segmented
from steadyhand.kinematics import IKError
from steadyhand.execution_offsets import load_offsets, parse_offsets, describe_offsets
from steadyhand.geometry import matrix_to_quaternion, quaternion_to_matrix, quaternion_angle
from steadyhand.board_relative import snapshot
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills
from steadyhand.vega_presets import configured_right_preset
from steadyhand.vision.wrist_servo import TemplateTracker, ServoWaypointError, run_xy_servo, DEFAULT_CENTERING_ITERATIONS
from steadyhand.wrist_part_profiles import (
    PART_NAMES,
    WORKING_ARM,
    TCP_FRAME,
    WRIST_CAMERA,
    default_template_dir,
    load_profiles,
    save_profile,
    file_sha256,
)
from tools.vega_wrist_fine_center import WristAOnlyCapture
from steadyhand.vision.wrist_review import select_pixel

ROOT = Path(__file__).resolve().parents[1]
COMPETITION_ACTIONS = ROOT / "configs" / "competition_actions.json"
GOAL_CLICK_MAX_ERROR_PX = 50.0
MOTION_STEPS = dict(max_translation_step_m=.020, max_orientation_step_rad=.08)


class GripNotVerifiedError(RuntimeError):
    """Gripper completed normally but did not report a grasp."""


class PickupPreflightError(IKError):
    """The complete pickup route failed before any descent or grip command."""


class PlacementPreflightError(IKError):
    """Placement route rejected while still at the successful pickup hover."""


def _session_execution_offsets(args):
    # In particular, drop teaching reuses competition-style pickup positioning.
    # Gate by CLI mode, not begin_part(competition=True), so NO teaching reads
    # this file, even if it is missing or malformed.
    if args.mode != "test":
        return None
    snapshot = getattr(args, "execution_offsets_json", None)
    return parse_offsets(json.loads(snapshot)) if snapshot is not None else load_offsets()


def _competition_center_backoff_m():
    """Return the measured forward correction for competition centering."""
    try:
        value = json.loads(COMPETITION_ACTIONS.read_text(encoding="utf-8"))
        millimetres = float(value.get("visual_center_backoff_mm", 15.0))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        millimetres = 15.0
    return max(0.0, min(30.0, millimetres)) / 1000.0


def release_wiggle_settings(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict):
        raise ValueError("release_wiggle must be an object")
    enabled = value.get("enabled", False)
    angle = value.get("angle_deg", 3.0)
    cycles = value.get("cycles", 2)
    if not isinstance(enabled, bool):
        raise ValueError("release_wiggle.enabled must be true or false")
    if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not math.isfinite(angle) or not 0 < angle <= 10:
        raise ValueError("release_wiggle.angle_deg must be finite and in (0, 10]")
    if isinstance(cycles, bool) or not isinstance(cycles, int) or not 1 <= cycles <= 5:
        raise ValueError("release_wiggle.cycles must be an integer 1..5")
    return {"enabled": enabled, "angle_deg": float(angle), "cycles": cycles}


def _is_visual_alignment_failure(exc):
    """Return whether an exception is safe to handle as a centering failure.

    Competition fallback is intentionally limited to wrist-image/tracker and
    local-servo calibration failures.  IK, TCP, camera, joint, floor, and
    transport failures still escape and stop the run rather than being hidden
    as a bad feature.
    """
    message = str(exc).lower()
    # A bounded local-servo arrival residual is recoverable: the controller
    # records the measured TCP pose and can either retry or use the coarse
    # grasp fallback.  This is distinct from a general TCP/IK failure.
    if "tcp missed servo waypoint by" in message:
        return True
    hardware_markers = (
        "ik ", "inverse kinematics", "joint", "tcp ", "floor", "camera",
        "estop", "timeout", "waypoint", "robot", "gripper", "motor",
        "non-finite tcp", "orientation drift",
    )
    if any(marker in message for marker in hardware_markers):
        return False
    visual_markers = (
        "feature", "tracking", "ambiguous", "centering", "jacobian",
        "probe", "pixel", "image", "template",
    )
    return any(marker in message for marker in visual_markers)


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _yaw_pose(pose, degrees):
    rotation = quaternion_to_matrix(pose.quaternion_wxyz)
    angle = math.radians(float(degrees))
    c, s = math.cos(angle), math.sin(angle)
    yaw = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    combined = tuple(
        tuple(sum(yaw[i][k] * rotation[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )
    return Pose(pose.position_m, matrix_to_quaternion(combined))


def _crop_template(rgb, uv, radius=20):
    import numpy as np
    array = np.asarray(rgb)
    u, v = (int(round(float(x))) for x in uv)
    if not (radius <= u < array.shape[1] - radius and radius <= v < array.shape[0] - radius):
        raise ValueError("feature is too close to the wrist image edge for a saved template")
    return array[v-radius:v+radius+1, u-radius:u+radius+1].copy(), (radius, radius)


def _load_place_template(settings):
    """Load a saved release-target template without touching pickup data."""
    path = _resolve(settings["template"])
    expected = settings.get("template_sha256")
    if expected and file_sha256(path) != expected:
        raise ValueError("placement template hash changed; re-teach placement CV")
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read placement template {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _write_overlay(path, rgb, feature=None, goal=None, *, label=None):
    import cv2
    import numpy as np
    image = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    if feature is not None:
        cv2.circle(image, tuple(int(round(v)) for v in feature), 22, (0, 255, 0), 3)
    if goal is not None:
        cv2.drawMarker(image, tuple(int(round(v)) for v in goal), (0, 0, 255), cv2.MARKER_CROSS, 32, 3)
    if label:
        cv2.putText(image, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.imwrite(str(path), image)


class PartSession:
    """One robot connection, fresh board registration and shared taught workflow."""

    def __init__(self, args, output, cfg, profiles):
        self.args, self.output, self.cfg, self.profiles = args, output, cfg, profiles
        self.release_wiggles = {}
        if args.mode == "test":
            raw = getattr(args, "release_wiggle_json", None)
            if raw is not None:
                self.release_wiggles[args.part] = release_wiggle_settings(json.loads(raw))
            elif COMPETITION_ACTIONS.is_file():
                entries = json.loads(COMPETITION_ACTIONS.read_text(encoding="utf-8")).get("parts", {})
                self.release_wiggles = {
                    part: release_wiggle_settings(entry.get("release_wiggle"))
                    for part, entry in entries.items()
                }
        self.execution_offsets = getattr(args, "execution_offsets", None)
        self.floor = float(load_vega_skills()["safety"]["min_tcp_z_m"])
        self.robot = VegaAdapter(cfg)
        self.cameras = VegaWristCameras()
        self.capture = WristAOnlyCapture(self.cameras, output, settle_s=.20, warmup_attempts=8)
        self.holding = False
        self.runtime = None
        self.targets = None
        self.head_scene = None
        self.head_observations = {}
        self.tracker = None
        self.goal = None
        self.goal_match = None
        self.reference_feature = None
        self.reference_match_score = None
        self.goal_match_score = None
        self.goal_match_error_px = None
        self.feature_tracking_mode = "strict"
        self.manual_alignment_override = False
        self.alignment_fallback_used = False
        self.alignment_verified = False
        self.no_cv_mode = False
        self.no_cv_used_recorded = False
        self.grasp_verified = False
        self.gripper_open_fraction = None
        self.step_mm = 5.0
        self.yaw = 0.
        self.history = []
        self.status = "created"
        self.last_error = None
        self.part = None
        self.action = None
        self.board_scene_paths = []
        self.drop_evidence_photos = []
        self.remote_safe = bool(getattr(args, "remote_safe", False))
        self.remote_checkpoint_index = 0
        self.place_cv_settings = None
        self.reference_pose = None
        self.successful_pickup_pose = None
        self.motion_faulted = False
        self.automatic_continuation_safe = False
        self.pickup_completed = False
        self.recovery_state_verified = False
        self.recovery_release_attempted = False
        self.recovery_blocked = False

    def start(self):
        self.status = "starting"
        self.robot.connect()
        self.retake()
        # Drop teaching deliberately reuses the saved pickup hover/grasp and
        # does not need a wrist frame.  Avoid making the procedure depend on a
        # live wrist-camera bridge that is irrelevant to this stage.
        mode = getattr(self.args, "mode", "calibrate")
        needs_wrist = mode not in ("drop", "test") or (
            mode == "test" and (not getattr(self.args, "no_cv", False)
                                or getattr(self.args, "place_cv", False)))
        if needs_wrist or self.remote_safe:
            self.cameras.connect()
        self.status = "ready"

    def close(self):
        cleanup_error = None
        try:
            try:
                self.cameras.close()
            finally:
                self.robot.close()
        except BaseException as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
            self.automatic_continuation_safe = False
            self.recovery_state_verified = False
            raise
        finally:
            summary = {
                "schema_version": 1,
                "status": self.status,
                "part": self.part,
                "action": self.action,
                "holding_may_be_true": bool(self.holding),
                "pickup_completed": bool(getattr(self, "pickup_completed", False)),
                "recovery_state_verified": bool(getattr(self, "recovery_state_verified", False)),
                "recovery_release_attempted": bool(getattr(self, "recovery_release_attempted", False)),
                "recovery_blocked": bool(getattr(self, "recovery_blocked", False)),
                "motion_faulted": bool(getattr(self, "motion_faulted", False)),
                "cleanup_error": cleanup_error,
                "automatic_continuation_safe": bool(getattr(self, "automatic_continuation_safe", False)
                                                     and not getattr(self, "motion_faulted", False)
                                                     and not self.holding),
                "board_reference": (snapshot(self.runtime[2])
                                    if getattr(self, "runtime", None) is not None else None),
                "last_error": self.last_error,
                "execution_offsets": self.execution_offsets.as_dict() if getattr(self, "execution_offsets", None) else None,
                "coarse_xy_m": list(self.coarse.position_m[:2]) if hasattr(self, "coarse") else None,
                "yaw_deg": self.yaw,
                "gripper_open_fraction": getattr(self, "gripper_open_fraction", None),
                "alignment_verified": getattr(self, "alignment_verified", False),
                "alignment_fallback_used": getattr(self, "alignment_fallback_used", False),
                "grasp_verified": getattr(self, "grasp_verified", False),
                "grasp_clearance_m": getattr(self, "last_grasp_clearance", None),
                "drop_release_photo": getattr(self, "drop_release_photo", None),
                "drop_evidence_photos": list(getattr(self, "drop_evidence_photos", [])),
                "board_scene_paths": list(self.board_scene_paths),
                "events_path": "events.jsonl",
                "calibration_path": "calibration/vega_board_manual.json",
                "profiles_path": str(self.args.profiles),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            (self.output / "run_summary.json").write_text(
                json.dumps(summary, indent=2, default=str) + "\n",
                encoding="utf-8",
            )

    def retake(self):
        if self.holding:
            raise RuntimeError("Retake blocked while a part may be held")
        from tools.vega_competition_pipeline import _capture_registered_board, _load_runtime, _task_targets
        runtime = _load_runtime()
        if self.runtime is not None and (self.runtime[2][3].get("calibration_sha256")
                                        == runtime[2][3].get("calibration_sha256")):
            # A rejected retake must retain this session's last accepted pose,
            # not jump back to the board position in the original JSON.
            runtime = runtime[0], runtime[1], self.runtime[2], self.runtime[3]
        self.runtime = runtime  # Clearance model is needed before camera-clear.
        self.runtime, scene = _capture_registered_board(
            self.robot, runtime, floor_m=self.floor, output=self.output,
            checkpoint=self.remote_checkpoint if self.remote_safe else None,
        )
        ready_q, _ = configured_right_preset(self.cfg, "right_ready")
        self.remote_checkpoint("before_right_ready")
        self.robot.move_joints(ready_q, speed_scale=self.args.speed_scale)
        self.remote_checkpoint("after_right_ready")
        # All menus share the reviewed physical task orientation. Saved
        # profiles add board-relative approach/grasp corrections below.
        teaching_task_data = dict(self.runtime[1])
        teaching_frame_override = not getattr(self.args, "head_reacquire", False)
        teaching_task_data["task_coordinate_mirror_y"] = False
        print("WRIST TARGET FRAME: validated physical orientation (task_coordinate_mirror_y=false).", flush=True)
        self.targets = _task_targets(self.runtime, teaching_task_data, .100, use_profiles=False)
        # During teaching/drop setup, do not let the generic dark-object
        # association select a similarly shaped object on the reflected side
        # of the board.  The reviewed task map (or a saved physical hover in
        # begin_part) is the authoritative coarse target.  Competition test
        # runs may still use fresh head detections for board translation.
        self.head_scene = scene
        if teaching_frame_override:
            self.head_observations = {}
            print(
                "WRIST COARSE TARGETS: reviewed task map only; "
                "live part association disabled.",
                flush=True,
            )
        else:
            try:
                from tools.vega_head_fallback import match_expected_parts
                self.head_observations = match_expected_parts(
                    scene, self.runtime, self.targets, task_data=teaching_task_data
                )
            except Exception as exc:
                self.head_observations = {}
                print(
                    "HEAD PART PERCEPTION UNAVAILABLE: keeping reviewed task "
                    f"coordinates ({type(exc).__name__}: {exc})",
                    flush=True,
                )
        for part, observation in self.head_observations.items():
            if observation.get("selection") != "head_detection":
                continue
            xy = observation.get("selected_xy_m")
            if not isinstance(xy, list) or len(xy) != 2:
                continue
            name = f"task.{part}.pick"
            target = self.targets.get(name)
            if target is None:
                continue
            x, y = (float(xy[0]), float(xy[1]))
            self.targets[name] = Pose(
                (x, y, self.surface(x, y) + .100),
                target.quaternion_wxyz,
            )
            print(
                "HEAD PART DETECTION: "
                f"{part} -> ({x:.4f}, {y:.4f}) m; coarse target updated",
                flush=True,
            )
        if self.output is not None:
            (self.output / "head_observations.json").write_text(
                json.dumps(self.head_observations, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        from steadyhand.board_relative import snapshot
        scene["registered_board_reference"] = snapshot(self.runtime[2])
        self.event("board_reference", {"board_reference": scene["registered_board_reference"]})
        scene_path = self.output / f"board_{time.time_ns()}.json"
        scene_path.write_text(json.dumps(scene, indent=2, default=str) + "\n")
        self.board_scene_paths.append(scene_path.name)
        status = self.runtime[2][3].get("registration", {}).get("status", "reference")
        print(f"BOARD FRAME: {status.upper()}; RIGHT_READY reached.", flush=True)

    def surface(self, x, y):
        from tools.vega_task_coordinate_reachability import calibrated_surface_z
        return calibrated_surface_z(x, y, self.runtime[2][3])

    def move(self, target, *, slow=False):
        if getattr(self, "motion_faulted", False):
            raise RuntimeError("Movement blocked after a hardware/motion fault; inspect before restarting")
        # Check the complete Cartesian segment before issuing its first waypoint.
        before = self.robot.get_tcp_pose()
        preflight_tcp_segmented(self.robot._kinematics, self.robot._read_joint_positions(),
                                before, target, **MOTION_STEPS)
        # Remote supervision changes confirmation/evidence only, not the
        # established trajectory, speed or manual jog size.
        if self.remote_safe:
            start_clearance = before.position_m[2] - self.surface(*before.position_m[:2])
            end_clearance = target.position_m[2] - self.surface(*target.position_m[:2])
            if end_clearance < start_clearance - .0005:
                self.remote_checkpoint("before_lowering", capture=True, target=target)
        speed = min(.42, self.args.speed_scale) if slow else self.args.speed_scale
        try:
            move_tcp_segmented(self.robot, target, speed_scale=speed,
                               **MOTION_STEPS,
                               waypoint_guard=self.remote_waypoint if self.remote_safe else None,
                               min_tcp_z_m=None)
        except Exception:
            self.motion_faulted = True
            raise

    def remote_waypoint(self, target):
        from steadyhand.remote_motion import needs_low_clearance_confirmation
        if needs_low_clearance_confirmation(self.robot.get_tcp_pose(), target, self.surface):
            self.remote_checkpoint("before_low_waypoint", target=target)

    def remote_checkpoint(self, label, *, capture=False, target=None):
        """Confirm low arm stages; only capture when explicitly requested."""
        if not getattr(self, "remote_safe", False):
            return
        self.remote_checkpoint_index += 1
        pose = self.robot.get_tcp_pose()
        record = {
            "index": self.remote_checkpoint_index,
            "label": str(label),
            "tcp_position_m": list(pose.position_m),
            "tcp_quaternion_wxyz": list(pose.quaternion_wxyz),
            "joint_positions_rad": [float(v) for v in self.robot._read_joint_positions()],
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if capture:
            try:
                _rgb, path = self.frame(f"remote checkpoint {label}")
                record["wrist_image"] = str(path.relative_to(ROOT))
            except Exception as exc:
                record["wrist_image_error"] = f"{type(exc).__name__}: {exc}"
                print(f"REMOTE CHECKPOINT IMAGE WARNING: {exc}", flush=True)
        self.event("remote_checkpoint", record)
        from steadyhand.remote_motion import needs_low_clearance_confirmation
        needs_prompt = label != "before_head_down" and needs_low_clearance_confirmation(
            pose, target, self.surface
        )
        if not needs_prompt:
            self.event("checkpoint_decision", {"checkpoint": label, "decision": "auto_continue"})
            return
        if target is not None:
            print("NEXT ARM TARGET =", target.position_m, flush=True)
        print(
            f"REMOTE-SAFE CHECKPOINT {self.remote_checkpoint_index}: {label}\n"
            f"  TCP={tuple(round(float(v), 6) for v in pose.position_m)}\n"
            "  Press Enter to continue, or type abort to stop safely.",
            flush=True,
        )
        while True:
            answer = input("REMOTE-SAFE> ").strip().lower()
            if answer in ("abort", "a", "q", "quit", "exit", "stop"):
                self.event("checkpoint_decision", {"checkpoint": label, "decision": "abort"})
                raise KeyboardInterrupt()
            if not answer:
                self.event("checkpoint_decision", {"checkpoint": label, "decision": "continue"})
                return
            print("Press Enter to continue or type abort; other input does not authorize motion.", flush=True)

    def frame(self, label="wrist"):
        # Drop/no-CV workflows start without a wrist stream. Opening is
        # idempotent, so optional inspection works in those workflows too.
        self.cameras.connect()
        rgb = self.capture()
        raw = self.output / f"{self.capture.index-1:03d}_wrist_a.png"
        print(f"{label.upper()} IMAGE: {raw}", flush=True)
        self._publish_live_image(rgb, raw, label)
        return rgb, raw

    def inspection_image(self, label):
        """Optional operator inspection must not discard a teaching session."""
        try:
            return self.frame(label)
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"IMAGE UNAVAILABLE: {exc}. No motion issued; retry image or abort.", flush=True)
            return None

    def _recoverable_servo_miss(self, exc):
        # Keep the servo's 8 mm bound. Only a small, structured residual from
        # an otherwise completed motion can abandon CV and start a NEW plan.
        if (not isinstance(exc, ServoWaypointError) or exc.measured_pose is None or
                exc.position_error_m is None or not .008 < exc.position_error_m <= .010):
            self.event("servo_replan_rejected", {"reason": "missing structured pose or residual outside 8..10 mm",
                       "position_error_m": getattr(exc, "position_error_m", None)})
            return False
        print(f"SERVO RECOVERY CHECK: {exc.position_error_m * 1000:.2f} mm miss; checking fresh stopped-arm state.", flush=True)
        try:
            pose = self.robot.stationary_tcp_pose()
        except (RuntimeError, ValueError) as state_exc:
            self.event("servo_replan_rejected", {"reason": str(state_exc)})
            raise
        drift = math.dist(pose.position_m, exc.measured_pose.position_m)
        rotation = quaternion_angle(pose.quaternion_wxyz, exc.measured_pose.quaternion_wxyz)
        if drift > .002 or rotation > .005:
            self.event("servo_replan_rejected", {"reason": "TCP changed after servo stopped",
                       "drift_m": drift, "rotation_rad": rotation})
            return False
        self.event("servo_replan_allowed", {"reason": str(exc), "measured_tcp": pose.position_m})
        print("CENTERING ABANDONED: stationary arm verified; saved grasp will be replanned from measured pose.", flush=True)
        return True

    def _publish_live_image(self, rgb, raw, label):
        """Publish a stable, optional laptop-pull path without affecting control."""
        live = ROOT / "runs" / "wrist_live"
        try:
            live.mkdir(parents=True, exist_ok=True)
            image_tmp = live / ".latest_wrist_a.png.tmp"
            manifest_tmp = live / ".latest_wrist_a.json.tmp"
            shutil.copyfile(raw, image_tmp)
            image_tmp.replace(live / "latest_wrist_a.png")
            source = str(raw.relative_to(ROOT)).replace("\\", "/")
            manifest = {
                "schema_version": 1,
                "part": self.part,
                "stage": str(label),
                "capture_index": int(self.capture.index - 1),
                "source_run": source,
                "image": "runs/wrist_live/latest_wrist_a.png",
                "image_shape": [int(value) for value in rgb.shape],
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            manifest_tmp.replace(live / "latest_wrist_a.json")
            print(
                "LIVE IMAGE (fixed laptop path): "
                "/home/dexmate/ROCO-SteadyHand-live/runs/wrist_live/latest_wrist_a.png",
                flush=True,
            )
        except (OSError, ValueError) as exc:
            # This is only an inspection convenience. A failed copy must never
            # turn a valid camera frame into a robot-control failure.
            print(f"LIVE IMAGE PUBLISH WARNING: {exc}", flush=True)

    def _invalidate_alignment(self, reason):
        if self.goal is not None:
            self.alignment_verified = False
            print(
                f"ALIGNMENT INVALIDATED ({reason}); run 'center' again before grab",
                flush=True,
            )

    def _gripper_speed(self):
        return int(round(float((self.cfg.get("gripper") or {}).get("open_speed_dps", 500))))

    def _set_gripper_fraction(self, fraction):
        """Move the empty right jaw to an absolute 0..1 opening fraction."""
        fraction = _number(fraction, 0.0, 1.0)
        if getattr(self, "remote_safe", False):
            self.remote_checkpoint("before_pregrasp_jaw_adjustment")
        self.robot.connect_gripper()
        result = self.robot._gripper.move_fraction(
            fraction, speed=self._gripper_speed()
        )
        measured = self.robot.gripper_position()
        try:
            measured = float(measured)
        except (TypeError, ValueError):
            measured = None
        self.gripper_open_fraction = fraction
        self.event(
            "gripper_opening",
            {"requested_fraction": fraction, "measured_fraction": measured},
        )
        print(
            "GRIPPER OPENING =",
            f"{fraction * 100:.1f}%",
            "MEASURED =",
            "unknown" if measured is None else f"{measured * 100:.1f}%",
            flush=True,
        )
        return result

    def _read_gripper_fraction(self):
        self.robot.connect_gripper()
        value = self.robot.gripper_position()
        value = float(value)
        if not math.isfinite(value):
            raise RuntimeError("gripper returned a non-finite position")
        return max(0.0, min(1.0, value))

    def event(self, kind, fields):
        if "feature_uv" in fields:
            import cv2
            raw = self.output / f"{self.capture.index-1:03d}_wrist_a.png"
            bgr = cv2.imread(str(raw))
            if bgr is not None:
                _write_overlay(raw.with_name(raw.stem + "_tracked.png"), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                               fields["feature_uv"], self.goal, label=self.part)
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"part": self.part, "event": kind, **fields}, default=str) + "\n")
        print(kind.upper(), json.dumps(fields, default=str), flush=True)

    def select(self, rgb, path, title):
        print(
            f"PART={self.part} | run the laptop helper "
            "`python tools/transfer_image.py --watch --open` to view new images",
            flush=True,
        )
        return select_pixel(rgb, path, title, use_viewer=not self.args.no_viewer)

    def teach_feature(self, *, choose_goal=True):
        """Capture and teach the first, trackable feature annotation.

        Feature-quality failures are annotation failures, so they stay inside
        this loop.  Camera, motion, and other hardware exceptions still escape
        to the session safety handler.
        """
        while True:
            rgb, path = self.frame(
                f"{self.part}: select a visible part feature, not the gripper"
            )
            feature = self.select(
                rgb,
                path,
                f"{self.part}: select a distinctive textured feature ON the selected part",
            )
            self.feature_tracking_mode = "strict"
            try:
                tracker = TemplateTracker(rgb, feature)
                _, score = tracker.locate(rgb)
            except (ValueError, RuntimeError) as exc:
                # Some valid part features have several nearly identical
                # nearby patches (smooth batteries and gear teeth are common
                # examples). Keep strict matching as the default, but accept
                # an operator-selected high-score patch in a bounded local
                # mode. The click-distance check plus the servo probe/return
                # gates still stop if the tracker drifts.
                message = str(exc).lower()
                tracker = None
                if isinstance(exc, RuntimeError) and "lost/ambiguous" in message:
                    try:
                        candidate = TemplateTracker(
                            rgb,
                            feature,
                            search_radius=80,
                            min_margin=0.0,
                        )
                        located, score = candidate.locate(rgb)
                        click_error = math.dist(tuple(located), tuple(feature))
                        if float(score) >= 0.90 and click_error <= 12.0:
                            tracker = candidate
                            self.feature_tracking_mode = "operator_local"
                            print(
                                "FEATURE ACCEPTED IN LOCAL MODE: "
                                f"score={float(score):.3f}, click_error={click_error:.1f}px; "
                                "probe/return checks remain enabled",
                                flush=True,
                            )
                    except (ValueError, RuntimeError):
                        tracker = None
                if tracker is None:
                    print(
                        "FEATURE REJECTED: this feature is weak or ambiguous; "
                        f"capture another image and choose a different feature ({exc})",
                        flush=True,
                    )
                    continue
            self.tracker = tracker
            # Preserve the pre-servo reference: a final aligned frame may
            # contain the jaw over the part and is a poor global template.
            self.reference_rgb = rgb.copy()
            self.reference_pose = self.robot.get_tcp_pose()
            self.reference_feature = tuple(float(v) for v in feature)
            self.reference_match_score = float(score)
            self.goal = None
            self.goal_match = None
            self.goal_match_score = None
            self.goal_match_error_px = None
            self.manual_alignment_override = False
            self.alignment_verified = False
            _write_overlay(
                path.with_name(path.stem + "_reference.png"),
                rgb,
                self.reference_feature,
                label=self.part,
            )
            self.event(
                "reference_feature",
                {
                    "feature_uv": self.reference_feature,
                    "score": self.reference_match_score,
                    "tracking_mode": self.feature_tracking_mode,
                },
            )
            print(
                "REFERENCE FEATURE ACCEPTED:",
                tuple(round(v, 1) for v in self.reference_feature),
                f"score={self.reference_match_score:.3f}",
                flush=True,
            )
            return self.reference_feature

    def teach_goal_feature(self):
        """Capture the aligned pose and require a second click on the same feature."""
        if self.tracker is None or self.reference_feature is None:
            raise RuntimeError("teach the first feature before selecting the goal")
        while True:
            rgb, path = self.frame(
                f"{self.part}: aligned grasp pose; select the SAME distinctive feature"
            )
            selected = self.select(
                rgb,
                path,
                f"{self.part}: select the SAME feature again at the desired grasp alignment",
            )
            try:
                located, score, error = self.tracker.verify_selected(
                    rgb, selected, max_error_px=GOAL_CLICK_MAX_ERROR_PX
                )
            except (ValueError, RuntimeError) as exc:
                print(
                    "GOAL FEATURE REJECTED: the second click was not verified "
                    "as the original feature; capture another image and retry "
                    f"({exc})",
                    flush=True,
                )
                continue
            # The operator's second annotation is the desired image coordinate;
            # the tracker match is retained as evidence that it is the same
            # feature rather than an arbitrary pixel.
            self.goal = tuple(float(v) for v in selected)
            self.goal_match = tuple(float(v) for v in located)
            self.goal_match_score = float(score)
            self.goal_match_error_px = float(error)
            self.manual_alignment_override = False
            self.alignment_verified = False
            _write_overlay(
                path.with_name(path.stem + "_goal.png"),
                rgb,
                self.goal_match,
                self.goal,
                label=self.part,
            )
            self.event(
                "goal_feature",
                {
                    "goal_uv": self.goal,
                    "matched_feature_uv": self.goal_match,
                    "score": self.goal_match_score,
                    "click_match_error_px": self.goal_match_error_px,
                },
            )
            print(
                "GOAL FEATURE ACCEPTED:",
                tuple(round(v, 1) for v in self.goal),
                "matched=",
                tuple(round(v, 1) for v in self.goal_match),
                f"score={self.goal_match_score:.3f}",
                f"click_error={self.goal_match_error_px:.1f}px",
                flush=True,
            )
            return self.goal

    def localize(self):
        if self.tracker is None or self.reference_feature is None:
            raise RuntimeError("teach a visible reference feature before centering")
        if self.goal is None:
            # A goal click is useful when the operator wants a particular
            # feature pixel, but the normal wrist-centering target is simply
            # the image center. This keeps `center` useful after the first
            # feature annotation and avoids a false prerequisite failure.
            if not hasattr(self, "reference_rgb"):
                raise RuntimeError("capture a reference image before centering")
            from steadyhand.vision.wrist_servo import image_center
            self.goal = tuple(float(v) for v in image_center(self.reference_rgb.shape))
            self.goal_match = None
            self.goal_match_score = None
            self.goal_match_error_px = None
            print(
                "CENTER TARGET: wrist-image center "
                f"= {tuple(round(v, 1) for v in self.goal)}",
                flush=True,
            )
        reference = self.robot.get_tcp_pose().quaternion_wxyz
        self.alignment_verified = False
        # Centering is deliberately retried with a gentler controller.  The
        # probe/return phase is sensitive to wrist-camera jitter, especially
        # when the jaws partly occlude a smooth battery.  Each retry returns to
        # the measured pose from before the attempt; no retry is issued for an
        # IK, TCP, camera, or other hardware error.
        origin = self.robot.get_tcp_pose()
        settings = (
            {
                "probe_m": .010, "gain": .45, "max_step_m": .008,
                "tolerance_px": 8., "max_iterations": DEFAULT_CENTERING_ITERATIONS,
            },
            {
                "probe_m": .008, "gain": .30, "max_step_m": .006,
                "tolerance_px": 12., "max_iterations": DEFAULT_CENTERING_ITERATIONS,
            },
        )
        last_error = None
        for attempt, values in enumerate(settings, 1):
            try:
                self.remote_checkpoint("before_wrist_centering")
                result = run_xy_servo(
                    self.robot, self.capture, floor_m=self.floor, goal_uv=self.goal,
                    max_radius_m=.060, speed_scale=.66, event=self.event,
                    checkpoint=self.remote_checkpoint if self.remote_safe else None,
                    waypoint_guard=self.remote_waypoint if self.remote_safe else None,
                    tracker_factory=lambda rgb, _: self._reacquire(rgb), surface_z=self.surface,
                    reference_quaternion_wxyz=reference, **values,
                )
                self.alignment_verified = result.get("status") == "converged"
                if self.alignment_verified:
                    self.alignment_fallback_used = False
                    if getattr(self.args, "mode", "calibrate") == "test":
                        # The wrist image goal is consistently about 15 mm
                        # forward of the physical grasp center on this setup.
                        # Apply the operator-editable correction only after a
                        # verified visual convergence; saved coarse hovers and
                        # calibration teaching remain unchanged.
                        backoff = _competition_center_backoff_m()
                        if backoff > 0.0:
                            current = self.robot.get_tcp_pose()
                            corrected_x = current.position_m[0] - backoff
                            clearance = current.position_m[2] - self.surface(
                                current.position_m[0], current.position_m[1]
                            )
                            corrected = Pose(
                                (
                                    corrected_x,
                                    current.position_m[1],
                                    self.surface(corrected_x, current.position_m[1]) + clearance,
                                ),
                                current.quaternion_wxyz,
                            )
                            self.move(corrected, slow=True)
                            result["competition_center_backoff_m"] = backoff
                            print(
                                "EXECUTION CENTER BACKOFF: "
                                f"moved {backoff * 1000:.0f} mm back from the visual goal",
                                flush=True,
                            )
                    print("ALIGNMENT VERIFIED: same feature reproduced at the taught goal", flush=True)
                    self.remote_checkpoint("after_wrist_centering")
                return result
            except (RuntimeError, ValueError) as exc:
                self.alignment_verified = False
                last_error = exc
                # A probe that the arm did not physically reach cannot be
                # used to estimate an image Jacobian.  Leave the TCP where
                # the adapter stopped and keep the teaching session alive so
                # the operator can use the already-supervised pose with
                # ``grab manual``.  In particular, do not issue a blind
                # return/retry after a missed physical waypoint.
                failure_text = str(exc).lower()
                if "tcp missed servo waypoint" in failure_text:
                    self.motion_faulted = True  # verification failure must remain a hard stop
                    if self._recoverable_servo_miss(exc):
                        self.motion_faulted = False
                    elif getattr(self.args, "competition", False):
                        raise
                if (
                    "tcp missed servo waypoint" in failure_text
                    or "centering stalled" in failure_text
                    or "pixel error increased" in failure_text
                    or "feature did not return" in failure_text
                    or "ill-conditioned" in failure_text
                    or "feature barely moved" in failure_text
                ):
                    self.alignment_fallback_used = True
                    self.event(
                        "alignment_fallback",
                        {
                            "reason": str(exc),
                            "fallback": "manual_current_pose",
                        },
                    )
                    print(
                        "VISUAL CENTERING STOPPED: the bounded visual servo "
                        "did not verify a final pixel alignment. No further "
                        "motion was issued; if the gripper is already over the "
                        "part, use 'grab manual' to continue teaching.",
                        flush=True,
                    )
                    return {
                        "status": "manual_fallback",
                        "reason": str(exc),
                        "tcp_position_m": self.robot.get_tcp_pose().position_m,
                    }
                if attempt >= len(settings) or not _is_visual_alignment_failure(exc):
                    raise
                print(
                    "CENTERING RETRY: wrist feature/servo observation was not reliable; "
                    "returning to the remembered hover and retrying with gentler motion "
                    f"({exc})",
                    flush=True,
                )
                # This is a bounded recovery to the pose where this attempt
                # began.  If that recovery itself fails, the hardware error
                # escapes rather than being treated as an annotation issue.
                self.move(origin, slow=True)
        raise last_error

    def _reacquire(self, rgb):
        self.tracker.locate(rgb)
        return self.tracker

    def _load_saved_feature(self, profile):
        """Read a new frame; never reuse another part's tracker on rejection."""
        self.tracker = None
        self.alignment_verified = False
        rgb, path = self.frame("saved template check")
        if list(rgb.shape[:2]) != profile["image_shape"]:
            raise ValueError("Wrist image resolution changed; re-teach this part's feature")
        tracker = TemplateTracker.from_saved_template(
            rgb, _load_template(profile), profile["template"].get("template_uv"),
            initial_uv=profile.get("feature_uv"),
        )
        self.tracker = tracker
        self.reference_rgb = rgb.copy()
        self.reference_pose = self.robot.get_tcp_pose()
        self.reference_feature = tuple(profile.get("feature_uv") or tracker.uv)
        self.reference_match_score = profile.get("reference_match_score")
        self.goal = tuple(profile["goal_uv"]) if profile.get("goal_uv") is not None else None
        _write_overlay(path.with_name(path.stem + "_match.png"), rgb, tracker.uv, self.goal, label=self.part)

    def use_recorded_grasp(self):
        """Restore the projected successful grasp, not the last CV probe pose."""
        if self.holding or getattr(self, "motion_faulted", False):
            raise RuntimeError("Recorded grasp blocked: holding or motion fault requires inspection")
        self.move(self.grasp_target, slow=True)
        self.no_cv_mode = True
        self.alignment_fallback_used = True
        self.alignment_verified = False
        self.tracker = None
        print("SAVED GRASP HOVER REACHED; taught depth and jaw opening are unchanged.", flush=True)

    def _visual_recovery(self, exc, *, competition=False):
        """A visual rejection permits inspection; a motion/camera fault does not."""
        if getattr(self, "motion_faulted", False) or not _is_visual_alignment_failure(exc):
            raise exc
        self.alignment_verified = False
        self.alignment_fallback_used = True
        self.event("alignment_fallback", {"reason": str(exc),
                   "fallback": "projected_taught_grasp" if competition else "operator_review"})
        if competition:
            self.use_recorded_grasp()
        else:
            print(f"WRIST ALIGNMENT NOT VERIFIED: {exc}\n"
                  "Arm left at its current hover; no grasp commanded. "
                  "Use saved for the taught grasp, center to retry, image to inspect, or abort.",
                  flush=True)

    def retry_saved_alignment(self, profile):
        self.no_cv_mode = False
        try:
            self._load_saved_feature(profile)
            return self.localize()
        except (RuntimeError, ValueError) as exc:
            self._visual_recovery(exc)
            return None

    def profile_pose(self, profile, action, *, no_cv=False, clearance=.100):
        from steadyhand.board_relative import resolve_profile_target
        nominal = self.targets[f"task.{self.part or profile['part']}.{action}"]
        if not profile:
            return nominal
        if action == "pick" and not profile.get("pickup_board"):
            print(f"LEGACY BOARD REFERENCE ASSUMED for {self.part}: "
                  "run vega_migrate_board_profiles.py to recover teaching logs; verify hover.", flush=True)
        xy, quat = resolve_profile_target(profile, self.runtime[2], self.runtime[3],
                                           nominal, action=action, no_cv=no_cv)
        return Pose((*xy, self.surface(*xy) + clearance), quat)

    def pickup_record(self):
        from steadyhand.board_relative import snapshot, make_record, record_target
        reference = snapshot(self.runtime[2])
        approach = self.reference_pose or self.coarse
        grasp = self.successful_pickup_pose
        if grasp is None:
            raise ValueError("No successful pickup pose recorded in this teaching session")
        return make_record(reference,
            approach=record_target(reference, approach, source="feature_image_TCP"),
            grasp=record_target(reference, grasp, source="confirmed_grasp_hover_TCP"))

    def _settle_pickup_hover(self, target):
        """Let endpoint readback settle before capturing the reference image."""
        previous = self.robot.get_tcp_pose()
        stable_reads = 0
        for _ in range(5):
            time.sleep(.1)
            actual = self.robot.get_tcp_pose()
            stable = (math.dist(previous.position_m, actual.position_m) <= .0005
                      and quaternion_angle(previous.quaternion_wxyz,
                                           actual.quaternion_wxyz) <= .002)
            stable_reads = stable_reads + 1 if stable else 0
            previous = actual
            if stable_reads >= 2:
                break
        error = math.dist(actual.position_m, target.position_m)
        registration = self.runtime[2][3].get("registration", {})
        self.event("pickup_hover_reached", {
            "requested_tcp": list(target.position_m),
            "measured_tcp": list(actual.position_m),
            "position_error_m": error,
            "orientation_error_rad": quaternion_angle(actual.quaternion_wxyz,
                                                       target.quaternion_wxyz),
            "approach_to_grasp_xy_m": math.dist(self.coarse.position_m[:2],
                                                self.grasp_target.position_m[:2]),
            "board_registration": registration,
        })
        print(f"PICKUP HOVER: measured error {error * 1000:.1f} mm; "
              f"board frame {registration.get('status', 'reference')}", flush=True)

    def begin_part(self, part, profile=None, *, initial_yaw=None, competition=False,
                   no_cv=False):
        if self.holding or getattr(self, "motion_faulted", False):
            raise RuntimeError("Next part blocked: holding or motion fault requires inspection")
        self.part, self.history = part, []
        self.pickup_completed = False
        self.recovery_state_verified = False
        self.automatic_continuation_safe = False
        self.recovery_blocked = False
        self.recovery_release_attempted = False
        self.tracker = None
        self.goal = self.goal_match = None
        self.reference_feature = None
        self.reference_match_score = None
        self.reference_rgb = None
        self.goal_match_score = self.goal_match_error_px = None
        self.alignment_verified = False
        self.manual_alignment_override = False
        manual_teaching = getattr(self.args, "mode", None) == "calibrate"
        self.no_cv_mode = bool(no_cv or manual_teaching)
        self.no_cv_used_recorded = False
        self.alignment_fallback_used = False
        self.calibration_hash_mismatch = False
        validate_task_coordinate_extent(self.runtime[1], names=[f"{part}.pick"])
        self.successful_pickup_pose = None
        self.reference_pose = None
        self.yaw = float(profile["yaw_deg"]) if profile else float(initial_yaw or 0.0)
        nominal = self.targets[f"task.{part}.pick"]
        if profile:
            coarse = self.profile_pose(profile, "pick", no_cv=manual_teaching)
        else:
            coarse = nominal if manual_teaching else _yaw_pose(nominal, self.yaw)
        self.grasp_target = self.profile_pose(profile, "pick", no_cv=True) if profile else coarse
        if not manual_teaching and getattr(self.args, "head_reacquire", False):
            observation = self.head_observations.get(part, {})
            if observation.get("selection") != "head_detection":
                raise ValueError("Head reacquisition has no unique nearby detection; part skipped")
            if profile:
                # A head detector observes the *part center*, not the taught
                # jaw/TCP offset. Apply only its residual displacement to both
                # projected taught poses; don't discard the successful grasp.
                delta = [float(observation["selected_xy_m"][i]) -
                         float(observation["expected_xy_m"][i]) for i in range(2)]
                if not all(math.isfinite(v) for v in delta) or math.hypot(*delta) > .060:
                    raise ValueError("Head part displacement exceeds the bounded local correction")
                def shifted(pose):
                    x, y = (pose.position_m[i] + delta[i] for i in range(2))
                    return Pose((x, y, self.surface(x, y) + .100), pose.quaternion_wxyz)
                coarse, self.grasp_target = shifted(coarse), shifted(self.grasp_target)
        self.coarse = coarse
        print(f"BOARD-RELATIVE {part} APPROACH: {tuple(round(v, 4) for v in coarse.position_m)}", flush=True)
        print(f"BOARD-RELATIVE {part} GRASP HOVER: {tuple(round(v, 4) for v in self.grasp_target.position_m)}", flush=True)
        self.event("resolved_pickup", {"approach_tcp": list(coarse.position_m),
            "grasp_hover_tcp": list(self.grasp_target.position_m),
            "registration": self.runtime[2][3].get("registration")})
        if profile:
            # A field recalibration changes the board surface/registration hash.
            # It must not invalidate a previously verified grasp profile: the
            # current board image and task transform are the best available
            # position, while the saved profile remains the grasp/wrist record.
            active_path = ROOT / "calibration" / "vega_board_manual.json"
            if not active_path.is_file():
                active_path = ROOT / "calibration" / "vega_board_manual_fallback.json"
            active_calibration = load_board_calibration(active_path, self.cfg)
            self.calibration_hash_mismatch = (
                active_calibration["sha256"] != profile.get("calibration_sha256")
            )
            if self.calibration_hash_mismatch:
                self.event(
                    "board_calibration_mismatch_continue",
                    {
                        "saved_calibration_sha256": profile.get("calibration_sha256"),
                        "active_calibration_sha256": active_calibration["sha256"],
                        "policy": "continue_using_current_board_target_and_saved_grasp_profile",
                    },
                )
        if profile and profile.get("gripper_open_fraction") is not None:
            opening = float(profile["gripper_open_fraction"])
            self.gripper_open_fraction = opening
            self._set_gripper_fraction(opening)
        self.coarse = coarse
        if no_cv or manual_teaching:
            self.remote_checkpoint("before_coarse_hover")
            self.move(self.grasp_target)
            self._settle_pickup_hover(self.grasp_target)
            self.coarse = self.grasp_target
            self.no_cv_used_recorded = bool(profile)
            if manual_teaching:
                print("MANUAL TEACHING HOVER: " + (
                    "board-projected saved grasp" if profile else "nominal task position"
                ) + "; no feature loaded or centering commanded.", flush=True)
            else:
                print("NO-CV COARSE HOVER: projected taught grasp", flush=True)
            self.remote_checkpoint("coarse_hover")
            return None
        self.remote_checkpoint("before_coarse_hover")
        self.move(coarse)
        self._settle_pickup_hover(coarse)
        self.coarse = coarse
        self.remote_checkpoint("coarse_hover")
        if profile:
            try:
                self._load_saved_feature(profile)
            except (RuntimeError, ValueError) as exc:
                # Keep identity metadata for an explicit manual approval, but
                # never call an ambiguous candidate a verified alignment.
                self.reference_feature = tuple(profile.get("feature_uv") or (0.0, 0.0))
                self.reference_match_score = profile.get("reference_match_score")
                self.goal = tuple(profile["goal_uv"]) if profile.get("goal_uv") is not None else None
                self._visual_recovery(exc, competition=competition)
                return None
        else:
            return None
        try:
            result = self.localize()
            if competition and not self.alignment_verified:
                if self.motion_faulted:
                    raise RuntimeError("Visual servo motion failed; automatic grasp blocked")
                self.use_recorded_grasp()
                print("CENTERING UNVERIFIED: returned to projected taught grasp", flush=True)
            return result
        except (RuntimeError, ValueError) as exc:
            self._visual_recovery(exc, competition=competition)
            return None

    def grab(self, clearance, *, allow_unverified=False):
        if getattr(self, "motion_faulted", False):
            raise RuntimeError("Grasp blocked after a motion fault; inspect before restarting")
        if clearance is None:
            raise ValueError("Set depth N first; N is millimetres below the 100 mm hover")
        if (self.goal is None or not self.alignment_verified) and not allow_unverified:
            raise RuntimeError("Verify same-feature alignment with 'center' before grab")
        if allow_unverified and (self.goal is None or not self.alignment_verified):
            if self.reference_feature is None and not self.no_cv_mode:
                raise RuntimeError("teach a visible reference feature before grab manual")
            # The operator has deliberately positioned the gripper at the
            # desired grasp pose. Capture the feature at this exact current
            # pose so a later competition run does not reuse an older goal
            # pixel and servo the arm away from the manual adjustment.
            captured_current_goal = False
            if self.tracker is not None and self.reference_feature is not None:
                try:
                    rgb, path = self.frame("manual alignment snapshot")
                    feature, score = self.tracker.locate(rgb)
                    self.goal = tuple(float(v) for v in feature)
                    self.goal_match = tuple(float(v) for v in feature)
                    self.goal_match_score = float(score)
                    self.goal_match_error_px = 0.0
                    captured_current_goal = True
                    _write_overlay(
                        path.with_name(path.stem + "_manual_goal.png"),
                        rgb,
                        feature,
                        feature,
                        label=self.part,
                    )
                    self.event(
                        "manual_current_pose_goal",
                        {
                            "goal_uv": self.goal,
                            "score": self.goal_match_score,
                            "source": "fresh_feature_at_current_pose",
                        },
                    )
                    print(
                        "MANUAL CURRENT-POSE GOAL:",
                        tuple(round(v, 1) for v in self.goal),
                        f"score={self.goal_match_score:.3f}",
                        flush=True,
                    )
                except (ValueError, RuntimeError) as exc:
                    if not _is_visual_alignment_failure(exc):
                        raise
                    print(
                        "MANUAL GOAL IMAGE NOT TRACKED; preserving the current "
                        f"supervised TCP pose ({exc})",
                        flush=True,
                    )
            if not captured_current_goal and self.goal is None and self.reference_feature is not None:
                # If no fresh match was available, preserve the original
                # supervised reference as the last-resort goal.
                self.goal = tuple(self.reference_feature)
                self.goal_match = tuple(self.reference_feature)
                self.goal_match_score = self.reference_match_score
                self.goal_match_error_px = 0.0
            self.manual_alignment_override = True
            self.alignment_verified = True
            self.event(
                "manual_alignment_override",
                {
                    "goal_uv": self.goal,
                    "reason": "operator_confirmed_current_pose_without_center",
                },
            )
            print(
                "MANUAL ALIGNMENT OVERRIDE: using the current supervised TCP pose; "
                "visual centering was skipped by the operator.",
                flush=True,
            )
        if self.holding:
            raise ValueError("A part may already be held; inspect it before another grab")
        hover = self.robot.get_tcp_pose()
        self.last_grasp_clearance = clearance
        self.pickup_hover = hover
        # Snapshot the operator's actual current hover pose.  In particular,
        # preserve a deliberate manual ``back`` adjustment made after visual
        # centering; the grasp approach must not reconstruct an older goal or
        # let the IK descent's small XY residual erase that adjustment.
        anchor_x, anchor_y, _ = hover.position_m
        x, y = anchor_x, anchor_y
        grasp = Pose((x, y, self.surface(x, y) + clearance), hover.quaternion_wxyz)
        # Validate the same segmented descent AND lift that move() executes.
        # A one-shot lift solve can fail even when every local step is valid.
        seed = self.robot._read_joint_positions()
        start = hover
        for stage, target in (("descent", grasp), ("lift", hover)):
            try:
                seed = preflight_tcp_segmented(self.robot._kinematics, seed, start, target,
                                               **MOTION_STEPS)
            except IKError as exc:
                self.event("pickup_preflight_failed", {
                    "stage": stage, "hover_tcp": list(hover.position_m),
                    "grasp_tcp": list(grasp.position_m),
                    "quaternion_wxyz": list(hover.quaternion_wxyz),
                    "grasp_clearance_m": clearance, "reason": str(exc),
                })
                raise PickupPreflightError(f"Pickup {stage} preflight failed: {exc}") from exc
            start = target
        self.robot.connect_gripper()
        if self.gripper_open_fraction is None:
            # Never travel to the hard-open stop during a competition grasp;
            # it forces a slow homing/opening cycle and is unnecessary for a
            # taught object.  Twenty percent is the validated safe fallback.
            self.gripper_open_fraction = .20
            self._set_gripper_fraction(self.gripper_open_fraction)
        else:
            self._set_gripper_fraction(self.gripper_open_fraction)
        self.remote_checkpoint("before_grasp_descent")
        self.move(grasp, slow=True)
        self.remote_checkpoint("grasp_height_jaws_open")
        # Read back the actual endpoint. A small upward Z residual may be late
        # telemetry; settle it before considering one retry of the taught Z.
        # Never compensate by requesting a position below the taught target.
        measured_grasp = self.robot.get_tcp_pose()
        if measured_grasp.position_m[2] - grasp.position_m[2] > .002:
            measured_grasp = self._stationary_grasp_readback()
        xy_error = math.dist(
            measured_grasp.position_m[:2], (anchor_x, anchor_y)
        )
        z_error = measured_grasp.position_m[2] - grasp.position_m[2]
        correct_height = (.002 < z_error <= .008 and
                          math.hypot(xy_error, z_error) <= .010)
        self.event(
            "grasp_approach_readback",
            {
                "anchor_xy_m": [anchor_x, anchor_y],
                "requested_grasp_tcp": list(grasp.position_m),
                "measured_grasp_tcp": list(measured_grasp.position_m),
                "xy_error_m": xy_error,
                "z_error_m": z_error,
                "height_correction_planned": correct_height,
            },
        )
        if xy_error > 0.001 or correct_height:
            if xy_error > 0.010:
                raise RuntimeError(
                    "Grasp approach drifted more than 10 mm from the manually "
                    "positioned hover; no grip command issued"
                )
            correction = Pose(
                (anchor_x, anchor_y, grasp.position_m[2] if correct_height else measured_grasp.position_m[2]),
                measured_grasp.quaternion_wxyz,
            )
            print(
                f"GRASP {'XYZ' if correct_height else 'XY'} CORRECTION: "
                f"XY residual {xy_error * 1000:.1f} mm; "
                f"height {z_error * 1000:+.1f} mm from taught target; "
                + ("one retry to taught height" if correct_height else "height unchanged"),
                flush=True,
            )
            self.move(correction, slow=True)
            measured_grasp = (self._stationary_grasp_readback() if correct_height
                              else self.robot.get_tcp_pose())
            corrected_error = math.dist(
                measured_grasp.position_m[:2], (anchor_x, anchor_y)
            )
            self.event(
                "grasp_approach_corrected",
                {
                    "measured_grasp_tcp": list(measured_grasp.position_m),
                    "xy_error_m": corrected_error,
                    "z_error_m": measured_grasp.position_m[2] - grasp.position_m[2],
                    "height_correction_attempted": correct_height,
                },
            )
            if corrected_error > 0.006:
                raise RuntimeError(
                    "Grasp approach could not return to the manually positioned "
                    f"hover anchor (XY residual {corrected_error * 1000:.1f} mm)"
                )
        final_z_error = measured_grasp.position_m[2] - grasp.position_m[2]
        self.event("grasp_height_readback", {
            "requested_z_m": grasp.position_m[2],
            "measured_z_m": measured_grasp.position_m[2],
            "z_error_m": final_z_error,
            "height_correction_attempted": correct_height,
        })
        if abs(final_z_error) > .002:
            print(f"GRASP HEIGHT RESIDUAL: {final_z_error * 1000:+.1f} mm "
                  "(positive = above taught target). No further height correction; "
                  "continuing the existing grip attempt.", flush=True)
        self.remote_checkpoint("before_gripper_close")
        self.holding = True  # Remains true on uncertain grip/error; no blind recovery.
        self.robot.grip(self.part)
        result = self.robot._gripper.last_grip_result()
        self.event("grip_result", {
            "requested_grasp_tcp": list(grasp.position_m),
            "measured_grasp_tcp": list(self.robot.get_tcp_pose().position_m),
            "result": result,
        })
        print("GRIP RESULT", json.dumps(result, default=str), flush=True)
        if not isinstance(result, dict) or not isinstance(result.get("gripped"), bool):
            raise RuntimeError("Gripper result is unknown; holding state requires inspection")
        if result.get("gripped") is not True:
            raise GripNotVerifiedError("Grip was not verified; stopped at grasp height for inspection")
        self.remote_checkpoint("grip_verified")
        self.remote_checkpoint("before_lift")
        self.move(hover, slow=True)
        self.remote_checkpoint("lifted_with_part")
        self.successful_pickup_pose = hover
        self.pickup_completed = True
        self.event("successful_pickup_pose", {"hover_tcp": list(hover.position_m),
            "quaternion_wxyz": list(hover.quaternion_wxyz)})
        return result

    def _stationary_grasp_readback(self):
        try:
            return self.robot.stationary_tcp_pose(settle_timeout_s=1.0)
        except Exception:
            self.motion_faulted = True
            raise

    def return_part(self, clearance, *, partial_release=False):
        if not self.holding:
            raise ValueError("No part is held")
        hover = self.robot.get_tcp_pose()
        x, y, _ = hover.position_m
        release = Pose(
            (x, y, self.surface(x, y) + clearance),
            hover.quaternion_wxyz,
        )
        retreat = hover
        pickup_hover = getattr(self, "pickup_hover", None)
        if (pickup_hover is not None and
                math.dist(pickup_hover.position_m[:2], hover.position_m[:2]) <= .010 and
                pickup_hover.position_m[2] > hover.position_m[2]):
            # A normal "no object gripped" result occurs before lift. Release
            # there, then retreat vertically to the original pickup height;
            # don't leave the arm at grasp height before the next attempt.
            retreat = Pose((x, y, pickup_hover.position_m[2]), hover.quaternion_wxyz)
        seed = self.robot._read_joint_positions()
        seed = preflight_tcp_segmented(self.robot._kinematics, seed, hover, release, **MOTION_STEPS)
        preflight_tcp_segmented(self.robot._kinematics, seed, release, retreat, **MOTION_STEPS)
        self.remote_checkpoint("before_return_descent")
        self.move(release, slow=True)
        self.remote_checkpoint("return_release_height")
        # Never use the hard-open stop to release a held part: it can push or
        # move the board.  The adapter measures the live holding position and
        # opens by at most five percentage points.
        release_result = self.robot.release_gripper(self.part)
        self.event("place_release", {
            "requested_release_tcp": list(release.position_m),
            "gripper_release": release_result,
            "settings": {
                "return_to_source": True,
                "clearance_m": float(clearance),
            },
        })
        self.holding = False
        self.remote_checkpoint("returned_part_released")
        self.remote_checkpoint("before_return_retreat")
        self.move(retreat, slow=True)
        self.remote_checkpoint("return_retreat_complete")

    def _place_visual_align(self, settings):
        """Align at the taught hover; low confidence never authorizes release."""
        from steadyhand.vision.placement import placement_digest, placement_tracker
        cv_settings = self.place_cv_settings
        using_corners = bool(cv_settings and cv_settings.get("method") == "white_board_corners_v1")
        from steadyhand.vision.board_corners import CornerVisualError
        visual_error = CornerVisualError if using_corners else ValueError
        origin = self.robot.stationary_tcp_pose(settle_timeout_s=2.) if using_corners else self.robot.get_tcp_pose()
        try:
            if not cv_settings or not cv_settings.get("enabled", False):
                raise visual_error("Placement feature reference is missing or disabled")
            if cv_settings.get("placement_sha256") != placement_digest(settings):
                raise visual_error("Placement feature reference belongs to different place settings")
            clearance = origin.position_m[2] - self.surface(*origin.position_m[:2])
            if abs(clearance - float(cv_settings.get("reference_clearance_m", -1))) > .008:
                raise visual_error("Placement feature reference was taught at a different height")
            from steadyhand.vision.board_corners import METHOD, CornerServo, rotate_reference
            if cv_settings.get("method") == METHOD:
                reference = rotate_reference(cv_settings, snapshot(self.runtime[2]))
                self.remote_checkpoint("before_placement_centering")
                servo = CornerServo(self.robot, self._move_place_corner,
                    self._placement_corner_frame, self.surface, self.event)
                result = servo.align(reference)
                # A correction is useful only if the corrected descent and
                # retreat are reachable. Check before accepting its XY.
                aligned = self.robot.stationary_tcp_pose(settle_timeout_s=2.)
                release = Pose((*aligned.position_m[:2],
                    self.surface(*aligned.position_m[:2]) + settings["clearance_m"]), aligned.quaternion_wxyz)
                try:
                    seed = preflight_tcp_segmented(self.robot._kinematics, self.robot._read_joint_positions(),
                                                   aligned, release, **MOTION_STEPS)
                    preflight_tcp_segmented(self.robot._kinematics, seed, release, aligned, **MOTION_STEPS)
                except IKError as exc:
                    raise CornerVisualError(f"Corrected placement is unreachable: {exc}") from exc
                self.event("place_cv_result", {"method": METHOD, "result": result})
                self.remote_checkpoint("after_placement_centering")
                return result
            template = _load_place_template(cv_settings)
            goal = tuple(float(v) for v in cv_settings["goal_uv"])
            self.remote_checkpoint("before_placement_centering")
            result = run_xy_servo(
                self.robot, self.capture, floor_m=self.floor, goal_uv=goal,
                probe_m=.006, gain=.35, max_step_m=.006, max_radius_m=.030,
                tolerance_px=8.0, max_iterations=DEFAULT_CENTERING_ITERATIONS,
                speed_scale=.66,
                checkpoint=self.remote_checkpoint if self.remote_safe else None,
                waypoint_guard=self.remote_waypoint if self.remote_safe else None,
                event=self.event, surface_z=self.surface,
                reference_quaternion_wxyz=origin.quaternion_wxyz,
                tracker_factory=lambda rgb, _uv: placement_tracker(rgb, template, cv_settings),
            )
            if result.get("status") != "converged":
                raise RuntimeError("Placement centering was not verified")
            self.event("place_cv_result", {"result": result, "goal_uv": goal})
            self.remote_checkpoint("after_placement_centering")
            return result
        except (RuntimeError, ValueError) as exc:
            if using_corners and not isinstance(exc, CornerVisualError):
                raise
            # Hardware/motion failures always propagate. Only visual failures
            # are eligible for an explicit operator-authorized saved-pose fallback.
            if not using_corners and any(word in str(exc).lower() for word in ("ik ", "tcp", "joint", "camera", "estop", "e-stop", "timeout", "motor")):
                raise
            self.event("place_cv_low_confidence", {"reason": str(exc), "release_authorized": False})
            print(f"PLACEMENT CV NOT VERIFIED: {exc}; no descent/release authorized.", flush=True)
            if getattr(self.args, "competition", False) and not self.remote_safe:
                # Competition has no operator at the terminal.  A visual-only
                # failure must not strand a held part or abort the whole run:
                # return to the previously verified release hover and use the
                # saved physical placement pose.  The event is explicit so a
                # later audit can distinguish CV success from this fallback.
                self.event("place_cv_decision", {
                    "decision": "saved_pose_auto_fallback",
                    "reason": str(exc),
                    "release_authorized": True,
                })
                print("PLACEMENT CV FALLBACK: returning to the taught release pose.", flush=True)
                self.move(origin, slow=True)
                return None
            while True:
                answer = input("Type saved to return to the taught hover and use its release pose, or abort: ").strip().lower()
                if answer == "saved":
                    self.event("place_cv_decision", {"decision": "saved_pose", "reason": str(exc)})
                    self.move(origin, slow=True)
                    return None
                if answer in ("abort", "stop", "q", "exit"):
                    raise KeyboardInterrupt()

    def _execution_target(self, kind, hover, clearance):
        """Apply a run's correction once, after CV, without modifying teaching."""
        offsets = getattr(self, "execution_offsets", None)
        if offsets is None:
            return hover, clearance
        offset = getattr(offsets, kind)
        corrected = offset.hover(hover, self.surface)
        depth = offset.clearance(clearance)
        if corrected != hover or depth != clearance:
            self.event("execution_offset", {
                "stage": kind,
                "offset_mm": offsets.as_dict()[kind],
                "original_hover_tcp": list(hover.position_m),
                "corrected_hover_tcp": list(corrected.position_m),
                "taught_clearance_m": clearance,
                "execution_clearance_m": depth,
            })
            # Reject an unreachable corrected descent before shifting the arm.
            # move() and grab() also preflight their entire segmented paths.
            x, y = corrected.position_m[:2]
            descent = Pose((x, y, self.surface(x, y) + depth), corrected.quaternion_wxyz)
            seed = self.robot._read_joint_positions()
            seed = preflight_tcp_segmented(self.robot._kinematics, seed,
                                           self.robot.get_tcp_pose(), corrected, **MOTION_STEPS)
            preflight_tcp_segmented(self.robot._kinematics, seed, corrected, descent, **MOTION_STEPS)
            if corrected != hover:
                self.move(corrected, slow=True)
        return corrected, depth

    def _wiggle_before_release(self):
        settings = self.release_wiggles.get(self.part, {})
        if not settings.get("enabled", False):
            return
        if not self.holding or self.motion_faulted:
            raise RuntimeError("Release wiggle needs a held part and healthy motion state")
        original = None
        try:
            self.robot.stationary_tcp_pose(settle_timeout_s=1.0)
            original = tuple(self.robot._read_joint_positions())
            angle = math.radians(settings["angle_deg"])
            targets = []
            for _ in range(settings["cycles"]):
                for delta in (angle, -angle):
                    target = list(original)
                    target[-1] += delta
                    targets.append(tuple(target))
            targets.append(original)
            for target in targets:
                self.robot._check_joint_limits(target)
            self.event("release_wiggle_start", {
                "part": self.part, "settings": settings, "original_joints_rad": list(original),
            })
            print(f"RELEASE WIGGLE {self.part}: joint 7 +/-{settings['angle_deg']:g} deg, "
                  f"{settings['cycles']} cycles; restore original angle before release", flush=True)
            for target in targets:
                self.robot.move_joints(target, speed_scale=min(.15, self.args.speed_scale))
            self.robot.stationary_tcp_pose(settle_timeout_s=1.0)
            measured = tuple(self.robot._read_joint_positions())
            error = abs(measured[-1] - original[-1])
            if error > .005:
                raise RuntimeError(f"Release wiggle joint 7 restoration error {math.degrees(error):.3f} deg; jaws remain closed")
            self.event("release_wiggle_restored", {
                "original_joints_rad": list(original), "measured_joints_rad": list(measured),
                "last_joint_error_deg": math.degrees(error),
            })
            print("RELEASE WIGGLE: original joint 7 angle restored and verified", flush=True)
        except BaseException:
            self.motion_faulted = True
            self.automatic_continuation_safe = False
            # Do not command a blind restoration through an E-stop or failed
            # motion. Keep the holding state and prohibit release/next action.
            self.robot.stop()
            raise

    def place(self, settings, *, partial_release=False, use_place_cv=False):
        if not self.holding or settings is None:
            raise ValueError("Place needs a verified held part and taught place settings")
        profile = dict(self.profiles["parts"][self.part], place=settings)
        from steadyhand.vision.placement import placement_digest
        previous_place = self.profiles["parts"][self.part].get("place")
        if previous_place and placement_digest(previous_place) != placement_digest(settings):
            profile.pop("placement_board", None)
            profile.pop("place_release_tcp_m", None)
        corner_settings = getattr(self, "place_cv_settings", None) or {}
        cv_clearance = (.100 if not use_place_cv or corner_settings.get("method") != "white_board_corners_v1"
                        else float(corner_settings["reference_clearance_m"]))
        if not .060 <= cv_clearance <= .150:
            raise ValueError("Invalid placement corner hover clearance")
        hover = self.profile_pose(profile, "place", clearance=cv_clearance)
        x, y = hover.position_m[:2]
        quat = hover.quaternion_wxyz
        release = Pose((x, y, self.surface(x, y)+settings["clearance_m"]), quat)
        # Validate placement descent before transporting the held part.
        seed = self.robot._read_joint_positions()
        try:
            seed = preflight_tcp_segmented(self.robot._kinematics, seed,
                                           self.robot.get_tcp_pose(), hover, **MOTION_STEPS)
            seed = preflight_tcp_segmented(self.robot._kinematics, seed, hover, release, **MOTION_STEPS)
            preflight_tcp_segmented(self.robot._kinematics, seed, release, hover, **MOTION_STEPS)
        except IKError as exc:
            raise PlacementPreflightError(str(exc)) from exc
        self.remote_checkpoint("before_place_hover")
        self.move(hover)
        self.remote_checkpoint("place_hover")
        if use_place_cv:
            alignment = self._place_visual_align(settings)
            # Re-read the TCP after the bounded wrist alignment.  Preserve the
            # taught clearance and current orientation while using any small
            # verified XY correction.
            if alignment is not None:
                aligned = self.robot.get_tcp_pose()
                x, y = aligned.position_m[:2]
                quat = aligned.quaternion_wxyz
                hover = Pose((x, y, self.surface(x, y) + cv_clearance), quat)
                release = Pose((x, y, self.surface(x, y) + settings["clearance_m"]), quat)
        # Placement has its own independent correction, after visual alignment
        # (or saved-pose fallback), so neither pickup nor CV can erase/double it.
        hover, clearance = self._execution_target("placement", hover, settings["clearance_m"])
        x, y = hover.position_m[:2]
        release = Pose((x, y, self.surface(x, y) + clearance), hover.quaternion_wxyz)
        self.remote_checkpoint("before_place_descent")
        self.move(release, slow=True)
        self.remote_checkpoint("place_release_height")
        self.remote_checkpoint("before_place_release")
        self._capture_drop_evidence("competition_release_before")
        self._wiggle_before_release()
        self.robot.release_gripper(self.part)
        self.holding = False
        self._capture_drop_evidence("competition_release_after")
        self.remote_checkpoint("place_released")
        self.remote_checkpoint("before_place_retreat")
        self.move(hover, slow=True)
        self.remote_checkpoint("place_retreat_complete")
        print("Placement release completed; insertion/assembly is not inferred.")

    def _drop_settings_from_pose(self, pose):
        """Convert the measured drop TCP pose to the profile's board offset."""
        target = self.targets[f"task.{self.part}.place"]
        _, ux, uy, _ = self.runtime[2]
        dx = float(pose.position_m[0]) - float(target.position_m[0])
        dy = float(pose.position_m[1]) - float(target.position_m[1])
        clearance = float(pose.position_m[2]) - self.surface(
            pose.position_m[0], pose.position_m[1]
        )
        if not math.isfinite(clearance):
            raise ValueError("drop clearance must be finite")
        if clearance > .100:
            raise ValueError("Release must be at or below the return hover so retreat moves upward")
        return {
            "offset_board_xy_m": [
                dx * float(ux[0]) + dy * float(ux[1]),
                dx * float(uy[0]) + dy * float(uy[1]),
            ],
            "clearance_m": clearance,
            "yaw_deg": float(self.yaw),
        }

    def _save_drop_profile(self, pose, settings, release_result):
        """Persist the operator-confirmed release immediately after release."""
        previous = self.profiles["parts"][self.part]
        profile = dict(previous)
        if previous.get("place") != settings:
            profile.pop("place_cv", None)
        profile["place"] = settings
        profile["place_verified"] = True
        profile["place_release_tcp_m"] = list(pose.position_m)
        from steadyhand.board_relative import snapshot, make_record, record_target
        reference = snapshot(self.runtime[2])
        profile["placement_board"] = make_record(reference,
            release=record_target(reference, pose, source="measured_release_TCP"))
        profile["place_release_gripper"] = release_result
        if getattr(self, "drop_release_photo", None):
            profile["place_release_photo"] = self.drop_release_photo
        if getattr(self, "drop_evidence_photos", None):
            profile["place_evidence_photos"] = list(self.drop_evidence_photos)
        profile["place_taught_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.profiles = save_profile(_resolve(self.args.profiles), self.cfg, profile)
        return profile

    def _placement_corner_frame(self):
        rgb, path = self.frame("placement board corners")
        self.place_corner_image_path = path
        self.event("place_corner_frame", {"image": str(path)})
        return rgb

    def _move_place_corner(self, pose):
        from steadyhand.vision.board_corners import CornerVisualError
        kin_cfg = self.robot._kinematics.config
        precision = {"position_tolerance_m": .0007, "orientation_tolerance_rad": .01}
        previous = {key: kin_cfg.get(key) for key in precision}
        try:
            # Coarse-navigation IK tolerances consume much of a small visual
            # correction. Match the existing wrist fine-centering solver only
            # for this optional hover movement, including its preflight.
            for key, value in precision.items():
                kin_cfg[key] = min(value, float(kin_cfg.get(key, value)))
            self.move(pose, slow=True)
        except IKError as exc:
            # move() preflights before commanding anything and latches faults
            # if an executed movement fails. Never hide the latter as vision.
            if self.motion_faulted:
                raise
            raise CornerVisualError(f"Optional corner correction is unreachable: {exc}") from exc
        finally:
            for key, value in previous.items():
                if value is None:
                    kin_cfg.pop(key, None)
                else:
                    kin_cfg[key] = value

    def _teach_board_corner_reference(self, settings, selected_uvs=None):
        """Learn at the actual release XY, at 100 mm, without changing the pose record."""
        from steadyhand.vision.board_corners import CornerServo, draw_corners, TEACH_PROBE_M
        from steadyhand.vision.placement import placement_digest
        import cv2
        print(f"BOARD CORNER TEACHING: two {TEACH_PROBE_M * 1000:g} mm XY measurements at this hover, then return; no grip/release.", flush=True)
        servo = CornerServo(self.robot, self._move_place_corner,
                            self._placement_corner_frame, self.surface, self.event)
        anchor = self.profile_pose(self.profiles["parts"][self.part], "place")
        reference, rgb = servo.teach(selected_uvs, anchor_xy=anchor.position_m[:2])
        directory = ROOT / "calibration" / "place_templates"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.part}_{time.time_ns()}_BOARD_CORNERS.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise OSError("Could not save placement corner reference image")
        overlay = path.with_name(path.stem + "_REVIEW.png")
        if not cv2.imwrite(str(overlay), cv2.cvtColor(draw_corners(rgb, reference), cv2.COLOR_RGB2BGR)):
            raise OSError("Could not save placement corner review image")
        reference.update(enabled=True, camera=WRIST_CAMERA,
            reference_image=str(path.relative_to(ROOT)), reference_image_sha256=file_sha256(path),
            review_image=str(overlay.relative_to(ROOT)),
            reference_board=snapshot(self.runtime[2]),
            placement_sha256=placement_digest(settings),
            created_at_utc=datetime.now(timezone.utc).isoformat())
        # Reload through save_profile; pickup/physical placement values are untouched.
        profile = copy.deepcopy(self.profiles["parts"][self.part])
        profile["place_cv"] = reference
        self.profiles = save_profile(_resolve(self.args.profiles), self.cfg, profile)
        self.place_cv_settings = reference
        self._publish_live_image(rgb, overlay, "PLACEMENT BOARD CORNERS SAVED")
        print(f"PLACEMENT CORNERS SAVED: {len(reference['corners'])} observed corners. Competition uses no probe movements.\n"
              f"CORNER REVIEW: {overlay}", flush=True)
        return reference

    def _optional_place_corner_teaching(self, settings, selected_uvs=None):
        from steadyhand.vision.board_corners import CornerVisualError
        try:
            return self._teach_board_corner_reference(settings, selected_uvs)
        except CornerVisualError as exc:
            # Images may fail; physical motion failures must not be relabeled
            # as annotation trouble or hidden after a part was released.
            self.event("place_corner_teaching_unavailable", {"reason": str(exc)})
            print(f"BOARD CORNERS NOT SAVED: {exc}. Physical release calibration is still saved.\n"
                  "You can retry corner teaching here without picking up the part again.", flush=True)
            return None

    def _capture_drop_evidence(self, label):
        """Archive a named wrist frame without making camera capture a gate."""
        try:
            self.cameras.connect()
            _rgb, raw = self.frame(f"drop evidence {label}")
            archive = ROOT / "runs" / "drop_release_positions"
            archive.mkdir(parents=True, exist_ok=True)
            stamp = time.time_ns()
            destination = archive / (
                f"DROP_EVIDENCE__{self.part}__{stamp}__{label}__WRIST_A.png"
            )
            shutil.copyfile(raw, destination)
            relative = str(destination.relative_to(ROOT)).replace("\\", "/")
            pose = self.robot.get_tcp_pose()
            record = {
                "label": str(label),
                "path": relative,
                "tcp_position_m": list(pose.position_m),
                "quaternion_wxyz": list(pose.quaternion_wxyz),
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            self.drop_evidence_photos.append(record)
            # Write after every successful capture so an interrupted teach
            # still leaves a self-describing evidence set for later CV work.
            manifest = self.output / "drop_evidence_manifest.json"
            manifest.write_text(
                json.dumps({
                    "schema_version": 1,
                    "part": self.part,
                    "photos": list(self.drop_evidence_photos),
                }, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            print(f"DROP EVIDENCE PHOTO [{label}] = {destination}", flush=True)
            return relative
        except Exception as exc:
            print(
                f"DROP EVIDENCE PHOTO WARNING [{label}]: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return None

    def _capture_drop_release_photo(self):
        """Keep the legacy release-photo field while using the evidence archive."""
        relative = self._capture_drop_evidence("release_before")
        self.drop_release_photo = relative
        return relative

    def _teaching_pickup(self, profile):
        """Keep a dry-run IK rejection inspectable without restarting teaching.

        Only the pre-motion pickup check is recoverable here. A failure after
        descent starts, or any hardware failure, still escapes to safe shutdown.
        """
        while True:
            try:
                self.grab(profile["grasp_clearance_m"], allow_unverified=True)
                return True
            except PickupPreflightError as exc:
                if not self._pickup_plan_retry(exc):
                    return False

    def _pickup_plan_retry(self, exc):
        print(f"PICKUP PLAN BLOCKED: {exc}\n"
              "No descent or grip was commanded. The arm remains at the pickup hover.\n"
              "Inspect before retrying; the saved calibration has not been changed.", flush=True)
        if self.remote_safe:
            self.inspection_image("pickup preflight blocked")
        while True:
            command = input("PICKUP PLAN [retry / image / abort]: ").strip().lower()
            if command == "retry":
                return True
            if command == "image":
                self.inspection_image("pickup preflight inspection")
            elif command in ("abort", "stop", "exit", "q"):
                return False
            else:
                print("Use retry to recheck IK, image to inspect, or abort to finish.", flush=True)

    def _plan_drop_retreat(self, release_pose):
        from steadyhand.executor import cartesian_waypoints
        retreat = Pose((*release_pose.position_m[:2],
            self.surface(*release_pose.position_m[:2]) + .100), release_pose.quaternion_wxyz)
        original_seed = tuple(self.robot._read_joint_positions())
        errors = []
        for step in (.020, .005, .002):
            seed = original_seed
            plan = []
            try:
                for waypoint in cartesian_waypoints(release_pose, retreat,
                        max_translation_step_m=step, max_orientation_step_rad=.08):
                    seed = tuple(float(v) for v in self.robot._kinematics.solve(waypoint, seed))
                    self.robot._check_joint_limits(seed)
                    plan.append((waypoint, seed))
            except IKError as exc:
                errors.append(str(exc))
                continue
            self.event("drop_retreat_plan", {
                "translation_step_mm": step * 1000, "waypoints": len(plan),
                "release_tcp": list(release_pose.position_m),
                "retreat_tcp": list(retreat.position_m),
                "joint_path_rad": [list(q) for _, q in plan],
            })
            print(f"RELEASE RETREAT READY: {len(plan)} solved waypoints "
                  f"({step * 1000:g} mm spacing); release orientation preserved", flush=True)
            return retreat, plan
        raise IKError("No vertical retreat could be solved at this orientation, including "
                      "5 mm and 2 mm waypoints. No release commanded. Try a small up N "
                      "or yaw N adjustment, then release again. Last solver error: " + errors[-1])

    def _execute_drop_retreat(self, plan):
        # Execute the exact joint path solved before opening the jaws; solving
        # again from noisy telemetry could select a different IK branch.
        try:
            for waypoint, joints in plan:
                if self.remote_safe:
                    self.remote_waypoint(waypoint)
                self.robot.move_joints(joints, speed_scale=min(.42, self.args.speed_scale))
        except BaseException:
            self.motion_faulted = True
            self.automatic_continuation_safe = False
            raise

    def teach_drop(self, part, profile):
        """Teach a physical drop position using an existing pickup profile.

        The pickup uses the saved arm hover and grasp calibration without
        wrist centering.  The part is then carried to its task drop target,
        descended 40 mm below the 100 mm hover, and adjusted interactively.
        ``release`` records the measured TCP and saves the profile before any
        retreat motion is attempted.
        """
        if profile is None:
            raise ValueError("No saved pickup profile. Teach and verify this part first.")
        _check_ready(profile, self.cfg, "pick", competition=True, no_cv=True)
        self.part, self.action = part, "drop_calibrate"
        self.begin_part(part, profile, competition=True, no_cv=True)
        self.yaw = float((profile.get("place") or {}).get("yaw_deg", 0.0))
        self.history = []
        self.drop_evidence_photos = []
        self.drop_release_photo = None
        self.grasp_verified = True
        if not self._teaching_pickup(profile):
            return 1
        self.grasp_verified = True
        self.remote_checkpoint("drop_pickup_complete")

        hover = self.profile_pose(profile, "place")
        def approach_drop():
            try:
                self.remote_checkpoint("before_drop_hover")
                self.move(hover)
                self.remote_checkpoint("drop_hover_100mm")
                self._capture_drop_evidence("nominal_hover_100mm")
                current = Pose((hover.position_m[0], hover.position_m[1],
                    self.surface(*hover.position_m[:2]) + .060), hover.quaternion_wxyz)
                self.remote_checkpoint("before_drop_initial_descent")
                self.move(current, slow=True)
                self._capture_drop_evidence("initial_40mm_below_hover")
                print(f"DROP CALIBRATION READY: {part}; 40 mm below hover.", flush=True)
            except IKError as exc:
                if self.motion_faulted:
                    raise
                print(f"NEXT DROP SEGMENT BLOCKED BEFORE MOTION: {exc}\n"
                      "Part remains held. Use small directional adjustments, target to retry, "
                      "image to inspect, or return to put it back. Release only at the intended destination.", flush=True)
        approach_drop()
        print(
            "Commands: forward/back/left/right N (mm) | step N | yaw N (deg) | "
            "down N (mm) | up N | undo | target | image | status | release | return | abort\n"
            "release saves the physical position immediately, then lifts vertically. "
            + ("Placement CV teaching is disabled for this part." if getattr(self.args, "skip_place_cv_teaching", False)
               else "Board corners are learned automatically at the hover."),
            flush=True,
        )
        while True:
            raw = input(f"drop {part}> ").strip().lower().split()
            if not raw:
                continue
            command = raw[0]
            if command in ("abort", "exit", "q"):
                return 3 if self.holding else 1
            if command == "image":
                self.inspection_image("drop inspection")
                continue
            if command == "target":
                approach_drop()
                continue
            if command == "release":
                try:
                    self.remote_checkpoint("before_drop_release")
                    release_pose = self.robot.get_tcp_pose()
                    settings = self._drop_settings_from_pose(release_pose)
                    retreat, retreat_plan = self._plan_drop_retreat(release_pose)
                    release_photo = self._capture_drop_release_photo()
                    release_result = self.robot.release_gripper(self.part)
                    self.holding = False
                    profile = self._save_drop_profile(release_pose, settings, release_result)
                    self._capture_drop_evidence("release_after")
                    self.event("drop_release", {
                        "release_tcp": list(release_pose.position_m),
                        "place": settings,
                        "release_photo": release_photo,
                        "gripper_release": release_result,
                    })
                    profile = self._save_drop_profile(
                        release_pose, settings, release_result
                    )
                    self.remote_checkpoint("drop_released")
                    self.status = "drop_profile_saved"
                    print("DROP PROFILE SAVED AFTER RELEASE", flush=True)
                    print(json.dumps({
                        "part": part,
                        "place": settings,
                        "place_release_tcp_m": list(release_pose.position_m),
                        "place_release_photo": release_photo,
                        "place_verified": profile["place_verified"],
                        "gripper_release": release_result,
                    }, indent=2, default=str), flush=True)
                    # Lift over the actual release, including manual XY/yaw
                    # adjustments. This is the exact reference used in runs.
                    self.remote_checkpoint("before_drop_retreat")
                    self._execute_drop_retreat(retreat_plan)
                    self.remote_checkpoint("drop_retreat_complete")
                    self._capture_drop_evidence("retreat_after_release")
                    if not getattr(self.args, "skip_place_cv_teaching", False):
                        self._finish_place_corner_teaching(settings, retreat)
                    return 0
                except (RuntimeError, ValueError) as exc:
                    if not self.holding or self.motion_faulted:
                        raise
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    print(f"RELEASE BLOCKED: {exc}; part remains held.", flush=True)
                    continue
            if command == "return":
                try:
                    current = self.robot.get_tcp_pose()
                    self.move(Pose((current.position_m[0], current.position_m[1],
                                    self.surface(*current.position_m[:2]) + .100),
                                   current.quaternion_wxyz), slow=True)
                    self.move(self.successful_pickup_pose or self.coarse, slow=True)
                    self.return_part(profile["grasp_clearance_m"])
                    self.status = "drop_cancelled_returned"
                    return 1
                except (RuntimeError, ValueError) as exc:
                    print(f"RETURN BLOCKED: {exc}; part remains held.", flush=True)
                    continue
            if command == "status":
                pose = self.robot.get_tcp_pose()
                print(json.dumps({
                    "tcp_position_m": list(pose.position_m),
                    "clearance_mm": (pose.position_m[2] - self.surface(*pose.position_m[:2])) * 1000,
                    "yaw_deg": self.yaw,
                    "holding": self.holding,
                }), flush=True)
                continue
            if command == "step" and len(raw) == 2:
                try:
                    self.step_mm = _number(raw[1], .001, math.inf)
                    print(f"STEP = {self.step_mm:g} mm", flush=True)
                except ValueError as exc:
                    print(f"COMMAND BLOCKED: {exc}", flush=True)
                continue
            if command == "undo":
                if not self.history:
                    print("No drop adjustment to undo", flush=True)
                    continue
                previous, old_yaw = self.history.pop()
                try:
                    self.move(previous, slow=True)
                    self.yaw = old_yaw
                except (RuntimeError, ValueError) as exc:
                    print(f"UNDO BLOCKED: {exc}", flush=True)
                continue
            try:
                current = self.robot.get_tcp_pose()
                old_yaw = self.yaw
                x, y, z = current.position_m
                if command in ("forward", "back", "left", "right"):
                    amount = self.step_mm if len(raw) == 1 else _number(raw[1], .001, math.inf)
                    amount /= 1000.0
                    dx, dy = {
                        "forward": (amount, 0.0), "back": (-amount, 0.0),
                        "left": (0.0, amount), "right": (0.0, -amount),
                    }[command]
                    x, y = x + dx, y + dy
                    clearance = z - self.surface(*current.position_m[:2])
                    target = Pose(
                        (x, y, self.surface(x, y) + clearance),
                        current.quaternion_wxyz,
                    )
                elif command in ("down", "up") and len(raw) == 2:
                    amount = _number(raw[1], .001, math.inf) / 1000.0
                    target_z = z + amount if command == "up" else z - amount
                    target = Pose((x, y, target_z), current.quaternion_wxyz)
                elif command == "yaw" and len(raw) == 2:
                    requested_yaw = _number(raw[1])
                    target = Pose(
                        current.position_m,
                        _yaw_pose(self.runtime[3], requested_yaw).quaternion_wxyz,
                    )
                else:
                    print("Unknown command; use help, directions, down N, yaw N, release, return, or abort.", flush=True)
                    continue
                self.move(target, slow=True)
                if command == "yaw":
                    self.yaw = requested_yaw
                self.history.append((current, old_yaw))
                self._capture_drop_evidence(
                    f"adjust_{command}_{len(self.drop_evidence_photos):03d}"
                )
                self.remote_checkpoint(f"drop_adjustment_{command}")
            except (RuntimeError, ValueError) as exc:
                print(f"COMMAND BLOCKED: {type(exc).__name__}: {exc}", flush=True)
                continue

    def teach(self, part, old=None):
        self.begin_part(part, old, initial_yaw=(old or {}).get("yaw_deg"))
        grasp = old.get("grasp_clearance_m") if old else None
        place = old.get("place") if old else None
        place_changed = False
        feature_selected = False
        self.gripper_open_fraction = old.get("gripper_open_fraction") if old else None
        self.grasp_verified = False
        self.last_grasp_clearance = grasp
        result = None
        print("Adjust the hover manually; center runs only when requested. "
              "Existing visual calibration is kept unless you use feature, then save.", flush=True)
        print("Commands:")
        print("  forward/back/left/right [N] | step N | yaw N | undo")
        print("  feature (new first annotation) | goal (second annotation of SAME feature)")
        print("  image | center (verify visual alignment) | jaw N (0..100% open)")
        print("  open N / close N (incremental percentage points) | depth N (mm below hover)")
        print("  grab [manual] | confirm yes/no | return | place-config X Y DEPTH YAW | save | abort")
        while True:
            raw = input(f"{part}> ").strip().lower().split()
            if not raw:
                continue

            command = raw[0]
            if command in ("abort", "exit", "q"):
                return 3 if self.holding else 1
            try:
                if command == "save":
                    if self.holding:
                        print("Type return to put the test part back before saving.")
                        continue
                    if not old and not feature_selected:
                        print("SAVE BLOCKED: teach this new part's visual reference with feature first.")
                        continue
                    if not self.alignment_verified or (feature_selected and self.goal is None):
                        print("SAVE BLOCKED: verify alignment with center, or use grab manual at the supervised grasp pose.")
                        continue
                    if grasp is None or not self.grasp_verified:
                        print("SAVE BLOCKED: teach depth, grab, and confirm the physical grasp first.")
                        continue
                    visual_fields = {}
                    if feature_selected:
                        try:
                            rgb, path = self.frame("final taught pose")
                            feature, final_score = self.tracker.locate(rgb)
                        except (ValueError, RuntimeError) as exc:
                            print(
                                "SAVE BLOCKED: the original feature was not confidently "
                                f"visible in the final image ({exc}); use image/feature/goal and retry.",
                                flush=True,
                            )
                            continue
                        _write_overlay(
                            path.with_name(path.stem + "_final.png"),
                            rgb, feature, self.goal, label=part,
                        )
                        # Replace visual calibration only after an explicit
                        # feature selection and save, never by opening teaching
                        # or by loading the old tracker for an explicit center.
                        template, anchor = _crop_template(self.reference_rgb, self.reference_feature)
                        import cv2
                        directory = default_template_dir(ROOT)
                        directory.mkdir(parents=True, exist_ok=True)
                        template_path = directory / f"{part}_{time.time_ns()}.png"
                        if not cv2.imwrite(str(template_path), cv2.cvtColor(template, cv2.COLOR_RGB2BGR)):
                            raise RuntimeError("Failed to save wrist template")
                        visual_fields = {
                            "feature_uv": list(self.reference_feature), "goal_uv": list(self.goal),
                            "goal_source": (
                                "operator_confirmed_current_pose"
                                if self.manual_alignment_override
                                else (
                                    "wrist_image_center"
                                    if self.goal_match is None
                                    else "same_feature_second_annotation"
                                )
                            ),
                            "goal_match_uv": list(self.goal_match) if self.goal_match is not None else None,
                            "goal_match_score": self.goal_match_score,
                            "goal_click_match_error_px": self.goal_match_error_px,
                            "reference_match_score": self.reference_match_score,
                            "feature_tracking_mode": self.feature_tracking_mode,
                            "final_match_uv": list(feature), "final_match_score": float(final_score),
                            "image_shape": list(self.reference_rgb.shape[:2]),
                            "template": {"path": str(template_path.relative_to(ROOT)), "sha256": file_sha256(template_path), "template_uv": list(anchor)},
                        }
                    else:
                        # The retained image still belongs to its original
                        # approach pose. Reproject that pose when recording a
                        # new physical grasp; don't attach it to today's hover.
                        self.reference_pose = self.profile_pose(old, "pick", no_cv=False)
                    cal = load_board_calibration(ROOT / "calibration/vega_board_manual.json", self.cfg)
                    profile = {**(old or {}), **visual_fields,
                        "part": part, "working_arm": WORKING_ARM, "tcp_frame": TCP_FRAME, "wrist_camera": WRIST_CAMERA,
                        "calibration_sha256": cal["sha256"],
                        "coarse_xy_m": list((self.reference_pose or self.coarse).position_m[:2]),
                        "hover_clearance_m": .100, "grasp_clearance_m": grasp, "yaw_deg": self.yaw,
                        "gripper_open_fraction": self.gripper_open_fraction,
                        "place": place, "grasp_verified": self.grasp_verified,
                        "place_verified": False if place_changed else bool((old or {}).get("place_verified", False)),
                        "localization_result": result, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    profile["pickup_board"] = self.pickup_record()
                    if place_changed:
                        # A new physical release pose invalidates any image
                        # reference taught for the previous pose.  Keep the
                        # pickup fields, but require menu 14 to teach a new
                        # placement reference before CV can be enabled again.
                        profile.pop("place_cv", None)
                        profile.pop("placement_board", None)
                        profile.pop("place_release_tcp_m", None)
                    self.profiles = save_profile(_resolve(self.args.profiles), self.cfg, profile)
                    handoff = self.output / f"{part}_handoff.json"
                    handoff.write_text(json.dumps(profile, indent=2, default=str)+"\n")
                    self.status = "profile_saved"
                    print("PROFILE SAVED; reused automatically:", handoff, flush=True)
                    print(json.dumps(profile, indent=2, default=str), flush=True)
                    return 0
                if command == "grab":
                    manual_override = len(raw) == 2 and raw[1] in (
                        "manual", "skip-center", "skip-centering"
                    )
                    try:
                        result = self.grab(grasp, allow_unverified=manual_override)
                    except (ValueError, RuntimeError) as exc:
                        if self.holding:
                            print(
                                "GRASP UNCERTAIN; the robot is left in holding state. "
                                f"Inspect the jaws and type return before anything else ({exc})",
                                flush=True,
                            )
                            continue
                        message = str(exc)
                        if "center" in message.lower() or "alignment" in message.lower():
                            print(
                                "GRAB BLOCKED: same-feature alignment is not verified. "
                                "Run 'center', or if the jaws are already manually positioned "
                                "correctly, type 'grab manual'. No grasp motion was issued.",
                                flush=True,
                            )
                        elif "depth" in message.lower():
                            print(
                                "GRAB BLOCKED: set a safe descent first with 'depth N' "
                                "(millimetres below the 100 mm hover), then retry 'grab'.",
                                flush=True,
                            )
                        else:
                            print(
                                f"GRAB NOT COMPLETED: {type(exc).__name__}: {exc} "
                                "No further automatic motion was issued; inspect and retry.",
                                flush=True,
                            )
                        continue
                    answer = input(
                        "Did the selected part physically lift and remain held? Type yes or no: "
                    ).strip().lower()
                    self.grasp_verified = answer in ("y", "yes")
                    if self.grasp_verified:
                        print("PHYSICAL GRASP CONFIRMED; type return before save.", flush=True)
                    else:
                        print("GRASP NOT CONFIRMED; keep holding state and type return for inspection.", flush=True)
                    continue
                if command == "confirm" and len(raw) == 2:
                    self.grasp_verified = raw[1] in ("y", "yes", "true", "1")
                    print("GRASP CONFIRMED =", self.grasp_verified, flush=True)
                    continue
                if command == "return":
                    self.return_part(grasp)
                    continue
                if self.holding:
                    print("A part is held. Use return or abort before adjusting the pose.")
                    continue
                if command in ("image", "capture", "snapshot"):
                    rgb, path = self.frame("manual wrist capture")
                    if self.tracker is None:
                        print("No active feature; use feature to teach one, or center to reuse the saved feature.")
                        continue
                    try:
                        feature, score = self.tracker.locate(rgb)
                        _write_overlay(path.with_name(path.stem + "_tracked.png"), rgb, feature, self.goal, label=part)
                        print("TRACKED FEATURE =", tuple(round(v, 1) for v in feature), f"score={score:.3f}", flush=True)
                    except (ValueError, RuntimeError) as exc:
                        print("FEATURE NOT VERIFIED IN THIS IMAGE:", exc, flush=True)
                    continue
                if command in ("feature", "reference"):
                    self.teach_feature()
                    feature_selected = True
                    self.no_cv_mode = False
                    self.remote_checkpoint("feature_selected")
                    continue
                if command in ("goal", "target"):
                    self.teach_goal_feature()
                    continue
                if command in ("center", "verify", "localize"):
                    try:
                        # Loading the old image/goal is also operator initiated.
                        # A new feature always takes precedence over the old one.
                        if self.tracker is None and old and not feature_selected:
                            self._load_saved_feature(old)
                        if self.tracker is None:
                            print("CENTER NOT STARTED: use feature to teach a visible reference feature first.")
                            continue
                        self.no_cv_mode = False
                        result = self.localize()
                    except RuntimeError as exc:
                        message = str(exc).lower()
                        if any(word in message for word in (
                            "feature", "lost", "ambiguous", "tracking",
                            "centering stalled", "pixel error increased",
                            "tcp missed servo waypoint", "feature did not return",
                            "ill-conditioned", "feature barely moved",
                        )):
                            self.alignment_fallback_used = True
                            print(
                                "VISUAL ALIGNMENT STOPPED: no further motion was issued; "
                                "capture another image or teach a different feature. "
                                "If the TCP is already over the part, use 'grab manual'; "
                                "competition pickup does not require centering.",
                                flush=True,
                            )
                            continue
                        raise
                    self.remote_checkpoint("wrist_alignment_result")
                    continue
                if command == "status":
                    status = {
                        "stage": self.status,
                        "feature_uv": self.reference_feature,
                        "goal_uv": self.goal,
                        "alignment_verified": self.alignment_verified,
                        "grasp_verified": self.grasp_verified,
                        "grasp_clearance_m": grasp,
                        "gripper_open_fraction": self.gripper_open_fraction,
                        "step_mm": self.step_mm,
                        "holding": self.holding,
                    }
                    print(json.dumps(status, default=str), flush=True)
                    continue
                if command in ("help", "h", "?"):
                    print("forward/back/left/right [N] | step N | yaw N | undo", flush=True)
                    print("feature | goal | image | center | jaw N | open N | close N", flush=True)
                    print("depth N | grab [manual] | confirm yes/no | return | save | abort", flush=True)
                    continue
                try:
                    if command == "step" and len(raw) == 2:
                        self.step_mm = _number(raw[1], .001, math.inf)
                        print(f"STEP = {self.step_mm:g} mm", flush=True)
                        continue
                    if command in ("jaw", "gripper") and len(raw) == 1:
                        measured = self._read_gripper_fraction()
                        print(f"GRIPPER OPENING = {measured * 100:.1f}%", flush=True)
                        continue
                    if command in ("jaw", "gripper") and len(raw) == 2:
                        self._set_gripper_fraction(_number(raw[1], 0., 100.) / 100.)
                        continue
                    if command in ("open", "close"):
                        amount = self.step_mm if len(raw) == 1 else _number(raw[1], .5, 100.)
                        current = self._read_gripper_fraction()
                        delta = amount / 100.0 * (1.0 if command == "open" else -1.0)
                        self._set_gripper_fraction(max(0.0, min(1.0, current + delta)))
                        continue
                    if command == "depth" and len(raw) == 2:
                        depth = _number(raw[1], 0., math.inf)
                        grasp = .100-depth/1000
                        self.last_grasp_clearance = grasp
                        self.grasp_verified = False
                        print(f"Grasp clearance above calibrated surface: {grasp*1000:.1f} mm", flush=True)
                        continue
                    if command == "place-config" and len(raw) == 5:
                        dx, dy = (_number(v)/1000 for v in raw[1:3])
                        depth = _number(raw[3], 0., math.inf)
                        yaw = _number(raw[4])
                        place = {"offset_board_xy_m": [dx, dy], "clearance_m": .100-depth/1000, "yaw_deg": yaw}
                        place_changed = True
                        print("Place settings recorded; task test must validate them before competition.")
                        continue
                    current = self.robot.get_tcp_pose()
                    if command == "undo" and len(raw) == 1:
                        if not self.history:
                            print("No adjustment to undo")
                            continue
                        target, old_yaw = self.history.pop()
                        self.move(target, slow=True)
                        self.yaw = old_yaw
                        self._invalidate_alignment("undo")
                        self.frame("after undo")
                        continue
                    if command == "yaw" and len(raw) == 2:
                        yaw = _number(raw[1])
                        target = Pose(current.position_m, _yaw_pose(self.runtime[3], yaw).quaternion_wxyz)
                    elif command in ("forward", "back", "left", "right") and len(raw) in (1, 2):
                        amount = self.step_mm if len(raw) == 1 else _number(raw[1], .001, math.inf)
                        amount /= 1000.0
                        dx, dy = {"forward": (amount, 0), "back": (-amount, 0), "left": (0, amount), "right": (0, -amount)}[command]
                        x, y = current.position_m[0]+dx, current.position_m[1]+dy
                        target = Pose((x, y, self.surface(x, y)+.100), current.quaternion_wxyz)
                        yaw = self.yaw
                    else:
                        print("Unknown command; use help, feature, goal, center, step N, jaw N, or a direction.")
                        continue
                except ValueError as exc:
                    print(exc)
                    continue
                self.move(target, slow=True)
                self.history.append((current, self.yaw))
                self.yaw = yaw
                self._invalidate_alignment(f"{command} adjustment")
                self.frame("after hover adjustment")
                self.remote_checkpoint(f"manual_adjustment_{command}")

            except (ValueError, RuntimeError) as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                print(
                    f"COMMAND BLOCKED: {type(exc).__name__}: {exc}",
                    "No further automatic motion was issued; inspect the current pose and retry.",
                    flush=True,
                )
                if self.holding:
                    print("A part may be held; type return or abort before another command.", flush=True)
                else:
                    print("Calibration remains active; type help for the next valid step.", flush=True)
                continue

    def teach_place_cv(self, part, profile):
        """Refresh corners above a saved release without picking up any part."""
        if not profile or not profile.get("place") or not profile.get("place_verified"):
            raise ValueError("Teach and verify the physical place position first")
        if self.holding:
            raise RuntimeError("Corner refresh requires an empty gripper")
        self.part, self.action = part, "place_cv"
        hover = self.profile_pose(profile, "place")
        print("REFRESH PLACEMENT CORNERS: saved 100 mm release hover; no pickup, descent or release.", flush=True)
        self.move(hover)
        return self._finish_place_corner_teaching(profile["place"], hover)

    def _finish_place_corner_teaching(self, settings, hover):
        """One calibration flow; a visual rejection never repeats the pickup."""
        selected_uvs = None
        while True:
            if self._optional_place_corner_teaching(settings, selected_uvs):
                self.status = "place_corner_profile_saved"
                return 0
            # A visual-only rejection can leave a completed 6 mm hover probe.
            # Restore the saved view *before* asking for image coordinates.
            self.move(hover)
            self.inspection_image("placement board corner selection")
            selected_uvs = None
            while True:
                raw = input("CORNERS [retry / image / select u v [u v ...] / skip]: ").strip().lower().split()
                if not raw or raw[0] in ("skip", "abort", "exit", "q"):
                    return 1
                if raw[0] == "image":
                    self.inspection_image("placement board corner inspection")
                elif raw[0] == "retry":
                    break
                elif raw[0] == "select":
                    try:
                        values = [float(v) for v in raw[1:]]
                        if len(values) not in (2, 4, 6, 8) or not all(math.isfinite(v) for v in values):
                            raise ValueError("Use select u v for each of 1–4 visible outer board corners")
                        selected_uvs = list(zip(values[::2], values[1::2]))
                        break
                    except ValueError as exc:
                        print(exc, flush=True)
                else:
                    print("Use retry, image, select u v [u v ...], or skip.", flush=True)

    def recover_action(self, reason):
        """Dispose of a possible hold only from a freshly verified stopped pose.

        No E-stop is cleared and no stale motion command is replayed. A failed
        gate leaves the holding/fault state latched for the next verification.
        """
        self.automatic_continuation_safe = False
        self.recovery_state_verified = False
        self.last_error = str(reason)
        current = self.robot.stationary_tcp_pose()
        self.event("recovery_stationary", {"tcp_position_m": current.position_m,
                                          "reason": str(reason), "holding": self.holding})
        # A prior missed target may be abandoned only after the adapter has
        # independently checked fresh state, joint limits, velocity and E-stop.
        self.motion_faulted = False
        if self.holding:
            from steadyhand.board_relative import local_xy
            reference = snapshot(self.runtime[2])
            x, y, z = current.position_m
            if max(abs(v) for v in local_xy(reference, (x, y))) > reference["board_size_m"] / 2:
                raise RuntimeError("Recovery release blocked: TCP is outside the board")
            clearance = z - self.surface(x, y)
            if not -.005 <= clearance <= .150:
                raise RuntimeError("Recovery release blocked: current height is outside the local board envelope")
            current_axis = [row[2] for row in quaternion_to_matrix(current.quaternion_wxyz)]
            ready_axis = [row[2] for row in quaternion_to_matrix(self.runtime[3].quaternion_wxyz)]
            tilt = math.acos(max(-1., min(1., sum(a*b for a, b in zip(current_axis, ready_axis)))))
            if tilt > .20:
                raise RuntimeError("Recovery release blocked: TCP tilt is not the taught downward orientation")
            release = Pose((x, y, self.surface(x, y) + .040), current.quaternion_wxyz)
            retreat = Pose((x, y, self.surface(x, y) + .100), current.quaternion_wxyz)
            seed = self.robot._read_joint_positions()
            seed = preflight_tcp_segmented(self.robot._kinematics, seed, current, release, **MOTION_STEPS)
            preflight_tcp_segmented(self.robot._kinematics, seed, release, retreat, **MOTION_STEPS)
            print("AUTOMATIC RECOVERY: vertical 40 mm release, small jaw opening, then hover.", flush=True)
            self.move(release, slow=True)
            measured = self.robot.stationary_tcp_pose()
            if (math.dist(measured.position_m, release.position_m) > .005 or
                    quaternion_angle(measured.quaternion_wxyz, release.quaternion_wxyz) > .03):
                self.motion_faulted = True
                raise RuntimeError("Recovery release waypoint was not reached; jaws remain unchanged")
            self.robot.connect_gripper()
            self.recovery_release_attempted = True
            result = self.robot.release_gripper(self.part)
            if not isinstance(result, dict):
                raise RuntimeError("Recovery release has no measured gripper result")
            before, target = float(result["from_fraction"]), float(result["to_fraction"])
            if not (math.isfinite(before) and math.isfinite(target) and
                    0 <= before < target <= 1 and .001 <= target - before <= .050001):
                raise RuntimeError("Recovery release did not request a bounded small opening")
            actual = float(self.robot.gripper_position())
            if not math.isfinite(actual) or abs(actual - target) > .01 or actual <= before + .001:
                raise RuntimeError("Recovery jaw opening was not confirmed; possible hold retained")
            self.holding = False
            self.recovery_release_attempted = False
            self.event("recovery_release", {"tcp_position_m": measured.position_m,
                "gripper_release": result, "measured_fraction": actual})
            self.move(retreat, slow=True)
        self.robot.stationary_tcp_pose()
        self.recovery_state_verified = True
        self.automatic_continuation_safe = True
        self.status = "pick_complete_recovery_released" if self.pickup_completed else "failed_recovered_empty"
        self.event("recovery_complete", {"status": self.status,
            "pickup_completed": self.pickup_completed, "holding_may_be_true": self.holding})
        return 0 if self.pickup_completed else 2

    def test(self, part, action, *, competition=False, no_cv=False, place_cv=False):
        try:
            return self._test_action(part, action, competition=competition,
                                     no_cv=no_cv, place_cv=place_cv)
        except (RuntimeError, ValueError, OSError) as exc:
            if getattr(self.args, "mode", None) != "test":
                raise
            try:
                return self.recover_action(exc)
            except (RuntimeError, ValueError, OSError, KeyError, TypeError) as recovery_exc:
                self.automatic_continuation_safe = False
                self.recovery_blocked = True
                self.last_error = f"{exc}; recovery blocked: {recovery_exc}"
                print(f"RECOVERY BLOCKED: {recovery_exc}. No blind restart or E-stop reset.", flush=True)
                raise exc from recovery_exc

    def _test_action(self, part, action, *, competition=False, no_cv=False, place_cv=False):
        self.part, self.action = part, action
        self.status = f"running:{part}.{action}"
        profile = self.profiles.get("parts", {}).get(part)
        _check_ready(profile, self.cfg, action, competition=competition, no_cv=no_cv)
        self.place_cv_settings = profile.get("place_cv") if place_cv else None
        self.begin_part(part, profile, competition=competition, no_cv=no_cv)
        if action == "localize":
            return 0 if self.alignment_verified else 1
        manual = False
        if not competition:
            if not self.alignment_verified and not no_cv:
                print("CV alignment was not verified. Inspect the current hover; "
                      "'grab manual' explicitly approves it, 'center' retries localization.", flush=True)
            while True:
                choice = input("grab / grab manual / saved / center / image / abort: ").strip().lower()
                if choice == "image":
                    self.frame("supervised pickup review")
                    continue
                if choice == "center":
                    self.retry_saved_alignment(profile)
                    continue
                if choice == "saved":
                    self.use_recorded_grasp()
                    continue
                if choice == "grab manual":
                    manual = True
                    break
                if choice == "grab":
                    if not self.no_cv_mode and not self.alignment_verified:
                        print("Use 'grab manual' to approve this unverified hover, or center/image/abort.")
                        continue
                    break
                if choice in ("abort", "stop", "exit", "q"):
                    self.status = "cancelled_before_grip"
                    return 1
                print("Choose grab, grab manual, saved, center, image, or abort.", flush=True)
        _, pickup_clearance = self._execution_target(
            "pickup", self.robot.get_tcp_pose(), profile["grasp_clearance_m"]
        )
        try:
            while True:
                try:
                    self.grab(
                        pickup_clearance,
                        allow_unverified=bool(
                            self.no_cv_mode or manual or (competition and self.alignment_fallback_used)
                        ),
                    )
                    break
                except PickupPreflightError as exc:
                    if competition:
                        raise
                    if not self._pickup_plan_retry(exc):
                        self.status = "cancelled_before_grip"
                        return 1
        except (RuntimeError, ValueError) as exc:
            if competition and self.holding and isinstance(exc, GripNotVerifiedError) and not self.motion_faulted:
                # The grasp routine marks holding before contact so an
                # uncertain result is never silently retried.  Make the
                # competition path self-cleaning when a safe return is still
                # possible; otherwise the run summary retains the hard-stop
                # holding flag for inspection.
                print(
                    f"COMPETITION GRASP UNCERTAIN: {exc}; attempting automatic return.",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    self.return_part(
                        pickup_clearance, partial_release=competition
                    )
                except Exception as return_exc:
                    self.last_error = f"{type(return_exc).__name__}: {return_exc}"
                    print(
                        "AUTOMATIC RETURN FAILED; holding state is preserved in the run summary.",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise
                self.status = "pick_failed_returned"
                self.automatic_continuation_safe = True
                return 2
            raise
        if not competition:
            confirmed = input(
                "Did the selected part physically lift and remain held? Type yes or no: "
            ).strip().lower() in ("y", "yes")
            if not confirmed:
                print("Pick not confirmed; type return to release it for inspection.", flush=True)
                if input("return / exit: ").strip().lower() == "return":
                    self.return_part(
                        pickup_clearance, partial_release=competition
                    )
                self.status = "pick_not_confirmed"
                return 2
        profile = dict(profile, grasp_verified=True)
        self.grasp_verified = True
        if not competition:
            self.profiles = save_profile(_resolve(self.args.profiles), self.cfg, profile)
        if action == "pick":
            self.status = "pick_complete_holding"
            if competition:
                print(
                    "COMPETITION PICK COMPLETE; automatically returning the part to its source.",
                    flush=True,
                )
                try:
                    self.return_part(
                        pickup_clearance, partial_release=competition
                    )
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    print(
                        "AUTOMATIC RETURN FAILED; holding state is preserved in the run summary.",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise
                self.status = "pick_complete_returned"
                return 0
            print("Pick complete. Part is held; next actions are blocked until it is returned.")
            if input("Type return to put it back at source, or exit to stop holding: ").strip() == "return":
                self.return_part(pickup_clearance)
                self.status = "pick_complete_returned"
                return 0
            return 3
        if not competition and input("Type place to test the taught transfer/descent/release: ").strip() != "place":
            return 3
        try:
            self.place(profile["place"], partial_release=competition, use_place_cv=place_cv)
        except PlacementPreflightError as exc:
            if not competition or self.motion_faulted:
                raise
            print(f"PLACE PLAN BLOCKED: {exc}. Pickup succeeded; returning to source and skipping placement.", flush=True)
            self.return_part(pickup_clearance, partial_release=True)
            self.status = "pick_complete_place_blocked_returned"
            self.automatic_continuation_safe = True
            self.last_error = str(exc)
            return 0
        self.status = "completed"
        if not competition:
            if input("Was placement correct? Type yes to enable it for competition: ").strip().lower() == "yes":
                profile["place_verified"] = True
                self.profiles = save_profile(_resolve(self.args.profiles), self.cfg, profile)
                self.status = "completed_place_verified"
            else:
                self.status = "completed_place_unverified"
        return 0


def _number(value, low=-math.inf, high=math.inf):
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"Value must be finite and in {low}..{high}")
    return result


def recover_competition_checkpoint(part, action, previous_output, output, *, speed_scale):
    """Reconnect for measured recovery, without camera-clear, ready or centering."""
    from tools.vega_competition_pipeline import _load_runtime
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    prior = json.loads((Path(previous_output) / "run_summary.json").read_text())
    if prior.get("part") != part or prior.get("action") != action:
        raise ValueError("Checkpoint summary does not identify this part/action; cannot infer a recovery")
    cfg = load_bundle("vega")["robot"]
    cfg["allow_robot_init_head_motion"] = True
    cfg["auto_clear_software_estop_on_connect"] = False
    cfg["gripper"] = {**cfg.get("gripper", {}), "require_cached_calibration": True}
    args = SimpleNamespace(mode="test", competition=True, remote_safe=False,
                           speed_scale=speed_scale, execution_offsets=None,
                           profiles="calibration/wrist_part_profiles.json")
    session = PartSession(args, output, cfg, {})
    session.part, session.action = part, action
    session.holding = prior.get("holding_may_be_true") is not False
    session.pickup_completed = prior.get("pickup_completed") is True
    session.recovery_release_attempted = bool(prior.get("recovery_release_attempted"))
    session.status = "checkpoint_recovery"
    try:
        session.runtime = _load_runtime()
        reference = prior.get("board_reference")
        if session.holding:
            if prior.get("recovery_release_attempted"):
                raise ValueError("A recovery opening was already attempted without confirmation; refusing to widen jaws again")
            if not isinstance(reference, dict):
                raise ValueError("Held-part recovery requires the interrupted run's board reference")
            plane = session.runtime[2][3]
            if (not reference.get("calibration_sha256") or
                    reference["calibration_sha256"] != plane.get("calibration_sha256")):
                raise ValueError("Board calibration changed since interruption; recovery release blocked")
            frame = (reference["center_base_xy_m"], reference["board_x_unit_base_xy"],
                     reference["board_y_unit_base_xy"], plane)
            snapshot(frame)  # validate the saved rigid frame before any connection
            session.runtime = (*session.runtime[:2], frame, session.runtime[3])
        session.robot.connect()
        session.recover_action(prior.get("last_error") or "interrupted competition action")
    except Exception as exc:
        session.last_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        session.close()
    return json.loads((output / "run_summary.json").read_text())


def _load_template(profile):
    import cv2
    path = _resolve(profile["template"]["path"])
    try:
        actual_hash = file_sha256(path)
    except OSError as exc:
        raise ValueError(f"Wrist template unavailable: {path}: {exc}") from exc
    if actual_hash != profile["template"]["sha256"]:
        raise ValueError("Wrist template hash changed; re-teach the part")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read wrist template {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _check_ready(profile, cfg, action, *, competition=False, no_cv=False):
    if profile is None:
        raise ValueError("No taught wrist profile. Use menu 3 first.")
    calibration_path = ROOT / "calibration/vega_board_manual.json"
    if not calibration_path.is_file():
        calibration_path = ROOT / "calibration/vega_board_manual_fallback.json"
    current = load_board_calibration(calibration_path, cfg)
    if current["sha256"] != profile["calibration_sha256"]:
        # A field recalibration is expected during onsite setup.  It changes
        # board registration and surface height, but it does not erase a
        # verified part's wrist feature, grasp depth, or gripper settings.
        # PartSession.begin_part() uses the fresh task target first (and the
        # saved hover as a bounded fallback) when this mismatch is present.
        print(
            "WARNING: board calibration changed since wrist teaching; "
            "continuing with the saved wrist/grasp profile and current field "
            "registration.",
            flush=True,
        )
    # Pickup tests/competition check the template at the hover, where a visual
    # rejection can offer the saved physical grasp. Localization-only requests
    # still require the visual asset up front.
    if not no_cv and action == "localize":
        _load_template(profile)
    if action != "localize" and profile.get("grasp_clearance_m") is None:
        raise ValueError("Grasp depth has not been taught")
    if action == "pick_place" and profile.get("place") is None:
        raise ValueError("Place settings have not been taught; use place-config in menu 3")
    if competition and (not profile.get("grasp_verified") or (action == "pick_place" and not profile.get("place_verified"))):
        raise ValueError("Run the individual task test successfully before enabling this competition action")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=PART_NAMES)
    parser.add_argument("--mode", choices=("calibrate", "test", "drop", "place-cv"), default="calibrate")
    parser.add_argument("--action", choices=("localize", "pick", "pick_place"), default="localize")
    parser.add_argument("--sequence", nargs="+", help="Explicit part.action sequence")
    parser.add_argument("--competition", action="store_true")
    # Internal snapshot passed by the pipeline to keep all parts/retries on the
    # same settings, even if the operator edits the JSON while a run is active.
    parser.add_argument("--execution-offsets-json", help=argparse.SUPPRESS)
    parser.add_argument("--release-wiggle-json", help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-cv", action="store_true",
        help="competition pickup from saved arm/task coordinates without wrist images",
    )
    parser.add_argument(
        "--place-cv", action="store_true",
        help="use a saved wrist release-target template during pick-place",
    )
    parser.add_argument("--skip-place-cv-teaching", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--head-reacquire", action="store_true",
        help="prefer the fresh head-camera task target before the saved hover",
    )
    parser.add_argument("--profiles", default="calibration/wrist_part_profiles.json")
    parser.add_argument("--output")
    parser.add_argument("--no-viewer", action="store_true", help="Enter image pixels in terminal instead of Tk viewer")
    parser.add_argument('--confirm-head-motion', action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument('--confirm-physical-motion', action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--speed-scale", type=float, default=.60)
    parser.add_argument(
        "--remote-safe", action="store_true",
        help="normal motion settings; confirm below 40 mm and capture once before lowering",
    )
    args = parser.parse_args(argv)
    minimum_speed = .10 if args.remote_safe else .25
    if not minimum_speed <= args.speed_scale <= .70:
        parser.error("--speed-scale must be .10..70 in remote-safe mode, otherwise .25..70")
    args.execution_offsets = _session_execution_offsets(args)
    if args.execution_offsets is not None:
        describe_offsets(args.execution_offsets)
    cfg = load_bundle("vega")["robot"]
    if cfg.get("working_arm") != WORKING_ARM or cfg["kinematics"].get("ee_frame") != TCP_FRAME:
        raise ValueError("Requires right arm / tip_r")
    cfg["allow_robot_init_head_motion"] = True
    cfg["auto_clear_software_estop_on_connect"] = not args.competition
    cfg["kinematics"]["near_target_fallback"] = True
    # The right-arm state stream routinely settles a few milliradians outside
    # the nominal 5 mrad gate even when the motion plugin has finished.  This
    # is the same supervised tolerance used by board calibration and prevents
    # a false E-stop during the wrist teaching/competition path.
    cfg["motion"]["joint_reached_tolerance_rad"] = max(
        float(cfg["motion"]["joint_reached_tolerance_rad"]), 0.020
    )
    profiles = load_profiles(_resolve(args.profiles), cfg)
    if args.mode in ("drop", "place-cv") and args.sequence:
        parser.error("--mode drop accepts one --part and does not use --sequence")
    actions = None if args.mode in ("drop", "place-cv") else (
        args.sequence or ([f"{args.part}.{args.action}"] if args.part else None)
    )
    if actions and args.mode == "test":
        for value in actions:
            part, action = value.split(".")
            if part not in PART_NAMES or action not in ("localize", "pick", "pick_place"):
                parser.error(f"Unknown action {value}")
            _check_ready(
                profiles["parts"].get(part), cfg, action,
                competition=args.competition, no_cv=args.no_cv,
            )
    if args.competition and not actions:
        parser.error("--competition requires explicit --sequence")
    if args.mode == "place-cv" and not args.part:
        parser.error("--mode place-cv requires --part")
    output = _resolve(args.output) if args.output else ROOT / "runs" / datetime.now(timezone.utc).strftime("wrist_parts_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    session = PartSession(args, output, cfg, profiles)
    if actions and len(actions) == 1:
        # Startup faults still need an attributable checkpoint, even if the
        # SDK or board camera fails before test() can identify the action.
        session.part, session.action = actions[0].split(".", 1)
    try:
        session.start()
        if args.mode == "drop":
            if not args.part:
                parser.error("--mode drop requires --part")
            return session.teach_drop(args.part, profiles["parts"].get(args.part))
        if args.mode == "place-cv":
            return session.teach_place_cv(args.part, profiles["parts"].get(args.part))
        if actions:
            for value in actions:
                part, action = value.split(".")
                result = (
                    session.teach(part, profiles["parts"].get(part))
                    if args.mode == "calibrate"
                    else session.test(
                        part, action, competition=args.competition, no_cv=args.no_cv,
                        place_cv=args.place_cv
                    )
                )
                if result:
                    return result
            return 0
        # Same ordering as option 2: camera -> ready -> choose a destination.
        choices = list(PART_NAMES) if args.mode == "calibrate" else [f"{part}.{action}" for part in PART_NAMES for action in ("pick", "pick_place")]
        while True:
            print("\n" + "\n".join(f"{i}. {name}" for i, name in enumerate(choices, 1)))
            choice = input("Choose number/name, r to retake board image, e to exit: ").strip()
            if choice in ("0", "e", "exit", "q"):
                return 0
            if choice == "r":
                session.retake()
                continue
            try:
                name = choices[int(choice)-1] if choice.isdigit() and 1 <= int(choice) <= len(choices) else choice
                if name not in choices:
                    raise ValueError("Unknown choice")
                if args.mode == "calibrate":
                    result = session.teach(name, session.profiles["parts"].get(name))
                else:
                    part, action = name.split(".")
                    result = session.test(part, action, place_cv=args.place_cv)
                if result == 3:
                    return result
            except ValueError as exc:
                print(f"BLOCKED: {exc}", flush=True)
    except (KeyboardInterrupt, EOFError, InterruptedError):
        session.status = "cancelled"
        session.last_error = "operator cancelled session"
        print("Session cancelled; no automatic recovery motion.", flush=True)
        return 1
    except Exception as exc:
        session.status = "failed"
        session.last_error = f"{type(exc).__name__}: {exc}"
        session.automatic_continuation_safe = bool(
            not session.holding and not session.motion_faulted and not session.recovery_blocked
            and (isinstance(exc, (IKError, ValueError)) or _is_visual_alignment_failure(exc))
            and "tcp missed servo waypoint" not in str(exc).lower())
        print(f"Session stopped: {type(exc).__name__}: {exc}. Inspect before retrying.", flush=True)
        return 2
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
