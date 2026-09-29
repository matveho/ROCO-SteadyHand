"""Teach or test a right-wrist visual grasp profile for one competition part.

The tool reuses the live path used by battery centering: fresh board image,
measured RIGHT_READY, coarse TCP hover, fresh physical-right ``wrist_a`` RGB,
bounded measured-Jacobian XY servo, and explicit software gripper gating.
It never operates the claw by hand.  ``grab`` is the only command that
descends or connects the gripper.
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
from steadyhand.board_calibration import load_board_calibration
from steadyhand.board_geometry import validate_task_coordinate_extent
from steadyhand.cameras.vega import VegaWristCameras
from steadyhand.config import load_bundle
from steadyhand.executor import move_tcp_segmented
from steadyhand.geometry import matrix_to_quaternion, quaternion_to_matrix, interpolate_pose, pose_distance
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills
from steadyhand.vega_presets import configured_right_preset
from steadyhand.vision.wrist_servo import TemplateTracker, run_xy_servo
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
        self.floor = float(load_vega_skills()["safety"]["min_tcp_z_m"])
        self.robot = VegaAdapter(cfg)
        self.cameras = VegaWristCameras()
        self.capture = WristAOnlyCapture(self.cameras, output, settle_s=.20, warmup_attempts=8)
        self.holding = False
        self.runtime = None
        self.targets = None
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

    def start(self):
        self.status = "starting"
        self.robot.connect()
        self.retake()
        self.cameras.connect()
        self.status = "ready"

    def close(self):
        try:
            try:
                self.cameras.close()
            finally:
                self.robot.close()
        finally:
            summary = {
                "schema_version": 1,
                "status": self.status,
                "part": self.part,
                "action": self.action,
                "holding_may_be_true": bool(self.holding),
                "last_error": self.last_error,
                "coarse_xy_m": list(self.coarse.position_m[:2]) if hasattr(self, "coarse") else None,
                "yaw_deg": self.yaw,
                "gripper_open_fraction": getattr(self, "gripper_open_fraction", None),
                "alignment_verified": getattr(self, "alignment_verified", False),
                "alignment_fallback_used": getattr(self, "alignment_fallback_used", False),
                "grasp_verified": getattr(self, "grasp_verified", False),
                "grasp_clearance_m": getattr(self, "last_grasp_clearance", None),
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
        from tools.vega_competition_pipeline import _capture_downward_head_frame, _load_runtime, _runtime_from_board_scene, _task_targets
        runtime = _load_runtime()
        scene = _capture_downward_head_frame(self.robot, floor_m=self.floor, bundle=runtime[0])
        self.runtime = _runtime_from_board_scene(runtime, scene)
        ready_q, _ = configured_right_preset(self.cfg, "right_ready")
        self.robot.move_joints(ready_q, speed_scale=self.args.speed_scale)
        self.targets = _task_targets(self.runtime, self.runtime[1], .100)
        scene_path = self.output / f"board_{time.time_ns()}.json"
        scene_path.write_text(json.dumps(scene, indent=2, default=str) + "\n")
        self.board_scene_paths.append(scene_path.name)
        print("FRESH BOARD REGISTERED; RIGHT_READY reached.", flush=True)

    def surface(self, x, y):
        from tools.vega_task_coordinate_reachability import calibrated_surface_z
        return calibrated_surface_z(x, y, self.runtime[2][3])

    def move(self, target, *, slow=False):
        # Check the complete Cartesian segment before issuing its first waypoint.
        before = self.robot.get_tcp_pose()
        distance, angle = pose_distance(before, target)
        count = max(1, math.ceil(distance/.020), math.ceil(angle/.08))
        seed = self.robot._read_joint_positions()
        for i in range(1, count + 1):
            pose = interpolate_pose(before, target, i/count)
            if pose.position_m[2] < self.floor + .005:
                raise ValueError("Target intersects the configured TCP floor + 5 mm; no motion issued")
            seed = self.robot._kinematics.solve(pose, seed)
        move_tcp_segmented(self.robot, target, speed_scale=.25 if slow else self.args.speed_scale,
                           max_translation_step_m=.020, max_orientation_step_rad=.08,
                           min_tcp_z_m=self.floor + .005)

    def frame(self, label="wrist"):
        rgb = self.capture()
        raw = self.output / f"{self.capture.index-1:03d}_wrist_a.png"
        print(f"{label.upper()} IMAGE: {raw}", flush=True)
        self._publish_live_image(rgb, raw, label)
        return rgb, raw

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
                    rgb, selected, max_error_px=18.0
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
        reference = _yaw_pose(self.runtime[3], self.yaw).quaternion_wxyz
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
                "tolerance_px": 8., "max_iterations": 12,
            },
            {
                "probe_m": .008, "gain": .30, "max_step_m": .006,
                "tolerance_px": 12., "max_iterations": 16,
            },
        )
        last_error = None
        for attempt, values in enumerate(settings, 1):
            try:
                result = run_xy_servo(
                    self.robot, self.capture, floor_m=self.floor, goal_uv=self.goal,
                    max_radius_m=.060, speed_scale=.45, event=self.event,
                    tracker_factory=lambda rgb, _: self._reacquire(rgb), surface_z=self.surface,
                    reference_quaternion_wxyz=reference, **values,
                )
                self.alignment_verified = result.get("status") == "converged"
                if self.alignment_verified:
                    self.alignment_fallback_used = False
                    print("ALIGNMENT VERIFIED: same feature reproduced at the taught goal", flush=True)
                return result
            except (RuntimeError, ValueError) as exc:
                self.alignment_verified = False
                last_error = exc
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

    def begin_part(self, part, profile=None, *, initial_yaw=None, competition=False,
                   no_cv=False):
        self.part, self.history = part, []
        self.no_cv_mode = bool(no_cv)
        self.no_cv_used_recorded = False
        self.alignment_fallback_used = False
        validate_task_coordinate_extent(self.runtime[1], names=[f"{part}.pick"])
        self.yaw = float(profile["yaw_deg"]) if profile else float(initial_yaw or 0.0)
        coarse = self.targets[f"task.{part}.pick"]
        self.coarse = coarse
        if profile and profile.get("gripper_open_fraction") is not None:
            opening = float(profile["gripper_open_fraction"])
            if competition:
                # Keep competition jaws away from both hard stops.  A zero
                # opening is too tight before descent, while a saved 100%
                # opening wastes time on a long travel.  The profile itself is
                # left unchanged; this is only a live competition clamp.
                clamped = min(.60, max(.20, opening))
                if clamped != opening:
                    print(
                        "COMPETITION GRIPPER OPENING OVERRIDE: "
                        f"using {clamped * 100:.0f}% pre-grasp opening",
                        flush=True,
                    )
                opening = clamped
            self.gripper_open_fraction = opening
            self._set_gripper_fraction(opening)
        self.coarse = coarse
        if no_cv:
            # First try the exact saved arm hover from teaching.  This is the
            # fastest no-camera path and remains useful when board vision or
            # the wrist stream is unavailable.  If the board moved enough for
            # that pose to be unreachable, fall back to the live task target.
            recorded = None
            if profile and profile.get("coarse_xy_m"):
                rx, ry = (float(v) for v in profile["coarse_xy_m"])
                recorded = Pose(
                    (rx, ry, self.surface(rx, ry) + float(profile.get("hover_clearance_m", .100))),
                    self.runtime[3].quaternion_wxyz,
                )
                if self.yaw:
                    recorded = _yaw_pose(recorded, self.yaw)
            candidates = (("recorded arm hover", recorded), ("live task target", coarse))
            last_error = None
            for label, target in candidates:
                if target is None:
                    continue
                try:
                    self.move(target)
                    self.coarse = target
                    self.no_cv_used_recorded = label == "recorded arm hover"
                    print(f"NO-CV COARSE HOVER: {label}", flush=True)
                    return None
                except (RuntimeError, ValueError) as exc:
                    last_error = exc
                    print(f"NO-CV COARSE HOVER FAILED ({label}): {exc}", flush=True)
            raise last_error or RuntimeError("no reachable no-CV coarse hover")
        self.move(coarse)
        if self.yaw:
            self.move(_yaw_pose(coarse, self.yaw), slow=True)
        self.coarse = coarse
        if profile:
            try:
                rgb, path = self.frame("saved template check")
                if list(rgb.shape[:2]) != profile["image_shape"]:
                    raise ValueError("Wrist resolution changed; re-teach this part")
                self.tracker = TemplateTracker.from_saved_template(
                    rgb,
                    _load_template(profile),
                    profile["template"].get("template_uv"),
                    initial_uv=profile.get("feature_uv"),
                )
                self.reference_rgb = rgb.copy()
                self.reference_feature = tuple(profile.get("feature_uv") or self.tracker.uv)
                self.reference_match_score = profile.get("reference_match_score")
                self.goal = tuple(profile["goal_uv"]) if profile.get("goal_uv") is not None else None
                _write_overlay(path.with_name(path.stem + "_match.png"), rgb, self.tracker.uv, self.goal, label=part)
            except (RuntimeError, ValueError) as exc:
                if not competition or not _is_visual_alignment_failure(exc):
                    raise
                # The saved profile still carries the feature/goal metadata;
                # no live match is required for the coarse-pose fallback.
                self.reference_feature = tuple(profile.get("feature_uv") or (0.0, 0.0))
                self.reference_match_score = profile.get("reference_match_score")
                self.goal = tuple(profile["goal_uv"]) if profile.get("goal_uv") is not None else None
                self.alignment_fallback_used = True
                self.alignment_verified = False
                self.event(
                    "alignment_fallback",
                    {
                        "reason": str(exc),
                        "fallback": "saved_coarse_hover",
                        "coarse_xy_m": list(self.coarse.position_m[:2]),
                    },
                )
                print(
                    "WRIST FEATURE MATCH FAILED; using the saved coarse hover "
                    "without visual centering.",
                    flush=True,
                )
                return None
        else:
            return None
        try:
            return self.localize()
        except (RuntimeError, ValueError) as exc:
            if not competition or not _is_visual_alignment_failure(exc):
                raise
            # The saved coarse board target is the remembered physical pose.
            # Return there, skip visual servoing, and let the bounded grasp use
            # that pose.  This is only a competition fallback; teaching still
            # requires successful centering or explicit supervised override.
            self.move(self.coarse, slow=True)
            self.alignment_fallback_used = True
            self.alignment_verified = False
            self.event(
                "alignment_fallback",
                {
                    "reason": str(exc),
                    "fallback": "saved_coarse_hover",
                    "coarse_xy_m": list(self.coarse.position_m[:2]),
                },
            )
            print(
                "VISUAL CENTERING FAILED; returned to the saved coarse hover and "
                "will attempt the grasp there without further centering.",
                flush=True,
            )
            return None

    def grab(self, clearance, *, allow_unverified=False):
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
        # Snapshot the operator's actual current hover pose.  In particular,
        # preserve a deliberate manual ``back`` adjustment made after visual
        # centering; the grasp approach must not reconstruct an older goal or
        # let the IK descent's small XY residual erase that adjustment.
        anchor_x, anchor_y, _ = hover.position_m
        x, y = anchor_x, anchor_y
        grasp = Pose((x, y, self.surface(x, y) + clearance), hover.quaternion_wxyz)
        # Validate both descent and return before opening/closing.
        if grasp.position_m[2] < self.floor + .005:
            raise ValueError("Grasp intersects TCP floor + 5 mm; no gripper command issued")
        seed = self.robot._read_joint_positions()
        for i in range(1, 11):
            seed = self.robot._kinematics.solve(interpolate_pose(hover, grasp, i/10), seed)
        self.robot._kinematics.solve(hover, seed)
        self.robot.connect_gripper()
        if self.gripper_open_fraction is None:
            # Never travel to the hard-open stop during a competition grasp;
            # it forces a slow homing/opening cycle and is unnecessary for a
            # taught object.  Twenty percent is the validated safe fallback.
            self.gripper_open_fraction = .20
            self._set_gripper_fraction(self.gripper_open_fraction)
        else:
            self._set_gripper_fraction(self.gripper_open_fraction)
        self.move(grasp, slow=True)
        # A Cartesian Z descent is solved as a sequence of joint targets.  On
        # Vega that can leave the measured TCP a few millimetres off in XY,
        # even though the requested grasp pose has the same XY as the hover.
        # Correct that bounded residual while the jaws are still open, before
        # applying grip pressure.  This is deliberately local and uses the
        # measured grasp height/orientation; it cannot create a new board move.
        measured_grasp = self.robot.get_tcp_pose()
        xy_error = math.dist(
            measured_grasp.position_m[:2], (anchor_x, anchor_y)
        )
        self.event(
            "grasp_approach_readback",
            {
                "anchor_xy_m": [anchor_x, anchor_y],
                "requested_grasp_tcp": list(grasp.position_m),
                "measured_grasp_tcp": list(measured_grasp.position_m),
                "xy_error_m": xy_error,
            },
        )
        if xy_error > 0.001:
            if xy_error > 0.010:
                raise RuntimeError(
                    "Grasp approach drifted more than 10 mm from the manually "
                    "positioned hover; no grip command issued"
                )
            correction = Pose(
                (anchor_x, anchor_y, measured_grasp.position_m[2]),
                measured_grasp.quaternion_wxyz,
            )
            print(
                "GRASP XY CORRECTION: returning to the exact manual hover "
                f"anchor ({xy_error * 1000:.1f} mm residual)",
                flush=True,
            )
            self.move(correction, slow=True)
            measured_grasp = self.robot.get_tcp_pose()
            corrected_error = math.dist(
                measured_grasp.position_m[:2], (anchor_x, anchor_y)
            )
            self.event(
                "grasp_approach_corrected",
                {
                    "measured_grasp_tcp": list(measured_grasp.position_m),
                    "xy_error_m": corrected_error,
                },
            )
            if corrected_error > 0.006:
                raise RuntimeError(
                    "Grasp approach could not return to the manually positioned "
                    f"hover anchor (XY residual {corrected_error * 1000:.1f} mm)"
                )
        self.holding = True  # Remains true on uncertain grip/error; no blind recovery.
        self.robot.grip(self.part)
        result = self.robot._gripper.last_grip_result()
        self.event("grip_result", {
            "requested_grasp_tcp": list(grasp.position_m),
            "measured_grasp_tcp": list(self.robot.get_tcp_pose().position_m),
            "result": result,
        })
        print("GRIP RESULT", json.dumps(result, default=str), flush=True)
        if not isinstance(result, dict) or result.get("gripped") is not True:
            raise RuntimeError("Grip was not verified; stopped at grasp height for inspection")
        self.move(hover, slow=True)
        return result

    def return_part(self, clearance, *, partial_release=False):
        if not self.holding:
            raise ValueError("No part is held")
        hover = self.robot.get_tcp_pose()
        x, y, _ = hover.position_m
        release = Pose(
            (x, y, self.surface(x, y) + clearance),
            hover.quaternion_wxyz,
        )
        if release.position_m[2] < self.floor + .005:
            raise ValueError("Return intersects TCP floor + 5 mm; retaining part")
        seed = self.robot._read_joint_positions()
        for i in range(1, 11):
            seed = self.robot._kinematics.solve(interpolate_pose(hover, release, i / 10), seed)
        self.robot._kinematics.solve(hover, seed)
        self.move(release, slow=True)
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
        self.move(hover, slow=True)

    def place(self, settings, *, partial_release=False):
        if not self.holding or settings is None:
            raise ValueError("Place needs a verified held part and taught place settings")
        target = self.targets[f"task.{self.part}.place"]
        _, ux, uy, _ = self.runtime[2]
        dx, dy = settings["offset_board_xy_m"]
        x = target.position_m[0] + ux[0]*dx + uy[0]*dy
        y = target.position_m[1] + ux[1]*dx + uy[1]*dy
        quat = _yaw_pose(self.runtime[3], settings["yaw_deg"]).quaternion_wxyz
        hover = Pose((x, y, self.surface(x, y)+.100), quat)
        release = Pose((x, y, self.surface(x, y)+settings["clearance_m"]), quat)
        if release.position_m[2] < self.floor + .005:
            raise ValueError("Place intersects TCP floor + 5 mm; retaining part")
        # Validate placement descent before transporting the held part.
        seed = self.robot._read_joint_positions()
        seed = self.robot._kinematics.solve(hover, seed)
        self.robot._kinematics.solve(release, seed)
        self.move(hover)
        self.move(release, slow=True)
        self.robot.release_gripper(self.part)
        self.holding = False
        self.move(hover, slow=True)
        print("Placement release completed; insertion/assembly is not inferred.")

    def teach(self, part, old=None):
        self.begin_part(part, None, initial_yaw=(old or {}).get("yaw_deg"))
        grasp = old.get("grasp_clearance_m") if old else None
        place = old.get("place") if old else None
        self.gripper_open_fraction = old.get("gripper_open_fraction") if old else None
        if self.gripper_open_fraction is not None:
            self._set_gripper_fraction(self.gripper_open_fraction)
        self.grasp_verified = False
        self.last_grasp_clearance = grasp
        self.teach_feature()
        result = None
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
                    if self.goal is None or not self.alignment_verified:
                        print("SAVE BLOCKED: run goal, then center, and verify alignment first.")
                        continue
                    if grasp is None or not self.grasp_verified:
                        print("SAVE BLOCKED: teach depth, grab, and confirm the physical grasp first.")
                        continue
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
                        rgb,
                        feature,
                        self.goal,
                        label=part,
                    )
                    # The saved template remains the uncluttered first image;
                    # final/goal images and events provide the audit trail.
                    template, anchor = _crop_template(self.reference_rgb, self.reference_feature)
                    import cv2
                    directory = default_template_dir(ROOT)
                    directory.mkdir(parents=True, exist_ok=True)
                    template_path = directory / f"{part}_{time.time_ns()}.png"
                    if not cv2.imwrite(str(template_path), cv2.cvtColor(template, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError("Failed to save wrist template")
                    cal = load_board_calibration(ROOT / "calibration/vega_board_manual.json", self.cfg)
                    profile = {
                        "part": part, "working_arm": WORKING_ARM, "tcp_frame": TCP_FRAME, "wrist_camera": WRIST_CAMERA,
                        "calibration_sha256": cal["sha256"], "coarse_xy_m": list(self.coarse.position_m[:2]),
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
                        "hover_clearance_m": .100, "grasp_clearance_m": grasp, "yaw_deg": self.yaw,
                        "gripper_open_fraction": self.gripper_open_fraction,
                        "place": place, "grasp_verified": self.grasp_verified, "place_verified": False,
                        "template": {"path": str(template_path.relative_to(ROOT)), "sha256": file_sha256(template_path), "template_uv": list(anchor)},
                        "localization_result": result, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
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
                    try:
                        feature, score = self.tracker.locate(rgb)
                        _write_overlay(path.with_name(path.stem + "_tracked.png"), rgb, feature, self.goal, label=part)
                        print("TRACKED FEATURE =", tuple(round(v, 1) for v in feature), f"score={score:.3f}", flush=True)
                    except (ValueError, RuntimeError) as exc:
                        print("FEATURE NOT VERIFIED IN THIS IMAGE:", exc, flush=True)
                    continue
                if command in ("feature", "reference"):
                    self.teach_feature()
                    continue
                if command in ("goal", "target"):
                    self.teach_goal_feature()
                    continue
                if command in ("center", "verify", "localize"):
                    try:
                        result = self.localize()
                    except RuntimeError as exc:
                        message = str(exc).lower()
                        if any(word in message for word in ("feature", "lost", "ambiguous", "tracking", "centering stalled")):
                            print(
                                "VISUAL ALIGNMENT STOPPED: no further motion was issued; "
                                "capture another image or teach a different feature. "
                                "If the TCP is already over the part, use 'grab manual'; "
                                "competition pickup does not require centering.",
                                flush=True,
                            )
                            continue
                        raise
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
                        self.step_mm = _number(raw[1], .5, 20.0)
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
                        depth = _number(raw[1], .001, 100.)
                        grasp = .100-depth/1000
                        self.last_grasp_clearance = grasp
                        self.grasp_verified = False
                        print(f"Grasp clearance above calibrated surface: {grasp*1000:.1f} mm", flush=True)
                        continue
                    if command == "place-config" and len(raw) == 5:
                        dx, dy = (_number(v, -50., 50.)/1000 for v in raw[1:3])
                        depth = _number(raw[3], .001, 100.)
                        yaw = _number(raw[4], -45., 45.)
                        place = {"offset_board_xy_m": [dx, dy], "clearance_m": .100-depth/1000, "yaw_deg": yaw}
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
                        yaw = _number(raw[1], -45., 45.)
                        target = Pose(current.position_m, _yaw_pose(self.runtime[3], yaw).quaternion_wxyz)
                    elif command in ("forward", "back", "left", "right") and len(raw) in (1, 2):
                        amount = self.step_mm if len(raw) == 1 else _number(raw[1], .1, 30.)
                        amount /= 1000.0
                        dx, dy = {"forward": (amount, 0), "back": (-amount, 0), "left": (0, amount), "right": (0, -amount)}[command]
                        x, y = current.position_m[0]+dx, current.position_m[1]+dy
                        if math.dist((x, y), self.coarse.position_m[:2]) > .060:
                            raise ValueError("Adjustment exceeds 60 mm local radius; fix coarse coordinates")
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

    def test(self, part, action, *, competition=False, no_cv=False):
        self.part, self.action = part, action
        self.status = f"running:{part}.{action}"
        profile = self.profiles.get("parts", {}).get(part)
        _check_ready(profile, self.cfg, action, competition=competition, no_cv=no_cv)
        self.begin_part(part, profile, competition=competition, no_cv=no_cv)
        if action == "localize":
            return 0
        if not competition and input("Type grab to test the taught descent/grip/lift; anything else cancels: ").strip() != "grab":
            self.status = "cancelled_before_grip"
            return 1
        try:
            self.grab(
                profile["grasp_clearance_m"],
                allow_unverified=bool(
                    no_cv or (competition and self.alignment_fallback_used)
                ),
            )
        except (RuntimeError, ValueError) as exc:
            if competition and self.holding:
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
                        profile["grasp_clearance_m"], partial_release=competition
                    )
                except Exception as return_exc:
                    self.last_error = f"{type(return_exc).__name__}: {return_exc}"
                    print(
                        "AUTOMATIC RETURN FAILED; holding state is preserved in the run summary.",
                        file=sys.stderr,
                        flush=True,
                    )
                    return 3
                self.status = "pick_failed_returned"
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
                        profile["grasp_clearance_m"], partial_release=competition
                    )
                self.status = "pick_not_confirmed"
                return 2
        profile = dict(profile, grasp_verified=True)
        self.grasp_verified = True
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
                        profile["grasp_clearance_m"], partial_release=competition
                    )
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    print(
                        "AUTOMATIC RETURN FAILED; holding state is preserved in the run summary.",
                        file=sys.stderr,
                        flush=True,
                    )
                    return 3
                self.status = "pick_complete_returned"
                return 0
            print("Pick complete. Part is held; next actions are blocked until it is returned.")
            if input("Type return to put it back at source, or exit to stop holding: ").strip() == "return":
                self.return_part(profile["grasp_clearance_m"])
                self.status = "pick_complete_returned"
                return 0
            return 3
        if not competition and input("Type place to test the taught transfer/descent/release: ").strip() != "place":
            return 3
        self.place(profile["place"], partial_release=competition)
        self.status = "completed"
        if not competition:
            if input("Was placement correct? Type yes to enable it for competition: ").strip().lower() == "yes":
                profile["place_verified"] = True
                self.profiles = save_profile(_resolve(self.args.profiles), self.cfg, profile)
                self.status = "completed_place_verified"
            else:
                self.status = "completed_place_unverified"
        return 0


def _number(value, low, high):
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"Value must be finite and in {low}..{high}")
    return result


def _load_template(profile):
    import cv2
    path = _resolve(profile["template"]["path"])
    if file_sha256(path) != profile["template"]["sha256"]:
        raise ValueError("Wrist template hash changed; re-teach the part")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read wrist template {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _check_ready(profile, cfg, action, *, competition=False, no_cv=False):
    if profile is None:
        raise ValueError("No taught wrist profile. Use menu 3 first.")
    current = load_board_calibration(ROOT / "calibration/vega_board_manual.json", cfg)
    if current["sha256"] != profile["calibration_sha256"]:
        raise ValueError("Board calibration changed since wrist teaching; re-teach this profile")
    if not no_cv:
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
    parser.add_argument("--mode", choices=("calibrate", "test"), default="calibrate")
    parser.add_argument("--action", choices=("localize", "pick", "pick_place"), default="localize")
    parser.add_argument("--sequence", nargs="+", help="Explicit part.action sequence")
    parser.add_argument("--competition", action="store_true")
    parser.add_argument(
        "--no-cv", action="store_true",
        help="competition pickup from saved arm/task coordinates without wrist images",
    )
    parser.add_argument("--profiles", default="calibration/wrist_part_profiles.json")
    parser.add_argument("--output")
    parser.add_argument("--no-viewer", action="store_true", help="Enter image pixels in terminal instead of Tk viewer")
    parser.add_argument("--confirm-head-motion", action="store_true")
    parser.add_argument("--confirm-physical-motion", action="store_true")
    parser.add_argument("--speed-scale", type=float, default=.38)
    args = parser.parse_args(argv)
    if not args.confirm_head_motion or not args.confirm_physical_motion:
        parser.error("requires --confirm-head-motion and --confirm-physical-motion")
    if not .25 <= args.speed_scale <= .70:
        parser.error("--speed-scale must be .25..70")
    cfg = load_bundle("vega")["robot"]
    if cfg.get("working_arm") != WORKING_ARM or cfg["kinematics"].get("ee_frame") != TCP_FRAME:
        raise ValueError("Requires right arm / tip_r")
    cfg["allow_robot_init_head_motion"] = True
    cfg["auto_clear_software_estop_on_connect"] = True
    # The right-arm state stream routinely settles a few milliradians outside
    # the nominal 5 mrad gate even when the motion plugin has finished.  This
    # is the same supervised tolerance used by board calibration and prevents
    # a false E-stop during the wrist teaching/competition path.
    cfg["motion"]["joint_reached_tolerance_rad"] = max(
        float(cfg["motion"]["joint_reached_tolerance_rad"]), 0.020
    )
    profiles = load_profiles(_resolve(args.profiles), cfg)
    actions = args.sequence or ([f"{args.part}.{args.action}"] if args.part else None)
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
    output = _resolve(args.output) if args.output else ROOT / "runs" / datetime.now(timezone.utc).strftime("wrist_parts_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    session = PartSession(args, output, cfg, profiles)
    try:
        session.start()
        if actions:
            for value in actions:
                part, action = value.split(".")
                result = (
                    session.teach(part, profiles["parts"].get(part))
                    if args.mode == "calibrate"
                    else session.test(
                        part, action, competition=args.competition, no_cv=args.no_cv
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
                    result = session.test(part, action)
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
        print(f"Session stopped: {type(exc).__name__}: {exc}. Inspect before retrying.", flush=True)
        return 2
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
