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
        return rgb, raw

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
        return select_pixel(rgb, path, title, use_viewer=not self.args.no_viewer)

    def teach_feature(self, *, choose_goal=True):
        rgb, path = self.frame("select a visible part edge, not the gripper")
        feature = self.select(rgb, path, "Select a textured feature ON the selected part")
        self.tracker = TemplateTracker(rgb, feature)
        # Preserve the pre-servo reference: a final aligned frame may contain
        # the jaw over the part and is a poor global reacquisition template.
        self.reference_rgb = rgb.copy()
        self.reference_feature = tuple(feature)
        if choose_goal:
            self.goal = self.select(rgb, path, "Select where that feature should lie for jaw alignment")
        return feature

    def localize(self):
        reference = _yaw_pose(self.runtime[3], self.yaw).quaternion_wxyz
        result = run_xy_servo(
            self.robot, self.capture, floor_m=self.floor, goal_uv=self.goal,
            probe_m=.008, gain=.65, max_step_m=.010, max_radius_m=.045,
            tolerance_px=5., max_iterations=8, speed_scale=.45, event=self.event,
            tracker_factory=lambda rgb, _: self._reacquire(rgb), surface_z=self.surface,
            reference_quaternion_wxyz=reference,
        )
        return result

    def _reacquire(self, rgb):
        self.tracker.locate(rgb)
        return self.tracker

    def begin_part(self, part, profile=None, *, initial_yaw=None):
        self.part, self.history = part, []
        validate_task_coordinate_extent(self.runtime[1], names=[f"{part}.pick"])
        self.yaw = float(profile["yaw_deg"]) if profile else float(initial_yaw or 0.0)
        coarse = self.targets[f"task.{part}.pick"]
        self.move(coarse)
        if self.yaw:
            self.move(_yaw_pose(coarse, self.yaw), slow=True)
        self.coarse = coarse
        if profile:
            rgb, path = self.frame("saved template check")
            if list(rgb.shape[:2]) != profile["image_shape"]:
                raise ValueError("Wrist resolution changed; re-teach this part")
            self.tracker = TemplateTracker.from_saved_template(rgb, _load_template(profile), profile["template"].get("template_uv"))
            self.goal = tuple(profile["goal_uv"])
            _write_overlay(path.with_name(path.stem + "_match.png"), rgb, self.tracker.uv, self.goal, label=part)
        else:
            self.teach_feature()
        return self.localize()

    def grab(self, clearance):
        if clearance is None:
            raise ValueError("Set depth N first; N is millimetres below the 100 mm hover")
        if self.holding:
            raise ValueError("A part may already be held; inspect it before another grab")
        hover = self.robot.get_tcp_pose()
        self.last_grasp_clearance = clearance
        x, y, _ = hover.position_m
        grasp = Pose((x, y, self.surface(x, y) + clearance), hover.quaternion_wxyz)
        # Validate both descent and return before opening/closing.
        if grasp.position_m[2] < self.floor + .005:
            raise ValueError("Grasp intersects TCP floor + 5 mm; no gripper command issued")
        seed = self.robot._read_joint_positions()
        for i in range(1, 11):
            seed = self.robot._kinematics.solve(interpolate_pose(hover, grasp, i/10), seed)
        self.robot._kinematics.solve(hover, seed)
        self.robot.connect_gripper()
        self.robot.open_gripper(self.part)
        self.move(grasp, slow=True)
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

    def return_part(self, clearance):
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
        self.robot.open_gripper(self.part)
        self.event("place_release", {
            "requested_release_tcp": list(release.position_m),
            "settings": {
                "return_to_source": True,
                "clearance_m": float(clearance),
            },
        })
        self.holding = False
        self.move(hover, slow=True)

    def place(self, settings):
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
        self.robot.open_gripper(self.part)
        self.holding = False
        self.move(hover, slow=True)
        print("Placement release completed; insertion/assembly is not inferred.")

    def teach(self, part, old=None):
        result = self.begin_part(part, None, initial_yaw=(old or {}).get("yaw_deg"))
        grasp = old.get("grasp_clearance_m") if old else None
        place = old.get("place") if old else None
        grip_verified = False
        print("Commands: forward/back/left/right N (mm), yaw N (degrees relative ready), undo,")
        print("  center, feature, image, depth N (mm below 100 mm hover), grab, return,")
        print("  place-config X Y DEPTH YAW (board XY offsets mm, depth mm, yaw degrees), save, abort")
        while True:
            raw = input(f"{part}> ").strip().lower().split()
            if not raw:
                continue
            command = raw[0]
            if command in ("abort", "exit", "q"):
                return 3 if self.holding else 1
            if command == "save":
                if self.holding:
                    print("Type return to put the test part back before saving its image profile.")
                    continue
                rgb, path = self.frame("final taught pose")
                feature, _ = self.tracker.locate(rgb)
                # The final observed feature is recorded for audit, while the
                # saved template remains the uncluttered pre-servo reference.
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
                    "feature_uv": list(self.reference_feature), "goal_uv": list(self.goal), "image_shape": list(self.reference_rgb.shape[:2]),
                    "hover_clearance_m": .100, "grasp_clearance_m": grasp, "yaw_deg": self.yaw,
                    "place": place, "grasp_verified": grip_verified, "place_verified": False,
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
                self.grab(grasp)
                grip_verified = True
                continue
            if command == "return":
                self.return_part(grasp)
                continue
            if self.holding:
                print("A part is held. Use return or abort before adjusting the taught pose.")
                continue
            if command == "image":
                rgb, path = self.frame()
                feature, _ = self.tracker.locate(rgb)
                _write_overlay(path.with_name(path.stem+"_tracked.png"), rgb, feature, self.goal, label=part)
                continue
            if command == "feature":
                self.teach_feature()
                continue
            if command == "center":
                result = self.localize()
                continue
            try:
                if command == "depth" and len(raw) == 2:
                    depth = _number(raw[1], 0.001, 100.)
                    grasp = .100-depth/1000
                    grip_verified = False
                    print(f"Grasp clearance above calibrated surface: {grasp*1000:.1f} mm")
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
                    target, yaw = self.history[-1]
                    self.move(target, slow=True)
                    self.history.pop()
                    changed_yaw = self.yaw != yaw
                    self.yaw = yaw
                    if changed_yaw:
                        self.teach_feature(choose_goal=False)
                    continue
                if command == "yaw" and len(raw) == 2:
                    yaw = _number(raw[1], -45., 45.)
                    target = Pose(current.position_m, _yaw_pose(self.runtime[3], yaw).quaternion_wxyz)
                elif command in ("forward", "back", "left", "right") and len(raw) == 2:
                    amount = _number(raw[1], .1, 30.)/1000
                    dx, dy = {"forward": (amount, 0), "back": (-amount, 0), "left": (0, amount), "right": (0, -amount)}[command]
                    x, y = current.position_m[0]+dx, current.position_m[1]+dy
                    if math.dist((x, y), self.coarse.position_m[:2]) > .060:
                        raise ValueError("Adjustment exceeds 60 mm local radius; fix coarse coordinates")
                    target = Pose((x, y, self.surface(x, y)+.100), current.quaternion_wxyz)
                    yaw = self.yaw
                else:
                    print("Unknown command; use save, grab, depth N, yaw N, or a direction N.")
                    continue
            except ValueError as exc:
                print(exc)
                continue
            self.move(target, slow=True)
            self.history.append((current, self.yaw))
            self.yaw = yaw
            grip_verified = False
            if command == "yaw":
                self.teach_feature(choose_goal=False)
            else:
                self.frame("adjusted pose")

    def test(self, part, action, *, competition=False):
        self.part, self.action = part, action
        self.status = f"running:{part}.{action}"
        profile = self.profiles.get("parts", {}).get(part)
        _check_ready(profile, self.cfg, action, competition=competition)
        self.begin_part(part, profile)
        if action == "localize":
            return 0
        if not competition and input("Type grab to test the taught descent/grip/lift; anything else cancels: ").strip() != "grab":
            self.status = "cancelled_before_grip"
            return 1
        self.grab(profile["grasp_clearance_m"])
        profile = dict(profile, grasp_verified=True)
        self.profiles = save_profile(_resolve(self.args.profiles), self.cfg, profile)
        if action == "pick":
            self.status = "pick_complete_holding"
            print("Pick complete. Part is held; next actions are blocked until it is returned.")
            if input("Type return to put it back at source, or exit to stop holding: ").strip() == "return":
                self.return_part(profile["grasp_clearance_m"])
                self.status = "pick_complete_returned"
                return 0
            return 3
        if not competition and input("Type place to test the taught transfer/descent/release: ").strip() != "place":
            return 3
        self.place(profile["place"])
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


def _check_ready(profile, cfg, action, *, competition=False):
    if profile is None:
        raise ValueError("No taught wrist profile. Use menu 3 first.")
    current = load_board_calibration(ROOT / "calibration/vega_board_manual.json", cfg)
    if current["sha256"] != profile["calibration_sha256"]:
        raise ValueError("Board calibration changed since wrist teaching; re-teach this profile")
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
    profiles = load_profiles(_resolve(args.profiles), cfg)
    actions = args.sequence or ([f"{args.part}.{args.action}"] if args.part else None)
    if actions and args.mode == "test":
        for value in actions:
            part, action = value.split(".")
            if part not in PART_NAMES or action not in ("localize", "pick", "pick_place"):
                parser.error(f"Unknown action {value}")
            _check_ready(profiles["parts"].get(part), cfg, action, competition=args.competition)
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
                result = session.teach(part, profiles["parts"].get(part)) if args.mode == "calibrate" else session.test(part, action, competition=args.competition)
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
