"""Head-camera-only fallback for coarse Vega part pickup.

This standalone path never imports or opens wrist cameras. It captures a fresh
head-camera board scene, matches detected dark objects against the reviewed
task coordinates, and lets an operator teach per-part XY, yaw, and descent
depth before an explicit gripper test.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.vega import VegaAdapter
from steadyhand.config import load_bundle
from steadyhand.executor import move_tcp_segmented
from steadyhand.board_geometry import board_relative_task_xy
from steadyhand.geometry import interpolate_pose, matrix_to_quaternion, pose_distance, quaternion_to_matrix
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills
from steadyhand.vega_presets import configured_right_preset
from steadyhand.wrist_part_profiles import PART_NAMES
from tools.vega_competition_pipeline import (
    _capture_downward_head_frame,
    _load_runtime,
    _runtime_from_board_scene,
    _task_targets,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = ROOT / "calibration" / "head_fallback_profiles.json"
HOVER_CLEARANCE_M = 0.100
DETECTION_RADIUS_M = 0.100
SPEED_SCALE = 0.42
CAN_INTERFACE = "can1"
CAN_BITRATE = "1000000"


def _finite_pair(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain two numbers")
    result = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _is_can_network_down(exc):
    text = repr(exc).lower()
    return (
        "network is down" in text
        or "error code 100" in text
        or "errno 100" in text
    )


def _bring_up_can1():
    """Best-effort recovery for a present but down SocketCAN interface.

    The robot image normally configures can1 before the competition process,
    but a reboot can leave it down.  Keep this local to the fallback so a
    failed pickup can be retried without tearing down the arm session.
    """
    if not Path(f"/sys/class/net/{CAN_INTERFACE}").exists():
        return "can1 is not present"
    commands = (
        ["ip", "link", "set", CAN_INTERFACE, "up"],
        ["ip", "link", "set", CAN_INTERFACE, "type", "can", "bitrate", CAN_BITRATE],
        ["ip", "link", "set", CAN_INTERFACE, "up"],
    )
    errors = []
    for command in commands:
        try:
            result = subprocess.run(
                command,
                check=False,
                text=True,
                capture_output=True,
                timeout=3.0,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc))
            continue
        if result.returncode == 0:
            return None
        detail = (result.stderr or result.stdout or "").strip()
        errors.append(detail or f"exit {result.returncode}")
    return "; ".join(errors)


def load_profiles(path):
    path = Path(path)
    if not path.is_file():
        return {"schema_version": 1, "camera": "head_camera", "parts": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("head fallback profiles must use schema_version 1")
    if not isinstance(value.get("parts", {}), dict):
        raise ValueError("head fallback profiles parts must be an object")
    return value


def save_profiles(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def _detection_positions(scene, runtime):
    board = scene.get("board") or {}
    raw = board.get("center_base_m_coarse")
    if not isinstance(raw, list) or len(raw) < 2:
        return []
    raw_center = _finite_pair(raw[:2], "scene board center")
    live_center = _finite_pair(runtime[2][0], "live board center")
    shift = (live_center[0] - raw_center[0], live_center[1] - raw_center[1])
    result = []
    for index, item in enumerate(scene.get("parts") or []):
        point = item.get("center_base_m_coarse")
        if not isinstance(point, list) or len(point) < 2:
            continue
        try:
            xy = (float(point[0]) + shift[0], float(point[1]) + shift[1])
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(v) for v in xy):
            board_xy = item.get("center_board_m")
            try:
                if not isinstance(board_xy, (list, tuple)) or len(board_xy) < 2:
                    raise ValueError("missing board-relative part center")
                board_xy = _finite_pair(board_xy[:2], "scene board-relative part center")
            except (TypeError, ValueError):
                board_xy = None
            result.append({"index": index, "xy_m": xy, "board_xy_m": board_xy})
    return result


def match_expected_parts(scene, runtime, expected_targets, task_data=None):
    """Match current detections, preferably in board coordinates.

    When task data is supplied, association is made in the live rectified
    board frame, so board translation/rotation does not turn a known part into
    a wrong robot-frame match.  The returned XY remains the freshly observed
    base-frame center used for coarse motion.
    """
    detections = _detection_positions(scene, runtime)
    used = set()
    observations = {}
    source_center = None
    rotation = 0.0
    mirror_x = False
    mirror_y = False
    if isinstance(task_data, dict):
        try:
            source_center = _finite_pair(
                task_data.get("source_board_center_xy_m"),
                "task source board center",
            )
            rotation = float(task_data.get("task_coordinate_rotation_deg", 0.0))
            mirror_x = bool(task_data.get("task_coordinate_mirror_x", False))
            mirror_y = bool(task_data.get("task_coordinate_mirror_y", False))
        except (TypeError, ValueError):
            source_center = None
    for part in PART_NAMES:
        expected_xy = tuple(float(v) for v in expected_targets[f"task.{part}.pick"].position_m[:2])
        expected_board_xy = None
        if source_center is not None:
            try:
                expected_source = (task_data["parts"][part]["pick"])
                expected_board_xy = board_relative_task_xy(
                    expected_source[:2], source_center,
                    rotation_deg=rotation, mirror_x=mirror_x, mirror_y=mirror_y,
                )
            except (KeyError, TypeError, ValueError):
                expected_board_xy = None
        scored = []
        for item in detections:
            if item["index"] in used:
                continue
            if expected_board_xy is not None and item.get("board_xy_m") is not None:
                distance = math.dist(item["board_xy_m"], expected_board_xy)
            else:
                distance = math.dist(item["xy_m"], expected_xy)
            scored.append((distance, item))
        candidates = sorted(scored, key=lambda entry: (entry[0], entry[1]["index"]))
        selected = None
        reason = "expected_coordinate"
        rejection = "no_detections" if not candidates else "outside_radius"
        if candidates and candidates[0][0] <= DETECTION_RADIUS_M:
            rejection = "ambiguous_candidates"
            if len(candidates) == 1 or candidates[1][0] - candidates[0][0] >= 0.015:
                selected = candidates[0][1]
                used.add(selected["index"])
                reason = "head_detection"
                rejection = None
        selected_xy = list(selected["xy_m"]) if selected else list(expected_xy)
        # With board coordinates available, apply the observed displacement
        # to the calibrated task target.  Do not combine a corrected board
        # center with the uncalibrated camera extrinsic's part axes/scale.
        if selected and expected_board_xy is not None and selected.get("board_xy_m") is not None:
            dx = selected["board_xy_m"][0] - expected_board_xy[0]
            dy = selected["board_xy_m"][1] - expected_board_xy[1]
            _, ux, uy, _ = runtime[2]
            selected_xy = [expected_xy[i] + dx * ux[i] + dy * uy[i] for i in range(2)]
        observations[part] = {
            "expected_xy_m": list(expected_xy),
            "detected_xy_m": list(selected["xy_m"]) if selected else None,
            "selected_xy_m": selected_xy,
            "selection": reason,
            "distance_to_expected_m": candidates[0][0] if candidates else None,
            "detection_index": selected["index"] if selected else None,
            "rejection_reason": rejection,
            "association_frame": "board" if expected_board_xy is not None else "coarse_base",
            "expected_board_xy_m": list(expected_board_xy) if expected_board_xy is not None else None,
            "candidates": [{"detection_index": item["index"], "distance_m": distance}
                           for distance, item in candidates],
        }
    return observations


class HeadFallbackSession:
    def __init__(self, args):
        self.args = args
        self.cfg = load_bundle("vega")["robot"]
        self.cfg["allow_robot_init_head_motion"] = True
        self.cfg["auto_clear_software_estop_on_connect"] = True
        self.floor = float(load_vega_skills()["safety"]["min_tcp_z_m"])
        self.robot = VegaAdapter(self.cfg)
        self.profiles_path = Path(args.profiles)
        if not self.profiles_path.is_absolute():
            self.profiles_path = ROOT / self.profiles_path
        self.profiles = load_profiles(self.profiles_path)
        self.runtime = None
        self.scene = None
        self.observations = {}
        self.targets = None
        self.ready_pose = None
        self.part = None
        self.profile = {}
        self.observation = None
        self.hover_pose = None
        self.grasp_clearance_m = None
        self.holding = False
        self.output = None

    def start(self):
        self.output = ROOT / "runs" / datetime.now(timezone.utc).strftime("head_fallback_%Y%m%dT%H%M%S_%fZ")
        self.output.mkdir(parents=True, exist_ok=False)
        self.robot.connect()
        self.refresh_board()
        self.ready_pose = configured_right_preset(self.cfg, "right_ready")[1]
        self.robot.move_joints(configured_right_preset(self.cfg, "right_ready")[0], speed_scale=SPEED_SCALE)

    def close(self):
        try:
            if self.output is not None:
                (self.output / "run_summary.json").write_text(json.dumps({
                    "schema_version": 1,
                    "status": "holding_requires_inspection" if self.holding else "closed",
                    "part": self.part,
                    "holding_may_be_true": bool(self.holding),
                    "profiles": str(self.profiles_path),
                }, indent=2) + "\n", encoding="utf-8")
        finally:
            self.robot.close()

    def refresh_board(self):
        runtime = _load_runtime()
        error = None
        for attempt in range(1, 4):
            try:
                scene = _capture_downward_head_frame(self.robot, floor_m=self.floor, bundle=runtime[0])
                runtime = _runtime_from_board_scene(runtime, scene)
                self.scene = scene
                break
            except Exception as exc:
                error = exc
                print(f"HEAD FALLBACK BOARD RETRY {attempt}: {type(exc).__name__}: {exc}", flush=True)
        else:
            if not self.args.expected_only:
                raise error
            print("HEAD FALLBACK: using expected coordinates only", flush=True)
            self.scene = {"board": {}, "parts": []}
        self.runtime = runtime
        self.targets = _task_targets(runtime, runtime[1], HOVER_CLEARANCE_M)
        self.observations = match_expected_parts(self.scene, runtime, self.targets)
        if self.output is not None:
            (self.output / "board_scene.json").write_text(json.dumps(self.scene, indent=2, default=str) + "\n", encoding="utf-8")
            (self.output / "observations.json").write_text(json.dumps(self.observations, indent=2) + "\n", encoding="utf-8")
        for part in PART_NAMES:
            obs = self.observations[part]
            print(f"{part}: {obs['selection']} expected={tuple(round(v, 4) for v in obs['expected_xy_m'])} selected={tuple(round(v, 4) for v in obs['selected_xy_m'])}", flush=True)

    def surface(self, x, y):
        from tools.vega_task_coordinate_reachability import calibrated_surface_z
        return calibrated_surface_z(x, y, self.runtime[2][3])

    def make_hover(self, x, y, yaw):
        base = self.ready_pose or configured_right_preset(self.cfg, "right_ready")[1]
        oriented = _yaw_pose(base, yaw)
        return Pose((float(x), float(y), self.surface(x, y) + HOVER_CLEARANCE_M), oriented.quaternion_wxyz)

    def move(self, target, *, slow=False):
        current = self.robot.get_tcp_pose()
        if current is None:
            raise RuntimeError("invalid current pose")
        distance, angle = pose_distance(current, target)
        count = max(1, math.ceil(distance / 0.025), math.ceil(angle / 0.12))
        seed = self.robot._read_joint_positions()
        for index in range(1, count + 1):
            waypoint = interpolate_pose(current, target, index / count)
            seed = self.robot._kinematics.solve(waypoint, seed)
        move_tcp_segmented(self.robot, target, speed_scale=0.25 if slow else SPEED_SCALE,
                           max_translation_step_m=0.025, max_orientation_step_rad=0.12,
                           min_tcp_z_m=None)

    def select_part(self, part):
        if self.holding:
            raise RuntimeError("a part may be held; type return or abort")
        self.part = part
        self.profile = (self.profiles.get("parts") or {}).get(part) or {}
        self.observation = self.observations[part]
        offset = _finite_pair(self.profile.get("offset_base_xy_m", [0, 0]), "offset_base_xy_m")
        yaw = float(self.profile.get("yaw_deg", 0.0))
        if not math.isfinite(yaw) or not -45 <= yaw <= 45:
            raise ValueError("saved fallback yaw must be -45..45 degrees")
        xy = (self.observation["selected_xy_m"][0] + offset[0], self.observation["selected_xy_m"][1] + offset[1])
        self.hover_pose = self.make_hover(*xy, yaw)
        depth = self.profile.get("grasp_depth_mm")
        if depth is None:
            self.grasp_clearance_m = None
        else:
            depth = float(depth)
            if not math.isfinite(depth) or not 0 <= depth <= 100:
                raise ValueError("saved grasp depth must be 0..100.0 mm below hover")
            self.grasp_clearance_m = HOVER_CLEARANCE_M - depth / 1000.0
        print(f"SELECTED {part}: {self.observation['selection']} target={tuple(round(v, 4) for v in self.hover_pose.position_m)}", flush=True)
        self.move(self.hover_pose)

    def adjust(self, direction, amount_mm):
        amount = float(amount_mm) / 1000.0
        if not math.isfinite(amount) or not 0.0001 <= amount <= 0.050:
            raise ValueError("adjustment must be 0.1..50 mm")
        dx, dy = {"forward": (amount, 0), "back": (-amount, 0), "left": (0, amount), "right": (0, -amount)}[direction]
        x, y = self.hover_pose.position_m[:2]
        self.hover_pose = self.make_hover(x + dx, y + dy, float(self.profile.get("yaw_deg", 0)))
        self.move(self.hover_pose, slow=True)
        # ``amount_mm`` arrives from the command parser as text.  Format the
        # validated numeric value so an otherwise successful jog cannot abort
        # with ``Unknown format code 'g'``.
        print(f"ADJUSTED {direction} {float(amount_mm):g} mm -> {self.hover_pose.position_m}", flush=True)

    def set_yaw(self, degrees):
        yaw = float(degrees)
        if not math.isfinite(yaw) or not -45 <= yaw <= 45:
            raise ValueError("yaw must be -45..45 degrees")
        x, y = self.hover_pose.position_m[:2]
        self.profile["yaw_deg"] = yaw
        self.hover_pose = self.make_hover(x, y, yaw)
        self.move(self.hover_pose, slow=True)

    def set_depth(self, depth_mm):
        depth = float(depth_mm)
        if not math.isfinite(depth) or not 0 <= depth <= 100:
            raise ValueError("depth must be 0..100.0 mm below hover")
        clearance = HOVER_CLEARANCE_M - depth / 1000.0
        self.grasp_clearance_m = clearance
        grasp_z = self.surface(*self.hover_pose.position_m[:2]) + clearance
        print(
            f"DEPTH ACCEPTED = {depth:.1f} mm below hover; "
            f"TCP is {clearance*1000:.1f} mm above surface at z={grasp_z:.4f} m",
            flush=True,
        )

    def save(self):
        if self.holding:
            raise RuntimeError("return the part before saving")
        expected = self.observation["selected_xy_m"]
        self.profiles.setdefault("parts", {})[self.part] = {
            "offset_base_xy_m": [self.hover_pose.position_m[0] - expected[0], self.hover_pose.position_m[1] - expected[1]],
            "grasp_depth_mm": None if self.grasp_clearance_m is None else (HOVER_CLEARANCE_M - self.grasp_clearance_m) * 1000,
            "yaw_deg": float(self.profile.get("yaw_deg", 0)),
            "source": self.observation["selection"],
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_profiles(self.profiles_path, self.profiles)
        print(f"HEAD FALLBACK PROFILE SAVED: {self.profiles_path}", flush=True)

    def grab(self):
        if self.grasp_clearance_m is None:
            raise RuntimeError("set depth N before grab")
        x, y = self.hover_pose.position_m[:2]
        grasp = Pose((x, y, self.surface(x, y) + self.grasp_clearance_m), self.hover_pose.quaternion_wxyz)
        # Connect and open while still safely at hover.  A CAN failure must
        # never leave the arm down at the part or abort the operator session.
        try:
            self._connect_gripper_with_recovery()
            self.robot.open_gripper(self.part)
        except Exception as exc:
            print(f"GRIPPER NOT READY: {type(exc).__name__}: {exc}", flush=True)
            if _is_can_network_down(exc):
                print(
                    "CAN recovery failed; run `ip link show can1` and bring can1 "
                    "up before retrying grab",
                    flush=True,
                )
            return

        descent_attempted = False
        self.holding = False
        result = None
        try:
            descent_attempted = True
            self.move(grasp, slow=True)
            # From this point on a failed CAN reply may still leave the part
            # between the jaws, so retain the holding state conservatively.
            self.holding = True
            result = self.robot.grip(self.part)
            print(f"GRIP RESULT = {json.dumps(result, default=str)}", flush=True)
        except Exception as exc:
            print(f"GRIP COMMAND FAILED: {type(exc).__name__}: {exc}", flush=True)
            result = None
        finally:
            if descent_attempted:
                try:
                    self.move(self.hover_pose, slow=True)
                except Exception as exc:
                    print(f"LIFT AFTER GRIP FAILED: {type(exc).__name__}: {exc}", flush=True)
        if not isinstance(result, dict) or result.get("gripped") is not True:
            print("GRIP NOT VERIFIED; holding state retained for inspection", flush=True)
        else:
            print("PICK VERIFIED; type return before another part", flush=True)

    def _connect_gripper_with_recovery(self):
        try:
            self.robot.connect_gripper()
            return
        except Exception as first_error:
            if not _is_can_network_down(first_error):
                raise
            print("CAN1 IS DOWN; attempting automatic can1 recovery", flush=True)
            recovery_error = _bring_up_can1()
            if recovery_error:
                raise RuntimeError(
                    f"can1 recovery failed: {recovery_error}; "
                    "bring can1 up, then retry grab"
                ) from first_error
            print("CAN1 RECOVERED; retrying gripper initialization", flush=True)
            self.robot.connect_gripper()

    def return_part(self):
        if not self.holding:
            raise RuntimeError("no part is marked held")
        x, y = self.hover_pose.position_m[:2]
        release = Pose((x, y, self.surface(x, y) + (self.grasp_clearance_m or 0.02)), self.hover_pose.quaternion_wxyz)
        self.move(release, slow=True)
        self.robot.release_gripper(self.part)
        self.holding = False
        self.move(self.hover_pose, slow=True)

    def run_part(self, part):
        self.select_part(part)
        print("Commands: forward/back/left/right N, yaw N, depth 0..100, grab, return, save, retake, status, abort")
        while True:
            raw = input(f"head-fallback {part}> ").strip().lower().split()
            if not raw:
                continue
            command = raw[0]
            try:
                if command in ("abort", "exit", "q"):
                    return 3 if self.holding else 1
                if command in ("forward", "back", "left", "right") and len(raw) == 2:
                    self.adjust(command, raw[1])
                elif command == "yaw" and len(raw) == 2:
                    self.set_yaw(raw[1])
                elif command == "depth" and len(raw) == 2:
                    self.set_depth(raw[1])
                elif command == "grab":
                    self.grab()
                elif command == "return":
                    self.return_part()
                elif command == "save":
                    self.save()
                    return 0
                elif command == "retake" and not self.holding:
                    self.refresh_board()
                    self.ready_pose = configured_right_preset(self.cfg, "right_ready")[1]
                    self.select_part(part)
                elif command == "status":
                    print(json.dumps({"target": self.hover_pose.position_m, "holding": self.holding, "grasp_clearance_m": self.grasp_clearance_m}, default=str))
                else:
                    print("Use forward/back/left/right N, yaw N, depth N, grab, return, save, retake, status, or abort")
            except Exception as exc:
                print(f"COMMAND FAILED (session remains active): {type(exc).__name__}: {exc}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "run"), default="calibrate")
    parser.add_argument("--part", choices=PART_NAMES)
    parser.add_argument("--profiles", default=str(DEFAULT_PROFILES.relative_to(ROOT)))
    parser.add_argument("--expected-only", action="store_true")
    parser.add_argument("--confirm-head-motion", action="store_true")
    parser.add_argument("--confirm-physical-motion", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_head_motion or not args.confirm_physical_motion:
        parser.error("requires --confirm-head-motion and --confirm-physical-motion")
    session = HeadFallbackSession(args)
    try:
        session.start()
        if args.part:
            if args.mode == "run" and args.part not in (session.profiles.get("parts") or {}):
                raise RuntimeError(f"no saved profile for {args.part}; calibrate it first")
            return session.run_part(args.part)
        choices = list(PART_NAMES)
        while True:
            print("\nHEAD FALLBACK PARTS")
            for index, part in enumerate(choices, 1):
                state = "profile" if part in (session.profiles.get("parts") or {}) else "uncalibrated"
                print(f"  {index}. {part} ({state})")
            raw = input("Choose part number/name, or 0 to exit: ").strip()
            if raw in ("0", "q", "exit", "quit"):
                return 0
            part = choices[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(choices) else raw
            if part not in choices:
                print("Unknown part")
                continue
            if args.mode == "run" and part not in (session.profiles.get("parts") or {}):
                print("No saved profile; calibrate this part first")
                continue
            result = session.run_part(part)
            if result == 3:
                return result
    except (KeyboardInterrupt, EOFError):
        print("Head fallback cancelled; inspect holding state before recovery")
        return 1
    except Exception as exc:
        print(f"HEAD FALLBACK STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
