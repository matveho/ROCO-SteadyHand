"""Operator menu for the calibrated Vega competition workflow.

The board is treated as movable state. Recalibration is the first menu action;
all position tests and task targets are rebuilt from the resulting calibration
file before any arm motion is planned.
"""

from collections import OrderedDict
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.vega import VegaAdapter
from steadyhand.cameras.vega import VegaHeadCamera
from steadyhand.config import load_bundle
from steadyhand.board_geometry import (
    configured_board_plane_z,
    validate_task_board_geometry,
    validate_task_coordinate_extent,
)
from steadyhand.executor import move_tcp_segmented, preflight_tcp_segmented
from steadyhand.kinematics import IKError
from steadyhand.operator_input import clean_choice
from steadyhand.execution_offsets import load_offsets, describe_offsets
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills
from steadyhand.vega_presets import configured_right_preset
from steadyhand.vega_camera_clear import move_camera_clear_for_image
from steadyhand.vision.scene import detect_head_task_scene
from steadyhand.wrist_part_profiles import PART_NAMES, load_profiles
from tools.vega_board_five_point_calibrate import main as run_five_point_calibration
from tools.vega_wrist_part_calibrate import main as run_wrist_part_calibration
from tools.vega_task_coordinate_reachability import (
    _finite_vector,
    calibrated_surface_z,
    _live_pose,
    _load_manual,
    _resolve_point,
)

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "calibration" / "vega_board_manual.json"
FALLBACK_CALIBRATION = ROOT / "calibration" / "vega_board_manual_fallback.json"
TASK_COORDINATES = ROOT / "configs" / "task_coordinates.json"
DEFAULT_TASK_CLEARANCE_MM = 100.0
# The onsite operator measured a consistent 12 mm forward bias in the task
# annotations.  Keep the source JSON immutable and apply this runtime correction
# to every task point (pick/place/connect/grade) after board registration.
TASK_FORWARD_OFFSET_M = 0.012
DEFAULT_PIPELINE_SPEED_SCALE = 0.38
COMPETITION_PLAN = ROOT / "configs" / "competition_plan.json"
COMPETITION_ACTIONS = ROOT / "configs" / "competition_actions.json"


COMPETITION_TASKS = OrderedDict([
    ("battery_size1_pick_v0", "legacy battery approach only; calibrated hover, no gripper"),
    ("battery_size1_pick_v1", "battery grasp and lift; experimental scaffold"),
    ("battery_size1_pick_place_v1", "battery pick, transfer, and place; experimental scaffold"),
    ("gear_60teeth_pick_place_v1", "gear pick/place; scaffold"),
    ("gear_20teeth_pick_place_v1", "gear pick/place; scaffold"),
    ("rod_16mm_pick_place_v1", "rod pick/snap place; scaffold"),
    ("bolt_8mm_pick_place_v1", "bolt pick/snap place; scaffold"),
    ("usb_a_pick_place_v1", "USB pick/snap place; scaffold"),
    ("hdmi_pick_place_v1", "HDMI pick/snap place; scaffold"),
    ("pin_pick_place_v1", "pin pick/snap place; scaffold"),
    ("battery_size5_pick_place_v1", "slim battery pick/place; scaffold"),
])


# Default scoring order is deliberately conservative: the large battery and
# larger, textured/circular parts are easiest to acquire first.  The JSON plan
# is the operator-editable source of truth for tomorrow's hardware evidence.
DEFAULT_PICK_PRIORITY = (
    "battery_size1", "gear_20teeth", "gear_60teeth", "pin", "bolt_8mm",
    "battery_size5", "rod_16mm", "usb_a", "hdmi",
)


TASK_ACTIONS = OrderedDict(
    (f"{part}.{action}",
     f"{action.replace('_', ' ')} test; requires a taught wrist profile")
    for part in PART_NAMES
    for action in ("pick", "pick_place")
)


COMPETITION_SEQUENCE_ACTIONS = OrderedDict(
    (str(index), f"{part}.pick_place")
    for index, part in enumerate(DEFAULT_PICK_PRIORITY, 1)
)


def _load_competition_plan():
    """Load small operator-tuning values without importing robot state."""
    defaults = {
        "pick_priority": list(DEFAULT_PICK_PRIORITY),
        "max_retries_per_part": 1,
        "default_action": "pick_place",
        "pipeline_speed_scale": DEFAULT_PIPELINE_SPEED_SCALE,
        "task_clearance_mm": DEFAULT_TASK_CLEARANCE_MM,
    }
    if not COMPETITION_PLAN.is_file():
        return defaults
    value = json.loads(COMPETITION_PLAN.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("competition plan must be a JSON object")
    if value.get("schema_version", 1) != 1:
        raise ValueError("unsupported competition plan schema_version")
    priority = value.get("pick_priority", defaults["pick_priority"])
    if not isinstance(priority, list) or len(priority) != len(PART_NAMES) or set(priority) != set(PART_NAMES):
        raise ValueError("competition plan pick_priority must list every known part exactly once")
    retries_raw = value.get("max_retries_per_part", defaults["max_retries_per_part"])
    try:
        retries_float = float(retries_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("competition plan max_retries_per_part must be an integer 0..2") from exc
    if not retries_float.is_integer():
        raise ValueError("competition plan max_retries_per_part must be an integer 0..2")
    retries = int(retries_float)
    if not 0 <= retries <= 2:
        raise ValueError("competition plan max_retries_per_part must be 0..2")
    action = value.get("default_action", defaults["default_action"])
    if action not in ("pick", "pick_place"):
        raise ValueError("competition plan default_action must be pick or pick_place")
    speed = float(value.get("pipeline_speed_scale", defaults["pipeline_speed_scale"]))
    clearance = float(value.get("task_clearance_mm", defaults["task_clearance_mm"]))
    if not 0.25 <= speed <= 0.70:
        raise ValueError("competition plan pipeline_speed_scale must be 0.25..0.70")
    if not 20.0 <= clearance <= 100.0:
        raise ValueError("competition plan task_clearance_mm must be 20..100")
    return {
        "pick_priority": priority,
        "max_retries_per_part": retries,
        "default_action": action,
        "pipeline_speed_scale": speed,
        "task_clearance_mm": clearance,
    }


def _load_competition_actions():
    """Load the one operator-editable competition routine configuration."""
    def strict_bool(value, label, default):
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        raise ValueError(f"{label} must be true or false")

    def strict_int(value, label, low, high):
        # bool is an int subclass, but accepting true as one attempt makes a
        # mistyped JSON file surprisingly dangerous.
        if isinstance(value, bool):
            raise ValueError(f"{label} must be an integer {low}..{high}")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be an integer {low}..{high}") from exc
        if not number.is_integer() or not low <= number <= high:
            raise ValueError(f"{label} must be an integer {low}..{high}")
        return int(number)

    defaults = {
        "order": list(DEFAULT_PICK_PRIORITY),
        "retries_per_action": 1,
        "max_attempts_per_action": 2,
        "use_wrist_cv": True,
        "retry_without_wrist_cv": True,
        "use_place_cv": True,
        "head_reacquire_on_failure": True,
        "pipeline_speed_scale": DEFAULT_PIPELINE_SPEED_SCALE,
        "task_clearance_mm": DEFAULT_TASK_CLEARANCE_MM,
        "visual_center_backoff_mm": 15.0,
        "parts": {
            part: {
                "enabled": True,
                "mode": "auto",
                "pick_enabled": True,
                "place_enabled": True,
                "use_wrist_pick_cv": True,
                "use_place_cv": True,
                "max_attempts": 2,
            }
            for part in PART_NAMES
        },
    }
    if not COMPETITION_ACTIONS.is_file():
        return defaults
    value = json.loads(COMPETITION_ACTIONS.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version", 1) != 1:
        raise ValueError("competition_actions.json must use schema_version 1")
    order = value.get("order", defaults["order"])
    if not isinstance(order, list) or set(order) != set(PART_NAMES) or len(order) != len(PART_NAMES):
        raise ValueError("competition_actions.json order must list every known part exactly once")
    parts = value.get("parts", defaults["parts"])
    if not isinstance(parts, dict) or set(parts) != set(PART_NAMES):
        raise ValueError("competition_actions.json parts must configure every known part")
    try:
        retries = strict_int(value.get("retries_per_action", defaults["retries_per_action"]),
                             "retries_per_action", 0, 2)
        attempts = strict_int(value.get("max_attempts_per_action", retries + 1),
                              "max_attempts_per_action", 1, 3)
        speed = float(value.get("pipeline_speed_scale", defaults["pipeline_speed_scale"]))
        clearance = float(value.get("task_clearance_mm", defaults["task_clearance_mm"]))
        backoff = float(value.get("visual_center_backoff_mm", defaults["visual_center_backoff_mm"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("competition_actions.json numeric settings are invalid") from exc
    if not 0 <= retries <= 2:
        raise ValueError("retries_per_action must be 0..2")
    if not 1 <= attempts <= 3:
        raise ValueError("max_attempts_per_action must be 1..3")
    retries = min(2, max(0, attempts - 1))
    if not 0.25 <= speed <= 0.70:
        raise ValueError("pipeline_speed_scale must be 0.25..0.70")
    if not 20.0 <= clearance <= 100.0:
        raise ValueError("task_clearance_mm must be 20..100")
    if not 0.0 <= backoff <= 30.0:
        raise ValueError("visual_center_backoff_mm must be 0..30")
    global_pick_cv = strict_bool(value.get("use_wrist_cv"), "use_wrist_cv", True)
    global_place_cv = strict_bool(value.get("use_place_cv"), "use_place_cv", True)
    normalized_parts = {}
    for part in PART_NAMES:
        entry = parts[part]
        if not isinstance(entry, dict):
            raise ValueError(f"competition_actions.json entry for {part} must be an object")
        mode = entry.get("mode", "auto")
        if mode not in ("auto", "pick", "pick_place"):
            raise ValueError(f"competition action mode for {part} must be auto, pick, or pick_place")
        part_attempts = strict_int(entry.get("max_attempts", attempts),
                                   f"max_attempts for {part}", 1, 3)
        normalized_parts[part] = {
            "enabled": strict_bool(entry.get("enabled"), f"{part}.enabled", True),
            "mode": mode,
            # These explicit fields are the preferred contract.  The older
            # enabled/mode/global fields remain accepted for old deployments.
            "pick_enabled": strict_bool(entry.get("pick_enabled"), f"{part}.pick_enabled", True),
            "place_enabled": strict_bool(entry.get("place_enabled"), f"{part}.place_enabled", mode != "pick"),
            "use_wrist_pick_cv": strict_bool(entry.get("use_wrist_pick_cv"), f"{part}.use_wrist_pick_cv", global_pick_cv),
            "use_place_cv": strict_bool(entry.get("use_place_cv"), f"{part}.use_place_cv", global_place_cv),
            "max_attempts": part_attempts,
        }
    return {
        "order": order,
        "retries_per_action": retries,
        "max_attempts_per_action": attempts,
        "use_wrist_cv": global_pick_cv,
        "retry_without_wrist_cv": strict_bool(value.get("retry_without_wrist_cv"), "retry_without_wrist_cv", True),
        "use_place_cv": global_place_cv,
        "head_reacquire_on_failure": strict_bool(value.get("head_reacquire_on_failure"), "head_reacquire_on_failure", True),
        "pipeline_speed_scale": speed,
        "task_clearance_mm": clearance,
        "visual_center_backoff_mm": backoff,
        "parts": normalized_parts,
    }


def _reload_operator_settings(args):
    """Validate and display editable settings; apply them to the live menu."""
    plan = _load_competition_plan()
    task_data = json.loads(TASK_COORDINATES.read_text(encoding="utf-8"))
    validate_task_board_geometry(task_data)
    cfg = load_bundle("vega")["robot"]
    profiles = load_profiles(ROOT / "calibration" / "wrist_part_profiles.json", cfg)
    if not getattr(args, "speed_scale_cli", False):
        args.speed_scale = plan["pipeline_speed_scale"]
    if not getattr(args, "clearance_mm_cli", False):
        args.clearance_mm = plan["task_clearance_mm"]
        args.clearance_m = args.clearance_mm / 1000.0
    print("OPERATOR SETTINGS RELOADED", flush=True)
    print(f"  speed_scale={args.speed_scale:.3f} clearance_mm={args.clearance_mm:.1f}", flush=True)
    print(f"  retries_per_part={plan['max_retries_per_part']}", flush=True)
    print("  priority=" + ", ".join(plan["pick_priority"]), flush=True)
    print(f"  wrist_profiles={len(profiles.get('parts') or {})}/{len(PART_NAMES)}", flush=True)
    print("  task_coordinates=valid reviewed 386 mm annotation map", flush=True)
    return 0


def _competition_sequence_actions():
    plan = _load_competition_plan()
    return OrderedDict(
        (str(index), f"{part}.pick_place")
        for index, part in enumerate(plan["pick_priority"], 1)
    )


def _load_runtime():
    bundle = load_bundle("vega")
    task_data = json.loads(TASK_COORDINATES.read_text(encoding="utf-8"))
    validate_task_board_geometry(task_data)
    path = CALIBRATION if CALIBRATION.is_file() else FALLBACK_CALIBRATION
    center, ux, uy, plane = _load_manual(path, bundle["robot"])
    _, ready_pose = configured_right_preset(bundle["robot"], "right_ready")
    plane["calibration_ready_quaternion"] = list(ready_pose.quaternion_wxyz)
    return bundle, task_data, (center, ux, uy, plane), ready_pose


def _board_targets(runtime, clearance_m):
    _, _, (center, _, _, plane), ready_pose = runtime
    dynamic_corners = plane.get("board_corners_xy") if isinstance(plane, dict) else None
    if dynamic_corners:
        targets = OrderedDict()
        targets["board.center"] = Pose(
            (center[0], center[1], calibrated_surface_z(center[0], center[1], plane) + clearance_m),
            ready_pose.quaternion_wxyz,
        )
        for label in ("TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT"):
            x, y = dynamic_corners[label]
            targets[f"board.{label.lower()}"] = Pose(
                (x, y, calibrated_surface_z(x, y, plane) + clearance_m),
                ready_pose.quaternion_wxyz,
            )
        return targets
    path = Path(plane.get("calibration_path", "")) if isinstance(plane, dict) else None
    if path is None or not path.is_file():
        path = CALIBRATION if CALIBRATION.is_file() else FALLBACK_CALIBRATION
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples = raw.get("samples") or {}
    targets = OrderedDict()
    for label in ("CENTER", "TOP_RIGHT", "BOTTOM_RIGHT", "BOTTOM_LEFT"):
        pose = (samples.get(label) or {}).get("tip_r_pose")
        if not isinstance(pose, dict):
            raise ValueError(f"calibration is missing corrected {label}")
        xyz = _finite_vector(pose.get("position_m"), 3, f"{label}.position_m")
        targets[f"board.{label.lower()}"] = Pose(
            (xyz[0], xyz[1], calibrated_surface_z(xyz[0], xyz[1], runtime[2][3]) + clearance_m),
            ready_pose.quaternion_wxyz,
        )
    return targets


def _task_targets(runtime, task_data, clearance_m, *, use_profiles=True):
    bundle, _, (center, ux, uy, plane), ready_pose = runtime
    source_center = _finite_vector(task_data.get("source_board_center_xy_m"), 2, "source board center")
    rotation_deg = float(task_data.get("task_coordinate_rotation_deg", 0.0))
    mirror_x = bool(task_data.get("task_coordinate_mirror_x", False))
    # One physically validated orientation across every menu.
    mirror_y = False
    targets = OrderedDict()
    for part in task_data["official_order"]:
        for kind in task_data["parts"][part]:
            if kind not in ("pick", "place", "connect", "grade"):
                continue
            name = f"{part}.{kind}"
            source_xyz = _resolve_point(name, task_data)[2]
            pose = _live_pose(
                source_xyz, source_center=source_center,
                live_center=center, ux=ux, uy=uy,
                surface_plane=plane, clearance_m=clearance_m,
                quat=ready_pose.quaternion_wxyz, rotation_deg=rotation_deg,
                mirror_x=mirror_x, mirror_y=mirror_y,
            )
            # Forward is +base-X on this robot. Recompute Z on the calibrated
            # plane after shifting so the hover remains parallel to the board.
            from steadyhand.board_relative import snapshot, rotation_between
            reference = plane.get("calibration_board_reference")
            delta = (TASK_FORWARD_OFFSET_M, 0.)
            if reference:
                delta = rotation_between(reference, snapshot(runtime[2])) @ delta
            shifted_x = pose.position_m[0] + delta[0]
            shifted_y = pose.position_m[1] + delta[1]
            targets[f"task.{name}"] = Pose(
                (
                    shifted_x,
                    shifted_y,
                    calibrated_surface_z(shifted_x, shifted_y, plane) + clearance_m,
                ),
                pose.quaternion_wxyz,
            )
    if use_profiles:
        from steadyhand.board_relative import resolve_profile_target
        profiles = load_profiles(ROOT / "calibration/wrist_part_profiles.json", bundle["robot"])
        for part, profile in profiles.get("parts", {}).items():
            for action in ("pick", "place"):
                name = f"task.{part}.{action}"
                if name not in targets or (action == "place" and not profile.get("place")):
                    continue
                xy, quat = resolve_profile_target(profile, runtime[2], ready_pose,
                    targets[name], action=action, no_cv=True)
                targets[name] = Pose((*xy, calibrated_surface_z(*xy, plane) + clearance_m), quat)

    return targets


def _choose(items, title, *, allow_all=False):
    names = list(items)
    print(f"\n{title}")
    for index, name in enumerate(names, 1):
        description = items[name] if isinstance(items, dict) else ""
        print(f"  {index}. {name}{(': ' + description) if description else ''}")
    print("  0. back")
    raw = clean_choice(input("Type a number or name (comma-separated for tests): "))
    if raw in ("0", "b", "back", ""):
        return []
    values = [part.strip() for part in raw.split(",")]
    selected = []
    for value in values:
        if allow_all and value.lower() == "all":
            return names
        try:
            index = int(value)
            if not 1 <= index <= len(names):
                raise ValueError
            name = names[index - 1]
        except ValueError:
            name = value
            if name not in names:
                print(f"Unknown option: {value}")
                return []
        if name not in selected:
            selected.append(name)
    return selected


def _make_test_targets(selected, runtime, task_data, clearance_m):
    validate_task_coordinate_extent(
        task_data,
        names=[name.removeprefix("task.") for name in selected if name.startswith("task.")],
    )
    targets = OrderedDict()
    board = _board_targets(runtime, clearance_m)
    task = _task_targets(runtime, task_data, clearance_m)
    available = OrderedDict(list(board.items()) + list(task.items()))
    for name in selected:
        targets[name] = available[name]
    return targets


def _capture_downward_head_frame(robot, *, floor_m, bundle, speed_scale=None,
                                 checkpoint=None, output=None):
    """Move the arm/head clear, capture one downward board frame, then return."""
    if checkpoint:
        checkpoint("before_camera_clear")
    move_camera_clear_for_image(robot, floor_m=floor_m,
                                speed_scale=.90 if speed_scale is None else speed_scale)
    if checkpoint:
        checkpoint("after_camera_clear")
        checkpoint("before_head_down")
    head_before = list(robot._robot.head.get_joint_pos())
    print("HEAD BEFORE =", head_before, flush=True)
    # Do not preserve a stale pitch/yaw from a previous operation.  The board
    # detector is calibrated for this complete downward pose; changing only
    # head_j1 can leave the camera looking sideways while the log appears to
    # show a successful head move.
    head_q = [0.55, 0.0, 0.0]
    move_head = getattr(robot._robot.head, "move_to_joint_pos", None)
    moved_head = False
    if callable(move_head):
        try:
            handle = move_head(head_q, velocity_scale=.45 if speed_scale is None else speed_scale)
            wait_fn = getattr(handle, "wait", None)
            if callable(wait_fn):
                state = wait_fn(timeout=5.0)
                if state not in (None, "finished"):
                    raise RuntimeError(f"head motion ended in state {state!r}")
            else:
                time.sleep(1.5)
            moved_head = True
        except RuntimeError as exc:
            print(
                f"HEAD MOTION HANDLE FAILED ({exc}); falling back to set_joint_pos",
                flush=True,
            )
    if not moved_head:
        robot._robot.head.set_joint_pos(
            head_q, wait_time=1.2, exit_on_reach=True,
            exit_on_reach_kwargs={"tolerance": 0.02},
        )
    measured_head_q = list(robot._robot.head.get_joint_pos())
    print("HEAD AFTER  =", measured_head_q, flush=True)
    if len(measured_head_q) != len(head_q) or not all(
        math.isfinite(float(q)) for q in measured_head_q
    ) or max(
        abs(float(measured_head_q[i]) - head_q[i]) for i in range(len(head_q))
    ) > 0.03:
        raise RuntimeError(
            f"downward head view was not reached: target={head_q} "
            f"measured={measured_head_q}. The head did not follow its command; "
            "no board image will be used. Check head motor/E-stop status before retrying; "
            "this is not a board-calibration error"
        )
    time.sleep(0.5)
    camera = VegaHeadCamera()
    try:
        camera.connect()
        frame = camera.read(include_depth=False, timeout_s=15.0)
    finally:
        camera.close()
    print("BOARD CAMERA FRAME CAPTURED", flush=True)
    if output is not None:
        import cv2
        path = Path(output) / f"head_board_{time.time_ns()}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(frame.left_rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("Failed to save head camera evidence")
    cfg = bundle["robot"]
    try:
        scene = detect_head_task_scene(
            frame.left_rgb,
            frame.camera_info,
            measured_head_q,
            plane_z_m=configured_board_plane_z(
                bundle["robot"], floor_m
            ),
            lift_m=float(cfg["kinematics"]["fixed_joint_values"]["Lift"]),
            torso_flip_rad=float(cfg["kinematics"]["fixed_joint_values"]["torso_flip"]),
            layout="unlabeled",
        )
    except (RuntimeError, ValueError) as exc:
        # Image geometry failure, after the camera and head succeeded. The
        # caller can recapture; never hide a head-motion or camera-read error.
        scene = {"board": {}, "parts": [], "perception_error": str(exc)}
        print(f"BOARD PERCEPTION REJECTED: {exc}", flush=True)
    if output is not None:
        scene["head_image_path"] = str(path)
        path.with_suffix(".json").write_text(json.dumps(scene, indent=2, default=str) + "\n")
    return scene


def _runtime_from_board_scene(runtime, scene):
    """Apply differential board motion; never substitute coarse absolute XY."""
    from copy import deepcopy
    from steadyhand.board_relative import register, rotation_between, rotate_quaternion
    bundle, task_data, (center, ux, uy, old_plane), ready_pose = runtime
    plane = deepcopy(old_plane)
    reference = plane.get("calibration_board_reference")
    try:
        if not isinstance(scene, dict) or not reference:
            raise ValueError("missing camera scene or reference observation")
        current = register(reference, plane["calibration_camera_board"], scene["board"])
    except (ValueError, TypeError, KeyError) as exc:
        plane["registration"] = {"status": "stale", "reason": str(exc),
                                 "source": "last_known_good_frame"}
        print(f"BOARD REGISTRATION STALE: {exc}; retaining the last known frame. "
              "Board motion has NOT been measured.", flush=True)
        return bundle, task_data, (center, ux, uy, plane), ready_pose
    plane["registration"] = current["registration"]
    rotation = rotation_between(reference, current)
    angle = math.atan2(rotation[1, 0], rotation[0, 0])
    from steadyhand.board_relative import base_xy, local_xy
    # Keep the empirical surface and its local residual anchors unchanged.
    # Board corner hover targets, unlike Z, move with the registered board.
    calibrated_corners = plane.get("calibration_board_corners_xy") or {}
    plane["board_corners_xy"] = {
        label: base_xy(current, local_xy(reference, calibrated_corners[label])
                       if label in calibrated_corners else point)
        for label, point in (("TOP_RIGHT", (.193, -.193)),
                             ("BOTTOM_RIGHT", (.193, .193)),
                             ("BOTTOM_LEFT", (-.193, .193)))
    }
    quat = plane.get("calibration_ready_quaternion", ready_pose.quaternion_wxyz)
    ready_pose = Pose(ready_pose.position_m, rotate_quaternion(quat, angle))
    print("BOARD REGISTRATION " + json.dumps(plane["registration"]), flush=True)
    return bundle, task_data, (tuple(current["center_base_xy_m"]),
        tuple(current["board_x_unit_base_xy"]), tuple(current["board_y_unit_base_xy"]), plane), ready_pose


def _capture_registered_board(robot, runtime, *, floor_m, output=None, checkpoint=None,
                              speed_scale=None):
    """Two image attempts for rejected geometry; hardware errors still escape."""
    scene = None
    updated = runtime
    if output is None:
        output = ROOT / "runs" / "board_relative_previews" / str(time.time_ns())
        output.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        scene = _capture_downward_head_frame(robot, floor_m=floor_m, bundle=runtime[0],
            output=output, checkpoint=checkpoint, speed_scale=speed_scale)
        updated = _runtime_from_board_scene(runtime, scene)
        if updated[2][3].get("registration", {}).get("status") == "fresh":
            break
        if attempt == 0:
            print("BOARD REGISTRATION: recapturing once before using the saved frame", flush=True)
    try:
        _board_target_evidence(updated, scene, output)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"BOARD REVIEW IMAGE unavailable: {exc}", flush=True)
    return updated, scene


def _board_target_evidence(runtime, scene, output):
    """Save projected taught targets; expose the review through the existing image helper."""
    import cv2
    import numpy as np
    import shutil
    from steadyhand.board_relative import snapshot, local_xy
    reference = snapshot(runtime[2])
    targets = _task_targets(runtime, runtime[1], .100)
    report = {"board_reference": reference, "targets": {
        name: {"position_m": list(pose.position_m), "quaternion_wxyz": list(pose.quaternion_wxyz)}
        for name, pose in targets.items()}}
    (Path(output) / "board_projected_targets.json").write_text(json.dumps(report, indent=2) + "\n")
    source = scene.get("head_image_path")
    if not source:
        return
    rgb = cv2.imread(str(source))
    if rgb is None:
        raise ValueError("head evidence image could not be read")
    pixels = np.float32([scene["board"]["corners_px"][k] for k in ("tl", "tr", "br", "bl")])
    board = np.float32([[-.193, -.193], [.193, -.193], [.193, .193], [-.193, .193]])
    transform = cv2.getPerspectiveTransform(board, pixels)
    cv2.polylines(rgb, [pixels.astype(np.int32)], True, (0, 255, 0), 2)
    for name, pose in targets.items():
        if not name.endswith((".pick", ".place")):
            continue
        xy = local_xy(reference, pose.position_m[:2])
        uv = cv2.perspectiveTransform(np.float32([[xy]]), transform)[0, 0]
        u, v = (int(round(float(n))) for n in uv)
        if not (0 <= u < rgb.shape[1] and 0 <= v < rgb.shape[0]):
            continue
        color = (0, 255, 255) if name.endswith(".pick") else (255, 0, 255)
        cv2.drawMarker(rgb, (u, v), color, cv2.MARKER_CROSS, 12, 2)
        cv2.putText(rgb, name[5:], (max(0, u-35), max(12, v-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
    status = reference["registration"]["status"].upper()
    cv2.putText(rgb, f"{status} BOARD | yellow pickup / purple place | taught TCP projections",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 1, cv2.LINE_AA)
    destination = Path(output) / "BOARD_PROJECTED_TARGETS.png"
    if not cv2.imwrite(str(destination), rgb):
        raise OSError("cannot save board overlay")
    live = ROOT / "runs/wrist_live"
    live.mkdir(parents=True, exist_ok=True)
    pending = live / ".board_projected_targets.tmp"
    shutil.copyfile(destination, pending)
    pending.replace(live / "latest_wrist_a.png")
    (live / "latest_wrist_a.json").write_text(json.dumps({"camera": "head_camera",
        "stage": "BOARD_PROJECTED_TARGETS", "registration": reference["registration"],
        "source_run": str(destination)}) + "\n")
    print(f"BOARD TARGET REVIEW: {destination}", flush=True)


def _prompt_next_location(current_name, available_targets):
    """Ask for the next location or a controlled session transition."""
    names = list(available_targets)
    print(f"\nREACHED {current_name}. Choose the next destination:", flush=True)
    for index, name in enumerate(names, 1):
        print(f"  {index}. {name}", flush=True)
    print("  r. retake board image (board may have moved)", flush=True)
    print("  e. exit location testing", flush=True)
    while True:
        received = input("Next location [number/name/r/e]: ")
        raw = clean_choice(received)
        lowered = raw.lower()
        if lowered in ("e", "exit", "q", "quit", "0", "back"):
            return "exit"
        if lowered in ("r", "retake", "image", "photo"):
            return "retake_image"
        try:
            index = int(raw)
            if 1 <= index <= len(names):
                return names[index - 1]
        except ValueError:
            pass
        if raw in available_targets:
            return raw
        print(f"Unrecognized input {received!r}. Choose 1..{len(names)}, a location name, r, or e.", flush=True)


def _run_motion_targets(
    targets, bundle, *, confirm_physical, check_only, speed_scale,
    available_targets=None, interactive_next=False, runtime=None,
    task_data=None, clearance_m=None, prompt_after_capture=False,
    remote_safe=False,
):
    if not targets and not prompt_after_capture:
        return 0
    if available_targets is None:
        available_targets = targets
    cfg = bundle["robot"]
    # The Vega motion plugin can finish with a stationary joint a few
    # milliradians outside the nominal 5 mrad state-stream gate.  Position
    # testing is supervised and already validates the measured TCP pose, so
    # use the same 20 mrad settled-state tolerance used by board and wrist
    # calibration.  This prevents a false E-stop/retry when the robot has
    # actually stopped at the requested Cartesian target.
    cfg.setdefault("motion", {})["joint_reached_tolerance_rad"] = max(
        float(cfg.get("motion", {}).get("joint_reached_tolerance_rad", 0.005)),
        0.020,
    )
    cfg["allow_robot_init_head_motion"] = bool(confirm_physical)
    cfg["auto_clear_software_estop_on_connect"] = True
    floor = float(load_vega_skills()["safety"]["min_tcp_z_m"])
    attempt = 0
    while True:
        attempt += 1
        robot = VegaAdapter(cfg)
        try:
            def guard(target):
                from steadyhand.remote_motion import confirm_low_clearance
                def surface(x, y):
                    if runtime is None:
                        raise ValueError("no board clearance model")
                    return calibrated_surface_z(x, y, runtime[2][3])
                confirm_low_clearance(robot.get_tcp_pose(), target, surface)

            if check_only:
                robot.prepare()
            else:
                robot.connect()

                def refresh_board_image():
                    nonlocal runtime, available_targets, targets
                    if runtime is None or task_data is None or clearance_m is None:
                        return
                    runtime, scene = _capture_registered_board(
                        robot, runtime, floor_m=floor,
                        checkpoint=(lambda label: guard(None) if label == "before_camera_clear" else None) if remote_safe else None,
                    )
                    available_targets = _make_test_targets(
                        list(available_targets), runtime, task_data, clearance_m
                    )
                    selected_names = list(targets)
                    targets = OrderedDict(
                        (name, available_targets[name])
                        for name in selected_names
                    )
                    print(
                        f"BOARD FRAME {runtime[2][3].get('registration', {}).get('status', 'reference').upper()}; targets rebuilt",
                        flush=True,
                    )

                # Every physical location session starts with a fresh board
                # image, then automatically returns to the known RIGHT_READY.
                refresh_board_image()

            ready_q, ready_pose = configured_right_preset(cfg, "right_ready")
            steps = dict(max_translation_step_m=.020, max_orientation_step_rad=.10)
            if check_only:
                seed, start = ready_q, ready_pose
                for name, target in targets.items():
                    seed = preflight_tcp_segmented(robot._kinematics, seed, start, target, **steps)
                    start = target
                    print(name, "TARGET =", tuple(round(float(v), 6) for v in target.position_m), flush=True)
                print("ALL SELECTED PATHS PREFLIGHTED", flush=True)
                return 0
            print("MOVING TO RIGHT_READY", flush=True)
            if remote_safe:
                guard(ready_pose)
            robot.move_joints(ready_q, speed_scale=float(speed_scale))

            pending = list(targets.items())
            if prompt_after_capture and not pending:
                action = _prompt_next_location("BOARD IMAGE", available_targets)
                while action == "retake_image":
                    refresh_board_image()
                    robot.move_joints(ready_q, speed_scale=float(speed_scale))
                    action = _prompt_next_location("BOARD IMAGE", available_targets)
                if action == "exit":
                    return 0
                pending = [(action, available_targets[action])]

            while pending:
                name, target = pending.pop(0)
                print(f"GO TO {name}: checking the segmented path from the measured pose", flush=True)
                try:
                    preflight_tcp_segmented(robot._kinematics, robot._read_joint_positions(),
                                            robot.get_tcp_pose(), target, **steps)
                except IKError as exc:
                    print(f"TARGET BLOCKED BEFORE MOTION: {exc}. Arm remains at its current pose.", flush=True)
                    if not interactive_next:
                        return 2
                    action = _prompt_next_location("CURRENT POSE (requested target not reached)", available_targets)
                    if action == "exit":
                        return 0
                    if action == "retake_image":
                        refresh_board_image()
                        robot.move_joints(ready_q, speed_scale=float(speed_scale))
                        action = _prompt_next_location("BOARD IMAGE", available_targets)
                        while action == "retake_image":
                            refresh_board_image()
                            robot.move_joints(ready_q, speed_scale=float(speed_scale))
                            action = _prompt_next_location("BOARD IMAGE", available_targets)
                        if action == "exit":
                            return 0
                    pending = [(action, available_targets[action])]
                    continue
                move_tcp_segmented(
                    robot, target, speed_scale=float(speed_scale),
                    **steps,
                    min_tcp_z_m=None,
                    waypoint_guard=guard if remote_safe else None,
                )
                actual = robot.get_tcp_pose()
                print(name, "MEASURED TIP_R =", tuple(round(float(v), 6) for v in actual.position_m), flush=True)
                if interactive_next:
                    action = _prompt_next_location(name, available_targets)
                    while action == "retake_image":
                        refresh_board_image()
                        ready_q, _ = configured_right_preset(cfg, "right_ready")
                        print("MOVING TO RIGHT_READY", flush=True)
                        robot.move_joints(ready_q, speed_scale=float(speed_scale))
                        action = _prompt_next_location("BOARD IMAGE", available_targets)
                    if action == "exit":
                        return 0
                    next_target = available_targets[action]
                    pending = [(action, next_target)]
            return 0
        except Exception as exc:
            if not any(marker in str(exc) for marker in (
                "IK did not converge", "initial target is not reachable",
                "no reachable supervised inset", "downward head view was not reached",
            )):
                raise
            print(f"POSITION IK RETRY {attempt}: {exc}", flush=True)
            print("Recovering through camera-clear, downward head view, fresh photo, and retrying.", flush=True)
            answer = input("Retry this position test? Type yes to continue, or no to stop: ").strip().lower()
            if answer not in ("y", "yes"):
                print("Position retry stopped by operator.", flush=True)
                return 2
        finally:
            robot.close()


def _run_competition_task(name, runtime, task_data, args):
    if name == "battery_size1_pick_v0":
        targets = _make_test_targets(["task.battery_size1.pick"], runtime, task_data, args.clearance_m)
        return _run_motion_targets(targets, runtime[0], confirm_physical=True,
                                   check_only=args.check_only, speed_scale=args.speed_scale,
                                   runtime=runtime, task_data=task_data, clearance_m=args.clearance_m)
    print(f"{name} is present in the pipeline as a preserved version entry.")
    print("It is intentionally not enabled until the calibrated approach, grasp, and verification are validated.")
    print("No arm or gripper motion was commanded.")
    return 0


def _available_position_names(runtime, task_data, clearance_m):
    board_names = OrderedDict((name, "corrected board reference") for name in (
        "board.center", "board.top_right", "board.bottom_right", "board.bottom_left"))
    task_names = OrderedDict((name, "organizer task coordinate") for name in _task_targets(runtime, task_data, clearance_m))
    return OrderedDict(list(board_names.items()) + list(task_names.items()))


def _flatten_action_names(values):
    names = []
    for value in values or ():
        names.extend(part.strip() for part in value.split(",") if part.strip())
    return names


def _recalibrate():
    print("Starting the five-point board calibration. Its output becomes the active runtime frame.")
    result = run_five_point_calibration(["--confirm-physical-motion"])
    if result != 0:
        return result
    _load_runtime()
    print(f"ACTIVE CALIBRATION UPDATED: {CALIBRATION}")
    return 0


def _run_wrist_calibration_menu(args):
    if args.check_only:
        print("Wrist calibration requires physical motion; remove --check-only.")
        return 2
    selected = _choose(
        OrderedDict((part, "teach wrist_a feature, jaw pixel, yaw, and grasp depth") for part in PART_NAMES),
        "WRIST CAMERA CALIBRATION",
    )
    for part in selected:
        command = [
            "--part", part,
            "--mode", "calibrate",
            "--confirm-head-motion",
            "--confirm-physical-motion",
            "--speed-scale", str(args.speed_scale),
        ]
        if getattr(args, "remote_safe", False):
            command.append("--remote-safe")
        result = run_wrist_part_calibration(command)
        if result:
            return result
    return 0


def _run_drop_calibration_menu(args):
    if args.check_only:
        print("Drop calibration requires physical motion; remove --check-only.")
        return 2
    selected = _choose(
        OrderedDict((part, "pick with saved calibration, teach drop hover/depth, release and save") for part in PART_NAMES),
        "DROP-OFF POSITION CALIBRATION",
    )
    for part in selected:
        command = [
            "--part", part,
            "--mode", "drop",
            "--confirm-head-motion",
            "--confirm-physical-motion",
            "--speed-scale", str(args.speed_scale),
        ]
        if getattr(args, "remote_safe", False):
            command.append("--remote-safe")
        result = run_wrist_part_calibration(command)
        if result:
            return result
    return 0


def _run_place_cv_menu(args):
    if args.check_only:
        print("Placement CV teaching requires physical motion; remove --check-only.")
        return 2
    selected = _choose(
        OrderedDict((part, "teach wrist release feature while holding the verified part") for part in PART_NAMES),
        "PLACEMENT CV TARGET TEACHING",
    )
    for part in selected:
        command = [
            "--part", part, "--mode", "place-cv",
            "--confirm-head-motion", "--confirm-physical-motion",
            "--speed-scale", str(args.speed_scale),
        ]
        if getattr(args, "remote_safe", False):
            command.append("--remote-safe")
        result = run_wrist_part_calibration(command)
        if result:
            return result
    return 0


def _run_head_preview_menu(args):
    if args.check_only:
        print("Head target preview requires physical motion; remove --check-only.")
        return 2
    selected = _choose(
        OrderedDict((part, "preview head-camera target; jaws never move") for part in PART_NAMES),
        "HEAD-CAMERA TARGET PREVIEW",
    )
    if not selected:
        return 0
    print("\nPREVIEW MODE — no pickup, release, or calibration changes")
    print("  1. Head target only, 100 mm hover (default; best wrist visibility)")
    print("  2. Head target + wrist centering, 100 mm hover")
    print("  3. Head target only, 40 mm hover (closer view; jaws may obscure the part)")
    print("  4. Head target + wrist centering, 40 mm hover (needs new pixels for this height)")
    print("  0. Back")
    while True:
        mode = clean_choice(input("Preview mode [1/2/3/4, Enter=1, 0=back]: ")).lower()
        if mode in ("0", "e", "abort", "back"):
            return 0
        if mode in ("", "1", "2", "3", "4"):
            break
        print("Choose a displayed mode; Enter selects 1.", flush=True)
    center = mode in ("2", "4")
    height = "40" if mode in ("3", "4") else "100"
    from tools.vega_head_target_preview import main as run_head_target_preview
    for part in selected:
        result = run_head_target_preview([
            "--part", part,
            "--confirm-head-motion", "--confirm-physical-motion",
            "--hover-clearance-mm", height,
            "--speed-scale", str(getattr(args, "speed_scale", DEFAULT_PIPELINE_SPEED_SCALE)),
        ] + (["--center"] if center else []) + (
            ["--remote-safe"] if getattr(args, "remote_safe", False) else []
        ))
        if result:
            return result
    return 0


def _run_task_tests_menu(args):
    if args.check_only:
        print("Task tests require physical motion; remove --check-only.")
        return 2
    selected = _choose(TASK_ACTIONS, "TASK TESTS (PICK / PICK-PLACE)")
    for name in selected:
        part, action = name.split(".", 1)
        result = run_wrist_part_calibration([
            "--part", part,
            "--mode", "test",
            "--action", action,
            "--confirm-head-motion",
            "--confirm-physical-motion",
            "--speed-scale", str(args.speed_scale),
        ] + (["--remote-safe"] if getattr(args, "remote_safe", False) else []))
        if result:
            return result
    return 0


def _sequence_indices(raw, available=None):
    available = available or COMPETITION_SEQUENCE_ACTIONS
    selected = []
    for value in raw.replace(" ", "").split(","):
        if not value:
            continue
        if "-" in value:
            start, end = (int(v) for v in value.split("-", 1))
            selected.extend(range(start, end + (1 if end >= start else -1), 1 if end >= start else -1))
        else:
            selected.append(int(value))
    if not selected or any(str(index) not in available for index in selected):
        raise ValueError("sequence choices must be numbered 1..9; ranges such as 1-5,8,9 are accepted")
    return list(dict.fromkeys(selected))


def _snapshot_execution_offsets(args):
    # Called anew for every menu run, then reused by every part and retry.
    args.execution_offsets = load_offsets()
    describe_offsets(args.execution_offsets)


def _run_competition_action(args, part, action, *, retries=0, no_cv=False, place_cv=False,
                            head_reacquire=True, retry_without_cv=True):
    """Run one gated action with automatic, bounded recovery.

    Competition execution is deliberately non-interactive after launch: a
    transient visual/IK setup failure is retried automatically, then the part
    is recorded as skipped so the next eligible part can be attempted.  A
    run that may still be holding a part remains a hard stop because issuing
    another grasp would be unsafe.
    """
    if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 2:
        raise ValueError("competition retries must be an integer 0..2 (maximum three attempts)")
    if getattr(args, "execution_offsets", None) is None:
        _snapshot_execution_offsets(args)
    command = [
        "--part", part, "--mode", "test", "--action", action,
        "--competition", "--confirm-head-motion", "--confirm-physical-motion",
        "--speed-scale", str(args.speed_scale),
        "--execution-offsets-json", json.dumps(args.execution_offsets.as_dict()),
    ]
    if no_cv:
        command.append("--no-cv")
    if place_cv and action == "pick_place":
        command.append("--place-cv")
    if getattr(args, "remote_safe", False):
        command.append("--remote-safe")
    attempt = 0
    while True:
        attempt += 1
        output = ROOT / "runs" / "competition_actions" / (
            f"{time.time_ns()}_{part}_{action}_attempt{attempt}"
        )
        print(f"\nCOMPETITION ACTION {part}.{action} (attempt {attempt})", flush=True)
        attempt_command = list(command)
        if attempt == 2 and not no_cv and retry_without_cv:
            # Second attempt: keep the saved calibrated arm pose but disable
            # wrist centering.  This isolates a bad visual feature from a
            # physically good taught grasp.
            attempt_command.append("--no-cv")
        if attempt >= 3 and head_reacquire:
            # Final attempt: fresh head-camera part association followed by
            # the aggressive bounded wrist path (unless the part is globally
            # configured no-CV, in which case the saved pose remains active).
            attempt_command.append("--head-reacquire")
        try:
            result = run_wrist_part_calibration(attempt_command + ["--output", str(output)])
        except Exception as exc:
            result = 2
            print(
                f"Wrist action raised {type(exc).__name__}: {exc}; inspect before retrying.",
                file=sys.stderr, flush=True,
            )
        summary_path = output / "run_summary.json"
        summary = {}
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                summary = {"status": "malformed"}
        print(f"COMPETITION RUN ARTIFACT = {output}", flush=True)
        if summary.get("holding_may_be_true"):
            print(
                "RUN SUMMARY SAYS A PART MAY BE HELD; refusing all automatic retries.",
                file=sys.stderr, flush=True,
            )
            return 3
        if result == 0:
            if not summary_path.is_file():
                print(
                    "Action returned success without run_summary.json; stopping for inspection.",
                    file=sys.stderr, flush=True,
                )
                return 2
            # Pick-only competition actions intentionally return the part to
            # its source so the next part can be attempted.  That path has a
            # distinct successful terminal status; treating it as an error
            # made option 6 stop after the first successful pickup.
            successful_statuses = {"completed"}
            if action == "pick":
                successful_statuses.add("pick_complete_returned")
            if summary.get("status") not in successful_statuses:
                print(
                    f"Action returned success but run summary is {summary.get('status')!r}; stopping for inspection.",
                    file=sys.stderr, flush=True,
                )
                return 2
            print(f"COMPLETE {part}.{action}", flush=True)
            return 0
        if result == 3:
            print(
                "ACTION LEFT A POSSIBLY HELD PART; retry is blocked. Inspect and recover manually through the gated tool.",
                file=sys.stderr, flush=True,
            )
            return result
        if summary.get("automatic_continuation_safe") is not True:
            print("ACTION STOPPED WITHOUT SAFE CONTINUATION: motion/hardware state "
                  "or missing summary requires inspection; no automatic retry.", flush=True)
            return 2
        if attempt > retries:
            print(
                f"FAILED {part}.{action}; automatic retry budget exhausted; continuing.",
                file=sys.stderr,
                flush=True,
            )
            return -1
        print(
            f"AUTOMATIC RETRY {part}.{action}: attempt {attempt + 1} of {retries + 1}",
            flush=True,
        )
        # Keep retries distinct in the robot log without requiring operator
        # input.  Do not sleep after a held-part stop (handled above).
        time.sleep(0.25)


def _profile_ready_for_action(profiles, part, action):
    profile = (profiles.get("parts") or {}).get(part)
    if not isinstance(profile, dict):
        return False, "no saved wrist profile"
    if not profile.get("grasp_verified"):
        return False, "grasp not verified"
    if action == "pick_place":
        if not profile.get("place"):
            return False, "place settings missing"
        if not profile.get("place_verified"):
            return False, "place not verified"
    return True, "ready"


def _priority_competition_actions(args, *, action=None, no_cv=False):
    """Build and run the score-first plan from only verified profiles."""
    _snapshot_execution_offsets(args)
    plan = _load_competition_plan()
    cfg = load_bundle("vega")["robot"]
    profiles = load_profiles(ROOT / "calibration" / "wrist_part_profiles.json", cfg)
    chosen_action = action or plan["default_action"]
    actions = []
    skipped = []
    for part in plan["pick_priority"]:
        ready, reason = _profile_ready_for_action(profiles, part, chosen_action)
        if ready:
            actions.append((part, chosen_action))
        else:
            skipped.append((part, reason))
    print("\nPRIORITY COMPETITION PLAN", flush=True)
    print(f"Action: {chosen_action}; bounded retries per part: {plan['max_retries_per_part']}", flush=True)
    if no_cv:
        print("Positioning: board-projected taught grasp hover (no wrist CV)", flush=True)
    print("Eligible order: " + (", ".join(f"{p}.{a}" for p, a in actions) or "none"), flush=True)
    if skipped:
        print("Not yet eligible:", flush=True)
        for part, reason in skipped:
            print(f"  {part}: {reason}", flush=True)
    if not actions:
        print("No verified actions are ready; teach and individually validate a part first.", file=sys.stderr)
        return 2
    if args.check_only:
        print("CHECK-ONLY: no robot, camera, or gripper motion will be commanded.", flush=True)
        return 0
    completed = 0
    skipped_run = 0
    failed_parts = []
    for part, current_action in actions:
        action_kwargs = {"retries": plan["max_retries_per_part"]}
        if no_cv:
            action_kwargs["no_cv"] = True
        result = _run_competition_action(args, part, current_action, **action_kwargs)
        if result == 0:
            completed += 1
            continue
        if result == -1:
            skipped_run += 1
            failed_parts.append((part, current_action))
            continue
        print(
            f"PLAN STOPPED after {completed} completed actions and {skipped_run} skipped actions.",
            file=sys.stderr, flush=True,
        )
        return result
    print(
        f"PLAN FINISHED: {completed} completed, {skipped_run} skipped; "
        f"{len(skipped)} parts were not eligible before motion.", flush=True,
    )
    return 0


def _configured_competition_run(args):
    """Run the single JSON-configured, calibration-gated competition routine."""
    _snapshot_execution_offsets(args)
    settings = _load_competition_actions()
    if not getattr(args, "speed_scale_cli", False):
        args.speed_scale = settings["pipeline_speed_scale"]
    if not getattr(args, "clearance_mm_cli", False):
        args.clearance_mm = settings["task_clearance_mm"]
        args.clearance_m = settings["task_clearance_mm"] / 1000.0
    cfg = load_bundle("vega")["robot"]
    profiles = load_profiles(ROOT / "calibration" / "wrist_part_profiles.json", cfg)
    actions = []
    skipped = []
    for part in settings["order"]:
        entry = settings["parts"][part]
        if not entry["enabled"] or not entry["pick_enabled"]:
            skipped.append((part, "disabled in competition_actions.json"))
            continue
        profile = (profiles.get("parts") or {}).get(part)
        if not isinstance(profile, dict) or not profile.get("grasp_verified"):
            skipped.append((part, "no verified pickup calibration"))
            continue
        mode = entry["mode"]
        if mode == "auto":
            mode = "pick_place" if (
                entry["place_enabled"] and profile.get("place") and profile.get("place_verified")
            ) else "pick"
        if mode == "pick_place" and not (
            entry["place_enabled"] and profile.get("place") and profile.get("place_verified")
        ):
            # A missing drop calibration must never prevent a verified pickup
            # from earning the first competition point.
            print(f"{part}: place calibration missing; using pickup action", flush=True)
            mode = "pick"
        actions.append((part, mode))
    print("\nCONFIGURED COMPETITION RUN", flush=True)
    print(f"Actions JSON: {COMPETITION_ACTIONS}", flush=True)
    print("Attempts per part: " + ", ".join(
        f"{part}={settings['parts'][part]['max_attempts']}"
        for part, _ in actions
    ), flush=True)
    print("Eligible order: " + (", ".join(f"{p}.{a}" for p, a in actions) or "none"), flush=True)
    for part, reason in skipped:
        print(f"Skipped: {part} ({reason})", flush=True)
    if not actions:
        print("No enabled verified actions are ready.", file=sys.stderr)
        return 2
    if args.check_only:
        print("CHECK-ONLY: no robot, camera, or gripper motion will be commanded.", flush=True)
        return 0
    completed = 0
    failed = []
    for part, action in actions:
        part_settings = settings["parts"][part]
        profile = (profiles.get("parts") or {}).get(part) or {}
        # Enabling the placement-CV switch is opt-in, but a missing taught
        # release reference is a normal state while profiles are being built.
        # Use the verified saved release pose until menu 14 has produced a
        # placement reference; never make a held part depend on an absent
        # template during a non-interactive competition run.
        use_place_cv = bool(
            settings["use_place_cv"]
            and part_settings["use_place_cv"]
            and profile.get("place_cv", {}).get("enabled", False)
        )
        result = _run_competition_action(
            args, part, action,
            retries=part_settings["max_attempts"] - 1,
            no_cv=not part_settings["use_wrist_pick_cv"],
            place_cv=use_place_cv,
            head_reacquire=settings["head_reacquire_on_failure"],
            retry_without_cv=settings["retry_without_wrist_cv"],
        )
        if result == 0:
            completed += 1
            continue
        if result not in (0, -1):
            return result
        failed.append((part, action))
    if failed:
        print("Failed parts were skipped after their bounded attempt budgets: "
              + ", ".join(f"{part}.{action}" for part, action in failed), flush=True)
    print(
        f"CONFIGURED RUN FINISHED: {completed}/{len(actions)} actions completed.",
        flush=True,
    )
    return 0


def _all_calibrated_competition_run(args, *, place_cv=False):
    """Run every verified pickup, using place CV only where it is available."""
    _snapshot_execution_offsets(args)
    settings = _load_competition_actions()
    cfg = load_bundle("vega")["robot"]
    profiles = load_profiles(ROOT / "calibration" / "wrist_part_profiles.json", cfg)
    actions = []
    for part in settings["order"]:
        profile = (profiles.get("parts") or {}).get(part)
        if not isinstance(profile, dict) or not profile.get("grasp_verified"):
            continue
        action = "pick_place" if (
            profile.get("place") and profile.get("place_verified")
        ) else "pick"
        actions.append((part, action, bool(
            place_cv and action == "pick_place" and profile.get("place_cv", {}).get("enabled", False)
        )))
    print("\nALL CALIBRATED COMPETITION RUN", flush=True)
    print("Placement CV:", "enabled where taught" if place_cv else "disabled", flush=True)
    print("Eligible order: " + (", ".join(f"{p}.{a}" for p, a, _ in actions) or "none"), flush=True)
    if not actions:
        print("No verified pickup profiles are available.", file=sys.stderr, flush=True)
        return 2
    if args.check_only:
        print("CHECK-ONLY: no robot, camera, or gripper motion will be commanded.", flush=True)
        return 0
    completed = 0
    for part, action, use_cv in actions:
        result = _run_competition_action(
            args, part, action,
            retries=2,
            no_cv=False,
            place_cv=use_cv,
            head_reacquire=True,
        )
        if result == 0:
            completed += 1
        elif result != -1:
            return result
    print(f"ALL-CALIBRATED RUN FINISHED: {completed}/{len(actions)} actions completed.", flush=True)
    return 0


def _run_competition_sequence(args, raw=None):
    if args.check_only:
        print("Competition runs require physical motion; remove --check-only.")
        return 2
    if raw is None:
        print("\nCOMPETITION RUN MODES")
        print("  p. priority pick/place run (verified parts, easiest-first, bounded retry)")
        print("  a. attempt pickup of all verified parts (score-first, no placement)")
        print("  c. custom proven sequence (choose numbered pick/place actions)")
        mode = input("Choose p, a, or c (0 to cancel): ").strip().lower()
        if mode in ("0", "q", "quit", "exit", ""):
            return 0
        if mode in ("p", "priority", "pick", "pick_place"):
            return _priority_competition_actions(args, action="pick_place")
        if mode in ("a", "all", "pickup", "pick_only"):
            return _priority_competition_actions(args, action="pick")
        if mode not in ("c", "custom"):
            print("Choose p for pick/place, a for all pickups, or c for a custom sequence.", file=sys.stderr)
            return 2
        print("\nCOMPETITION SEQUENCE ACTIONS")
        available_actions = _competition_sequence_actions()
        for index, action in available_actions.items():
            print(f"  {index}. {action}")
        raw = input("Choose actions (for example 1-5,8,9), or 0 to cancel: ").strip()
    if raw in ("0", "", "q", "quit", "exit"):
        return 0
    available_actions = _competition_sequence_actions()
    try:
        indices = _sequence_indices(raw, available_actions)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    plan = _load_competition_plan()
    _snapshot_execution_offsets(args)
    for index in indices:
        part, action = available_actions[str(index)].split(".", 1)
        result = _run_competition_action(
            args, part, action, retries=plan["max_retries_per_part"],
        )
        if result not in (0, -1):
            return result
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirm-head-motion", action="store_true")
    p.add_argument("--confirm-physical-motion", action="store_true")
    p.add_argument("--check-only", action="store_true", help="preflight menu selections without moving")
    p.add_argument("--speed-scale", type=float, default=None)
    p.add_argument(
        "--remote-safe", action="store_true",
        help="normal motion settings; confirm arm motion below 40 mm clearance",
    )
    p.add_argument(
        "--clearance-mm", type=float, default=None,
        help="TCP clearance above the calibrated board surface (default: configs/competition_plan.json)",
    )
    actions = p.add_mutually_exclusive_group()
    actions.add_argument("--recalibrate", action="store_true",
                         help="run five-point calibration directly, without the menu")
    actions.add_argument("--test-positions", nargs="+", metavar="POINT",
                         help="test named board/task points directly; use all for every point")
    actions.add_argument("--competition-task", nargs="+", metavar="TASK",
                         help="run named competition task versions directly")
    actions.add_argument("--wrist-calibrate", metavar="PART",
                         choices=PART_NAMES,
                         help="teach one wrist_a part profile directly")
    actions.add_argument("--drop-calibrate", metavar="PART",
                         choices=PART_NAMES,
                         help="teach one physical drop position from its saved pickup profile")
    actions.add_argument("--place-cv-calibrate", metavar="PART",
                         choices=PART_NAMES,
                         help="teach one wrist-camera placement target while holding the part")
    actions.add_argument("--task-test", metavar="ACTION",
                         choices=list(TASK_ACTIONS),
                         help="run one wrist-backed pick or pick-place test directly")
    actions.add_argument("--competition-sequence", metavar="SEQUENCE",
                         help="run numbered sequence choices such as 1-5,8,9 directly")
    actions.add_argument(
        "--competition-run", action="store_true",
        help="run the single JSON-configured routine in configs/competition_actions.json",
    )
    actions.add_argument(
        "--competition-all", choices=("place_cv", "no_place_cv"),
        help="run every verified pickup; optionally use saved placement CV",
    )
    actions.add_argument(
        "--competition-plan",
        choices=("priority_pick_place", "priority_pick", "priority_pick_no_cv"),
        help="run verified profiles in the operator-configured easiest-first order",
    )
    actions.add_argument(
        "--versions", action="store_true",
        help="list and launch an archived rollback checkout; no robot motion",
    )
    actions.add_argument(
        "--list-versions", action="store_true",
        help="print archived rollback checkouts without motion",
    )
    args = p.parse_args(argv)
    args.speed_scale_cli = args.speed_scale is not None
    args.clearance_mm_cli = args.clearance_mm is not None
    operator_plan = _load_competition_plan()
    if args.speed_scale is None:
        args.speed_scale = operator_plan["pipeline_speed_scale"]
    if args.clearance_mm is None:
        args.clearance_mm = operator_plan["task_clearance_mm"]
    if args.versions or args.list_versions:
        from tools.vega_version_menu import main as run_version_menu
        return run_version_menu(["--list"] if args.list_versions else [])
    if not args.check_only and not args.confirm_physical_motion:
        p.error("physical pipeline requires --confirm-physical-motion")
    if not 20.0 <= args.clearance_mm <= 100.0:
        p.error("--clearance-mm must be 20..100")
    args.clearance_m = float(args.clearance_mm) / 1000.0

    if args.recalibrate:
        if args.check_only:
            p.error("--recalibrate cannot be combined with --check-only")
        return _recalibrate()

    if args.check_only and (
        args.wrist_calibrate is not None
        or args.drop_calibrate is not None
        or args.place_cv_calibrate is not None
        or args.task_test is not None
        or args.competition_sequence is not None
    ):
        p.error("wrist/task/competition actions cannot be combined with --check-only")

    if args.wrist_calibrate is not None:
        command = [
            "--part", args.wrist_calibrate, "--mode", "calibrate",
            "--confirm-head-motion", "--confirm-physical-motion",
            "--speed-scale", str(args.speed_scale),
        ]
        if args.remote_safe:
            command.append("--remote-safe")
        return run_wrist_part_calibration(command)

    if args.drop_calibrate is not None:
        command = [
            "--part", args.drop_calibrate, "--mode", "drop",
            "--confirm-head-motion", "--confirm-physical-motion",
            "--speed-scale", str(args.speed_scale),
        ]
        if args.remote_safe:
            command.append("--remote-safe")
        return run_wrist_part_calibration(command)

    if args.place_cv_calibrate is not None:
        command = [
            "--part", args.place_cv_calibrate, "--mode", "place-cv",
            "--confirm-head-motion", "--confirm-physical-motion",
            "--speed-scale", str(args.speed_scale),
        ]
        if args.remote_safe:
            command.append("--remote-safe")
        return run_wrist_part_calibration(command)

    if args.task_test is not None:
        part, action = args.task_test.split(".", 1)
        return run_wrist_part_calibration([
            "--part", part, "--mode", "test", "--action", action,
            "--confirm-head-motion", "--confirm-physical-motion",
            "--speed-scale", str(args.speed_scale),
        ] + (["--remote-safe"] if args.remote_safe else []))

    if args.competition_sequence is not None:
        return _run_competition_sequence(args, args.competition_sequence)

    if args.competition_plan is not None:
        action = "pick" if args.competition_plan != "priority_pick_place" else "pick_place"
        return _priority_competition_actions(
            args, action=action,
            no_cv=(args.competition_plan == "priority_pick_no_cv"),
        )

    if args.competition_run:
        return _configured_competition_run(args)

    if args.competition_all is not None:
        return _all_calibrated_competition_run(
            args, place_cv=args.competition_all == "place_cv"
        )

    if args.test_positions is not None or args.competition_task is not None:
        try:
            runtime = _load_runtime()
            task_data = runtime[1]
        except Exception as exc:
            print(f"No usable five-point calibration: {exc}", file=sys.stderr)
            return 2
        if args.test_positions is not None:
            available = _available_position_names(runtime, task_data, args.clearance_m)
            selected = list(available) if "all" in [x.lower() for x in args.test_positions] else _flatten_action_names(args.test_positions)
            unknown = [name for name in selected if name not in available]
            if unknown:
                print("Unknown position(s): " + ", ".join(unknown), file=sys.stderr)
                print("Available positions: " + ", ".join(available), file=sys.stderr)
                return 2
            targets = _make_test_targets(selected, runtime, task_data, args.clearance_m)
            return _run_motion_targets(
                targets, runtime[0], confirm_physical=args.confirm_physical_motion,
                check_only=args.check_only, speed_scale=args.speed_scale,
                runtime=runtime, remote_safe=args.remote_safe,
                task_data=task_data, clearance_m=args.clearance_m,
            )
        selected = _flatten_action_names(args.competition_task)
        unknown = [name for name in selected if name not in COMPETITION_TASKS]
        if unknown:
            print("Unknown competition task(s): " + ", ".join(unknown), file=sys.stderr)
            print("Available tasks: " + ", ".join(COMPETITION_TASKS), file=sys.stderr)
            return 2
        for name in selected:
            result = _run_competition_task(name, runtime, task_data, args)
            if result:
                return result
        return 0

    while True:
        print("\n=== VEGA COMPETITION PIPELINE ===")
        print("  1. Recalibrate moved board (camera + CENTER/TR/BR/BL + heights)")
        print("  2. Test calibrated board/task positions")
        print("  3. Wrist camera calibration (per-part feature / yaw / grasp depth)")
        print("  4. Task tests (all part pick and pick-place actions)")
        print("  5. Run configured competition routine (configs/competition_actions.json)")
        print("  6. Run all calibrated (with placement CV where taught)")
        print("  7. Run all calibrated (without placement CV)")
        print("  8. Reload operator settings / show readiness")
        print("  9. Legacy competition task versions")
        print(" 10. Calibrate drop-off position (saved pickup -> 40 mm descent -> release/save)")
        print(" 11. Remote-safe pickup calibration (confirm below 40 mm)")
        print(" 12. Remote-safe drop calibration (confirm below 40 mm)")
        print(" 13. Head-camera target preview (100 or 40 mm hover, never grabs)")
        print(" 14. Teach placement CV target (held part; no automatic release)")
        print(" 15. Launch an archived rollback version")
        print("  0. Exit")
        choice = input("Select an option: ").strip()
        if choice in ("0", "q", "quit", "exit"):
            return 0
        if choice == "1":
            if args.check_only:
                print("--check-only does not run physical recalibration.")
                continue
            try:
                _recalibrate()
            except (KeyboardInterrupt, EOFError):
                print("Calibration cancelled.")
            except Exception as exc:
                print(f"Calibration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if choice == "8":
            try:
                _reload_operator_settings(args)
            except Exception as exc:
                print(f"Operator settings rejected: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        try:
            runtime = _load_runtime()
            task_data = runtime[1]
        except Exception as exc:
            print(f"No usable five-point calibration: {exc}")
            print("Choose recalibrate first.")
            continue
        if choice == "2":
            try:
                available = _available_position_names(runtime, task_data, args.clearance_m)
                all_targets = _make_test_targets(
                    list(available), runtime, task_data, args.clearance_m
                )
                if args.check_only:
                    selected = _choose(available, "CALIBRATED POSITION TESTS", allow_all=True)
                    targets = OrderedDict((name, all_targets[name]) for name in selected)
                    _run_motion_targets(
                        targets, runtime[0],
                        confirm_physical=args.confirm_physical_motion,
                        check_only=True, speed_scale=args.speed_scale,
                    )
                else:
                    # The first prompt is intentionally after the fresh image,
                    # matching the retake-image flow used later in the session.
                    _run_motion_targets(
                        OrderedDict(), runtime[0],
                        confirm_physical=args.confirm_physical_motion,
                        check_only=False, speed_scale=args.speed_scale,
                        available_targets=all_targets, interactive_next=True,
                        runtime=runtime, task_data=task_data,
                        clearance_m=args.clearance_m, prompt_after_capture=True,
                        remote_safe=args.remote_safe,
                    )
            except Exception as exc:
                print(f"Position test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if choice == "3":
            try:
                _run_wrist_calibration_menu(args)
            except (KeyboardInterrupt, EOFError):
                print("Wrist calibration cancelled.")
            except Exception as exc:
                print(f"Wrist calibration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if choice == "4":
            try:
                _run_task_tests_menu(args)
            except (KeyboardInterrupt, EOFError):
                print("Task test cancelled.")
            except Exception as exc:
                print(f"Task test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if choice == "5":
            try:
                _configured_competition_run(args)
            except (KeyboardInterrupt, EOFError):
                print("Competition sequence cancelled.")
            except Exception as exc:
                print(f"Competition sequence failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if choice == "6":
            try:
                _all_calibrated_competition_run(args, place_cv=True)
            except (KeyboardInterrupt, EOFError):
                print("All-calibrated competition run cancelled.")
            except Exception as exc:
                print(f"All-calibrated competition run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if choice == "7":
            try:
                _all_calibrated_competition_run(args, place_cv=False)
            except (KeyboardInterrupt, EOFError):
                print("All-calibrated no-place-CV run cancelled.")
            except Exception as exc:
                print(f"All-calibrated no-place-CV run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if choice == "10":
            try:
                _run_drop_calibration_menu(args)
            except (KeyboardInterrupt, EOFError):
                print("Drop calibration cancelled.")
            except Exception as exc:
                print(f"Drop calibration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if choice == "11":
            previous_remote_safe = args.remote_safe
            args.remote_safe = True
            try:
                _run_wrist_calibration_menu(args)
            except (KeyboardInterrupt, EOFError):
                print("Remote-safe pickup calibration cancelled.")
            except Exception as exc:
                print(f"Remote-safe pickup calibration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            finally:
                args.remote_safe = previous_remote_safe
            continue
        if choice == "12":
            previous_remote_safe = args.remote_safe
            args.remote_safe = True
            try:
                _run_drop_calibration_menu(args)
            except (KeyboardInterrupt, EOFError):
                print("Remote-safe drop calibration cancelled.")
            except Exception as exc:
                print(f"Remote-safe drop calibration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            finally:
                args.remote_safe = previous_remote_safe
            continue
        if choice == "13":
            try:
                _run_head_preview_menu(args)
            except (KeyboardInterrupt, EOFError):
                print("Head target preview cancelled.")
            except Exception as exc:
                print(f"Head target preview failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if choice == "14":
            try:
                _run_place_cv_menu(args)
            except (KeyboardInterrupt, EOFError):
                print("Placement CV teaching cancelled.")
            except Exception as exc:
                print(f"Placement CV teaching failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if choice == "15":
            try:
                from tools.vega_version_menu import main as run_version_menu
                run_version_menu([])
            except (KeyboardInterrupt, EOFError):
                print("Version selection cancelled.")
            except Exception as exc:
                print(f"Archived version launch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if choice == "9":
            selected = _choose(COMPETITION_TASKS, "COMPETITION TASK VERSIONS")
            if selected:
                for name in selected:
                    try:
                        _run_competition_task(name, runtime, task_data, args)
                    except Exception as exc:
                        print(f"Task {name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        print("Unknown menu option.")


if __name__ == "__main__":
    raise SystemExit(main())
