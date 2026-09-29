"""Operator menu for the calibrated Vega competition workflow.

The board is treated as movable state. Recalibration is the first menu action;
all position tests and task targets are rebuilt from the resulting calibration
file before any arm motion is planned.
"""

from collections import OrderedDict
import argparse
import json
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
)
from steadyhand.board_calibration import (
    board_geometry_signature,
    compare_board_geometry,
    orthonormalize_xy_axes,
)
from steadyhand.executor import move_tcp_segmented
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills
from steadyhand.vega_presets import configured_right_preset
from steadyhand.vega_camera_clear import move_camera_clear_for_image
from steadyhand.vision.scene import detect_head_task_scene
from tools.vega_board_five_point_calibrate import main as run_five_point_calibration
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
DEFAULT_PIPELINE_SPEED_SCALE = 0.38


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
    ("battery_size5_pick_place_v1", "large battery pick/place; scaffold"),
])


def _load_runtime():
    bundle = load_bundle("vega")
    task_data = json.loads(TASK_COORDINATES.read_text(encoding="utf-8"))
    validate_task_board_geometry(task_data)
    path = CALIBRATION if CALIBRATION.is_file() else FALLBACK_CALIBRATION
    center, ux, uy, plane = _load_manual(path, bundle["robot"])
    _, ready_pose = configured_right_preset(bundle["robot"], "right_ready")
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


def _task_targets(runtime, task_data, clearance_m):
    bundle, _, (center, ux, uy, plane), ready_pose = runtime
    source_center = _finite_vector(task_data.get("source_board_center_xy_m"), 2, "source board center")
    rotation_deg = float(task_data.get("task_coordinate_rotation_deg", 0.0))
    targets = OrderedDict()
    for part in task_data["official_order"]:
        for kind in task_data["parts"][part]:
            if kind not in ("pick", "place", "connect", "grade"):
                continue
            name = f"{part}.{kind}"
            source_xyz = _resolve_point(name, task_data)[2]
            targets[f"task.{name}"] = _live_pose(
                source_xyz, source_center=source_center,
                live_center=center, ux=ux, uy=uy,
                surface_plane=plane, clearance_m=clearance_m,
                quat=ready_pose.quaternion_wxyz, rotation_deg=rotation_deg,
            )
    return targets


def _choose(items, title, *, allow_all=False):
    names = list(items)
    print(f"\n{title}")
    for index, name in enumerate(names, 1):
        description = items[name] if isinstance(items, dict) else ""
        print(f"  {index}. {name}{(': ' + description) if description else ''}")
    print("  0. back")
    raw = input("Type a number or name (comma-separated for tests): ").strip()
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
    targets = OrderedDict()
    board = _board_targets(runtime, clearance_m)
    task = _task_targets(runtime, task_data, clearance_m)
    available = OrderedDict(list(board.items()) + list(task.items()))
    for name in selected:
        targets[name] = available[name]
    return targets


def _capture_downward_head_frame(robot, *, floor_m, bundle):
    """Move the arm/head clear, capture one downward board frame, then return."""
    move_camera_clear_for_image(robot, floor_m=floor_m, speed_scale=0.90)
    head_q = list(robot._robot.head.get_joint_pos())
    print("HEAD BEFORE =", head_q, flush=True)
    head_q[0] = 0.55
    move_head = getattr(robot._robot.head, "move_to_joint_pos", None)
    moved_head = False
    if callable(move_head):
        try:
            handle = move_head(head_q, velocity_scale=0.45)
            wait_fn = getattr(handle, "wait", None)
            if callable(wait_fn):
                wait_fn(timeout=5.0)
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
    if abs(float(measured_head_q[0]) - 0.55) > 0.03:
        raise RuntimeError(
            f"downward head view was not reached: target=0.55 "
            f"measured={measured_head_q[0]}"
        )
    time.sleep(0.5)
    camera = VegaHeadCamera()
    try:
        camera.connect()
        frame = camera.read(include_depth=False, timeout_s=15.0)
    finally:
        camera.close()
    print("BOARD CAMERA FRAME CAPTURED", flush=True)
    cfg = bundle["robot"]
    return detect_head_task_scene(
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


def _runtime_from_board_scene(runtime, scene):
    """Replace only the live XY board registration from a fresh head image."""
    import numpy as np

    bundle, task_data, (_, _, _, old_plane), ready_pose = runtime
    board = scene.get("board") or {}
    reference_signature = old_plane.get("camera_geometry_signature")
    current_signature = board.get("corners_px")
    geometry = compare_board_geometry(
        reference_signature,
        board_geometry_signature(current_signature),
    )
    if not geometry.get("valid", True):
        raise RuntimeError(
            "board image retake is not compatible with the calibrated board "
            "geometry; run full five-point calibration: " + geometry["reason"]
        )
    matrix = np.asarray(board.get("T_base_board_center"), dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise RuntimeError("fresh board image did not provide a finite board transform")
    center = matrix[:3, 3].copy()
    ux = matrix[:3, 0].copy()
    uy = matrix[:3, 1].copy()
    ux[2] = 0.0
    uy[2] = 0.0
    ux_xy, uy_xy, _, _ = orthonormalize_xy_axes(ux[:2], uy[:2])

    calibration_cfg = (bundle["robot"].get("board_calibration") or {})
    corrections = calibration_cfg.get("camera_target_corrections_m") or {}
    center_correction = corrections.get("CENTER", (0.0, 0.0))
    center[:2] += np.asarray(center_correction, dtype=float)

    plane = {
        "coefficients": tuple(old_plane["coefficients"]),
        "anchors": [],
        "camera_geometry_signature": reference_signature,
        "last_retake_geometry": geometry,
    }
    raw_corners = board.get("corners_base_m_coarse") or {}
    corrected_corners = {}
    for label, key in (("TOP_RIGHT", "tr"), ("BOTTOM_RIGHT", "br"), ("BOTTOM_LEFT", "bl")):
        point = raw_corners.get(key)
        if not isinstance(point, list) or len(point) != 3:
            raise RuntimeError(f"fresh board image is missing corner {key}")
        correction = corrections.get(label, (0.0, 0.0))
        corrected_corners[label] = (
            float(point[0]) + float(correction[0]),
            float(point[1]) + float(correction[1]),
        )
    plane["board_corners_xy"] = corrected_corners
    return bundle, task_data, (
        tuple(float(v) for v in center[:2]),
        tuple(float(v) for v in ux_xy),
        tuple(float(v) for v in uy_xy),
        plane,
    ), ready_pose


def _prompt_next_location(current_name, available_targets):
    """Ask for the next location or a controlled session transition."""
    names = list(available_targets)
    print(f"\nREACHED {current_name}. Choose the next destination:", flush=True)
    for index, name in enumerate(names, 1):
        print(f"  {index}. {name}", flush=True)
    print("  r. retake board image (board may have moved)", flush=True)
    print("  e. exit location testing", flush=True)
    while True:
        raw = input("Next location [number/name/r/e]: ").strip()
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
        print("Choose a listed location, r to retake the board image, or e for exit.", flush=True)


def _run_motion_targets(
    targets, bundle, *, confirm_physical, check_only, speed_scale,
    available_targets=None, interactive_next=False, runtime=None,
    task_data=None, clearance_m=None, prompt_after_capture=False,
):
    if not targets and not prompt_after_capture:
        return 0
    if available_targets is None:
        available_targets = targets
    cfg = bundle["robot"]
    cfg["allow_robot_init_head_motion"] = bool(confirm_physical)
    cfg["auto_clear_software_estop_on_connect"] = True
    floor = float(load_vega_skills()["safety"]["min_tcp_z_m"])
    attempt = 0
    while True:
        attempt += 1
        robot = VegaAdapter(cfg)
        try:
            if check_only:
                robot.prepare()
            else:
                robot.connect()

                def refresh_board_image():
                    nonlocal runtime, available_targets, targets
                    scene = _capture_downward_head_frame(
                        robot, floor_m=floor, bundle=bundle
                    )
                    if runtime is None or task_data is None or clearance_m is None:
                        return
                    runtime = _runtime_from_board_scene(runtime, scene)
                    available_targets = _make_test_targets(
                        list(available_targets), runtime, task_data, clearance_m
                    )
                    selected_names = list(targets)
                    targets = OrderedDict(
                        (name, available_targets[name])
                        for name in selected_names
                    )
                    print(
                        "BOARD FRAME REFRESHED; targets rebuilt from the new image",
                        flush=True,
                    )

                # Every physical location session starts with a fresh board
                # image, then automatically returns to the known RIGHT_READY.
                refresh_board_image()

            ready_q, _ = configured_right_preset(cfg, "right_ready")
            for name, target in targets.items():
                robot._kinematics.solve(target, ready_q)
                print(name, "TARGET =", tuple(round(float(v), 6) for v in target.position_m), flush=True)
            print("ALL SELECTED TARGETS PREFLIGHTED", flush=True)
            if check_only:
                return 0
            print("MOVING TO RIGHT_READY", flush=True)
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
                robot._kinematics.solve(target, robot._read_joint_positions())
                move_tcp_segmented(
                    robot, target, speed_scale=float(speed_scale),
                    max_translation_step_m=0.06, max_orientation_step_rad=0.20,
                    min_tcp_z_m=floor,
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
                    robot._kinematics.solve(next_target, robot._read_joint_positions())
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
                                   check_only=args.check_only, speed_scale=args.speed_scale)
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirm-head-motion", action="store_true")
    p.add_argument("--confirm-physical-motion", action="store_true")
    p.add_argument("--check-only", action="store_true", help="preflight menu selections without moving")
    p.add_argument("--speed-scale", type=float, default=DEFAULT_PIPELINE_SPEED_SCALE)
    p.add_argument(
        "--clearance-mm", type=float, default=DEFAULT_TASK_CLEARANCE_MM,
        help="TCP clearance above the calibrated board surface (default: 100 mm)",
    )
    actions = p.add_mutually_exclusive_group()
    actions.add_argument("--recalibrate", action="store_true",
                         help="run five-point calibration directly, without the menu")
    actions.add_argument("--test-positions", nargs="+", metavar="POINT",
                         help="test named board/task points directly; use all for every point")
    actions.add_argument("--competition-task", nargs="+", metavar="TASK",
                         help="run named competition task versions directly")
    args = p.parse_args(argv)
    if not args.check_only and (not args.confirm_head_motion or not args.confirm_physical_motion):
        p.error("physical pipeline requires --confirm-head-motion and --confirm-physical-motion")
    if not 20.0 <= args.clearance_mm <= 100.0:
        p.error("--clearance-mm must be 20..100")
    args.clearance_m = float(args.clearance_mm) / 1000.0

    if args.recalibrate:
        if args.check_only:
            p.error("--recalibrate cannot be combined with --check-only")
        return _recalibrate()

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
        print("  3. Run competition task version")
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
                    )
            except Exception as exc:
                print(f"Position test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if choice == "3":
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
