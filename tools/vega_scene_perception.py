"""Canonical live head-perception stage for the competition Vega.

Behavior:
  1. Ensure the head-camera publisher is running.
  2. Construct Robot() and command head joints to [0.55, 0, 0].
  3. Read RGB only from the ZED left camera.
  4. Run the SAME board/part detector used by offline snapshot validation.
  5. Print + save board pixels/base coordinates and all part coordinates.

No arm or gripper commands are sent.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.cameras.vega import VegaHeadCamera
from steadyhand.config import load_bundle
from steadyhand.skill_config import load_vega_skills
from steadyhand.vision.scene import detect_head_task_scene, render_scene_overlay

ROOT = Path(__file__).resolve().parents[1]


def _publisher_running():
    try:
        result = subprocess.run(
            ["pgrep", "-f", "[d]exsensor launch --sensor head_camera"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except OSError:
        return False


def _ensure_publisher(log_path):
    """Start dexsensor only when the known head publisher is absent."""
    if _publisher_running():
        print("HEAD CAMERA PUBLISHER already running", flush=True)
        return False
    executable = shutil.which("dexsensor")
    if executable is None:
        raise RuntimeError("dexsensor executable not found")
    log_path = Path(log_path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("ab", buffering=0)
    try:
        subprocess.Popen(
            [executable, "launch", "--sensor", "head_camera"],
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        stream.close()
    print("STARTED HEAD CAMERA PUBLISHER; log =", log_path, flush=True)
    time.sleep(3.0)
    return True


def _connect_camera_with_retry(log_path):
    last = None
    for attempt in range(1, 4):
        camera = VegaHeadCamera()
        try:
            camera.connect()
            return camera
        except Exception as exc:
            last = exc
            try:
                camera.close()
            except Exception:
                pass
            if attempt < 3:
                print(f"camera connect attempt {attempt} failed: {exc}; retrying", flush=True)
                time.sleep(2.0)
    raise RuntimeError(f"head camera did not become ready; see {log_path}: {last}")


def _aim_head_down(robot, target_head, *, settle_s):
    """Explicitly enable/command all three head joints and verify readback."""
    import numpy as np

    if not robot.has_component("head"):
        raise RuntimeError("Robot reports no head component")

    head = robot.head
    before = np.asarray(head.get_joint_pos(), dtype=float)
    print("HEAD BEFORE =", before.tolist(), flush=True)

    # Command all 3 joints; do not inherit arbitrary J2/J3 state.
    target = np.asarray(target_head, dtype=float)
    move_fn = getattr(head, "move_to_joint_pos", None)
    if move_fn is not None:
        handle = move_fn(target, velocity_scale=0.45)
        # dexcontrol MotionHandle API varies slightly; block when supported.
        wait_fn = getattr(handle, "wait", None)
        if callable(wait_fn):
            wait_fn(timeout=5.0)
        else:
            time.sleep(1.5)
    else:
        head.set_joint_pos(
            target,
            wait_time=1.5,
            exit_on_reach=True,
            exit_on_reach_kwargs={"tolerance": 0.02},
        )
    time.sleep(float(settle_s))

    after = np.asarray(head.get_joint_pos(), dtype=float)
    print("HEAD AFTER  =", after.tolist(), flush=True)
    err = np.max(np.abs(after - np.asarray(target_head, dtype=float)))
    if not np.all(np.isfinite(after)) or err > 0.05:
        raise RuntimeError(
            f"Head did not reach requested pose {np.asarray(target_head).tolist()}; "
            f"readback={after.tolist()}, max_error={float(err):.4f} rad"
        )
    return after


def _save_frame(output, index, frame, scene):
    import cv2
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    stem = f"{index:03d}"
    np.save(output / f"{stem}_head_left_rgb.npy", frame.left_rgb)
    overlay = render_scene_overlay(frame.left_rgb, scene)
    cv2.imwrite(
        str(output / f"{stem}_scene_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )
    (output / f"{stem}_scene.json").write_text(
        json.dumps(scene, indent=2) + "\n", encoding="utf-8"
    )
    # Stable latest files are useful to the future autonomous runner / agents.
    (output / "latest_scene.json").write_text(
        json.dumps(scene, indent=2) + "\n", encoding="utf-8"
    )
    cv2.imwrite(
        str(output / "latest_scene_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )


def _print_scene(scene, index):
    board = scene["board"]
    print(f"FRAME {index}", flush=True)
    print("BOARD PIXELS =", board["corners_px"], flush=True)
    print(
        "BOARD CENTER BASE =",
        tuple(round(float(v), 5) for v in board["center_base_m_coarse"]),
        flush=True,
    )
    print(
        "BOARD CORNERS BASE =",
        {k: tuple(round(float(v), 5) for v in p)
         for k, p in board["corners_base_m_coarse"].items()},
        flush=True,
    )
    for part in scene["parts"]:
        label = part.get("name") or f"part_{part['index']}"
        print(
            f"{label}: image_px={tuple(round(v,1) for v in part['center_image_px'])} "
            f"board_mm_from_tl={tuple(round(v,1) for v in part['center_board_mm_from_tl'])} "
            f"base_m={tuple(round(v,4) for v in part['center_base_m_coarse'])}",
            flush=True,
        )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head-j1", type=float, default=0.55)
    p.add_argument("--head-j2", type=float, default=0.0)
    p.add_argument("--head-j3", type=float, default=0.0)
    p.add_argument("--settle-s", type=float, default=0.6)
    p.add_argument("--plane-z", type=float, default=None)
    p.add_argument("--layout", choices=("unlabeled", "final"), default="unlabeled")
    p.add_argument("--frames", type=int, default=1,
                   help="number of detections; 0 means run until Ctrl-C")
    p.add_argument("--interval-s", type=float, default=0.5)
    p.add_argument("--output")
    p.add_argument("--publisher-log", default="~/head_camera.log")
    p.add_argument(
        "--release-software-estop",
        action="store_true",
        help="explicitly deactivate the robot software E-stop before head motion",
    )
    args = p.parse_args(argv)

    if args.frames < 0:
        p.error("--frames must be >=0")
    if args.interval_s < 0:
        p.error("--interval-s must be >=0")

    import cv2  # Fail before Robot() if the image dependency is missing.
    import numpy as np

    cfg = load_bundle("vega")["robot"]
    os.environ.setdefault("ROBOT_NAME", cfg["robot_name"])
    # ROBOT_NAME must be established before importing/constructing Robot().
    from dexcontrol.robot import Robot

    safety = load_vega_skills()["safety"]
    plane_z = float(safety["min_tcp_z_m"] if args.plane_z is None else args.plane_z)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(args.output) if args.output else ROOT / "runs" / f"scene_{stamp}"

    _ensure_publisher(args.publisher_log)
    camera = None
    robot = None
    try:
        robot = Robot()

        # Robot() can connect while the software E-stop remains active. In that
        # state head commands are silently ineffective. Never proceed to CV
        # unless control is actually enabled.
        estop_state = robot.estop.get_state() if robot.has_component("estop") else None
        print("SOFTWARE ESTOP STATE =", estop_state, flush=True)
        if estop_state:
            if not args.release_software_estop:
                raise RuntimeError(
                    "Software E-stop is active, so the head cannot move. "
                    "Re-run with --release-software-estop only after confirming "
                    "the physical workspace is clear and the physical E-stop is accessible."
                )
            print("DEACTIVATING SOFTWARE ESTOP for head positioning", flush=True)
            robot.estop.deactivate()
            time.sleep(0.5)
            estop_after = robot.estop.get_state()
            print("SOFTWARE ESTOP AFTER =", estop_after, flush=True)
            if estop_after:
                raise RuntimeError("Software E-stop remained active after deactivate()")

        target_head = np.asarray(
            [args.head_j1, args.head_j2, args.head_j3], dtype=float
        )
        head_q = _aim_head_down(
            robot,
            target_head,
            settle_s=max(float(args.settle_s), 1.0),
        )

        # Only connect/read the camera after verified head readback.
        camera = _connect_camera_with_retry(args.publisher_log)
        index = 0
        while args.frames == 0 or index < args.frames:
            frame = camera.read(include_depth=False)
            # Refuse to run board CV on an obviously dead/black head stream.
            frame_mean = float(np.asarray(frame.left_rgb).mean())
            frame_std = float(np.asarray(frame.left_rgb).std())
            if frame_mean < 5.0 or frame_std < 3.0:
                raise RuntimeError(
                    f"Head RGB looks unpowered/invalid "
                    f"(mean={frame_mean:.2f}, std={frame_std:.2f})"
                )
            print(
                f"HEAD RGB STATS mean={frame_mean:.1f} std={frame_std:.1f}",
                flush=True,
            )
            # Read back each frame so repeated perception remains consistent
            # if another process has moved the head.
            head_q = np.asarray(robot.head.get_joint_pos(), dtype=float)
            scene = detect_head_task_scene(
                frame.left_rgb,
                frame.camera_info,
                head_q,
                plane_z_m=plane_z,
                lift_m=float(cfg["kinematics"]["fixed_joint_values"]["Lift"]),
                torso_flip_rad=float(
                    cfg["kinematics"]["fixed_joint_values"]["torso_flip"]
                ),
                layout=args.layout,
            )
            scene["captured_at_utc"] = datetime.now(timezone.utc).isoformat()
            scene["robot_name"] = cfg["robot_name"]
            scene["base_frame"] = cfg["kinematics"]["base_frame"]
            scene["coordinate_status"] = (
                "board mm exact-from-known-size; base coordinates coarse because "
                "absolute head extrinsic originated on organizer source robot"
            )
            _print_scene(scene, index)
            _save_frame(output, index, frame, scene)
            index += 1
            if args.frames == 0 or index < args.frames:
                time.sleep(float(args.interval_s))

        print("WROTE", output.resolve(), flush=True)
        return 0
    finally:
        if camera is not None:
            camera.close()
        if robot is not None:
            robot.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
