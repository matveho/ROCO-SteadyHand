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
    args = p.parse_args(argv)

    if args.frames < 0:
        p.error("--frames must be >=0")
    if args.interval_s < 0:
        p.error("--interval-s must be >=0")

    import cv2  # Fail before Robot() if the image dependency is missing.
    import numpy as np
    from dexcontrol.robot import Robot

    cfg = load_bundle("vega")["robot"]
    os.environ.setdefault("ROBOT_NAME", cfg["robot_name"])
    safety = load_vega_skills()["safety"]
    plane_z = float(safety["min_tcp_z_m"] if args.plane_z is None else args.plane_z)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(args.output) if args.output else ROOT / "runs" / f"scene_{stamp}"

    _ensure_publisher(args.publisher_log)
    camera = None
    robot = None
    try:
        robot = Robot()
        target_head = np.asarray(
            [args.head_j1, args.head_j2, args.head_j3], dtype=float
        )
        robot.head.set_joint_pos(
            target_head,
            wait_time=1.2,
            exit_on_reach=True,
            exit_on_reach_kwargs={"tolerance": 0.02},
        )
        time.sleep(float(args.settle_s))
        head_q = np.asarray(robot.head.get_joint_pos(), dtype=float)
        print("HEAD Q =", head_q.tolist(), flush=True)

        camera = _connect_camera_with_retry(args.publisher_log)
        index = 0
        while args.frames == 0 or index < args.frames:
            frame = camera.read(include_depth=False)
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
