"""Validate live head board registration against physical arm motion.

Benchmark:
  live head RGB -> T_base_board -> low vertical-claw board center
  -> +100 mm board-X -> center -> +100 mm board-Y -> center

This is a COARSE registration/axis test. It intentionally uses safe interior
points, one fixed Z plane, one fixed vertical-claw orientation, no gripper, and
fast free-space motion. Fine XY accuracy belongs to the wrist camera.
"""

import argparse
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.vega import VegaAdapter
from steadyhand.cameras.vega import VegaHeadCamera
from steadyhand.config import load_bundle
from steadyhand.board_geometry import configured_board_plane_z
from steadyhand.executor import move_tcp_segmented
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills
from steadyhand.vision.scene import detect_head_task_scene
from steadyhand.vega_camera_clear import move_camera_clear
from tools.vega_board_benchmark import _yaw_quat
from tools.vega_scene_perception import _ensure_publisher


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offset-m", type=float, default=0.100,
                   help="interior board-axis offset; default 100 mm")
    p.add_argument("--hover-z", type=float, default=0.550,
                   help="fixed low TCP plane")
    p.add_argument("--speed-scale", type=float, default=0.90,
                   help="free-space arm speed")
    p.add_argument("--claw-yaw-deg", type=float, default=0.0)
    p.add_argument("--settle-s", type=float, default=0.5)
    p.add_argument("--publisher-log", default="~/head_camera.log")
    p.add_argument("--confirm-physical-motion", action="store_true")
    args = p.parse_args(argv)

    if not args.confirm_physical_motion:
        p.error("--confirm-physical-motion is required")
    if not 0.05 <= float(args.offset_m) <= 0.12:
        p.error("--offset-m must be 0.05..0.12 m")
    if not 0.45 <= float(args.speed_scale) <= 1.0:
        p.error("--speed-scale must be 0.45..1.0")

    import numpy as np

    cfg = load_bundle("vega")["robot"]
    safety = dict(load_vega_skills().get("safety") or {})
    floor = float(safety["min_tcp_z_m"])
    if not floor + 0.03 <= float(args.hover_z) <= floor + 0.10 + 1e-9:
        p.error(
            f"--hover-z must stay 3-10 cm above floor {floor:.6f}; "
            f"got {float(args.hover_z):.6f}"
        )

    if cfg["working_arm"] != "right" or cfg["kinematics"]["ee_frame"] != "tip_r":
        raise SystemExit("benchmark requires right arm / tip_r")

    # Robot() is known to move the head. Opt in, then explicitly put the full
    # head pose back into the verified downward view before perception.
    cfg["allow_robot_init_head_motion"] = True
    cfg["auto_clear_software_estop_on_connect"] = True

    # This benchmark is clear free-space motion. Larger joint chunks reduce
    # controller stop/start behavior while the server motion plugin still owns
    # trajectory smoothing.
    cfg["motion"]["max_step_rad"] = max(float(cfg["motion"]["max_step_rad"]), 0.45)

    _ensure_publisher(
        args.publisher_log,
        robot_name=cfg["robot_name"],
    )

    robot = VegaAdapter(cfg)
    camera = None
    try:
        robot.connect()

        move_camera_clear(robot, floor_m=floor, speed_scale=0.90)

        target_head = np.asarray([0.55, 0.0, 0.0], dtype=float)
        print("HEAD BEFORE =", robot._robot.head.get_joint_pos(), flush=True)
        robot._robot.head.set_joint_pos(
            target_head,
            wait_time=1.2,
            exit_on_reach=True,
            exit_on_reach_kwargs={"tolerance": 0.02},
        )
        time.sleep(float(args.settle_s))
        head_q = np.asarray(robot._robot.head.get_joint_pos(), dtype=float)
        print("HEAD AFTER  =", head_q.tolist(), flush=True)

        camera = VegaHeadCamera()
        camera.connect()
        print("WAITING FOR HEAD CAMERA FRAMES", flush=True)
        frame = camera.read(include_depth=False, timeout_s=15.0)

        scene = detect_head_task_scene(
            frame.left_rgb,
            frame.camera_info,
            head_q,
            plane_z_m=configured_board_plane_z(cfg, floor),
            lift_m=float(cfg["kinematics"]["fixed_joint_values"]["Lift"]),
            torso_flip_rad=float(
                cfg["kinematics"]["fixed_joint_values"]["torso_flip"]
            ),
            layout="unlabeled",
        )
        print(
            f"DETECTED PART-LIKE REGIONS = {len(scene['parts'])} (not gated)",
            flush=True,
        )

        T = np.asarray(scene["board"]["T_base_board_center"], dtype=float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            raise RuntimeError("invalid T_base_board from live perception")

        center = T[:3, 3].copy()
        bx = T[:3, 0].copy()
        by = T[:3, 1].copy()
        bx[2] = 0.0
        by[2] = 0.0
        bx /= np.linalg.norm(bx)
        by /= np.linalg.norm(by)

        px = center + float(args.offset_m) * bx
        py = center + float(args.offset_m) * by

        print(
            "BOARD CENTER BASE =",
            tuple(round(float(v), 5) for v in center),
            flush=True,
        )
        print(
            "BOARD +X UNIT IN BASE =",
            tuple(round(float(v), 4) for v in bx),
            flush=True,
        )
        print(
            "BOARD +Y UNIT IN BASE =",
            tuple(round(float(v), 4) for v in by),
            flush=True,
        )

        quat = _yaw_quat(math.radians(float(args.claw_yaw_deg)))
        targets = {
            "CENTER": Pose(
                (float(center[0]), float(center[1]), float(args.hover_z)), quat
            ),
            "BOARD_X_PLUS": Pose(
                (float(px[0]), float(px[1]), float(args.hover_z)), quat
            ),
            "BOARD_Y_PLUS": Pose(
                (float(py[0]), float(py[1]), float(args.hover_z)), quat
            ),
        }

        # Plan all three exact interior targets once before any arm motion.
        # No yaw/Z/inset search: if an exact point fails, stop and inspect
        # instead of silently changing the benchmark.
        kin_cfg = robot._kinematics.config
        kin_cfg["position_tolerance_m"] = 0.010
        kin_cfg["orientation_tolerance_rad"] = 0.12
        kin_cfg["max_seed_delta_rad"] = 2.4
        kin_cfg["max_iterations"] = 160
        seed = robot._read_joint_positions()
        for name, target in targets.items():
            robot._kinematics.solve(target, seed)
            print(
                f"{name} TARGET =",
                tuple(round(float(v), 4) for v in target.position_m),
                flush=True,
            )

        # Keep the same low Z and same quaternion throughout. 100 mm axis moves
        # fit in one Cartesian segment; the initial approach may use a few.
        sequence = (
            ("CENTER", targets["CENTER"]),
            ("BOARD_X_PLUS", targets["BOARD_X_PLUS"]),
            ("CENTER", targets["CENTER"]),
            ("BOARD_Y_PLUS", targets["BOARD_Y_PLUS"]),
            ("CENTER", targets["CENTER"]),
        )
        for name, target in sequence:
            print(f"MOVE {name}", flush=True)
            move_tcp_segmented(
                robot,
                target,
                speed_scale=float(args.speed_scale),
                max_translation_step_m=0.20,
                max_orientation_step_rad=0.80,
                min_tcp_z_m=floor,
            )
            actual = robot.get_tcp_pose()
            print(
                f"REACHED {name} =",
                tuple(round(float(v), 4) for v in actual.position_m),
                flush=True,
            )

        print("BOARD AXIS BENCHMARK COMPLETE", flush=True)
        return 0
    # Perception/planning failures happen before arm motion and must not assert
    # software e-stop. VegaAdapter motion methods already e-stop if an actual
    # arm command fails or is interrupted.
    finally:
        if camera is not None:
            try:
                camera.close()
            except BaseException:
                pass
        try:
            robot.close()
        except BaseException as close_error:
            print(f"SHUTDOWN WARNING: {close_error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
