"""Capture wrist views or run one fixed-height right-wrist XY centering benchmark.

Default is camera capture only; Robot() is constructed only with --execute.
Motion also needs --camera (verified physical right) and --confirm-physical-motion.
Requires the onboard numpy, OpenCV, wrist_cameras and dexcontrol installations.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.vega import VegaAdapter
from steadyhand.cameras.vega import VegaWristCameras
from steadyhand.config import load_bundle
from steadyhand.executor import move_tcp_segmented
from steadyhand.skill_config import load_vega_skills
from steadyhand.vision.wrist_servo import run_xy_servo

ROOT = Path(__file__).resolve().parents[1]


class WristCapture:
    def __init__(self, cameras, output, camera, settle_s=0.25):
        self.cameras, self.output, self.camera = cameras, output, camera
        self.settle_s = settle_s
        self.last_identity = {}
        self.index = 0

    def __call__(self):
        import cv2
        import numpy as np

        time.sleep(self.settle_s)
        pair = self.cameras.read(timeout=3.0, fresh=True)
        metadata = {}
        for label in ("wrist_a", "wrist_b"):
            frame = getattr(pair, label)
            identity = (frame.frame_id, frame.timestamp_ns)
            if (label == self.camera and any(v is not None for v in identity)
                    and self.last_identity.get(label) == identity):
                raise RuntimeError(f"{label}: repeated frame identity; refusing stale feedback")
            self.last_identity[label] = identity
            rgb = np.asarray(frame.rgb)
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ValueError(f"{label}: expected uint8 HxWx3 RGB, got {rgb.shape}/{rgb.dtype}")
            # Combined-camera boot can yield a few all/near-black startup frames.
            # Never feed those into tracking or save them as a valid observation.
            if float(rgb.mean()) < 4.0 or float(rgb.std()) < 2.0:
                raise RuntimeError(
                    f"{label}: startup/invalid dark frame "
                    f"(mean={float(rgb.mean()):.2f}, std={float(rgb.std()):.2f}); retry capture"
                )
            filename = f"{self.index:03d}_{label}.png"
            if not cv2.imwrite(str(self.output / filename), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Failed to save {filename}")
            metadata[label] = {"file": filename, "shape": list(rgb.shape),
                               "frame_id": frame.frame_id, "timestamp_ns": frame.timestamp_ns,
                               "received_monotonic_ns": frame.received_monotonic_ns}
        (self.output / f"{self.index:03d}_frames.json").write_text(json.dumps(metadata, indent=2, default=int)+"\n")
        self.index += 1
        return None if self.camera is None else getattr(pair, self.camera).rgb.copy()


def move_to_board(robot, args, floor):
    from tools.vega_board_benchmark import _vertical_target_for_point

    if args.board_xy is not None:
        center = (*args.board_xy, floor)
    else:
        source = Path(args.registration)
        if not source.is_absolute():
            source = ROOT / source
        record = json.loads(source.read_text())
        if record.get("base_frame") != "vega_1u_base_link" or record.get("robot_name") != robot.config["robot_name"]:
            raise ValueError("Board registration has wrong base frame or robot name")
        center = record["center_base_m"]
    if len(center) != 3 or not all(math.isfinite(float(v)) for v in center):
        raise ValueError("Board center must contain three finite values")
    print("PLANNING BOARD CENTER (corners skipped)", flush=True)
    target = _vertical_target_for_point(robot, "center", center, center, args.hover_z,
                                        floor, math.radians(args.claw_yaw_deg))
    move_tcp_segmented(robot, target, speed_scale=args.speed_scale,
                       max_translation_step_m=0.20, max_orientation_step_rad=0.80,
                       min_tcp_z_m=floor)
    print("REACHED BOARD REGION", robot.get_tcp_pose().position_m, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", choices=("wrist_a", "wrist_b"), default="wrist_a",
                   help="camera used for servo; competition-unit default wrist_a=RIGHT")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm-physical-motion", action="store_true", help="also acknowledges Robot() head homing")
    p.add_argument("--move-to-board", action="store_true", help="first move to saved coarse board center")
    p.add_argument("--registration", default="calibration/vega_board_live.json")
    p.add_argument("--board-xy", nargs=2, type=float, metavar=("X", "Y"), help="override coarse board XY in base metres")
    p.add_argument("--hover-z", type=float, default=0.55)
    p.add_argument("--claw-yaw-deg", type=float, default=0.0)
    p.add_argument("--head-j1", type=float, default=0.55)
    p.add_argument("--feature", nargs=2, type=float, metavar=("U", "V"), help="visible feature pixel in INITIAL hover image; default nearest textured corner")
    p.add_argument("--goal-pixel", nargs=2, type=float, metavar=("U", "V"), help="default image center; later use taught jaw-alignment pixel")
    p.add_argument("--probe-m", type=float, default=0.012)
    p.add_argument("--gain", type=float, default=0.65)
    p.add_argument("--max-step-m", type=float, default=0.015)
    p.add_argument("--max-radius-m", type=float, default=0.06)
    p.add_argument("--tolerance-px", type=float, default=5)
    p.add_argument("--max-iterations", type=int, default=8)
    p.add_argument("--speed-scale", type=float, default=0.45)
    p.add_argument("--output", help="new run directory; default runs/wrist_servo_<UTC>")
    args = p.parse_args(argv)
    if args.execute and not args.confirm_physical_motion:
        p.error("--execute requires --confirm-physical-motion")
    if args.board_xy and not args.move_to_board:
        p.error("--board-xy requires --move-to-board")
    # Validate all motion options before Robot() can home its head.
    limits = (args.probe_m, args.gain, args.max_step_m, args.max_radius_m,
              args.tolerance_px, args.speed_scale, args.hover_z, args.claw_yaw_deg, args.head_j1)
    if not all(math.isfinite(v) for v in limits):
        p.error("motion settings must be finite")
    for coords in (args.board_xy, args.feature, args.goal_pixel):
        if coords is not None and not all(math.isfinite(v) for v in coords):
            p.error("XY and pixel coordinates must be finite")
    if not (0.006 <= args.probe_m <= 0.015 and 0 < args.gain <= 1
            and 0 < args.max_step_m <= 0.02 and args.probe_m <= args.max_radius_m <= 0.10
            and args.tolerance_px > 0 and 1 <= args.max_iterations <= 20
            and 0.45 <= args.speed_scale <= 1):
        p.error("invalid servo limits; see --help")
    import cv2  # Fail on missing image dependency before any robot motion.
    output = Path(args.output) if args.output else ROOT / "runs" / datetime.now(timezone.utc).strftime("wrist_servo_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "arguments.json").write_text(json.dumps(vars(args), indent=2)+"\n")
    print("RUN OUTPUT", output.resolve(), flush=True)

    goal_pixel = None
    capture = None

    def event(kind, fields):
        nonlocal goal_pixel
        record = {"event": kind, "time_monotonic": time.monotonic(), **fields}
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record)+"\n")
        print(kind.upper(), json.dumps(fields), flush=True)
        if "goal_uv" in fields:
            goal_pixel = fields["goal_uv"]
        if "feature_uv" in fields and capture is not None:
            raw = output / f"{capture.index-1:03d}_{args.camera}.png"
            annotated = cv2.imread(str(raw))
            if annotated is not None:
                uv = tuple(int(round(v)) for v in fields["feature_uv"])
                cv2.circle(annotated, uv, 20, (0, 255, 0), 2)
                if goal_pixel is not None:
                    goal = tuple(int(round(v)) for v in goal_pixel)
                    cv2.drawMarker(annotated, goal, (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
                cv2.imwrite(str(raw.with_name(raw.stem+"_tracked.png")), annotated)

    cameras, robot = VegaWristCameras(), None
    try:
        cameras.connect()
        capture = WristCapture(cameras, output, args.camera)
        capture()  # Both views saved before connecting Robot().
        if not args.execute:
            print("CAPTURE ONLY: view 000_wrist_a.png and 000_wrist_b.png; identify physical right.")
            return 0
        cfg = load_bundle("vega")["robot"]
        if cfg["working_arm"] != "right" or cfg["kinematics"]["ee_frame"] != "tip_r":
            raise ValueError("This benchmark requires the right arm / tip_r")
        floor = float(load_vega_skills()["safety"]["min_tcp_z_m"])
        if not floor+0.06 <= args.hover_z <= floor+0.10+1e-9:
            raise ValueError("--hover-z must be 60–100 mm above task floor")
        cfg["allow_robot_init_head_motion"] = True
        cfg["auto_clear_software_estop_on_connect"] = True
        cfg["motion"]["max_step_rad"] = max(float(cfg["motion"]["max_step_rad"]), 0.30)
        robot = VegaAdapter(cfg)
        robot.connect()
        import numpy as np
        head_q = np.asarray(robot._robot.head.get_joint_pos(), dtype=float)
        head_q[0] = args.head_j1
        robot._robot.head.set_joint_pos(head_q, wait_time=1.2, exit_on_reach=True,
                                       exit_on_reach_kwargs={"tolerance": 0.02})
        fine_config = dict(robot._kinematics.config)
        if args.move_to_board:
            move_to_board(robot, args, floor)
        # Coarse planner's 10 mm tolerance would swallow a 12 mm probe and
        # small corrections. Restore normal solver, then tighten position.
        robot._kinematics.config.clear()
        robot._kinematics.config.update(fine_config)
        robot._kinematics.config["position_tolerance_m"] = 0.0007
        robot._kinematics.config["orientation_tolerance_rad"] = 0.01
        result = run_xy_servo(robot, capture, floor_m=floor, feature_uv=args.feature,
                              goal_uv=args.goal_pixel, probe_m=args.probe_m, gain=args.gain,
                              max_step_m=args.max_step_m, max_radius_m=args.max_radius_m,
                              tolerance_px=args.tolerance_px, max_iterations=args.max_iterations,
                              speed_scale=args.speed_scale, event=event)
        (output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
        return 0
    except BaseException as exc:
        if robot is not None:
            try:
                robot.stop()
            except BaseException as stop_error:
                print(f"STOP FAILED: {stop_error}; use physical e-stop", file=sys.stderr)
        event("stopped", reason=str(exc), exception=type(exc).__name__)
        (output / "result.json").write_text(json.dumps({"status": "stopped", "reason": str(exc)}, indent=2)+"\n")
        raise
    finally:
        try:
            cameras.close()
        finally:
            if robot is not None:
                robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
