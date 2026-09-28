"""Fine XY validation using the verified physical-right wrist camera (wrist_a).

This tool deliberately starts from an ALREADY ESTABLISHED safe low hover. It
never performs global/board navigation, never changes Z or TCP orientation,
never connects the gripper, and never descends/inserts.

The caller must supply the coarse base-frame XY that the preceding motion stage
established. The live TCP must already be close to that XY and 60-120 mm above
the measured task floor before any XY probe is allowed.

The validation target is IMAGE CENTER ONLY. That proves the local visual-servo
loop; it is not a jaw-alignment calibration.
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
from steadyhand.skill_config import load_vega_skills
from steadyhand.vision.wrist_servo import run_xy_servo

ROOT = Path(__file__).resolve().parents[1]


class WristAOnlyCapture:
    """Fresh wrist_a RGB capture with startup-black and stale-frame rejection."""

    def __init__(
        self,
        cameras,
        output: Path,
        *,
        settle_s: float = 0.20,
        warmup_attempts: int = 8,
    ):
        self.cameras = cameras
        self.output = Path(output)
        self.settle_s = float(settle_s)
        self.warmup_attempts = int(warmup_attempts)
        self.last_identity = None
        self.index = 0

    def _log_capture(self, record):
        with (self.output / "capture_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=int) + "\n")

    def __call__(self):
        import cv2
        import numpy as np

        if self.settle_s:
            time.sleep(self.settle_s)

        last_dark = None
        for attempt in range(1, self.warmup_attempts + 1):
            pair = self.cameras.read(timeout=3.0, fresh=True)
            frame = pair.wrist_a
            identity = (frame.frame_id, frame.timestamp_ns)
            if (
                any(value is not None for value in identity)
                and self.last_identity == identity
            ):
                self._log_capture(
                    {
                        "accepted": False,
                        "reason": "stale",
                        "frame_id": frame.frame_id,
                        "timestamp_ns": frame.timestamp_ns,
                    }
                )
                raise RuntimeError("wrist_a repeated frame identity; refusing stale feedback")
            self.last_identity = identity

            rgb = np.asarray(frame.rgb)
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ValueError(
                    f"wrist_a expected uint8 HxWx3 RGB, got {rgb.shape}/{rgb.dtype}"
                )

            mean = float(rgb.mean())
            std = float(rgb.std())
            metadata = {
                "accepted": False,
                "attempt": attempt,
                "frame_id": frame.frame_id,
                "timestamp_ns": frame.timestamp_ns,
                "received_monotonic_ns": frame.received_monotonic_ns,
                "shape": list(rgb.shape),
                "mean": mean,
                "std": std,
            }
            if mean < 4.0 or std < 2.0:
                metadata["reason"] = "startup_or_invalid_dark"
                self._log_capture(metadata)
                last_dark = (mean, std)
                time.sleep(0.10)
                continue

            filename = f"{self.index:03d}_wrist_a.png"
            ok = cv2.imwrite(
                str(self.output / filename),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
            if not ok:
                raise RuntimeError(f"Failed to save {filename}")
            metadata.update({"accepted": True, "file": filename})
            self._log_capture(metadata)
            (self.output / f"{self.index:03d}_wrist_a.json").write_text(
                json.dumps(metadata, indent=2, default=int) + "\n",
                encoding="utf-8",
            )
            self.index += 1
            return rgb.copy()

        mean, std = last_dark or (float("nan"), float("nan"))
        raise RuntimeError(
            f"wrist_a remained black/invalid for {self.warmup_attempts} fresh frames "
            f"(last mean={mean:.2f}, std={std:.2f})"
        )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--coarse-xy",
        nargs=2,
        type=float,
        required=True,
        metavar=("X", "Y"),
        help="base-frame XY already established by the preceding coarse motion",
    )
    p.add_argument(
        "--feature",
        nargs=2,
        type=float,
        metavar=("U", "V"),
        help="optional feature pixel in the initial wrist_a image; default is a nearby textured feature",
    )
    p.add_argument("--probe-m", type=float, default=0.008)
    p.add_argument("--gain", type=float, default=0.65)
    p.add_argument("--max-step-m", type=float, default=0.010)
    p.add_argument("--max-radius-m", type=float, default=0.040)
    p.add_argument("--tolerance-px", type=float, default=5.0)
    p.add_argument("--max-iterations", type=int, default=6)
    p.add_argument("--speed-scale", type=float, default=0.45)
    p.add_argument("--start-xy-tolerance-m", type=float, default=0.015)
    p.add_argument("--settle-s", type=float, default=0.20)
    p.add_argument("--warmup-attempts", type=int, default=8)
    p.add_argument("--output", help="new run directory; default runs/wrist_fine_<UTC>")
    p.add_argument("--confirm-physical-motion", action="store_true")
    args = p.parse_args(argv)

    if not args.confirm_physical_motion:
        p.error("--confirm-physical-motion is required")
    scalars = (
        *args.coarse_xy,
        args.probe_m,
        args.gain,
        args.max_step_m,
        args.max_radius_m,
        args.tolerance_px,
        args.speed_scale,
        args.start_xy_tolerance_m,
        args.settle_s,
    )
    if not all(math.isfinite(float(v)) for v in scalars):
        p.error("all motion/servo settings must be finite")
    if args.feature is not None and not all(math.isfinite(float(v)) for v in args.feature):
        p.error("--feature requires two finite pixels")
    if not (
        0.006 <= args.probe_m <= 0.012
        and 0 < args.gain <= 1
        and 0 < args.max_step_m <= 0.015
        and args.probe_m <= args.max_radius_m <= 0.060
        and args.tolerance_px > 0
        and 1 <= args.max_iterations <= 12
        and 0.45 <= args.speed_scale <= 0.70
        and 0.003 <= args.start_xy_tolerance_m <= 0.030
        and 0 <= args.settle_s <= 2.0
        and 2 <= args.warmup_attempts <= 20
    ):
        p.error("servo bounds invalid; see --help")

    # Fail before Robot() if image dependencies are missing.
    import cv2
    import numpy as np

    cfg = load_bundle("vega")["robot"]
    if cfg["working_arm"] != "right" or cfg["kinematics"]["ee_frame"] != "tip_r":
        raise ValueError("Fine wrist benchmark requires working_arm=right and ee_frame=tip_r")
    wrist_map = cfg["cameras"]["wrists"]["api_label_to_physical_mount"]
    if wrist_map.get("wrist_a") != "right_wrist":
        raise ValueError("Competition mapping must verify wrist_a=right_wrist before motion")

    floor = float(load_vega_skills()["safety"]["min_tcp_z_m"])
    output = (
        Path(args.output)
        if args.output
        else ROOT / "runs" / datetime.now(timezone.utc).strftime("wrist_fine_%Y%m%dT%H%M%S_%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "arguments.json").write_text(
        json.dumps(vars(args), indent=2) + "\n",
        encoding="utf-8",
    )
    print("RUN OUTPUT", output.resolve(), flush=True)
    print("CAMERA wrist_a = VERIFIED PHYSICAL RIGHT", flush=True)
    print("GOAL = image center (SERVO VALIDATION ONLY; NOT jaw alignment)", flush=True)

    capture = None
    robot = None
    goal_pixel = None

    def event(kind, fields):
        nonlocal goal_pixel
        record = {"event": kind, "time_monotonic": time.monotonic(), **fields}
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(kind.upper(), json.dumps(fields), flush=True)
        if "goal_uv" in fields:
            goal_pixel = fields["goal_uv"]
        if "feature_uv" in fields and capture is not None:
            raw = output / f"{capture.index - 1:03d}_wrist_a.png"
            annotated = cv2.imread(str(raw))
            if annotated is not None:
                uv = tuple(int(round(v)) for v in fields["feature_uv"])
                cv2.circle(annotated, uv, 20, (0, 255, 0), 2)
                if goal_pixel is not None:
                    goal = tuple(int(round(v)) for v in goal_pixel)
                    cv2.drawMarker(
                        annotated,
                        goal,
                        (0, 0, 255),
                        cv2.MARKER_CROSS,
                        25,
                        2,
                    )
                cv2.imwrite(
                    str(raw.with_name(raw.stem + "_tracked.png")),
                    annotated,
                )

    cameras = VegaWristCameras()
    try:
        cameras.connect()
        capture = WristAOnlyCapture(
            cameras,
            output,
            settle_s=args.settle_s,
            warmup_attempts=args.warmup_attempts,
        )

        # Warm the verified RIGHT stream before constructing Robot(); black
        # startup frames are discarded and never become servo observations.
        capture()

        cfg["allow_robot_init_head_motion"] = True
        cfg["auto_clear_software_estop_on_connect"] = True
        robot = VegaAdapter(cfg)
        robot.connect()

        start = robot.get_tcp_pose()
        coarse_xy = tuple(float(v) for v in args.coarse_xy)
        start_xy_error = math.dist(start.position_m[:2], coarse_xy)
        min_hover_z = floor + 0.060
        max_hover_z = floor + 0.120
        start_record = {
            "coarse_xy_base_m": coarse_xy,
            "measured_tcp_position_m": start.position_m,
            "measured_tcp_quaternion_wxyz": start.quaternion_wxyz,
            "xy_error_to_supplied_coarse_m": start_xy_error,
            "allowed_hover_z_m": [min_hover_z, max_hover_z],
        }
        (output / "start_state.json").write_text(
            json.dumps(start_record, indent=2) + "\n",
            encoding="utf-8",
        )
        print("START STATE", json.dumps(start_record), flush=True)

        if start_xy_error > float(args.start_xy_tolerance_m):
            raise RuntimeError(
                f"Live TCP is {start_xy_error:.4f} m from supplied --coarse-xy; "
                "establish the coarse hover first"
            )
        if not min_hover_z <= float(start.position_m[2]) <= max_hover_z:
            raise RuntimeError(
                f"Live TCP z={float(start.position_m[2]):.4f} m is not a low safe hover "
                f"({min_hover_z:.4f}..{max_hover_z:.4f} m); establish the hover first"
            )

        # Small 8-10 mm probes require tighter IK convergence than coarse board
        # navigation. This changes solver tolerances only; no pose is commanded here.
        robot._kinematics.config["position_tolerance_m"] = 0.0007
        robot._kinematics.config["orientation_tolerance_rad"] = 0.01

        result = run_xy_servo(
            robot,
            capture,
            floor_m=floor,
            feature_uv=args.feature,
            goal_uv=None,  # Explicitly image center: validation, not jaw alignment.
            probe_m=args.probe_m,
            gain=args.gain,
            max_step_m=args.max_step_m,
            max_radius_m=args.max_radius_m,
            tolerance_px=args.tolerance_px,
            max_iterations=args.max_iterations,
            speed_scale=args.speed_scale,
            event=event,
        )
        result["validation_goal"] = "image_center_only_not_jaw_alignment"
        result["supplied_coarse_xy_base_m"] = coarse_xy
        (output / "result.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        print("WRIST_A FINE CENTER PASS", flush=True)
        return 0
    except BaseException as exc:
        # Do not assert software E-stop for camera/tracking/validation failures.
        # VegaAdapter.move_tcp() already stops the robot on an actual motion
        # command failure or interruption.
        record = {"status": "stopped", "reason": str(exc), "exception": type(exc).__name__}
        try:
            (output / "result.json").write_text(
                json.dumps(record, indent=2) + "\n",
                encoding="utf-8",
            )
        except BaseException:
            pass
        print("WRIST_A FINE CENTER STOPPED", json.dumps(record), flush=True)
        raise
    finally:
        try:
            cameras.close()
        finally:
            if robot is not None:
                robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
