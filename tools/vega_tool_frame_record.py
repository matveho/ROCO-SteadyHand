"""Interactive no-motion recorder for Vega A/B/C/D tool-frame calibration.

This tool NEVER constructs dexcontrol.robot.Robot and never sends a motion,
mode, E-stop, gripper, head, or camera command.  It creates a read-only arm
state subscriber directly, computes modeled tip_r pose from measured joints
using the tracked Pinocchio model, and writes analyzer-compatible JSON.

Robot repositioning between A/B/C/D is entirely external to this process.

Stages:
  A_REFERENCE      physical claw center on a repeatable reference mark
  B_FORWARD        same intended tip orientation, translated +base X
  C_PIVOT_PITCH    physical claw center returned to A mark, orientation changed
  D_PIVOT_ROLL     physical claw center returned to A mark, second nonparallel change

External measurements are optional.  If an A->B physical center delta is
entered (or both A/B absolute centers are entered), a translation_check is
included.  A/C/D are assigned pivot_group="center_mark" so the merged
vega_tool_frame_calibration.py analyzer can solve the fixed center offset.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.config import load_bundle
from steadyhand.kinematics import PinocchioArmKinematics
from tools.vega_tool_frame_calibration import (
    EXPECTED_BASE_FRAME,
    EXPECTED_JOINT_NAMES,
    EXPECTED_RECORDER_MODE,
    EXPECTED_RECORDER_TOOL,
    EXPECTED_ROBOT_NAME,
    EXPECTED_TCP_FRAME,
    EXPECTED_WORKING_ARM,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "calibration" / "vega_tool_frame.template.json"
DEFAULT_OUTPUT = ROOT / "calibration" / "vega_tool_frame_measurements.json"

STAGES = (
    (
        "A_REFERENCE",
        "Place the PHYSICAL claw center on the repeatable reference mark. "
        "This pose is part of pivot group center_mark.",
    ),
    (
        "B_FORWARD",
        "Using an EXTERNAL procedure, move +base-X 50-100 mm while preserving "
        "the intended tip quaternion. Do not use B as a pivot pose.",
    ),
    (
        "C_PIVOT_PITCH",
        "Return the PHYSICAL claw center to the exact A mark and change "
        "orientation about one horizontal axis by roughly 8-12 deg.",
    ),
    (
        "D_PIVOT_ROLL",
        "Keep the PHYSICAL claw center on the A mark and change orientation "
        "about a second nonparallel horizontal axis by roughly 8-12 deg.",
    ),
)
PIVOT_LABELS = {"A_REFERENCE", "C_PIVOT_PITCH", "D_PIVOT_ROLL"}


def _finite_vector(values, size, name):
    try:
        result = tuple(float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} finite numbers") from exc
    if len(result) != size or any(not math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


def _unit_vector(values, name):
    vec = _finite_vector(values, 3, name)
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be nonzero")
    return tuple(v / norm for v in vec)


def parse_optional_vector(text, *, size=3, scale=1.0, name="vector"):
    """Parse blank/none as None; otherwise parse comma/space separated values."""
    raw = str(text).strip()
    if not raw or raw.lower() in {"none", "skip", "unknown", "n/a"}:
        return None
    parts = raw.replace(",", " ").split()
    values = _finite_vector(parts, size, name)
    return tuple(float(v) * float(scale) for v in values)


def _prompt_optional_vector(prompt, *, size=3, scale=1.0, name="vector"):
    while True:
        raw = input(prompt).strip()
        try:
            return parse_optional_vector(
                raw, size=size, scale=scale, name=name
            )
        except ValueError as exc:
            print(f"INVALID: {exc}", flush=True)


def _pose_record(pose):
    return {
        "position_m": [float(v) for v in pose.position_m],
        "quaternion_wxyz": [float(v) for v in pose.quaternion_wxyz],
    }


def make_observation(
    label,
    *,
    pose,
    joint_positions_rad,
    joint_timestamp_ns,
    physical_center_base_m=None,
    physical_axis_base=None,
    physical_x_axis_base=None,
    recorded_at_utc=None,
):
    if label not in {stage[0] for stage in STAGES}:
        raise ValueError(f"unknown calibration stage {label!r}")
    q = _finite_vector(joint_positions_rad, 7, "joint_positions_rad")
    stamp = int(joint_timestamp_ns)
    if stamp <= 0:
        raise ValueError("joint_timestamp_ns must be positive")

    obs = {
        "label": label,
        "modeled_tip_pose": _pose_record(pose),
        "joint_positions_rad": [float(v) for v in q],
        "joint_timestamp_ns": stamp,
        "recorded_at_utc": recorded_at_utc
        or datetime.now(timezone.utc).isoformat(),
    }
    if label in PIVOT_LABELS:
        obs["pivot_group"] = "center_mark"

    if physical_center_base_m is not None:
        obs["physical_claw_center_base_m"] = list(
            _finite_vector(
                physical_center_base_m, 3, "physical_claw_center_base_m"
            )
        )
    if physical_axis_base is not None:
        obs["physical_claw_axis_base"] = list(
            _unit_vector(physical_axis_base, "physical_claw_axis_base")
        )
    if physical_x_axis_base is not None:
        if physical_axis_base is None:
            raise ValueError(
                "physical_claw_x_axis_base requires physical_claw_axis_base"
            )
        x_axis = _unit_vector(
            physical_x_axis_base, "physical_claw_x_axis_base"
        )
        z_axis = tuple(obs["physical_claw_axis_base"])
        alignment = abs(sum(a * b for a, b in zip(x_axis, z_axis)))
        if alignment > 0.98:
            raise ValueError(
                "physical claw X and Z axes are nearly parallel; "
                "measure a distinct jaw/reference direction"
            )
        obs["physical_claw_x_axis_base"] = list(x_axis)
    return obs


def _find_observation(observations, label):
    for obs in observations:
        if obs.get("label") == label:
            return obs
    return None


def _physical_center(obs):
    if obs is None:
        return None
    value = obs.get("physical_claw_center_base_m")
    if value is None:
        return None
    return _finite_vector(value, 3, "physical_claw_center_base_m")


def resolve_ab_delta(observations, explicit_delta_m=None):
    if explicit_delta_m is not None:
        return _finite_vector(
            explicit_delta_m, 3, "A_to_B physical claw-center delta"
        ), "operator_entered_delta"

    a = _physical_center(_find_observation(observations, "A_REFERENCE"))
    b = _physical_center(_find_observation(observations, "B_FORWARD"))
    if a is None or b is None:
        return None, None
    return tuple(bi - ai for ai, bi in zip(a, b)), "difference_of_absolute_centers"


def _fallback_context():
    return {
        "purpose": (
            "offline calibration of modeled tip_r to operator-defined "
            "physical claw center/frame"
        ),
        "frame_convention": {
            "base_axes": "x forward/away from robot, y robot-left, z up",
            "physical_claw_center": "center used by operator for alignment",
            "physical_claw_axis_base": (
                "unit vector from wrist toward fingertips"
            ),
            "physical_claw_x_axis_base": (
                "unit vector along a marked jaw/reference direction; "
                "required only to resolve yaw/full frame"
            ),
        },
        "measurement_protocol": [instruction for _, instruction in STAGES],
    }


def _load_template_context():
    """Load only calibration instructions, never historical numeric evidence."""
    fallback = _fallback_context()
    if not TEMPLATE_PATH.is_file():
        return fallback
    try:
        data = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    return {
        key: deepcopy(data.get(key, fallback[key]))
        for key in (
            "purpose",
            "frame_convention",
            "measurement_protocol",
        )
    }


def _validate_runtime_identity(cfg):
    kin = cfg.get("kinematics") or {}
    checks = (
        ("robot_name", cfg.get("robot_name"), EXPECTED_ROBOT_NAME),
        ("working_arm", cfg.get("working_arm"), EXPECTED_WORKING_ARM),
        ("base_frame", kin.get("base_frame"), EXPECTED_BASE_FRAME),
        ("tcp_frame", kin.get("ee_frame"), EXPECTED_TCP_FRAME),
    )
    for field, actual, expected in checks:
        if actual != expected:
            raise ValueError(
                f"tool-frame recorder {field}={actual!r}, expected {expected!r}"
            )
    joint_names = tuple(kin.get("right_arm_joint_names") or ())
    if joint_names != EXPECTED_JOINT_NAMES:
        raise ValueError(
            "tool-frame recorder right_arm_joint_names do not match the "
            f"required order {EXPECTED_JOINT_NAMES!r}"
        )


def build_analyzer_payload(
    observations,
    *,
    explicit_ab_delta_m=None,
    translation_tolerance_m=0.003,
    robot_name=EXPECTED_ROBOT_NAME,
):
    if not observations:
        raise ValueError("at least one captured observation is required")
    labels = [obs.get("label") for obs in observations]
    if any(not label for label in labels) or len(labels) != len(set(labels)):
        raise ValueError("captured observations require unique labels")

    if robot_name != EXPECTED_ROBOT_NAME:
        raise ValueError(
            f"tool-frame recorder robot_name={robot_name!r}, "
            f"expected {EXPECTED_ROBOT_NAME!r}"
        )

    payload = {
        "schema_version": 1,
        **_load_template_context(),
        "recorder": {
            "tool": EXPECTED_RECORDER_TOOL,
            "mode": EXPECTED_RECORDER_MODE,
            "robot_name": EXPECTED_ROBOT_NAME,
            "base_frame": EXPECTED_BASE_FRAME,
            "working_arm": EXPECTED_WORKING_ARM,
            "tcp_frame": EXPECTED_TCP_FRAME,
            "joint_names": list(EXPECTED_JOINT_NAMES),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": (
                "Robot repositioning between observations was external to this "
                "recorder. This recorder only subscribed to arm state and ran FK."
            ),
        },
        "observations": [deepcopy(obs) for obs in observations],
        "translation_checks": [],
    }

    delta, source = resolve_ab_delta(
        observations, explicit_delta_m=explicit_ab_delta_m
    )
    if delta is not None:
        if float(translation_tolerance_m) <= 0:
            raise ValueError("translation_tolerance_m must be > 0")
        payload["translation_checks"].append(
            {
                "from": "A_REFERENCE",
                "to": "B_FORWARD",
                "measured_physical_center_delta_m": [float(v) for v in delta],
                "explanation_tolerance_m": float(translation_tolerance_m),
                "measurement_source": source,
            }
        )
    return payload


def write_payload(
    path,
    observations,
    *,
    explicit_ab_delta_m=None,
    translation_tolerance_m=0.003,
    robot_name=None,
):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    payload = build_analyzer_payload(
        observations,
        explicit_ab_delta_m=explicit_ab_delta_m,
        translation_tolerance_m=translation_tolerance_m,
        robot_name=robot_name,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    return path, payload


class ReadOnlyVegaTipReader:
    """Direct arm-state subscriber + local FK.

    Intentionally does not use VegaAdapter.connect() or dexcontrol.robot.Robot,
    because Robot() has a documented head-motion side effect on this robot.
    """

    def __init__(self, cfg, *, active_timeout_s=5.0, fresh_timeout_s=3.0):
        self.cfg = deepcopy(cfg)
        self.active_timeout_s = float(active_timeout_s)
        self.fresh_timeout_s = float(fresh_timeout_s)
        self._query_owner = None
        self._arm = None
        self._kinematics = None
        self._joint_names = ()
        self._last_timestamp_ns = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def connect(self):
        if self._arm is not None:
            return

        robot_name = self.cfg.get("robot_name")
        env_name = os.environ.get("ROBOT_NAME")
        if robot_name and env_name and str(robot_name) != str(env_name):
            raise RuntimeError(
                "Configured robot_name disagrees with ROBOT_NAME; refusing read"
            )
        if robot_name and not env_name:
            os.environ["ROBOT_NAME"] = str(robot_name)
        if not os.environ.get("ROBOT_NAME"):
            raise RuntimeError("ROBOT_NAME is not set")

        expected = self.cfg.get("sdk_version")
        if expected:
            try:
                installed = importlib.metadata.version("dexcontrol")
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError("dexcontrol is not installed") from exc
            if installed != expected and not self.cfg.get(
                "allow_sdk_version_mismatch", False
            ):
                raise RuntimeError(
                    f"dexcontrol {installed} is installed, expected {expected}"
                )

        # Lazy imports keep offline tests independent from robot packages.
        #
        # IMPORTANT: there is intentionally NO import of dexcontrol.robot.Robot.
        # Robot() is the API with the documented automatic head movement.
        from dexbot_utils import RobotInfo
        from dexcontrol.core.arm import Arm
        from dexcontrol.core.robot_query_interface import RobotQueryInterface

        signature = inspect.signature(Arm)
        required = {"name", "robot_info"}
        if not required.issubset(signature.parameters):
            raise RuntimeError(
                "Installed dexcontrol Arm constructor is not the verified "
                "(name, robot_info) API. Refusing to fall back to Robot(), "
                "because this recorder must remain no-motion."
            )

        try:
            _validate_runtime_identity(self.cfg)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        component_name = "right_arm"

        kin_cfg = dict(self.cfg["kinematics"])
        self._joint_names = tuple(kin_cfg["right_arm_joint_names"])
        limits = self.cfg["arm_joint_limits_rad"]["right"]
        kin_cfg["joint_limits_rad"] = limits

        urdf_path = Path(self.cfg["urdf_path"]).expanduser()
        if not urdf_path.is_absolute():
            urdf_path = ROOT / urdf_path
        self._kinematics = PinocchioArmKinematics(
            str(urdf_path),
            kin_cfg["ee_frame"],
            self._joint_names,
            kin_cfg,
        )

        try:
            # QueryInterface owns the shared DexComm node only. It does not
            # construct robot components or issue motion commands.
            self._query_owner = RobotQueryInterface.create()
            robot_info = RobotInfo()
            self._arm = Arm(name=component_name, robot_info=robot_info)

            names = tuple(self._arm.get_joint_name())
            if names != self._joint_names:
                raise RuntimeError(
                    f"SDK joint order {names!r} differs from configured "
                    f"order {self._joint_names!r}"
                )
            if not self._arm.wait_for_active(timeout=self.active_timeout_s):
                raise RuntimeError(
                    f"{component_name} state did not become active within "
                    f"{self.active_timeout_s:.1f}s"
                )
            # First capture also requires a fresh timestamp transition.
            self._last_timestamp_ns = None
        except BaseException:
            self.close()
            raise

    def _raw_sample(self):
        if self._arm is None:
            raise RuntimeError("read-only Vega state reader is not connected")
        joints = _finite_vector(
            self._arm.get_joint_pos(), 7, "Vega joint positions"
        )
        stamp = int(self._arm.get_timestamp_ns())
        if stamp <= 0:
            raise RuntimeError("Vega joint state timestamp is invalid")
        return joints, stamp

    def capture(self):
        if self._arm is None or self._kinematics is None:
            raise RuntimeError("read-only Vega state reader is not connected")

        # Require a state newer than the last captured sample.  On the first
        # capture, require one timestamp transition after entering capture().
        initial_q, initial_stamp = self._raw_sample()
        required_newer_than = (
            initial_stamp
            if self._last_timestamp_ns is None
            else self._last_timestamp_ns
        )
        deadline = time.monotonic() + self.fresh_timeout_s
        joints, stamp = initial_q, initial_stamp
        while stamp <= required_newer_than:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Joint-state timestamp did not advance during read-only "
                    f"capture: current={stamp}, required>{required_newer_than}"
                )
            time.sleep(0.02)
            joints, stamp = self._raw_sample()

        self._last_timestamp_ns = stamp
        pose = self._kinematics.forward(joints)
        return {
            "pose": pose,
            "joint_positions_rad": joints,
            "joint_timestamp_ns": stamp,
        }

    def close(self):
        errors = []
        if self._arm is not None:
            try:
                # Arm.shutdown() tears down this client's subscriber/publisher.
                # It does not call Arm.stop() and this client has never sent a
                # target or command.
                self._arm.shutdown()
            except BaseException as exc:
                errors.append(exc)
            finally:
                self._arm = None
        if self._query_owner is not None:
            try:
                self._query_owner.close()
            except BaseException as exc:
                errors.append(exc)
            finally:
                self._query_owner = None
        self._kinematics = None
        if errors:
            raise RuntimeError(
                "Read-only Vega state cleanup failed: "
                + "; ".join(str(e) for e in errors)
            )


def _print_capture(label, sample):
    pose = sample["pose"]
    print(
        f"{label} tip_r XYZ m = "
        f"{tuple(round(float(v), 6) for v in pose.position_m)}",
        flush=True,
    )
    print(
        f"{label} tip_r QUAT wxyz = "
        f"{tuple(round(float(v), 8) for v in pose.quaternion_wxyz)}",
        flush=True,
    )
    print(
        f"{label} JOINTS rad = "
        f"{tuple(round(float(v), 7) for v in sample['joint_positions_rad'])}",
        flush=True,
    )
    print(
        f"{label} JOINT TIMESTAMP ns = {sample['joint_timestamp_ns']}",
        flush=True,
    )


def _external_measurements(label, *, length_scale):
    unit = "mm" if math.isclose(length_scale, 0.001) else "m"
    print(
        "External measurements are optional. Blank/skip/unknown omits a field.",
        flush=True,
    )
    center = _prompt_optional_vector(
        f"{label} physical claw CENTER in base XYZ [{unit}; blank=unknown]: ",
        scale=length_scale,
        name=f"{label} physical center",
    )
    axis = _prompt_optional_vector(
        f"{label} physical claw AXIS base XYZ "
        "(unitless wrist->fingertips; blank=unknown): ",
        scale=1.0,
        name=f"{label} physical axis",
    )
    if axis is not None:
        axis = _unit_vector(axis, f"{label} physical axis")
    x_axis = _prompt_optional_vector(
        f"{label} physical claw X/jaw reference axis base XYZ "
        "(unitless; blank=unknown): ",
        scale=1.0,
        name=f"{label} physical x axis",
    )
    if x_axis is not None:
        if axis is None:
            print(
                "X-axis ignored because no physical claw axis was supplied.",
                flush=True,
            )
            x_axis = None
        else:
            x_axis = _unit_vector(x_axis, f"{label} physical x axis")
    return center, axis, x_axis


def _confirm_capture(label, instruction):
    print("", flush=True)
    print("=" * 72, flush=True)
    print(label, flush=True)
    print(instruction, flush=True)
    print(
        "This recorder will NOT move the robot. Reposition using your external "
        "procedure, wait until stationary, then return here.",
        flush=True,
    )
    while True:
        answer = input(
            f"Press ENTER to record current {label} state, or type 'abort': "
        ).strip().lower()
        if answer in {"abort", "a", "q", "quit"}:
            raise KeyboardInterrupt()
        if answer == "":
            return
        print("Type only ENTER to capture, or 'abort'.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT.relative_to(ROOT)),
        help="analyzer-compatible JSON output; rewritten atomically after each capture",
    )
    parser.add_argument(
        "--external-length-unit",
        choices=("mm", "m"),
        default="mm",
        help="unit used for entered physical center positions/deltas (default mm)",
    )
    parser.add_argument(
        "--translation-tolerance-mm",
        type=float,
        default=3.0,
        help="A->B physical-center prediction tolerance for analyzer (default 3 mm)",
    )
    parser.add_argument("--active-timeout-s", type=float, default=5.0)
    parser.add_argument("--fresh-timeout-s", type=float, default=3.0)
    args = parser.parse_args(argv)

    if not math.isfinite(args.translation_tolerance_mm) or args.translation_tolerance_mm <= 0:
        parser.error("--translation-tolerance-mm must be finite and > 0")
    if not math.isfinite(args.active_timeout_s) or args.active_timeout_s <= 0:
        parser.error("--active-timeout-s must be finite and > 0")
    if not math.isfinite(args.fresh_timeout_s) or args.fresh_timeout_s <= 0:
        parser.error("--fresh-timeout-s must be finite and > 0")

    cfg = load_bundle("vega")["robot"]
    try:
        _validate_runtime_identity(cfg)
    except ValueError as exc:
        parser.error(str(exc))

    length_scale = 0.001 if args.external_length_unit == "mm" else 1.0
    tolerance_m = float(args.translation_tolerance_mm) / 1000.0

    print("VEGA TOOL-FRAME A/B/C/D RECORDER", flush=True)
    print("MODE = READ_ONLY_NO_MOTION_COMMANDS", flush=True)
    print(
        "No Robot() object, no arm/head/gripper command, and no E-stop state "
        "change is issued by this tool.",
        flush=True,
    )
    print(
        "All A/B/C/D repositioning must be performed externally by the operator.",
        flush=True,
    )

    observations = []
    explicit_ab_delta_m = None

    with ReadOnlyVegaTipReader(
        cfg,
        active_timeout_s=args.active_timeout_s,
        fresh_timeout_s=args.fresh_timeout_s,
    ) as reader:
        for label, instruction in STAGES:
            _confirm_capture(label, instruction)
            sample = reader.capture()
            _print_capture(label, sample)

            center, axis, x_axis = _external_measurements(
                label, length_scale=length_scale
            )
            observation = make_observation(
                label,
                pose=sample["pose"],
                joint_positions_rad=sample["joint_positions_rad"],
                joint_timestamp_ns=sample["joint_timestamp_ns"],
                physical_center_base_m=center,
                physical_axis_base=axis,
                physical_x_axis_base=x_axis,
            )
            observations.append(observation)

            if label == "B_FORWARD":
                unit = args.external_length_unit
                delta = _prompt_optional_vector(
                    "A->B PHYSICAL claw-center DELTA base dX dY dZ "
                    f"[{unit}; blank=derive from absolute A/B centers if present, "
                    "otherwise omit translation check]: ",
                    scale=length_scale,
                    name="A->B physical center delta",
                )
                explicit_ab_delta_m = delta

            out, payload = write_payload(
                args.output,
                observations,
                explicit_ab_delta_m=explicit_ab_delta_m,
                translation_tolerance_m=tolerance_m,
                robot_name=cfg.get("robot_name"),
            )
            print(
                f"SAVED {label}: {out.resolve()} "
                f"({len(payload['observations'])}/4 poses)",
                flush=True,
            )

    out, payload = write_payload(
        args.output,
        observations,
        explicit_ab_delta_m=explicit_ab_delta_m,
        translation_tolerance_m=tolerance_m,
        robot_name=cfg.get("robot_name"),
    )
    print("", flush=True)
    print("RECORDER COMPLETE", flush=True)
    print("OUTPUT =", out.resolve(), flush=True)
    print("OBSERVATIONS =", [o["label"] for o in payload["observations"]], flush=True)
    if payload["translation_checks"]:
        print(
            "A->B TRANSLATION CHECK =",
            payload["translation_checks"][0],
            flush=True,
        )
    else:
        print(
            "A->B TRANSLATION CHECK = omitted (no external physical delta available)",
            flush=True,
        )
    print("", flush=True)
    print("READY ANALYZER COMMAND:", flush=True)
    print(
        "python3 tools/vega_tool_frame_calibration.py "
        f"{out.resolve()} "
        "--json-output runs/vega_tool_frame_analysis.json",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
