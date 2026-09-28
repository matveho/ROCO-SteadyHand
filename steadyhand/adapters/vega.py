"""DexMate Vega U hardware adapter.

This is the real hardware boundary for Vega. It intentionally remains gated by
configuration because constructing dexcontrol Robot() moves the head and motion
must not start from guessed robot/URDF/frame settings.

Verified competition-unit control assumptions used here:
- dexcontrol Robot() exposes left_arm/right_arm joint position state;
- public control is joint-position only;
- move_to_joint_pos() delegates trajectory generation/smoothing/gravity
  compensation to the robot-server motion plugin and returns a MotionHandle;
- set_joint_pos(wait_time=0) requires continuous high-frequency streaming and
  set_joint_pos(wait_time>0) is deprecated, so neither path is used here;
- software e-stop is available through robot.estop;
- the competition gripper is a separate CAN device.
"""

import importlib.metadata
import math
import os
from pathlib import Path
import sys
import time

from ..cameras.vega import VegaHeadCamera, VegaWristCameras
from ..grippers.vega import VegaCanGripper
from ..kinematics import PinocchioArmKinematics
from .base import (
    AdapterCapabilities,
    HardwareUnavailableError,
    RobotAdapter,
    RobotObservation,
)


class VegaAdapter(RobotAdapter):
    robot_id = "vega"

    def __init__(self, config):
        super().__init__(config)
        self._robot = None
        self._arm = None
        self._kinematics = None
        self._gripper = None
        self._head_camera = None
        self._wrist_cameras = None
        self._joint_names = ()
        self._joint_limits = ()
        self._active_motion_handle = None

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            joint_motion=True,
            tcp_motion=True,
            cameras=True,
            force_torque=True,
            tactile=False,
        )

    # ------------------------------------------------------------------
    # Motion/control connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self._robot is not None:
            return
        if not self.config.get("allow_robot_init_head_motion"):
            raise HardwareUnavailableError(
                "Refusing Robot(): it automatically moves the head. Set "
                "allow_robot_init_head_motion=true only after clearing the "
                "workspace and confirming this behavior onsite."
            )
        self.prepare()

        robot_name = self.config.get("robot_name")
        env_name = os.environ.get("ROBOT_NAME")
        if robot_name and env_name and robot_name != env_name:
            raise ValueError("Configured robot_name disagrees with ROBOT_NAME; refusing to connect")
        if not robot_name and not env_name:
            raise ValueError("Set robot_name or ROBOT_NAME to the verified physical robot name")
        if robot_name and not env_name:
            os.environ["ROBOT_NAME"] = str(robot_name)

        expected = self.config.get("sdk_version")
        if expected:
            try:
                installed = importlib.metadata.version("dexcontrol")
            except importlib.metadata.PackageNotFoundError as exc:
                raise HardwareUnavailableError("dexcontrol is not installed") from exc
            if (
                installed != expected
                and not self.config.get("allow_sdk_version_mismatch", False)
            ):
                raise HardwareUnavailableError(
                    f"dexcontrol {installed} is installed, expected {expected}; "
                    "verify the onsite API before allowing a version mismatch"
                )

        from dexcontrol.robot import Robot

        # IMPORTANT: the supplied field manual documents that Robot() moves
        # the head to home. _validate_motion_config requires explicit opt-in.
        try:
            self._robot = Robot()
            self._arm = self._select_arm(self._robot)
            if not callable(getattr(self._arm, "move_to_joint_pos", None)):
                raise HardwareUnavailableError(
                    "Installed dexcontrol arm has no move_to_joint_pos(); "
                    "refusing to fall back to unverified streaming/interpolation"
                )
            names = tuple(self._arm.get_joint_name())
            if names != self._joint_names:
                raise ValueError(
                    f"SDK joint order {names!r} differs from configured URDF order "
                    f"{self._joint_names!r}; verify the mapping before commanding"
                )
            self._read_joint_positions()
            stamp = self._state_timestamp()
            self._wait_for_joint_state(newer_than=stamp)
            estop = self._read_estop_status()
            if estop["button_pressed"]:
                raise RuntimeError(
                    "Physical e-stop is active; release it before commanding"
                )
            if estop["software_estop_enabled"]:
                if not self.config.get("auto_clear_software_estop_on_connect", False):
                    raise RuntimeError(
                        "Software e-stop is active; clear it before commanding or "
                        "set auto_clear_software_estop_on_connect=true for an "
                        "operator-supervised physical test"
                    )
                print("CLEARING SOFTWARE E-STOP on connect", flush=True)
                self._robot.estop.deactivate()
                deadline = time.monotonic() + min(
                    float(self.config["motion"]["joint_timeout_s"]), 3.0
                )
                while True:
                    estop = self._read_estop_status()
                    if estop["button_pressed"]:
                        raise RuntimeError(
                            "Physical e-stop became active while clearing software e-stop"
                        )
                    if not estop["software_estop_enabled"]:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "Software e-stop remained active after deactivate()"
                        )
                    time.sleep(0.01)
                print("SOFTWARE E-STOP CLEARED", flush=True)
        except BaseException:
            self._stop_after_failure()
            try:
                self.close()
            except BaseException as exc:
                print(f"Vega connection cleanup failed: {exc}", file=sys.stderr)
            raise

    def prepare(self):
        """Validate local settings and load IK before Robot() can move the head."""
        self._validate_motion_config()
        kin = dict(self.config["kinematics"])
        joint_names = kin["right_arm_joint_names"]
        self._joint_names = tuple(joint_names)
        self._joint_limits = tuple(
            tuple(float(x) for x in bound)
            for bound in self.config["arm_joint_limits_rad"]["right"]
        )
        kin["joint_limits_rad"] = self._joint_limits
        urdf_path = Path(self.config["urdf_path"]).expanduser()
        if not urdf_path.is_absolute():
            urdf_path = Path(__file__).resolve().parents[2] / urdf_path
        self._kinematics = PinocchioArmKinematics(
            str(urdf_path),
            kin["ee_frame"],
            joint_names,
            kin,
        )

    def connect_gripper(self) -> None:
        if self._gripper is not None:
            return
        cfg = dict(self.config.get("gripper") or {})
        if not cfg.get("driver_path"):
            cfg["driver_path"] = self.config.get("gripper_driver_path")
        if not cfg.get("scope"):
            cfg["scope"] = "right"
        if cfg.get("scope") != "right":
            raise ValueError("competition Vega manipulation is locked to the right gripper")
        gripper = VegaCanGripper(cfg)
        gripper.connect()
        self._gripper = gripper

    def close(self) -> None:
        errors = []
        try:
            self.close_cameras()
        except BaseException as exc:
            errors.append(exc)
        if self._gripper is not None:
            try:
                self._gripper.close()
            except BaseException as exc:
                errors.append(exc)
            finally:
                self._gripper = None
        if self._robot is not None:
            try:
                self._robot.shutdown()
            except BaseException as exc:
                errors.append(exc)
            finally:
                self._robot = None
                self._arm = None
                self._kinematics = None
                self._active_motion_handle = None
        if errors:
            raise RuntimeError("Vega shutdown failed: " + "; ".join(str(exc) for exc in errors)) from errors[0]

    def stop(self) -> None:
        """Software emergency stop plus CAN jaw halt when available."""
        errors = []
        handle = self._active_motion_handle
        if handle is not None:
            try:
                if not handle.is_done:
                    handle.cancel()
            except BaseException as exc:
                errors.append(exc)
            finally:
                self._active_motion_handle = None
        if self._robot is not None:
            try:
                self._robot.estop.activate()
                deadline = time.monotonic() + float(self.config["motion"]["joint_timeout_s"])
                while not self._read_estop_status()["software_estop_enabled"]:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Software e-stop was NOT confirmed; use physical e-stop")
                    time.sleep(0.01)
            except BaseException as exc:
                errors.append(exc)
        if self._gripper is not None:
            try:
                self._gripper.halt()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("Vega stop failed: " + "; ".join(str(exc) for exc in errors)) from errors[0]

    def _stop_after_failure(self):
        try:
            self.stop()
        except BaseException as exc:
            print(f"STOP FAILED: {exc}. Use physical e-stop.", file=sys.stderr)

    def clear_estop(self) -> None:
        self._require_robot()
        self._robot.estop.deactivate()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def observe(self) -> RobotObservation:
        self._require_robot()
        q = self._read_joint_positions()
        extras = {
            "joint_velocity": tuple(float(x) for x in self._arm.get_joint_vel()),
        }
        try:
            extras["joint_current"] = tuple(
                float(x) for x in self._arm.get_joint_current()
            )
        except Exception:
            # Current read is useful but should not make state acquisition fail.
            extras["joint_current"] = None
        try:
            extras["wrench"] = tuple(
                float(x) for x in self._arm.wrench_sensor.get_wrench_state()
            )
        except Exception:
            extras["wrench"] = None
        try:
            extras["wrist_buttons"] = self._arm.wrench_sensor.get_button_state()
        except Exception:
            extras["wrist_buttons"] = None

        return RobotObservation(
            timestamp_s=time.monotonic(),
            joint_positions=q,
            extras=extras,
        )

    def read_wrench(self):
        self._require_robot()
        return _finite_vector(self._arm.wrench_sensor.get_wrench_state(), 6, "wrist wrench")

    def get_tcp_pose(self):
        self._require_robot()
        q = self._read_joint_positions()
        return self._kinematics.forward(q)

    # ------------------------------------------------------------------
    # Joint / TCP motion
    # ------------------------------------------------------------------

    def move_joints(self, joint_positions, *, speed_scale: float = 1.0) -> None:
        try:
            self._move_joints(joint_positions, speed_scale=speed_scale)
        except BaseException:
            self._stop_after_failure()
            raise

    def _move_joints(self, joint_positions, *, speed_scale: float = 1.0) -> None:
        self._require_robot()
        target = _finite_vector(joint_positions, 7, "Vega joint target")
        if not (0 < float(speed_scale) <= 1.0):
            raise ValueError("speed_scale must be in (0, 1]")

        self._check_joint_limits(target)

        current = self._read_joint_positions()
        delta = [b - a for a, b in zip(current, target)]
        worst = max(abs(x) for x in delta)

        motion = self.config["motion"]
        max_total = float(motion["max_total_delta_rad"])
        if worst > max_total:
            raise ValueError(
                f"Refusing joint target {worst:.3f} rad from current state; "
                f"limit is {max_total:.3f} rad"
            )

        timeout = float(motion["joint_timeout_s"])
        stamp = self._state_timestamp()

        # IMPORTANT: move_to_joint_pos() is itself a complete, smoothed
        # robot-server trajectory. Do not subdivide one goal into a sequence of
        # blocking motion-plugin goals here. Every MotionHandle converges to a
        # terminal waypoint before returning; waiting for "finished" at each
        # artificial joint chunk forces the arm to decelerate to zero, pause,
        # then accelerate again. That was the source of the visible stop/start
        # motion during onsite Cartesian tests.
        #
        # Safety is enforced by the absolute joint limits and max_total_delta
        # above, then by independent measured endpoint validation below. The
        # server owns interpolation/smoothing/gravity compensation for the
        # entire move.
        handle = self._arm.move_to_joint_pos(
            target,
            relative=False,
            velocity_scale=float(speed_scale),
        )
        self._active_motion_handle = handle
        state = handle.wait(timeout=timeout)
        if state != "finished":
            raise RuntimeError(
                f"Vega motion plugin ended target in state {state!r}: "
                f"{getattr(handle, 'message', '')}"
            )
        self._active_motion_handle = None
        # Once the server motion plugin reports a terminal "finished" state,
        # measured joint state should settle promptly. Do not burn a second
        # full motion timeout here; fail quickly with measured diagnostics.
        self._wait_for_joint_state(
            target=target,
            newer_than=stamp,
            timeout_s=min(timeout, 3.0),
        )

    def move_tcp(self, pose, *, speed_scale: float = 1.0) -> None:
        try:
            self._require_robot()
            seed = self._read_joint_positions()
            target_q = self._kinematics.solve(pose, seed)
            self._move_joints(target_q, speed_scale=speed_scale)
        except BaseException:
            self._stop_after_failure()
            raise

    # ------------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------------

    def open_gripper(self, part_name=None) -> None:
        self._require_gripper()
        self._gripper.open()

    def close_gripper(self, part_name=None) -> None:
        """Empty-jaw close. For objects use grip(), which is current limited."""
        self._require_gripper()
        self._gripper.close_empty()

    def grip(self, part_name=None, *, current_a=None) -> None:
        self._require_gripper()
        self._gripper.grip(current_a=current_a)

    def verify_grasp(self, part_name):
        """Use the official driver's current/stall-based grip result."""
        self._require_gripper()
        result = self._gripper.last_grip_result()
        if result is None:
            return None
        if isinstance(result, dict) and "gripped" in result:
            return bool(result["gripped"])
        return None

    def gripper_status(self):
        self._require_gripper()
        return self._gripper.status()

    def gripper_position(self):
        self._require_gripper()
        return self._gripper.position()

    # ------------------------------------------------------------------
    # No-motion camera path
    # ------------------------------------------------------------------

    def connect_cameras(self, *, head=True, wrists=True) -> None:
        """Connect requested cameras without constructing dexcontrol Robot()."""
        opened_head = False
        try:
            if head and self._head_camera is None:
                self._head_camera = VegaHeadCamera()
                self._head_camera.connect()
                opened_head = True

            if wrists and self._wrist_cameras is None:
                self._wrist_cameras = VegaWristCameras()
                self._wrist_cameras.connect()
        except Exception:
            if opened_head and self._head_camera is not None:
                self._head_camera.close()
                self._head_camera = None
            if wrists and self._wrist_cameras is not None:
                self._wrist_cameras.close()
                self._wrist_cameras = None
            raise

    def read_cameras(self, *, head=True, wrists=True):
        out = {}
        if head:
            if self._head_camera is None:
                raise RuntimeError("Head camera is not connected")
            out["head"] = self._head_camera.read()
        if wrists:
            if self._wrist_cameras is None:
                raise RuntimeError("Wrist cameras are not connected")
            out["wrists"] = self._wrist_cameras.read()
        return out

    def close_cameras(self) -> None:
        errors = []
        if self._wrist_cameras is not None:
            try:
                self._wrist_cameras.close()
            except BaseException as exc:
                errors.append(exc)
            finally:
                self._wrist_cameras = None
        if self._head_camera is not None:
            try:
                self._head_camera.close()
            except BaseException as exc:
                errors.append(exc)
            finally:
                self._head_camera = None
        if errors:
            raise RuntimeError("Camera shutdown failed: " + "; ".join(str(exc) for exc in errors)) from errors[0]

    # ------------------------------------------------------------------

    def _validate_motion_config(self):
        if self.config.get("working_arm") != "right":
            raise ValueError("competition Vega manipulation is locked to working_arm='right'")
        if not self.config.get("urdf_path"):
            raise ValueError("vega.urdf_path is required for physical IK")
        motion = self.config.get("motion") or {}
        for field in ("max_step_rad", "max_total_delta_rad", "joint_reached_tolerance_rad", "joint_timeout_s"):
            _positive(motion.get(field), f"vega.motion.{field}")
        if float(motion["joint_reached_tolerance_rad"]) >= float(motion["max_step_rad"]):
            raise ValueError("joint_reached_tolerance_rad must be smaller than max_step_rad")
        limits = (self.config.get("arm_joint_limits_rad") or {}).get(self.config["working_arm"])
        if not isinstance(limits, (list, tuple)) or len(limits) != 7:
            raise ValueError("arm_joint_limits_rad must supply 7 verified [lower, upper] pairs for the working arm")
        for bound in limits:
            lo, hi = _finite_vector(bound, 2, "joint limit")
            if lo >= hi:
                raise ValueError("Joint limits require lower < upper")
        kin = self.config.get("kinematics") or {}
        if kin.get("backend") != "pinocchio":
            raise ValueError("Only pinocchio kinematics is currently implemented")
        if not kin.get("ee_frame"):
            raise ValueError("vega.kinematics.ee_frame must be verified onsite")

    def _read_joint_positions(self):
        values = _finite_vector(self._arm.get_joint_pos(), 7, "Vega joint state")
        self._check_joint_limits(values)
        return values

    def _check_joint_limits(self, values):
        for index, (value, (lo, hi)) in enumerate(zip(values, self._joint_limits), 1):
            if not lo <= value <= hi:
                raise ValueError(f"Joint {index} value {value:.5f} rad outside [{lo:.5f}, {hi:.5f}]")

    def _state_timestamp(self):
        value = self._arm.get_timestamp_ns()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError("Vega arm state has no valid timestamp; refusing stale/unknown state")
        return value

    def _read_estop_status(self):
        """Require real E-stop state; get_status() reports false when unavailable."""
        raw = self._robot.estop.get_state()
        if not isinstance(raw, dict) or "software_estop_enabled" not in raw:
            raise RuntimeError("E-stop state is unavailable or malformed")
        physical_keys = (
            "left_base_estop_enabled",
            "right_base_estop_enabled",
            "torso_estop_enabled",
            "remote_estop_enabled",
        )
        if any(key not in raw for key in physical_keys):
            raise RuntimeError("E-stop state lacks physical button channels")
        return {
            "software_estop_enabled": bool(raw["software_estop_enabled"]),
            "button_pressed": any(bool(raw[key]) for key in physical_keys),
        }

    def _wait_for_joint_state(self, *, target=None, newer_than, timeout_s=None):
        motion = self.config["motion"]
        timeout_s = (
            float(motion["joint_timeout_s"])
            if timeout_s is None
            else float(timeout_s)
        )
        deadline = time.monotonic() + timeout_s
        tolerance = float(motion["joint_reached_tolerance_rad"])
        last_values = None
        last_stamp = None
        last_fresh = False
        last_error = None
        while True:
            values = self._read_joint_positions()
            stamp = self._state_timestamp()
            fresh = stamp > newer_than
            max_error = (
                None
                if target is None
                else max(abs(a - b) for a, b in zip(values, target))
            )
            reached = target is None or max_error <= tolerance
            last_values = values
            last_stamp = stamp
            last_fresh = fresh
            last_error = max_error
            if fresh and reached:
                return values
            if time.monotonic() >= deadline:
                detail = (
                    f"fresh={last_fresh}, state_timestamp_ns={last_stamp}, "
                    f"required_newer_than_ns={newer_than}"
                )
                if target is not None:
                    detail += (
                        f", max_joint_error_rad={last_error:.6f}, "
                        f"tolerance_rad={tolerance:.6f}, "
                        f"target={tuple(round(float(v), 6) for v in target)}, "
                        f"measured={tuple(round(float(v), 6) for v in last_values)}"
                    )
                try:
                    velocity = _finite_vector(
                        self._arm.get_joint_vel(), 7, "Vega joint velocity"
                    )
                    detail += (
                        f", max_abs_joint_velocity_rad_s="
                        f"{max(abs(v) for v in velocity):.6f}"
                    )
                except Exception:
                    pass
                raise RuntimeError(
                    "Joint motion/state timeout after "
                    f"{timeout_s:.1f}s: {detail}; stop and inspect"
                )
            # Read-only polling; SDK owns the active command stream above.
            time.sleep(0.01)

    def _select_arm(self, robot):
        return robot.right_arm

    def _require_robot(self):
        if self._robot is None or self._arm is None or self._kinematics is None:
            raise RuntimeError("Vega motion adapter is not connected")

    def _require_gripper(self):
        if self._gripper is None:
            raise RuntimeError(
                "Vega CAN gripper is not connected; call connect_gripper() "
                "after bringing can1 up"
            )


def connect(config):
    adapter = VegaAdapter(config)
    adapter.connect()
    return adapter


def _positive(value, name):
    if value is None:
        raise ValueError(f"{name} must be physically validated and set")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return number


def _finite_vector(values, size, name):
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} finite values") from exc
    if len(result) != size or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def connect_cameras(config=None, *, head=True, wrists=True):
    adapter = VegaAdapter(config or {})
    adapter.connect_cameras(head=head, wrists=wrists)
    return adapter
