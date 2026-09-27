"""DexMate Vega U hardware adapter.

This is the real hardware boundary for Vega. It intentionally remains gated by
configuration because constructing dexcontrol Robot() moves the head and motion
must not start from guessed robot/URDF/frame settings.

Verified/public control assumptions used here:
- dexcontrol Robot() exposes left_arm/right_arm joint position state and
  set_joint_pos();
- public control is joint-position only;
- wait_time=0 is NOT used here because it requires a continuous 100-500 Hz
  command loop;
- software e-stop is available through robot.estop;
- the competition gripper is a separate CAN device.
"""

import importlib.metadata
import math
import os
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
        self._validate_motion_config()

        robot_name = self.config.get("robot_name")
        if robot_name and not os.environ.get("ROBOT_NAME"):
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
        self._robot = Robot()
        self._arm = self._select_arm(self._robot)

        kin = self.config["kinematics"]
        joint_names = (
            kin["left_arm_joint_names"]
            if self.config["working_arm"] == "left"
            else kin["right_arm_joint_names"]
        )
        self._kinematics = PinocchioArmKinematics(
            self.config["urdf_path"],
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
        gripper = VegaCanGripper(cfg)
        gripper.connect()
        self._gripper = gripper

    def close(self) -> None:
        self.close_cameras()
        if self._gripper is not None:
            self._gripper.close()
            self._gripper = None
        if self._robot is not None:
            try:
                self._robot.shutdown()
            finally:
                self._robot = None
                self._arm = None
                self._kinematics = None

    def stop(self) -> None:
        """Software emergency stop plus CAN jaw halt when available."""
        if self._gripper is not None:
            self._gripper.halt()
        if self._robot is not None:
            self._robot.estop.activate()

    def clear_estop(self) -> None:
        self._require_robot()
        self._robot.estop.deactivate()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def observe(self) -> RobotObservation:
        self._require_robot()
        q = tuple(float(x) for x in self._arm.get_joint_pos())
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
        return tuple(float(x) for x in self._arm.wrench_sensor.get_wrench_state())

    def get_tcp_pose(self):
        self._require_robot()
        q = tuple(float(x) for x in self._arm.get_joint_pos())
        return self._kinematics.forward(q)

    # ------------------------------------------------------------------
    # Joint / TCP motion
    # ------------------------------------------------------------------

    def move_joints(self, joint_positions, *, speed_scale: float = 1.0) -> None:
        self._require_robot()
        target = tuple(float(x) for x in joint_positions)
        if len(target) != 7 or any(not math.isfinite(x) for x in target):
            raise ValueError("Vega joint target must contain 7 finite values")
        if not (0 < float(speed_scale) <= 1.0):
            raise ValueError("speed_scale must be in (0, 1]")

        limits = (self.config.get("arm_joint_limits_rad") or {}).get(
            self.config["working_arm"]
        )
        if limits:
            if len(limits) != 7:
                raise ValueError("Configured arm joint limits must contain 7 pairs")
            for index, (value, bound) in enumerate(zip(target, limits), 1):
                lo, hi = (float(bound[0]), float(bound[1]))
                if not lo <= value <= hi:
                    raise ValueError(
                        f"Joint {index} target {value:.4f} rad outside "
                        f"[{lo:.4f}, {hi:.4f}]"
                    )

        current = tuple(float(x) for x in self._arm.get_joint_pos())
        delta = [b - a for a, b in zip(current, target)]
        worst = max(abs(x) for x in delta)

        motion = self.config["motion"]
        max_total = float(motion["max_total_delta_rad"])
        if worst > max_total:
            raise ValueError(
                f"Refusing joint target {worst:.3f} rad from current state; "
                f"limit is {max_total:.3f} rad"
            )

        max_step = float(motion["max_step_rad"])
        wait = float(motion["step_wait_time_s"]) / float(speed_scale)
        steps = max(1, int(math.ceil(worst / max_step)))

        import numpy as np

        for index in range(1, steps + 1):
            alpha = index / steps
            waypoint = np.asarray(
                [a + alpha * d for a, d in zip(current, delta)],
                dtype=float,
            )
            self._arm.set_joint_pos(waypoint, wait_time=wait)

    def move_tcp(self, pose, *, speed_scale: float = 1.0) -> None:
        self._require_robot()
        seed = tuple(float(x) for x in self._arm.get_joint_pos())
        target_q = self._kinematics.solve(pose, seed)
        self.move_joints(target_q, speed_scale=speed_scale)

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
        if self._wrist_cameras is not None:
            self._wrist_cameras.close()
            self._wrist_cameras = None
        if self._head_camera is not None:
            self._head_camera.close()
            self._head_camera = None

    # ------------------------------------------------------------------

    def _validate_motion_config(self):
        if not self.config.get("allow_robot_init_head_motion"):
            raise HardwareUnavailableError(
                "Refusing Robot(): it automatically moves the head. Set "
                "allow_robot_init_head_motion=true only after clearing the "
                "workspace and confirming this behavior onsite."
            )
        if self.config.get("working_arm") not in ("left", "right"):
            raise ValueError("vega.working_arm must be left or right")
        if not self.config.get("urdf_path"):
            raise ValueError("vega.urdf_path is required for physical IK")
        motion = self.config.get("motion") or {}
        if motion.get("step_wait_time_s") is None:
            raise ValueError(
                "vega.motion.step_wait_time_s must be physically validated"
            )
        if float(motion.get("max_step_rad", 0)) <= 0:
            raise ValueError("vega.motion.max_step_rad must be > 0")
        if float(motion.get("max_total_delta_rad", 0)) <= 0:
            raise ValueError("vega.motion.max_total_delta_rad must be > 0")
        kin = self.config.get("kinematics") or {}
        if kin.get("backend") != "pinocchio":
            raise ValueError("Only pinocchio kinematics is currently implemented")
        if not kin.get("ee_frame"):
            raise ValueError("vega.kinematics.ee_frame must be verified onsite")

    def _select_arm(self, robot):
        return robot.left_arm if self.config["working_arm"] == "left" else robot.right_arm

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


def connect_cameras(config=None, *, head=True, wrists=True):
    adapter = VegaAdapter(config or {})
    adapter.connect_cameras(head=head, wrists=wrists)
    return adapter
