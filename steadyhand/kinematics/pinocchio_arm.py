"""Pinocchio-based fixed-base arm FK/IK.

Vendor control is joint-position only, so Cartesian task execution needs a
local IK layer. This module lazy-imports numpy/pinocchio so offline config and
tests do not require robot dependencies.

The solver operates only on the selected 7 arm joints. All other URDF joints
remain at neutral or at explicitly configured fixed_joint_values.
"""

from pathlib import Path
import math

from ..geometry import matrix_to_quaternion, quaternion_to_matrix
from ..models import Pose


class IKError(RuntimeError):
    pass


class PinocchioArmKinematics:
    def __init__(self, urdf_path, ee_frame, arm_joint_names, config=None):
        self.config = dict(config or {})
        self.urdf_path = str(Path(urdf_path).expanduser())
        self.ee_frame = str(ee_frame)
        self.arm_joint_names = tuple(arm_joint_names)

        if len(self.arm_joint_names) != 7 or len(set(self.arm_joint_names)) != 7:
            raise ValueError("Vega arm_joint_names must contain exactly 7 distinct joints")
        self.base_frame = self.config.get("base_frame")
        if not self.base_frame:
            raise ValueError(
                "kinematics.base_frame must name the verified URDF frame in "
                "which runtime robot_base target poses are measured"
            )
        self._validate_solver_config()

        import numpy as np
        import pinocchio as pin

        self.np = np
        self.pin = pin
        if not Path(self.urdf_path).is_file():
            raise FileNotFoundError(self.urdf_path)

        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        if not self.model.existFrame(self.ee_frame):
            names = [frame.name for frame in self.model.frames]
            raise ValueError(
                f"EE frame {self.ee_frame!r} not in URDF; "
                f"available frame count={len(names)}"
            )
        self.frame_id = self.model.getFrameId(self.ee_frame)
        if not self.model.existFrame(self.base_frame):
            raise ValueError(f"Base frame {self.base_frame!r} not in URDF")
        self.base_frame_id = self.model.getFrameId(self.base_frame)
        base = self.model.frames[self.base_frame_id]
        if base.parentJoint != 0:
            raise ValueError(
                f"Base frame {self.base_frame!r} must be fixed to the URDF root; "
                "a movable base requires an explicit calibrated transform"
            )
        # A fixed frame can still be translated/rotated from the URDF root.
        # Pinocchio oMf is in that root; callers use the configured base frame.
        self._root_M_base = base.placement.copy()

        self._joint_ids = []
        self._q_idx = []
        self._v_idx = []
        for name in self.arm_joint_names:
            jid = self._joint_id(name)
            joint = self.model.joints[jid]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(
                    f"Joint {name!r} is not a scalar 1-DoF joint "
                    f"(nq={joint.nq}, nv={joint.nv})"
                )
            self._joint_ids.append(jid)
            self._q_idx.append(joint.idx_q)
            self._v_idx.append(joint.idx_v)

        chain = set()
        jid = self.model.frames[self.frame_id].parentJoint
        while jid != 0:
            chain.add(jid)
            jid = self.model.parents[jid]
        wrong_chain = [name for name, jid in zip(self.arm_joint_names, self._joint_ids)
                       if jid not in chain]
        if wrong_chain:
            raise ValueError(
                f"EE frame {self.ee_frame!r} is not downstream of all selected "
                "arm joints: " + ", ".join(wrong_chain)
            )

        self._lower = np.asarray(self.model.lowerPositionLimit)[self._q_idx].copy()
        self._upper = np.asarray(self.model.upperPositionLimit)[self._q_idx].copy()
        configured_limits = self.config.get("joint_limits_rad")
        if configured_limits is not None:
            limits = np.asarray(configured_limits, dtype=float)
            if (limits.shape != (7, 2) or not np.all(np.isfinite(limits))
                    or np.any(limits[:, 0] >= limits[:, 1])):
                raise ValueError("kinematics.joint_limits_rad must be 7 finite [lower, upper] pairs")
            self._lower = np.maximum(self._lower, limits[:, 0])
            self._upper = np.minimum(self._upper, limits[:, 1])
        if (np.any(np.isnan(self._lower)) or np.any(np.isnan(self._upper))
                or np.any(self._lower >= self._upper)):
            raise ValueError("URDF and configured arm joint limits have an invalid or empty intersection")

        self._base_q = pin.neutral(self.model)
        fixed_values = self.config.get("fixed_joint_values", {})
        if not isinstance(fixed_values, dict):
            raise ValueError("kinematics.fixed_joint_values must map joint names to measured values")
        self._require_chain_values(fixed_values)
        for name, value in fixed_values.items():
            jid = self._joint_id(name)
            if jid in self._joint_ids:
                raise ValueError(f"Configured fixed joint {name!r} is an active arm joint")
            joint = self.model.joints[jid]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(f"Configured fixed joint {name!r} is not scalar")
            value = float(value)
            lower = self.model.lowerPositionLimit[joint.idx_q]
            upper = self.model.upperPositionLimit[joint.idx_q]
            if not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(
                    f"Configured fixed joint {name!r} value {value} is non-finite "
                    f"or outside URDF limits [{lower}, {upper}]"
                )
            self._base_q[joint.idx_q] = value

    def _joint_id(self, name):
        # Pinocchio returns model.njoints (not zero) for an unknown name.
        jid = self.model.getJointId(name)
        if not 0 < jid < self.model.njoints:
            raise ValueError(f"Joint {name!r} not found in URDF")
        return jid

    def _validate_solver_config(self):
        for name, default in (("position_tolerance_m", 0.003),
                              ("orientation_tolerance_rad", 0.05),
                              ("damping", 1e-6), ("integration_step", 0.15)):
            value = float(self.config.get(name, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"kinematics.{name} must be finite and positive")
        if float(self.config.get("integration_step", 0.15)) > 1:
            raise ValueError("kinematics.integration_step must be <= 1")
        iterations = float(self.config.get("max_iterations", 250))
        if not math.isfinite(iterations) or iterations < 1 or not iterations.is_integer():
            raise ValueError("kinematics.max_iterations must be a positive integer")
        max_delta = self.config.get("max_seed_delta_rad")
        if max_delta is not None and (not math.isfinite(float(max_delta)) or float(max_delta) <= 0):
            raise ValueError("kinematics.max_seed_delta_rad must be finite and positive")

    def _require_chain_values(self, fixed_values):
        """Require an explicit value for every non-active movable chain joint."""
        active = set(self._joint_ids)
        jid = self.model.frames[self.frame_id].parentJoint
        missing = []
        while jid != 0:
            joint = self.model.joints[jid]
            name = self.model.names[jid]
            if joint.nq > 0 and jid not in active and name not in fixed_values:
                missing.append(name)
            jid = self.model.parents[jid]
        if missing:
            raise ValueError(
                "IK chain has non-arm movable joints with unknown physical "
                "values: " + ", ".join(reversed(missing)) +
                ". Add them to kinematics.fixed_joint_values after verifying "
                "the real robot."
            )

    def _full_q(self, arm_q):
        np = self.np
        arm_q = np.asarray(arm_q, dtype=float)
        if arm_q.shape != (7,) or not np.all(np.isfinite(arm_q)):
            raise ValueError("arm_q must be 7 finite joint values")
        invalid = np.flatnonzero((arm_q < self._lower) | (arm_q > self._upper))
        if invalid.size:
            i = int(invalid[0])
            raise ValueError(
                f"Joint {self.arm_joint_names[i]!r} value {arm_q[i]} is outside "
                f"limits [{self._lower[i]}, {self._upper[i]}]"
            )
        q = self._base_q.copy()
        for idx, value in zip(self._q_idx, arm_q):
            q[idx] = value
        return q

    def _arm_q(self, q):
        return tuple(float(q[idx]) for idx in self._q_idx)

    def forward(self, arm_q):
        pin = self.pin
        q = self._full_q(arm_q)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        placement = self._root_M_base.actInv(self.data.oMf[self.frame_id])
        rotation = tuple(
            tuple(float(placement.rotation[i, j]) for j in range(3))
            for i in range(3)
        )
        return Pose(
            position_m=tuple(float(x) for x in placement.translation),
            quaternion_wxyz=matrix_to_quaternion(rotation),
        )

    def solve(self, target_pose, seed_arm_q):
        np, pin = self.np, self.pin
        q = self._full_q(seed_arm_q)

        r = np.asarray(
            quaternion_to_matrix(target_pose.quaternion_wxyz), dtype=float
        )
        p = np.asarray(target_pose.position_m, dtype=float)
        if p.shape != (3,) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(r)):
            raise ValueError("IK target must contain a finite position and quaternion")
        desired = self._root_M_base * pin.SE3(r, p)
        p, r = desired.translation, desired.rotation

        pos_tol = float(self.config.get("position_tolerance_m", 0.003))
        orn_tol = float(self.config.get("orientation_tolerance_rad", 0.05))
        max_iter = int(self.config.get("max_iterations", 250))
        damp = float(self.config.get("damping", 1e-6))
        dt = float(self.config.get("integration_step", 0.15))

        final_pos = final_orn = None
        for iteration in range(max_iter + 1):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            current = self.data.oMf[self.frame_id]

            relative = current.actInv(desired)
            err = pin.log6(relative).vector

            final_pos = float(np.linalg.norm(current.translation - p))
            r_err = current.rotation.T @ r
            cos_angle = max(-1.0, min(1.0, float((np.trace(r_err) - 1.0) / 2.0)))
            final_orn = float(np.arccos(cos_angle))
            if final_pos <= pos_tol and final_orn <= orn_tol:
                answer = self._arm_q(q)
                self._check_seed_delta(answer, seed_arm_q)
                return answer
            if iteration == max_iter:
                break

            j = self._local_error_jacobian(q, relative)
            if not np.all(np.isfinite(j)) or not np.all(np.isfinite(err)):
                raise IKError("IK produced a non-finite error/Jacobian")

            lhs = j @ j.T + damp * np.eye(6)
            v_arm = -j.T @ np.linalg.solve(lhs, err)
            if not np.all(np.isfinite(v_arm)):
                raise IKError("IK produced a non-finite joint update")
            v = np.zeros(self.model.nv)
            for idx, value in zip(self._v_idx, v_arm):
                v[idx] = value
            q = pin.integrate(self.model, q, v * dt)

            # Clamp active scalar joints to URDF limits.
            q[self._q_idx] = np.clip(q[self._q_idx], self._lower, self._upper)

        raise IKError(
            f"IK did not converge after {max_iter} iterations "
            f"(position_error={final_pos:.5f} m, "
            f"orientation_error={final_orn:.5f} rad)"
        )

    def _local_error_jacobian(self, q, relative):
        # e(q) = log(current(q)^-1 * desired), in the current EE frame.
        # Its derivative is -Jlog6(relative^-1) * J_LOCAL. The leading minus
        # here and in the damped Newton update are both required.
        pin = self.pin
        j_full = pin.computeFrameJacobian(
            self.model, self.data, q, self.frame_id, pin.ReferenceFrame.LOCAL
        )
        return (-pin.Jlog6(relative.inverse()) @ j_full)[:, self._v_idx]

    def _check_seed_delta(self, solution, seed):
        max_delta = self.config.get("max_seed_delta_rad")
        if max_delta is None:
            return
        worst = max(abs(float(a) - float(b)) for a, b in zip(solution, seed))
        if worst > float(max_delta):
            raise IKError(
                f"IK solution is {worst:.3f} rad from seed, exceeding "
                f"max_seed_delta_rad={float(max_delta):.3f}"
            )
