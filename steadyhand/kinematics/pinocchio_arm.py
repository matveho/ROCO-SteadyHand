"""Pinocchio-based fixed-base arm FK/IK.

Vendor control is joint-position only, so Cartesian task execution needs a
local IK layer. This module lazy-imports numpy/pinocchio so offline config and
tests do not require robot dependencies.

The solver operates only on the selected 7 arm joints. All other URDF joints
remain at neutral or at explicitly configured fixed_joint_values.
"""

from pathlib import Path

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

        if len(self.arm_joint_names) != 7:
            raise ValueError("Vega arm_joint_names must contain exactly 7 joints")

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

        self._joint_ids = []
        self._q_idx = []
        self._v_idx = []
        for name in self.arm_joint_names:
            jid = self.model.getJointId(name)
            if jid == 0:
                raise ValueError(f"Joint {name!r} not found in URDF")
            joint = self.model.joints[jid]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(
                    f"Joint {name!r} is not a scalar 1-DoF joint "
                    f"(nq={joint.nq}, nv={joint.nv})"
                )
            self._joint_ids.append(jid)
            self._q_idx.append(joint.idx_q)
            self._v_idx.append(joint.idx_v)

        self._base_q = pin.neutral(self.model)
        for name, value in self.config.get("fixed_joint_values", {}).items():
            jid = self.model.getJointId(name)
            if jid == 0:
                raise ValueError(f"Configured fixed joint {name!r} not in URDF")
            joint = self.model.joints[jid]
            if joint.nq != 1:
                raise ValueError(f"Configured fixed joint {name!r} is not scalar")
            self._base_q[joint.idx_q] = float(value)

    def _full_q(self, arm_q):
        np = self.np
        arm_q = np.asarray(arm_q, dtype=float)
        if arm_q.shape != (7,) or not np.all(np.isfinite(arm_q)):
            raise ValueError("arm_q must be 7 finite joint values")
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
        placement = self.data.oMf[self.frame_id]
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
        desired = pin.SE3(r, p)

        pos_tol = float(self.config.get("position_tolerance_m", 0.003))
        orn_tol = float(self.config.get("orientation_tolerance_rad", 0.05))
        max_iter = int(self.config.get("max_iterations", 250))
        damp = float(self.config.get("damping", 1e-6))
        dt = float(self.config.get("integration_step", 0.15))

        final_pos = final_orn = None
        for iteration in range(max_iter):
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

            j_full = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.frame_id,
                pin.ReferenceFrame.LOCAL,
            )
            j_err = -pin.Jlog6(relative.inverse()) @ j_full
            j = j_err[:, self._v_idx]

            lhs = j @ j.T + damp * np.eye(6)
            v_arm = -j.T @ np.linalg.solve(lhs, err)
            v = np.zeros(self.model.nv)
            for idx, value in zip(self._v_idx, v_arm):
                v[idx] = value
            q = pin.integrate(self.model, q, v * dt)

            # Clamp active scalar joints to URDF limits.
            for qidx in self._q_idx:
                lower = self.model.lowerPositionLimit[qidx]
                upper = self.model.upperPositionLimit[qidx]
                if np.isfinite(lower):
                    q[qidx] = max(q[qidx], lower)
                if np.isfinite(upper):
                    q[qidx] = min(q[qidx], upper)

        raise IKError(
            f"IK did not converge after {max_iter} iterations "
            f"(position_error={final_pos:.5f} m, "
            f"orientation_error={final_orn:.5f} rad)"
        )

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
