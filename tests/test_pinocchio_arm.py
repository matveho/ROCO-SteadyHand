"""Numerical kinematics regression tests, not a claim of Vega calibration.

The generated URDF has seven bounded arm joints, two non-arm chain joints,
an unrelated continuous joint (nq != nv), a rotated/translated base frame,
and an offset/rotated tool. Run with the robot's NumPy + Pinocchio interpreter;
standard-library CI explicitly skips the numerical tests when unavailable.
"""

import math
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.geometry import quaternion_angle, quaternion_to_matrix
from steadyhand.kinematics.pinocchio_arm import IKError, PinocchioArmKinematics
from steadyhand.models import Pose
from steadyhand.executor import preflight_tcp_segmented

try:
    import numpy as np
    import pinocchio as pin
    if not hasattr(pin, "buildModelFromUrdf"):
        raise ImportError("the imported pinocchio is not the robotics library")
except ImportError:
    np = pin = None


NAMES = tuple(f"arm_j{i}" for i in range(1, 8))
SEED = (0.2, 0.4, -0.3, -0.7, 0.25, 0.45, -0.25)
LIMITS = ((-3, 3), (-0.4, 1.55), (-3, 3), (-3, 0.244),
          (-3, 3), (-1.396, 1.396), (-1.378, 1.117))


def synthetic_urdf():
    text = ['<robot name="kinematics_regression"><link name="root"/>']

    def joint(name, parent, child, kind, xyz, axis="0 0 1", bounds=None, rpy="0 0 0"):
        text.append(f'<link name="{child}"/><joint name="{name}" type="{kind}">'
                    f'<parent link="{parent}"/><child link="{child}"/>'
                    f'<origin xyz="{xyz}" rpy="{rpy}"/><axis xyz="{axis}"/>')
        if kind != "fixed":
            positions = "" if bounds is None else f' lower="{bounds[0]}" upper="{bounds[1]}"'
            text.append(f'<limit effort="20" velocity="2"{positions}/>')
        text.append('</joint>')

    joint("A_unrelated", "root", "A_branch", "continuous", "0 0 0")
    joint("base_mount", "root", "calibrated_base", "fixed", "0.3 -0.2 0.1", rpy="0.1 -0.2 0.4")
    joint("Lift", "root", "lift_link", "prismatic", "0 0 0", bounds=(0, 0.8))
    joint("torso_flip", "lift_link", "torso", "revolute", "0 0 0.1", "0 1 0", (-1, 1))
    origins = ("0 0 0.2", "0.08 0 0.12", "0.14 0 0.02", "0.16 0 0",
               "0.13 0.04 0.01", "0.09 0 0.01", "0.07 0 0")
    axes = ("0 0 1", "0 1 0", "1 0 0", "0 1 0", "0 0 1", "0 1 0", "1 0 0")
    parent = "torso"
    for i, (name, origin, axis, bounds) in enumerate(zip(NAMES, origins, axes, LIMITS)):
        child = f"arm_link{i + 1}"
        joint(name, parent, child, "revolute", origin, axis, bounds)
        parent = child
    joint("tool_mount", parent, "tool", "fixed", "0.09 0.02 -0.015", rpy="0.1 0.2 -0.3")
    text.append('</robot>')
    return "".join(text)


class ConfigWithoutVendorTests(unittest.TestCase):
    def test_explicit_base_frame_required_before_importing_vendor_stack(self):
        with self.assertRaisesRegex(ValueError, "base_frame"):
            PinocchioArmKinematics("unused", "tool", NAMES)

    def test_duplicate_names_rejected_before_importing_vendor_stack(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            PinocchioArmKinematics("unused", "tool", ["same"] * 7)

    def test_invalid_solver_tuning_rejected_before_importing_vendor_stack(self):
        for key, value in (("position_tolerance_m", float("nan")),
                           ("orientation_tolerance_rad", 0), ("damping", -1),
                           ("integration_step", 1.1), ("max_iterations", 0),
                           ("max_iterations", 1.5), ("max_seed_delta_rad", float("inf"))):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                PinocchioArmKinematics("unused", "tool", NAMES, {"base_frame": "root", key: value})


@unittest.skipUnless(pin is not None, "Numerical tests require robotics Pinocchio and NumPy")
class PinocchioNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.urdf = Path(cls.folder.name) / "seven_joint_test.urdf"
        cls.urdf.write_text(synthetic_urdf(), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def make_kin(self, names=NAMES, frame="tool", **overrides):
        cfg = {"base_frame": "calibrated_base",
               "fixed_joint_values": {"Lift": 0.23, "torso_flip": 0.1},
               "position_tolerance_m": 1e-7, "orientation_tolerance_rad": 1e-7,
               "integration_step": 0.35, "max_iterations": 500, "max_seed_delta_rad": 0.5}
        cfg.update(overrides)
        return PinocchioArmKinematics(self.urdf, frame, names, cfg)

    def assert_pose_close(self, actual, wanted, tolerance=2e-7):
        np.testing.assert_allclose(actual.position_m, wanted.position_m, atol=tolerance, rtol=0)
        self.assertLessEqual(quaternion_angle(actual.quaternion_wxyz, wanted.quaternion_wxyz), tolerance)

    def test_fk_ik_roundtrip_with_nontrivial_tool_and_base_frames(self):
        kin = self.make_kin()
        wanted_q = np.asarray(SEED) + np.array([0.025, 0.01, -0.02, 0.015, 0.02, -0.01, 0.02])
        target = kin.forward(wanted_q)
        solved = kin.solve(target, SEED)
        self.assert_pose_close(kin.forward(solved), target)
        for q, (lower, upper) in zip(solved, LIMITS):
            self.assertLessEqual(lower, q)
            self.assertLessEqual(q, upper)

    def test_small_cartesian_displacements_in_each_base_axis(self):
        kin = self.make_kin()
        current = kin.forward(SEED)
        for axis in range(3):
            with self.subTest(axis=axis):
                p = list(current.position_m)
                p[axis] += 0.001
                target = Pose(tuple(p), current.quaternion_wxyz)
                self.assert_pose_close(kin.forward(kin.solve(target, SEED)), target)

    def test_base_transform_and_wxyz_output_match_independent_root_fk(self):
        kin = self.make_kin()
        root_kin = self.make_kin(base_frame="root")
        base_pose, root_pose = kin.forward(SEED), root_kin.forward(SEED)
        root_tool = pin.SE3(np.array(quaternion_to_matrix(root_pose.quaternion_wxyz)),
                            np.array(root_pose.position_m))
        expected = kin.model.frames[kin.base_frame_id].placement.inverse() * root_tool
        np.testing.assert_allclose(base_pose.position_m, expected.translation, atol=1e-12)
        np.testing.assert_allclose(quaternion_to_matrix(base_pose.quaternion_wxyz), expected.rotation, atol=1e-12)

    def test_error_jacobian_matches_central_differences_in_all_seven_columns(self):
        kin = self.make_kin()
        q = kin._full_q(SEED)
        target_q = kin._full_q(np.asarray(SEED) + np.array([.1, -.04, .03, .04, -.02, .04, .02]))
        pin.forwardKinematics(kin.model, kin.data, target_q)
        pin.updateFramePlacements(kin.model, kin.data)
        desired = kin.data.oMf[kin.frame_id].copy()

        def error(full_q):
            pin.forwardKinematics(kin.model, kin.data, full_q)
            pin.updateFramePlacements(kin.model, kin.data)
            return pin.log6(kin.data.oMf[kin.frame_id].actInv(desired)).vector.copy()

        error(q)
        relative = kin.data.oMf[kin.frame_id].actInv(desired)
        analytic = kin._local_error_jacobian(q, relative)
        finite_difference = np.zeros((6, 7))
        epsilon = 1e-7
        for col, velocity_index in enumerate(kin._v_idx):
            dq = np.zeros(kin.model.nv)
            dq[velocity_index] = epsilon
            finite_difference[:, col] = (error(pin.integrate(kin.model, q, dq))
                                         - error(pin.integrate(kin.model, q, -dq))) / (2 * epsilon)
        np.testing.assert_allclose(analytic, finite_difference, atol=1e-7, rtol=1e-6)

    def test_q_and_v_indices_differ_and_only_selected_arm_values_change(self):
        kin = self.make_kin()
        self.assertNotEqual(kin.model.nq, kin.model.nv)
        self.assertNotEqual(kin._q_idx, kin._v_idx)
        target = kin.forward(np.asarray(SEED) + 0.01)
        solved = kin.solve(target, SEED)
        full = kin._full_q(solved)
        inactive = set(range(kin.model.nq)) - set(kin._q_idx)
        for idx in inactive:
            self.assertEqual(full[idx], kin._base_q[idx])

    def test_joint_mapping_uses_configured_order(self):
        kin = self.make_kin(names=tuple(reversed(NAMES)))
        normal = self.make_kin()
        self.assert_pose_close(kin.forward(tuple(reversed(SEED))), normal.forward(SEED))

    def test_bad_frame_and_joint_names_fail_with_actionable_errors(self):
        for kwargs, pattern in (({"frame": "absent"}, "EE frame"),
                                ({"base_frame": "absent"}, "Base frame"),
                                ({"base_frame": "arm_link1"}, "fixed to the URDF root"),
                                ({"frame": "arm_link6"}, "downstream"),
                                ({"names": NAMES[:-1] + ("absent",)}, "not found")):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, pattern):
                self.make_kin(**kwargs)

    def test_fixed_chain_values_must_be_explicit_finite_and_in_limits(self):
        for fixed, pattern in (({}, "Lift.*torso_flip"),
                               ({"Lift": 0.2}, "torso_flip"),
                               ({"Lift": float("nan"), "torso_flip": 0}, "non-finite"),
                               ({"Lift": 1, "torso_flip": 0}, "outside URDF"),
                               ({"Lift": 0.2, "torso_flip": 0, "arm_j1": 0}, "active arm"),
                               ({"Lift": 0.2, "torso_flip": 0, "absent": 0}, "not found")):
            with self.subTest(fixed=fixed), self.assertRaisesRegex(ValueError, pattern):
                self.make_kin(fixed_joint_values=fixed)

    def test_invalid_seed_cannot_be_returned_as_success(self):
        kin = self.make_kin()
        target = kin.forward(SEED)
        for value in (LIMITS[1][0] - 0.01, LIMITS[1][1] + 0.01, float("nan")):
            seed = list(SEED)
            seed[1] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                kin.solve(target, seed)

    def test_command_limits_intersect_urdf_limits(self):
        limits = [list(pair) for pair in LIMITS]
        limits[1] = [0.3, 0.5]
        kin = self.make_kin(joint_limits_rad=limits)
        self.assertAlmostEqual(kin._lower[1], 0.3)
        self.assertAlmostEqual(kin._upper[1], 0.5)
        seed = list(SEED)
        seed[1] = 0.29
        with self.assertRaisesRegex(ValueError, "outside limits"):
            kin.forward(seed)
        limits[1] = [2, 3]
        with self.assertRaisesRegex(ValueError, "empty intersection"):
            self.make_kin(joint_limits_rad=limits)

    def test_unreachable_pose_fails_with_residuals(self):
        kin = self.make_kin(max_iterations=3)
        current = kin.forward(SEED)
        with self.assertRaisesRegex(IKError, "position_error=.*orientation_error="):
            kin.solve(Pose((20, 20, 20), current.quaternion_wxyz), SEED)


@unittest.skipUnless(pin is not None, "Numerical tests require robotics Pinocchio and NumPy")
class VegaPickupRegressionTests(unittest.TestCase):
    """Operator's failed menu-12 hover; no vendor control or robot connection."""
    seed = (.2265978455543518, -.17115746438503265, -2.197451591491699,
            -.9340024590492249, .9935635924339294, -.8889415860176086, .9849573373794556)
    hover = Pose((.6066043805040553, -.14009201208247296, .5589450232631612),
                 (.8199953126488716, .04858657587239977, .06911024666811323, .5661014093643891))
    steps = dict(max_translation_step_m=.020, max_orientation_step_rad=.08)

    def kin(self, enabled=True):
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root / 'configs/robots/vega.json').read_text())
        settings = dict(cfg['kinematics'], joint_limits_rad=cfg['arm_joint_limits_rad']['right'],
                        near_target_fallback=enabled)
        return PinocchioArmKinematics(root / cfg['urdf_path'], 'tip_r',
                                      settings['right_arm_joint_names'], settings)

    def test_logged_hover_fk_and_descent_failure_are_reproducible(self):
        kin = self.kin(enabled=False)
        np.testing.assert_allclose(kin.forward(self.seed).position_m, self.hover.position_m, atol=1e-12)
        grasp = Pose((*self.hover.position_m[:2], .46191301569882637), self.hover.quaternion_wxyz)
        with self.assertRaisesRegex(IKError, 'waypoint 5/5.*did not converge'):
            preflight_tcp_segmented(kin, self.seed, self.hover, grasp, **self.steps)

    def test_logged_pickup_and_segmented_lift_recover_with_original_limits(self):
        from steadyhand.executor import cartesian_waypoints
        from steadyhand.geometry import pose_distance
        kin = self.kin()
        for z in (.46191301569882637, .4639450232631612):
            with self.subTest(z=z), mock.patch.object(kin, '_near_target_solve', wraps=kin._near_target_solve) as recover:
                grasp = Pose((*self.hover.position_m[:2], z), self.hover.quaternion_wxyz)
                seed = self.seed
                for start, end in ((self.hover, grasp), (grasp, self.hover)):
                    for point in cartesian_waypoints(start, end, **self.steps):
                        solved = kin.solve(point, seed)
                        dp, da = pose_distance(kin.forward(solved), point)
                        self.assertLessEqual(dp, .002)
                        self.assertLessEqual(da, .020)
                        self.assertLessEqual(max(abs(a-b) for a, b in zip(solved, seed)), 1.5)
                        self.assertTrue(np.all(np.asarray(solved) >= kin._lower))
                        self.assertTrue(np.all(np.asarray(solved) <= kin._upper))
                        seed = solved
                self.assertGreater(recover.call_count, 0)

    def test_normal_success_and_distant_failure_do_not_use_recovery(self):
        kin = self.kin()
        with mock.patch.object(kin, '_near_target_solve', side_effect=AssertionError('unexpected recovery')):
            kin.solve(self.hover, self.seed)
            with self.assertRaises(IKError):
                kin.solve(Pose((20, 20, 20), self.hover.quaternion_wxyz), self.seed)

    def test_recovery_cannot_bypass_original_seed_delta(self):
        kin = self.kin()
        kin.config['max_seed_delta_rad'] = .02
        grasp = Pose((*self.hover.position_m[:2], .46191301569882637), self.hover.quaternion_wxyz)
        desired = kin._root_M_base * pin.SE3(np.asarray(quaternion_to_matrix(grasp.quaternion_wxyz)),
                                            np.asarray(grasp.position_m))
        self.assertIsNone(kin._near_target_solve(desired, self.seed, .002, .02, 500))


if __name__ == "__main__":
    unittest.main()
