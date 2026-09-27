"""No hardware dependencies: frame conventions and fail-closed motion inputs."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from steadyhand.config import load_bundle
from steadyhand.executor import object_pose_to_tcp
from steadyhand.geometry import (
    compose, interpolate_pose, invert_rigid, matrix_to_pose,
    matrix_to_quaternion, pose_distance, pose_to_matrix, quaternion_angle,
    quaternion_slerp, quaternion_to_matrix, transform_point,
)
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills, skill_for_part, validate_skill
from steadyhand.targets import load_runtime_targets, require_goal


IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


class TargetsGeometryTests(unittest.TestCase):
    def setUp(self):
        self.tasks = load_bundle("vega")["tasks"]
        self.pose = {"position_m": [0.1, -0.2, 0.3],
                     "quaternion_wxyz": [1, 0, 0, 0]}
        self.targets = {
            "schema_version": 1, "robot_id": "vega",
            "pose_frame": "robot_base", "base_frame": "verified_base",
            "parts": {"battery_size1": {"pick_pose": self.pose,
                                          "place_pose": self.pose}},
        }

    def load(self, value, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "targets.json"
            path.write_text(json.dumps(value), encoding="utf-8-sig")
            return load_runtime_targets(path, self.tasks, **kwargs)

    def skill(self):
        skill = skill_for_part(load_vega_skills(), "battery_size1")
        skill.update(T_part_tcp=copy.deepcopy(IDENTITY), grip_current_a=0.1)
        return skill

    def test_targets_use_task_release_semantics_and_explicit_base(self):
        _, goals = self.load(self.targets, expected_base_frame="verified_base")
        goal = require_goal(goals, "battery_size1")
        self.assertEqual(goal.release_mode, "open")
        self.assertEqual(goal.pick_pose.position_m, (0.1, -0.2, 0.3))
        self.assertEqual(goal.pick_pose.quaternion_wxyz, (1, 0, 0, 0))

    def test_wrong_or_missing_target_base_rejected(self):
        for base in (None, "camera", "robot_base"):
            with self.subTest(base=base):
                self.targets["base_frame"] = base
                with self.assertRaisesRegex(ValueError, "base_frame"):
                    self.load(self.targets, expected_base_frame="verified_base")

    def test_missing_selected_pose_fails_explicitly(self):
        self.targets["parts"]["battery_size1"]["place_pose"] = None
        _, goals = self.load(self.targets)
        with self.assertRaisesRegex(ValueError, "place_pose is missing"):
            require_goal(goals, "battery_size1")
        with self.assertRaisesRegex(ValueError, "No runtime pose"):
            require_goal(goals, "pin")

    def test_malformed_target_records_and_unknown_names_rejected(self):
        for parts in ({"battery_size1": []}, {"battery_size_1": {}},
                      {"battery_size1": {"pick_poses": self.pose}}):
            with self.subTest(parts=parts):
                self.targets["parts"] = parts
                with self.assertRaises(ValueError):
                    self.load(self.targets)
        with self.assertRaisesRegex(ValueError, "JSON object"):
            self.load([])

    def test_wrong_explicit_units_and_quaternion_order_rejected(self):
        for key, value in (("position_units", "mm"), ("quaternion_order", "xyzw"),
                           ("pose_frame", "camera")):
            target = copy.deepcopy(self.targets)
            target[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.load(target)

    def test_duplicate_json_target_keys_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "targets.json"
            path.write_text('{"schema_version": 1, "schema_version": 2}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_runtime_targets(path, self.tasks)

    def test_direct_and_json_poses_reject_nan_wrong_shapes_and_bad_quaternions(self):
        for position, quat in (([0, 0], [1, 0, 0, 0]),
                               ([0, 0, float("nan")], [1, 0, 0, 0]),
                               ([0, 0, 0], [0, 0, 0, 0]),
                               ([0, 0, 0], [1, float("inf"), 0, 0]),
                               ([0, 0, 0], [1, 1, 0, 0]),
                               ([0, False, 0], [1, 0, 0, 0])):
            with self.subTest(position=position, quat=quat):
                with self.assertRaises(ValueError):
                    Pose(position, quat)
                self.targets["parts"]["battery_size1"]["pick_pose"] = {
                    "position_m": position, "quaternion_wxyz": quat}
                with self.assertRaises(ValueError):
                    self.load(self.targets)

    def test_wxyz_rotation_and_transform_order(self):
        # Base->part is a 90-degree Z rotation. A TCP offset along part X
        # must become base Y, and must not be added in base X.
        part = Pose((1, 2, 3), (math.sqrt(0.5), 0, 0, math.sqrt(0.5)))
        relative = copy.deepcopy(IDENTITY)
        relative[0][3] = 0.1
        tcp = object_pose_to_tcp(part, {"T_part_tcp": relative})
        for a, b in zip(tcp.position_m, (1, 2.1, 3)):
            self.assertAlmostEqual(a, b)
        self.assertAlmostEqual(quaternion_angle(part.quaternion_wxyz,
                                                tcp.quaternion_wxyz), 0)
        self.assertAlmostEqual(transform_point(pose_to_matrix(part), (1, 0, 0))[1], 3)

    def test_legacy_offset_preserves_submitted_world_axis_semantics(self):
        part = Pose((1, 2, 3), (math.sqrt(0.5), 0, 0, math.sqrt(0.5)))
        tcp = object_pose_to_tcp(part, {
            "legacy_ee_offset_m": [0.1, 0, 0],
            "legacy_ee_orientation_wxyz": [0, 1, 0, 0],
        })
        self.assertEqual(tcp.position_m, (1.1, 2, 3))
        self.assertEqual(tcp.quaternion_wxyz, (0, 1, 0, 0))

    def test_rigid_transform_and_quaternion_round_trips_near_pi(self):
        for q in ((0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
                  (0.5, -0.5, 0.5, -0.5)):
            with self.subTest(q=q):
                pose = Pose((0.1, -0.2, 0.3), q)
                t = pose_to_matrix(pose)
                reconstructed = matrix_to_pose(t)
                dp, da = pose_distance(pose, reconstructed)
                self.assertLess(dp, 1e-12)
                self.assertLess(da, 1e-7)
                identity = compose(t, invert_rigid(t))
                for i in range(4):
                    for j in range(4):
                        self.assertAlmostEqual(identity[i][j], IDENTITY[i][j])

    def test_slerp_antipodes_and_half_turn(self):
        self.assertEqual(quaternion_slerp((1, 0, 0, 0), (-1, 0, 0, 0), 0.5),
                         (1, 0, 0, 0))
        half = quaternion_slerp((1, 0, 0, 0), (0, 0, 0, 1), 0.5)
        self.assertAlmostEqual(quaternion_angle((1, 0, 0, 0), half), math.pi/2)
        pose = interpolate_pose(Pose((0, 0, 0), (1, 0, 0, 0)),
                                Pose((0.2, 0.4, 0.6), (0, 0, 0, 1)), 0.5)
        self.assertEqual(pose.position_m, (0.1, 0.2, 0.3))
        self.assertAlmostEqual(quaternion_angle(half, pose.quaternion_wxyz), 0)

    def test_bad_quaternions_and_extrapolation_do_not_hide_as_zero_angle(self):
        for q in ((0, 0, 0, 0), (float("nan"), 0, 0, 0),
                  (float("inf"), 0, 0, 0), (1, 0, 0)):
            with self.subTest(q=q):
                for operation in (quaternion_to_matrix,
                                  lambda value: quaternion_angle(value, (1, 0, 0, 0)),
                                  lambda value: quaternion_slerp(value, (1, 0, 0, 0), 0.5)):
                    with self.assertRaises(ValueError):
                        operation(q)
        for fraction in (-0.1, 1.1, float("nan")):
            with self.assertRaisesRegex(ValueError, "fraction"):
                quaternion_slerp((1, 0, 0, 0), (1, 0, 0, 0), fraction)

    def test_nonrigid_transforms_cannot_turn_into_plausible_tcp_poses(self):
        for bad in (-1, 2, float("nan")):
            transform = copy.deepcopy(IDENTITY)
            transform[0][0] = bad
            with self.subTest(bad=bad):
                for operation in (matrix_to_pose, invert_rigid):
                    with self.assertRaises(ValueError):
                        operation(transform)
                with self.assertRaises(ValueError):
                    matrix_to_quaternion([row[:3] for row in transform[:3]])

    def test_shipped_simulation_geometry_is_not_motion_calibration(self):
        skill = self.skill()
        skill["T_part_tcp"] = None
        skill["legacy_geometry_verified"] = False
        validate_skill(skill, require_calibrated=False)
        with self.assertRaisesRegex(ValueError, "legacy_geometry_verified"):
            validate_skill(skill)
        skill["legacy_geometry_verified"] = True
        validate_skill(skill)

    def test_measured_geometry_and_positive_current_are_required(self):
        skill = self.skill()
        validate_skill(skill)
        for current in (None, 0, -0.1, float("nan"), "0.1"):
            skill["grip_current_a"] = current
            with self.subTest(current=current), self.assertRaisesRegex(ValueError, "grip_current_a"):
                validate_skill(skill)

    def test_invalid_steps_search_and_transform_fail_before_hardware(self):
        for key in ("hover_pick_m", "retract_m", "max_cartesian_step_m",
                    "max_orientation_step_rad"):
            skill = self.skill()
            skill[key] = 0
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_skill(skill)
        for search in ({"type": "grid", "n": 2.5, "extent_xy_m": [1, 1]},
                       {"type": "grid", "n": 3, "extent_xy_m": [0, -0.1]},
                       {"type": "grid", "n": 3, "extent_xy_m": [0, float("nan")]},
                       {"type": "spiral"}):
            skill = self.skill()
            skill["search"] = search
            with self.subTest(search=search), self.assertRaisesRegex(ValueError, "search"):
                validate_skill(skill)
        skill = self.skill()
        skill["T_part_tcp"][0][0] = -1
        with self.assertRaisesRegex(ValueError, "right-handed"):
            validate_skill(skill)


if __name__ == "__main__":
    unittest.main()
