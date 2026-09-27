"""Executor safety/fidelity tests independent of vendor libraries."""

import math
import unittest

from steadyhand.adapters.mock import MockAdapter
from steadyhand.executor import (
    ContactLimitExceeded,
    ExecutionError,
    execute_part,
    move_tcp_segmented,
)
from steadyhand.models import PartGoal, Pose


def skill(*, snap=False):
    return {
        "hover_pick_m": 0.02,
        "hover_place_m": 0.02,
        "retract_m": 0.03,
        "legacy_geometry_verified": True,
        "legacy_ee_offset_m": [0.0, 0.0, 0.0],
        "legacy_ee_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "T_part_tcp": None,
        "gripper_open_fraction": 1,
        "grip_current_a": 0.3,
        "grasp_settle_s": 0.0,
        "release_settle_s": 0.0,
        "max_cartesian_step_m": 0.01,
        "max_orientation_step_rad": 0.1,
        "preinsert_m": 0.01,
        "insertion_reached_tolerance_m": 0.002,
        "insertion_reached_tolerance_rad": 0.02,
        "max_contact_step_m": 0.002 if snap else None,
        "search": {"type": "grid", "n": 2,
                   "extent_xy_m": [0.001, 0.001]} if snap else None,
    }


def goal(mode="open"):
    return PartGoal(
        "battery_size1" if mode == "open" else "bolt_8mm",
        mode,
        Pose((0.01, 0.0, 0.02), (1, 0, 0, 0)),
        Pose((0.03, 0.0, 0.02), (1, 0, 0, 0)),
        None,
    )


class UnknownVerificationRobot(MockAdapter):
    def verify_grasp(self, part_name):
        self.commands.append(("verify_grasp", part_name))
        return None

    def verify_place(self, part_name):
        self.commands.append(("verify_place", part_name))
        return None


class ExecutorSafetyTests(unittest.TestCase):
    def test_segmented_motion_bounds_position_and_orientation(self):
        robot = MockAdapter()
        target = Pose(
            (0.025, 0.0, 0.0),
            (math.cos(0.15), 0.0, 0.0, math.sin(0.15)),
        )
        move_tcp_segmented(
            robot,
            target,
            speed_scale=0.2,
            max_translation_step_m=0.01,
            max_orientation_step_rad=0.1,
        )
        poses = [command[1] for command in robot.commands
                 if command[0] == "move_tcp"]
        # A floating-point boundary may conservatively add one segment.
        self.assertIn(len(poses), (3, 4))
        previous = Pose((0, 0, 0), (1, 0, 0, 0))
        from steadyhand.geometry import pose_distance
        for pose in poses:
            distance, angle = pose_distance(previous, pose)
            self.assertLessEqual(distance, 0.01 + 1e-12)
            self.assertLessEqual(angle, 0.1 + 1e-12)
            previous = pose

    def test_tcp_floor_refuses_target_below_limit(self):
        robot = MockAdapter()
        robot.connect()
        with self.assertRaisesRegex(ExecutionError, "configured floor"):
            move_tcp_segmented(
                robot,
                Pose((0.0, 0.0, -0.001), (1.0, 0.0, 0.0, 0.0)),
                speed_scale=0.2,
                max_translation_step_m=0.01,
                max_orientation_step_rad=0.1,
                min_tcp_z_m=0.0,
            )
        self.assertFalse(
            any(command[0] == "move_tcp" for command in robot.commands),
            "floor violation must be rejected before any TCP command",
        )

    def test_unknown_grasp_stops_before_lift_without_operator_verifier(self):
        robot = UnknownVerificationRobot()
        robot.connect()
        with self.assertRaisesRegex(ExecutionError, "grasp verification"):
            execute_part(robot, goal(), skill())
        names = [command[0] for command in robot.commands]
        self.assertIn("stop", names)
        self.assertNotIn("open_gripper", names[names.index("grip") + 1:])

    def test_false_placement_is_failure_not_completed_result(self):
        class BadPlace(MockAdapter):
            def verify_place(self, part_name):
                return False
        robot = BadPlace()
        robot.connect()
        with self.assertRaisesRegex(ExecutionError, "placement verification"):
            execute_part(
                robot, goal(), skill(), verify=lambda stage, name: True
            )
        self.assertEqual(robot.commands[-1][0], "stop")

    def test_cached_grip_success_does_not_skip_lift_verification(self):
        robot = MockAdapter()
        robot.connect()
        called = []
        with self.assertRaisesRegex(ExecutionError, "lift verification"):
            execute_part(
                robot, goal(), skill(),
                verify=lambda stage, name: called.append(stage) or False,
            )
        self.assertEqual(called, ["lift"])

    def test_baseexception_from_motion_stops(self):
        class AbortRobot(MockAdapter):
            def move_tcp(self, pose, *, speed_scale=0.2):
                raise SystemExit(7)
        robot = AbortRobot()
        robot.connect()
        with self.assertRaises(SystemExit):
            execute_part(robot, goal(), skill())
        self.assertEqual(robot.commands[-1][0], "stop")

    def test_snap_gates_fail_before_opening_gripper(self):
        robot = MockAdapter()
        robot.connect()
        with self.assertRaisesRegex(ExecutionError, "force_delta_limit"):
            execute_part(robot, goal("snap"), skill(snap=True), safety={})
        names = [command[0] for command in robot.commands]
        self.assertNotIn("open_gripper", names)
        self.assertEqual(names[-1], "stop")

    def test_tcp_arrival_cannot_authorize_insertion_release(self):
        robot = MockAdapter()
        robot.connect()
        safety = {"force_delta_limit": 10.0, "force_axes": [0, 1, 2],
                  "wrench_units": "N,Nm", "wrench_frame": "measured_sensor"}
        with self.assertRaisesRegex(ExecutionError, "physical verifier"):
            execute_part(
                robot, goal("snap"), skill(snap=True), safety=safety,
                verify=lambda stage, name: True if stage == "lift" else None,
            )
        opens = [command for command in robot.commands if command[0] == "open_gripper"]
        self.assertEqual(len(opens), 1, "must not release after mere TCP arrival")

    def test_contact_limit_stops_without_blind_retract(self):
        class ContactRobot(MockAdapter):
            reads = 0
            def read_wrench(self):
                self.reads += 1
                return (0, 0, 0, 0, 0, 0) if self.reads < 3 else (2, 0, 0, 0, 0, 0)
        robot = ContactRobot()
        robot.connect()
        safety = {"force_delta_limit": 1.0, "force_axes": [0],
                  "wrench_units": "N,Nm", "wrench_frame": "measured_sensor"}
        with self.assertRaises(ContactLimitExceeded):
            execute_part(robot, goal("snap"), skill(snap=True), safety=safety,
                         verify=lambda stage, name: True)
        self.assertEqual(robot.commands[-1][0], "stop")

    def test_even_grid_attempts_nominal_candidate_first(self):
        robot = MockAdapter()
        robot.connect()
        safety = {"force_delta_limit": 10.0, "force_axes": [0, 1, 2],
                  "wrench_units": "N,Nm", "wrench_frame": "measured_sensor"}
        events = []
        result = execute_part(
            robot, goal("snap"), skill(snap=True), safety=safety,
            verify=lambda stage, name: True,
            event=lambda phase, state, details: events.append((phase, state, details)),
        )
        started = [details for phase, state, details in events
                   if phase == "insertion_search" and state == "candidate_started"]
        self.assertEqual((started[0]["dx_m"], started[0]["dy_m"]), (0.0, 0.0))
        self.assertEqual(result.insertion_candidate, (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
