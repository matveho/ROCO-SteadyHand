import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from steadyhand.execution_offsets import (
    DirectionOffset, ExecutionOffsets, load_offsets, parse_offsets,
)
from steadyhand.models import Pose
from tools import vega_competition_pipeline as pipeline
from tools import vega_wrist_part_calibrate as wrist


QUAT = (1., 0., 0., 0.)


class ExecutionOffsetTests(unittest.TestCase):
    def test_robot_directions_surface_slope_and_depth_signs(self):
        surface = lambda x, y: .5 - .2 * x + .01 * y
        hover = Pose((.4, -.1, surface(.4, -.1) + .1), QUAT)
        for forward, right, down in ((12, 5, 3), (-12, -5, -3)):
            offset = DirectionOffset(forward, right, down)
            corrected = offset.hover(hover, surface)
            x, y, z = corrected.position_m
            self.assertAlmostEqual(x, .4 + forward / 1000)
            self.assertAlmostEqual(y, -.1 - right / 1000)
            self.assertAlmostEqual(z - surface(x, y), .1)
            self.assertAlmostEqual(offset.clearance(.02), .02 - down / 1000)
            self.assertEqual(corrected.quaternion_wxyz, QUAT)

    def test_zero_offsets_are_identity_and_operator_file_is_valid(self):
        pose = Pose((.4, -.1, .6), QUAT)
        surface = mock.Mock(side_effect=AssertionError("unneeded surface read"))
        self.assertIs(DirectionOffset().hover(pose, surface), pose)
        self.assertEqual(DirectionOffset().clearance(.008), .008)
        self.assertIsInstance(load_offsets(), ExecutionOffsets)

    def test_reject_invalid_edits_and_accept_windows_bom(self):
        data = ExecutionOffsets().as_dict()
        for value in (True, "5", None, float("nan"), float("inf"), 101, -101):
            bad = copy.deepcopy(data)
            bad["pickup"]["forward_mm"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_offsets(bad)
        bad = copy.deepcopy(data)
        bad["placement"]["forwards_mm"] = 4
        with self.assertRaises(ValueError):
            parse_offsets(bad)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "offsets.json"
            path.write_text(json.dumps(data), encoding="utf-8-sig")
            self.assertEqual(load_offsets(path), ExecutionOffsets())
            path.write_text(json.dumps(data).replace('"forward_mm": 0.0', '"forward_mm": 0, "forward_mm": 9'))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_offsets(path)

    def test_teaching_never_reads_offsets_even_with_competition_flag_or_bad_snapshot(self):
        with mock.patch.object(wrist, "load_offsets", side_effect=AssertionError("calibration read offsets")):
            for mode in ("calibrate", "drop", "place-cv"):
                args = SimpleNamespace(mode=mode, competition=True, execution_offsets_json="invalid")
                self.assertIsNone(wrist._session_execution_offsets(args))
            # A prevalidated competition snapshot never rereads the file.
            args = SimpleNamespace(mode="test", execution_offsets_json=json.dumps(ExecutionOffsets().as_dict()))
            self.assertEqual(wrist._session_execution_offsets(args), ExecutionOffsets())

    def session(self):
        s = wrist.PartSession.__new__(wrist.PartSession)
        s.part = "battery_size1"
        s.execution_offsets = ExecutionOffsets(DirectionOffset(10, 6, 3), DirectionOffset(-7, -4, -2))
        s.surface = lambda x, y: .5 - .2 * x + .01 * y
        s.robot = SimpleNamespace(
            pose=Pose((.4, -.1, s.surface(.4, -.1) + .1), QUAT),
            _kinematics=SimpleNamespace(solve=lambda pose, seed: seed),
            _read_joint_positions=lambda: [0.] * 7,
            release_gripper=mock.Mock(),
        )
        s.robot.get_tcp_pose = lambda: s.robot.pose
        s.moves = []
        def move(pose, slow=False):
            s.moves.append(pose)
            s.robot.pose = pose
        s.move = move
        s.event = mock.Mock()
        s.remote_checkpoint = mock.Mock()
        s._capture_drop_evidence = mock.Mock()
        s.cfg = {}
        s.holding = False
        s.alignment_fallback_used = False
        return s

    def test_pick_cv_and_no_cv_apply_once_after_alignment_and_return_at_corrected_depth(self):
        for no_cv in (False, True):
            with self.subTest(no_cv=no_cv):
                s = self.session()
                original_profile = {"part": s.part, "grasp_clearance_m": .02, "grasp_verified": True}
                s.profiles = {"parts": {s.part: copy.deepcopy(original_profile)}}
                # begin_part includes CV (or no-CV fallback). Its final TCP is
                # distinct from the original pose to catch corrections applied
                # before, rather than after, the servo.
                aligned = Pose((.43, -.12, s.surface(.43, -.12) + .1), QUAT)
                s.begin_part = lambda *a, **kw: setattr(s.robot, "pose", aligned)
                grabs = []
                def grab(clearance, **kw):
                    grabs.append((s.robot.pose, clearance))
                    s.holding = True
                s.grab = grab
                with mock.patch.object(wrist, "_check_ready"), mock.patch.object(wrist, "save_profile") as save:
                    self.assertEqual(s.test(s.part, "pick", competition=True, no_cv=no_cv), 0)
                save.assert_not_called()
                self.assertEqual(s.profiles["parts"][s.part], original_profile)
                anchor, depth = grabs[0]
                self.assertAlmostEqual(anchor.position_m[0], .44)
                self.assertAlmostEqual(anchor.position_m[1], -.126)
                self.assertAlmostEqual(depth, .017)
                release = s.moves[-2]
                self.assertEqual(release.position_m[:2], anchor.position_m[:2])
                self.assertAlmostEqual(release.position_m[2], s.surface(.44, -.126) + .017)
                self.assertFalse(s.holding)

    def test_placement_correction_after_cv_or_fallback_independent_of_pickup(self):
        for cv_result in ("disabled", "converged", "fallback"):
            with self.subTest(cv_result=cv_result):
                s = self.session()
                s.holding = True
                s.targets = {f"task.{s.part}.place": Pose((.5, .1, .6), QUAT)}
                # Deliberately rotated board axes: last-minute directions must
                # still be robot-relative, not board-relative.
                s.runtime = (None, None, (None, (0., 1.), (-1., 0.), None), Pose((0., 0., 0.), QUAT))
                settings = {"offset_board_xy_m": [.01, .02], "clearance_m": .015, "yaw_deg": 0.}
                original = copy.deepcopy(settings)
                x, y = .48, .11
                if cv_result == "converged":
                    x, y = .49, .13
                def align(_settings):
                    s.robot.pose = Pose((x, y, s.surface(x, y) + .1), QUAT)
                    return {"status": "converged"} if cv_result == "converged" else None
                s._place_visual_align = align
                s.place(settings, use_place_cv=cv_result != "disabled")
                release = s.moves[-2]
                self.assertAlmostEqual(release.position_m[0], x - .007)
                self.assertAlmostEqual(release.position_m[1], y + .004)
                self.assertAlmostEqual(release.position_m[2], s.surface(x - .007, y + .004) + .017)
                retreat = s.moves[-1]
                self.assertAlmostEqual(retreat.position_m[2] - s.surface(*retreat.position_m[:2]), .1)
                self.assertEqual(settings, original)
                s.robot.release_gripper.assert_called_once_with(s.part)

    def test_unreachable_correction_does_not_move_or_release_held_part(self):
        s = self.session()
        s.holding = True
        s.robot._kinematics.solve = mock.Mock(side_effect=RuntimeError("IK did not converge"))
        with self.assertRaisesRegex(RuntimeError, "IK"):
            s._execution_target("placement", s.robot.pose, .01)
        self.assertTrue(s.holding)
        self.assertEqual(s.moves, [])
        s.robot.release_gripper.assert_not_called()

    def test_retry_snapshot_does_not_change_mid_run_next_run_reloads(self):
        args = SimpleNamespace(speed_scale=.38)
        first = ExecutionOffsets(DirectionOffset(3, 2, 1))
        second = ExecutionOffsets(DirectionOffset(-3, -2, -1))
        seen = []
        with tempfile.TemporaryDirectory() as td:
            def run(command):
                seen.append(json.loads(command[command.index("--execution-offsets-json") + 1]))
                output = Path(command[command.index("--output") + 1])
                output.mkdir(parents=True)
                success = len(seen) % 2 == 0
                (output / "run_summary.json").write_text(json.dumps({
                    "status": "pick_complete_returned" if success else "failed",
                    "holding_may_be_true": False,
                }))
                return 0 if success else 2
            with mock.patch.object(pipeline, "ROOT", Path(td)), \
                 mock.patch.object(pipeline, "load_offsets", side_effect=[first, second]) as loader, \
                 mock.patch.object(pipeline, "run_wrist_part_calibration", side_effect=run), \
                 mock.patch.object(pipeline.time, "sleep"):
                pipeline._snapshot_execution_offsets(args)
                self.assertEqual(pipeline._run_competition_action(args, "battery_size1", "pick", retries=1), 0)
                self.assertEqual(seen, [first.as_dict(), first.as_dict()])
                loader.assert_called_once()
                pipeline._snapshot_execution_offsets(args)
                self.assertEqual(args.execution_offsets, second)

    def test_invalid_offsets_stop_check_only_before_any_action(self):
        args = SimpleNamespace(check_only=True)
        with mock.patch.object(pipeline, "load_offsets", side_effect=ValueError("bad offset")), \
             mock.patch.object(pipeline, "_run_competition_action") as run:
            with self.assertRaisesRegex(ValueError, "bad offset"):
                pipeline._configured_competition_run(args)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
