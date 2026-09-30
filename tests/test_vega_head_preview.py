import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import cv2

from steadyhand.models import Pose
from steadyhand.wrist_part_profiles import PART_NAMES
from tools.vega_head_fallback import match_expected_parts
from tools import vega_head_target_preview as preview


class HeadPreviewTests(unittest.TestCase):
    def setUp(self):
        self.ready = Pose((.5, 0., .7), (1., 0., 0., 0.))
        self.data = {'source_board_center_xy_m': [.193, .193],
                     'task_coordinate_rotation_deg': 0., 'parts': {
                         p: {'pick': [.02, .02, 0.]} for p in PART_NAMES}}
        self.data['parts']['battery_size1']['pick'] = [.193, .193, 0.]
        self.runtime = ({'robot': {'motion': {}}}, self.data,
            ((.5, 0.), (0., -1.), (-1., 0.), {'coefficients': (0., 0., .5), 'anchors': []}), self.ready)
        self.expected = Pose((.5, 0., .6), self.ready.quaternion_wxyz)
        self.targets = {f'task.{p}.pick': self.expected for p in PART_NAMES}
        self.scene = {'board': {'center_base_m_coarse': [3., 4., .5],
            'corners_px': {'tl': [20, 20], 'tr': [220, 20], 'br': [220, 220], 'bl': [20, 220]}},
            'parts': [{'index': 1, 'center_image_px': [130, 120],
                'center_board_m': [.0193, 0., 0.], 'center_base_m_coarse': [3.3, 4.2, .5],
                'quad_image_px': [[123, 113], [137, 113], [137, 127], [123, 127]]}]}

    def test_board_displacement_uses_calibrated_axes_not_coarse_extrinsic(self):
        obs = match_expected_parts(self.scene, self.runtime, self.targets, self.data)['battery_size1']
        self.assertEqual(obs['selection'], 'head_detection')
        self.assertEqual(obs['association_frame'], 'board')
        self.assertEqual(obs['detection_index'], 1)  # image label, not list index 0
        np.testing.assert_allclose(obs['selected_xy_m'], [.5, -.0193])

    def test_manual_pixel_and_detection_produce_same_target(self):
        obs = preview.head_pixel_target(self.scene, self.runtime, self.data,
                                       self.expected, 'battery_size1', [130, 120])
        np.testing.assert_allclose(obs['selected_xy_m'], [.5, -.0193], atol=1e-7)
        target = preview.preview_target(self.runtime, 'battery_size1', obs, .04)
        self.assertAlmostEqual(target.position_m[2], .54)

    def test_manual_pixel_respects_reviewed_180_degree_task_mapping(self):
        data = copy.deepcopy(self.data)
        data['task_coordinate_rotation_deg'] = 180.
        data['parts']['battery_size1']['pick'] = [.193 - .0193, .193, 0.]
        obs = preview.head_pixel_target(self.scene, self.runtime, data,
                                       self.expected, 'battery_size1', [130, 120])
        np.testing.assert_allclose(obs['selected_xy_m'], self.expected.position_m[:2], atol=1e-7)

    def test_ambiguous_ties_do_not_crash_or_select_arbitrarily(self):
        scene = copy.deepcopy(self.scene)
        other = copy.deepcopy(scene['parts'][0])
        other['index'] = 2
        other['center_board_m'] = [-.0193, 0., 0.]
        scene['parts'].append(other)
        obs = match_expected_parts(scene, self.runtime, self.targets, self.data)['battery_size1']
        self.assertEqual(obs['rejection_reason'], 'ambiguous_candidates')
        with self.assertRaises(ValueError):
            preview.preview_target(self.runtime, 'battery_size1', obs, .04)

    def test_bad_pixels_and_saved_coordinate_fallback_do_not_authorize_preview(self):
        for uv in ([0, 0], [210, 210], [float('nan'), 120]):
            with self.assertRaises(ValueError):
                preview.head_pixel_target(self.scene, self.runtime, self.data,
                                         self.expected, 'battery_size1', uv)
        with self.assertRaises(ValueError):
            preview.preview_target(self.runtime, 'battery_size1',
                {'selection': 'expected_coordinate', 'selected_xy_m': [.5, 0.]}, .04)

    def run_preview(self, *, review=False, abort=False, servo_failure=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'head.png'
            cv2.imwrite(str(raw), np.full((240, 240, 3), 220, np.uint8))
            scene = copy.deepcopy(self.scene)
            scene['head_image_path'] = str(raw)
            robot = SimpleNamespace(pose=self.ready, connect=mock.Mock(), close=mock.Mock(), moves=[])
            robot.get_tcp_pose = lambda: robot.pose
            robot._read_joint_positions = lambda: [0.] * 7
            robot._kinematics = SimpleNamespace(solve=lambda pose, seed: seed)
            def move(p, speed_scale):
                robot.pose = p
                robot.moves.append(p)
            robot.move_tcp = move
            robot.move_joints = mock.Mock()
            def capture_factory(cameras, output, **kwargs):
                class Capture:
                    index = 0
                    def __call__(self):
                        path = output / f'{self.index:03d}_wrist_a.png'
                        image = np.full((240, 240, 3), 220, np.uint8)
                        cv2.imwrite(str(path), image)
                        self.index += 1
                        return image
                return Capture()
            runtime = copy.deepcopy(self.runtime)
            updated = runtime if review else copy.deepcopy(runtime)
            with mock.patch.multiple(preview, ROOT=root,
                _load_runtime=mock.Mock(return_value=runtime),
                _runtime_from_board_scene=mock.Mock(return_value=updated),
                _capture_downward_head_frame=mock.Mock(return_value=scene),
                _task_targets=mock.Mock(return_value=self.targets),
                VegaAdapter=mock.Mock(return_value=robot),
                VegaWristCameras=mock.Mock(),
                WristAOnlyCapture=capture_factory,
                configured_right_preset=mock.Mock(return_value=([0.] * 7, self.ready)),
                load_vega_skills=mock.Mock(return_value={'safety': {'min_tcp_z_m': .45}})), \
                mock.patch.object(preview, 'run_xy_servo', side_effect=preview.ServoWaypointError('TCP missed servo waypoint by >8 mm')) as servo, \
                mock.patch('builtins.input', side_effect=['abort' if abort else '1'] if review else
                           AssertionError('High preview must not ask for confirmation')) as prompt:
                code = preview.main(['--part', 'battery_size1', '--remote-safe',
                                     '--output', str(root / 'out')] +
                                    (['--center', '--feature', '130', '120', '--goal-pixel', '140', '120'] if servo_failure else []))
            report = json.loads((root / 'out' / 'preview.json').read_text())
            self.assertTrue((root / 'out' / 'HEAD_TARGET_REVIEW.png').is_file())
            if abort:
                self.assertEqual(code, 1)
                self.assertEqual(robot.moves, [])
                robot.move_joints.assert_not_called()
            elif servo_failure:
                self.assertEqual(code, 2)
                self.assertEqual(report['status'], 'head_target_reached_wrist_not_verified')
                self.assertTrue(report['head_target_reached'])
                self.assertAlmostEqual(robot.pose.position_m[2], .60)
                servo.assert_called_once()
            else:
                self.assertEqual(code, 0)
                self.assertEqual(report['status'], 'completed_no_gripper_motion')
                self.assertAlmostEqual(robot.pose.position_m[2], .60)
            self.assertEqual(prompt.call_count, int(review))
            return report

    def test_unique_detection_preview_runs_without_motion_prompts(self):
        report = self.run_preview()
        self.assertEqual(report['head_observation']['selection'], 'head_detection')

    def test_rejected_registration_requires_target_review(self):
        report = self.run_preview(review=True)
        self.assertEqual(report['head_observation']['selection'], 'operator_head_pixel')

    def test_review_abort_never_approaches_target(self):
        self.run_preview(review=True, abort=True)

    def test_servo_miss_preserves_head_success_without_claiming_center_success(self):
        self.run_preview(servo_failure=True)


if __name__ == '__main__':
    unittest.main()
