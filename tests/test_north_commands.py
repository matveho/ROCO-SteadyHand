import math
import unittest

from steadyhand.adapters.north_commands import joint_step


class NorthCommandTests(unittest.TestCase):
    def setUp(self):
        self.reference = {"left_arm": [0.1] * 7, "right_arm": [-0.2] * 7,
                          "motor": [0.3] * 7}
        self.limits = [(-1, 1)] * 7

    def test_identity_retains_both_arms_and_body_without_hands_or_chassis(self):
        result = joint_step(self.reference)
        self.assertEqual(len(result), 3)
        for name, values in self.reference.items():
            self.assertEqual(result[f"/action/{name}/joint_angle"], values)
        result['/action/motor/joint_angle'][0] = 0
        self.assertEqual(self.reference['motor'][0], .3)

    def test_single_arm_target_preserves_other_arm_and_body(self):
        target = list(self.reference['right_arm']); target[6] += .005
        result = joint_step(self.reference, arm='right_arm', target=target,
                            limits=self.limits, max_delta_rad=.006)
        self.assertEqual(result['/action/right_arm/joint_angle'], target)
        self.assertEqual(result['/action/left_arm/joint_angle'], self.reference['left_arm'])
        self.assertEqual(result['/action/motor/joint_angle'], self.reference['motor'])

    def test_bad_shape_nonfinite_missing_body_rejected(self):
        for name in self.reference:
            for values in ([0] * 6, [0] * 8, [math.nan] * 7, [math.inf] * 7):
                with self.subTest(name=name, values=values), self.assertRaises(ValueError):
                    joint_step({**self.reference, name: values})
        with self.assertRaises(KeyError):
            joint_step({'left_arm': [0] * 7, 'right_arm': [0] * 7})

    def test_step_bounds_and_limits_fail_closed(self):
        defaults = dict(arm='left_arm', target=[.1] * 7,
                        limits=self.limits, max_delta_rad=.006)
        for invalid in ({'target': [.11] * 7}, {'target': [2] * 7},
                        {'limits': [(0, 0)] * 7}, {'limits': [(-1, math.inf)] * 7},
                        {'limits': [(0, 1)] * 6}, {'max_delta_rad': math.nan},
                        {'max_delta_rad': 0}, {'arm': 'motor'}, {'limits': None}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                joint_step(self.reference, **{**defaults, **invalid})
        with self.assertRaises(ValueError):
            joint_step(self.reference, target=[.1] * 7)


if __name__ == '__main__':
    unittest.main()
