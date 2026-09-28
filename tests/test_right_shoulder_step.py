import unittest
from steadyhand.models import Pose
from tools.vega_right_shoulder_step import plan_step


class FakeFK:
    def __init__(self, descending=False):
        self.descending = descending
        self.seen = []

    def forward(self, q):
        self.seen.append(tuple(q))
        if q[0] > 3:
            raise ValueError('joint limit')
        return Pose((0.3, -0.26, 1 + (-1 if self.descending else 1)*q[0]), (1,0,0,0))


class ShoulderStepTests(unittest.TestCase):
    def test_only_right_joint_one_changes(self):
        kin = FakeFK()
        seed = [0, .1, .2, .3, .4, .5, .6]
        goal, samples = plan_step(kin, seed, .1, .456)
        self.assertEqual(goal, [.1, .1, .2, .3, .4, .5, .6])
        self.assertEqual(seed[0], 0)
        self.assertEqual(len(samples), 101)
        self.assertTrue(all(list(q[1:]) == seed[1:] for q in kin.seen))

    def test_rejects_lowering_or_low_path(self):
        with self.assertRaisesRegex(ValueError, 'lower'):
            plan_step(FakeFK(True), [0]*7, .1, .456)
        with self.assertRaisesRegex(ValueError, '0.30'):
            plan_step(FakeFK(), [-.3]*7, .1, .456)

    def test_limits_and_invalid_requests(self):
        with self.assertRaisesRegex(ValueError, 'joint limit'):
            plan_step(FakeFK(), [2.95]*7, .1, .456)
        for delta in (-.1, 0, .51, float('nan')):
            with self.assertRaises(ValueError):
                plan_step(FakeFK(), [0]*7, delta, .456)


if __name__ == '__main__':
    unittest.main()
