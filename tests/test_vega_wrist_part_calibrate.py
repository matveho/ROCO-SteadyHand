import unittest

from steadyhand.models import Pose
from tools.vega_wrist_part_calibrate import PartSession


class _FakeRobot:
    def __init__(self):
        self.pose = Pose((0.10, 0.20, 0.60), (1.0, 0.0, 0.0, 0.0))
        self.opened = []

    def get_tcp_pose(self):
        return self.pose

    def open_gripper(self, part):
        self.opened.append(part)


class WristPartReturnTests(unittest.TestCase):
    def test_return_part_releases_and_clears_holding_state(self):
        session = PartSession.__new__(PartSession)
        session.holding = True
        session.part = "battery_size1"
        session.robot = _FakeRobot()
        session.surface = lambda x, y: 0.50

        moves = []
        events = []
        session.move = lambda pose, slow=False: moves.append((pose, slow))
        session.event = lambda kind, fields: events.append((kind, fields))

        session.return_part(0.02)

        self.assertFalse(session.holding)
        self.assertEqual(session.robot.opened, ["battery_size1"])
        self.assertEqual(len(moves), 2)
        self.assertEqual(moves[0][0].position_m, (0.10, 0.20, 0.52))
        self.assertTrue(moves[0][1])
        self.assertEqual(moves[1][0].position_m, (0.10, 0.20, 0.60))
        self.assertTrue(moves[1][1])
        self.assertEqual(events[0][0], "place_release")
        self.assertEqual(
            events[0][1]["settings"],
            {"return_to_source": True, "clearance_m": 0.02},
        )


if __name__ == "__main__":
    unittest.main()
