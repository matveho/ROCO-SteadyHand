"""In-memory RobotAdapter for testing shared control flow without hardware."""

from ..models import Pose
from .base import AdapterCapabilities, RobotAdapter, RobotObservation


class MockAdapter(RobotAdapter):
    robot_id = "mock"

    def __init__(self, config=None):
        super().__init__(config or {})
        self.connected = False
        self.commands = []
        self.joints = ()
        self.tcp_pose = Pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        self.wrench = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def capabilities(self):
        return AdapterCapabilities(
            joint_motion=True,
            tcp_motion=True,
            cameras=True,
            force_torque=True,
            tactile=True,
        )

    def connect(self):
        self.connected = True
        self.commands.append(("connect",))

    def close(self):
        self.connected = False
        self.commands.append(("close",))

    def stop(self):
        self.commands.append(("stop",))

    def observe(self):
        self.commands.append(("observe",))
        return RobotObservation(joint_positions=tuple(self.joints))

    def move_joints(self, joint_positions, *, speed_scale=0.2):
        self.joints = tuple(float(v) for v in joint_positions)
        self.commands.append(("move_joints", self.joints, float(speed_scale)))

    def move_tcp(self, pose, *, speed_scale=0.2):
        self.tcp_pose = pose
        self.commands.append(("move_tcp", pose, float(speed_scale)))

    def get_tcp_pose(self):
        return self.tcp_pose

    def open_gripper(self, part_name=None):
        self.commands.append(("open_gripper", part_name))

    def close_gripper(self, part_name=None):
        self.commands.append(("close_gripper", part_name))

    def grip(self, part_name=None, *, current_a=None):
        self.commands.append(("grip", part_name, current_a))

    def read_wrench(self):
        return self.wrench

    def verify_grasp(self, part_name):
        self.commands.append(("verify_grasp", part_name))
        return True

    def verify_place(self, part_name):
        self.commands.append(("verify_place", part_name))
        return True
