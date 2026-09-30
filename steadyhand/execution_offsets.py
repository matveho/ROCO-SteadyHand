"""Immutable, execution-only corrections; never part of a taught profile."""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path

from steadyhand.config import read_json
from steadyhand.models import Pose


DEFAULT_PATH = Path(__file__).resolve().parents[1] / "competition_offsets.json"


@dataclass(frozen=True)
class DirectionOffset:
    forward_mm: float = 0.0
    right_mm: float = 0.0
    down_mm: float = 0.0

    def hover(self, pose, surface):
        """Shift in robot coordinates while retaining measured hover clearance."""
        if self.forward_mm == 0 and self.right_mm == 0:
            return pose
        x, y, z = pose.position_m
        nx, ny = x + self.forward_mm / 1000, y - self.right_mm / 1000
        nz = z + surface(nx, ny) - surface(x, y)
        if not all(math.isfinite(v) for v in (nx, ny, nz)):
            raise ValueError("Execution offset produced a non-finite hover")
        return Pose((nx, ny, nz), pose.quaternion_wxyz)

    def clearance(self, taught_clearance):
        return float(taught_clearance) - self.down_mm / 1000


@dataclass(frozen=True)
class ExecutionOffsets:
    pickup: DirectionOffset = field(default_factory=DirectionOffset)
    placement: DirectionOffset = field(default_factory=DirectionOffset)

    def as_dict(self):
        return {"schema_version": 1, **asdict(self)}


def parse_offsets(value):
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("competition_offsets.json: schema_version must be 1")
    if set(value) - {"schema_version", "_help", "pickup", "placement"}:
        raise ValueError("competition_offsets.json: unknown field (check spelling)")
    groups = {}
    for name in ("pickup", "placement"):
        group = value.get(name)
        fields = {"forward_mm", "right_mm", "down_mm"}
        if not isinstance(group, dict) or set(group) != fields:
            raise ValueError(f"competition_offsets.json: {name} needs forward_mm, right_mm, down_mm")
        for key, number in group.items():
            if type(number) not in (int, float) or not math.isfinite(number) or not -100 <= number <= 100:
                raise ValueError(f"competition_offsets.json: {name}.{key} must be a finite number from -100 to 100 mm")
        groups[name] = DirectionOffset(**group)
    return ExecutionOffsets(**groups)


def load_offsets(path=DEFAULT_PATH):
    return parse_offsets(read_json(path))


def describe_offsets(offsets):
    for name in ("pickup", "placement"):
        group = getattr(offsets, name)
        print(f"EXECUTION OFFSETS {name}: forward={group.forward_mm:+g} mm, "
              f"right={group.right_mm:+g} mm, down={group.down_mm:+g} mm "
              "(robot-relative; calibration unchanged)", flush=True)
