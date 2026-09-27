"""Robot-specific execution parameters layered over task semantics."""

import json
from pathlib import Path

from .config import WORKSPACE


def load_vega_skills(path=None):
    path = Path(path) if path else WORKSPACE / "configs" / "skills" / "vega.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("robot_id") != "vega":
        raise ValueError("Invalid Vega skill configuration")
    if not isinstance(value.get("defaults"), dict):
        raise ValueError("Vega skills require defaults")
    if not isinstance(value.get("parts"), dict):
        raise ValueError("Vega skills require parts mapping")
    return value


def skill_for_part(config, part_name):
    merged = dict(config["defaults"])
    merged.update(config["parts"].get(part_name, {}))
    return merged
