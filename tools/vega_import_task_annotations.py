"""Normalize reviewed annotation pixels onto the fixed 386 mm board.

The input is the full annotation project, not its mixed-scale task export.
A 180-degree rotation matches the operator's robot-facing view of this dataset.
No robot/calibration access or hardware imports occur here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steadyhand.board_geometry import BOARD_SIZE_M, BOARD_MOTION_MODEL, validate_task_board_geometry, validate_task_coordinate_extent
from steadyhand.wrist_part_profiles import PART_NAMES

ROOT = Path(__file__).resolve().parents[1]


def _legacy_secondary_points():
    """Keep old organizer-only connect/grade values available for audits.

    They are deliberately stored outside ``parts`` so the live pipeline never
    treats them as physical targets.  Older offline tools can still resolve
    them while the reviewed annotation export remains the active pick/place
    source.
    """
    path = ROOT / "configs" / "task_coordinates.organizer_reference.json"
    if not path.is_file():
        return {}
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    result = {}
    for part, values in (source.get("parts") or {}).items():
        for kind in ("connect", "grade"):
            point = values.get(kind)
            if isinstance(point, list) and len(point) == 3:
                result.setdefault(part, {})[kind] = list(point)
    return result


def convert(project, *, rotation_deg=180):
    parts = {part: {} for part in PART_NAMES}
    evidence = {}
    for state_name, kind in (("initial", "pick"), ("final", "place")):
        state = project["states"][state_name]
        width, height = state["rectified_size_px"]
        if width < 2 or height < 2:
            raise ValueError("Missing rectified board size")
        found = set()
        for record in state["annotations"]:
            part = record["part_name"]
            if part not in parts or part in found or record.get("closed") is not True:
                raise ValueError(f"Unknown, duplicate or open {state_name} annotation: {part}")
            u, v = map(float, record["center_board_px"])
            if not math.isfinite(u+v) or not 0 <= u <= width-1 or not 0 <= v <= height-1:
                raise ValueError(f"Out of board annotation: {part}")
            parts[part][kind] = [u/(width-1)*BOARD_SIZE_M, v/(height-1)*BOARD_SIZE_M, 0.0]
            found.add(part)
        if found != set(PART_NAMES):
            raise ValueError(f"Missing {state_name} annotations: {set(PART_NAMES)-found}")
        evidence[state_name] = {"rectified_size_px": [width,height], "original_effective_dimensions_m": state.get("effective_dimensions_m"), "image_sha256": state.get("image_sha256")}
    result = {
        "schema_version": 1, "source_pose_frame": "board_local_annotation",
        "position_units": "m", "quaternion_order": "wxyz",
        "board_width_m": BOARD_SIZE_M, "board_height_m": BOARD_SIZE_M,
        "board_motion_model": BOARD_MOTION_MODEL,
        "source_board_center_xy_m": [BOARD_SIZE_M/2]*2,
        "task_coordinate_rotation_deg": rotation_deg,
        "task_coordinate_mirror_x": False,
        "board_coordinate_convention": {"origin": "rectified image top-left", "x_positive": "image right", "y_positive": "image down", "runtime_x_positive": "robot-view board right", "runtime_y_positive": "robot-view board near edge"},
        "registration_notes": "Per-image rectified pixel fractions normalized to 386 mm. Rotate 180 degrees about board center to place batteries near robot, slim battery on right and rod destination near-left. Annotation Z is ignored; accepted board calibration owns Z.",
        "official_order": list(PART_NAMES), "parts": parts,
        "legacy_secondary_points": _legacy_secondary_points(),
        "physical_layout_expectations": {
            "small_battery": "battery_size5.pick",
            "large_battery": "battery_size1.pick",
            "rod": "rod_16mm.place",
            "near_robot": ["battery_size5.pick", "battery_size1.pick"],
        },
        "annotation_provenance": evidence,
        "excluded_legacy_points": "Organizer connect/grade points are retained only in task_coordinates.organizer_reference.json; they are not validated physical insertion targets.",
    }
    validate_task_board_geometry(result)
    validate_task_coordinate_extent(result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default="tools/outputs/initial.json")
    p.add_argument("--output", default="configs/task_coordinates.json")
    args=p.parse_args(argv)
    source=ROOT/args.project
    result=convert(json.loads(source.read_text()))
    result["annotation_provenance"]["project"]={"path":str(source.relative_to(ROOT)), "sha256":hashlib.sha256(source.read_bytes()).hexdigest()}
    (ROOT/args.output).write_text(json.dumps(result,indent=2)+"\n")
    print("WROTE",ROOT/args.output)

if __name__=="__main__":
    main()
