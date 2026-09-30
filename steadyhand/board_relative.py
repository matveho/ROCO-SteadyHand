"""Board-referenced taught targets and bounded differential registration.

No robot or camera calls live here. The measured board basis may be left handed;
only the *change* between two bases is required to be a proper rotation.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math

import numpy as np

from steadyhand.board_geometry import BOARD_SIZE_M
from steadyhand.models import Pose


CORNER_KEYS = ("tl", "tr", "br", "bl")


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be finite {shape}") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite {shape}")
    return result


def frame_arrays(reference):
    center = _array(reference.get("center_base_xy_m"), (2,), "board center")
    axes = np.column_stack((
        _array(reference.get("board_x_unit_base_xy"), (2,), "board X"),
        _array(reference.get("board_y_unit_base_xy"), (2,), "board Y"),
    ))
    if not np.allclose(axes.T @ axes, np.eye(2), atol=1e-6):
        raise ValueError("board basis must be orthonormal (no shear)")
    return center, axes


def snapshot(frame):
    center, ux, uy, plane = frame
    value = {
        "center_base_xy_m": list(center),
        "board_x_unit_base_xy": list(ux),
        "board_y_unit_base_xy": list(uy),
        "board_size_m": BOARD_SIZE_M,
        "calibration_sha256": plane.get("calibration_sha256"),
        "registration": deepcopy(plane.get("registration", {"status": "reference"})),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    frame_arrays(value)
    return value


def local_xy(reference, xy):
    center, axes = frame_arrays(reference)
    return (axes.T @ (_array(xy, (2,), "target XY") - center)).tolist()


def base_xy(reference, xy):
    center, axes = frame_arrays(reference)
    return (center + axes @ _array(xy, (2,), "board target XY")).tolist()


def rotation_between(reference, current):
    _, old = frame_arrays(reference)
    _, new = frame_arrays(current)
    rotation = new @ old.T
    if np.linalg.det(rotation) < 0:
        raise ValueError("board handedness changed; refusing reflected target")
    if abs(math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))) > 30:
        raise ValueError("board axes changed by more than 30 degrees; refusing an ambiguous target")
    return rotation


def rotate_quaternion(quaternion, angle_rad):
    w, x, y, z = _array(quaternion, (4,), "TCP quaternion")
    norm = np.linalg.norm((w, x, y, z))
    if abs(norm - 1.0) > .01:
        raise ValueError("TCP quaternion is not normalized")
    c, s = math.cos(angle_rad / 2), math.sin(angle_rad / 2)
    return tuple(float(v / norm) for v in (c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w))


def record_target(reference, pose, *, source):
    return {
        "board_xy_m": local_xy(reference, pose.position_m[:2]),
        "tcp_pose": {"position_m": list(pose.position_m),
                     "quaternion_wxyz": list(pose.quaternion_wxyz)},
        "source": source,
    }


def validate_record(record):
    reference = record.get("board_reference")
    frame_arrays(reference)
    if record.get("schema_version") != 1:
        raise ValueError("unsupported board-relative target version")
    for key in ("approach", "grasp", "release"):
        if key not in record:
            continue
        target = record[key]
        xy = _array(target.get("board_xy_m"), (2,), "board target")
        pose = target["tcp_pose"]
        xyz = pose.get("position_m")
        if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
            raise ValueError("recorded TCP must have three coordinates")
        _array(xyz[:2], (2,), "recorded TCP XY")
        if not (xyz[2] is None and target.get("z_not_recorded") is True):
            _array(xyz, (3,), "recorded TCP")
        rotate_quaternion(pose.get("quaternion_wxyz"), 0)
        if np.linalg.norm(np.asarray(base_xy(reference, xy)) - xyz[:2]) > 1e-6:
            raise ValueError("board target disagrees with its recorded TCP/reference")
    return record


def resolve_record(record, current_reference, *, kind):
    validate_record(record)
    target = record[kind]
    rotation = rotation_between(record["board_reference"], current_reference)
    angle = math.atan2(rotation[1, 0], rotation[0, 0])
    return (base_xy(current_reference, target["board_xy_m"]),
            rotate_quaternion(target["tcp_pose"]["quaternion_wxyz"], angle))


def make_record(reference, **targets):
    return {"schema_version": 1, "board_reference": deepcopy(reference), **targets}


def reference_from_calibration(calibration):
    reference = snapshot((calibration["center_base_xy_m"],
                     calibration["board_x_unit_base_xy"],
                     calibration["board_y_unit_base_xy"],
                     {"calibration_sha256": calibration["sha256"],
                      "registration": {"status": "reference", "source": calibration["path"]}}))
    reference["captured_at_utc"] = calibration["generated_at_utc"]
    return reference


def _corners(board):
    points = _array([board["corners_base_m_coarse"][k] for k in CORNER_KEYS],
                    (4, 3), "projected board corners")
    if np.ptp(points[:, 2]) > .002:
        raise ValueError("camera corners are not on the fixed table plane")
    return points


def register(reference, reference_board, board):
    """Estimate small rigid board motion from head-pose-corrected corners.

    Reference edge lengths set the metric scale to the known 386 mm span.
    Corner order is never permuted and a reflected/large rotation is rejected.
    Absolute extrinsic bias cancels; only the relative displacement is applied
    to the physically taught frame. This cannot correct head encoder bias.
    """
    ref3, live3 = _corners(reference_board), _corners(board)
    if abs(float(ref3[:, 2].mean() - live3[:, 2].mean())) > .002:
        raise ValueError("camera projection plane changed; relative registration unavailable")
    ref, live = ref3[:, :2], live3[:, :2]
    # Differential changes only: don't replace operator axes with camera axes.
    rc, lc = ref.mean(axis=0), live.mean(axis=0)
    r, l = ref - rc, live - lc
    u, _, vt = np.linalg.svd(r.T @ l)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        raise ValueError("board corner handedness reversed")
    angle = math.atan2(rotation[1, 0], rotation[0, 0])
    widths = [math.dist(ref[0], ref[1]), math.dist(ref[3], ref[2])]
    heights = [math.dist(ref[0], ref[3]), math.dist(ref[1], ref[2])]
    if min(widths + heights) < .20 or max(widths + heights) > .65:
        raise ValueError("reference camera board span is implausible")
    scale = BOARD_SIZE_M / float(np.mean(widths + heights))
    ref_edges = np.linalg.norm(np.roll(ref, -1, axis=0) - ref, axis=1)
    live_edges = np.linalg.norm(np.roll(live, -1, axis=0) - live, axis=1)
    edge_scale = live_edges / ref_edges
    residual = float(np.sqrt(np.mean(np.sum((l - r @ rotation.T)**2, axis=1))) * scale)
    delta = (lc - rc) * scale
    if np.any(np.abs(edge_scale - 1) > .15) or residual > .015:
        raise ValueError(f"board corner geometry changed (fit residual {residual*1000:.1f} mm)")
    if abs(math.degrees(angle)) > 12:
        raise ValueError("board rotation exceeds 12 degrees; corner identity cannot be trusted")
    if np.linalg.norm(delta) > .10:
        raise ValueError("board shift exceeds 100 mm; inspect the detected outline")
    center, axes = frame_arrays(reference)
    new_axes = rotation @ axes
    result = deepcopy(reference)
    result.update(center_base_xy_m=(center + delta).tolist(),
                  board_x_unit_base_xy=new_axes[:, 0].tolist(),
                  board_y_unit_base_xy=new_axes[:, 1].tolist())
    result["registration"] = {
        "status": "fresh", "source": "differential_four_corner_386mm",
        "translation_m": delta.tolist(), "rotation_deg": math.degrees(angle),
        "fit_residual_m": residual, "edge_scale": edge_scale.tolist(),
        "metric_scale": scale, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    result["captured_at_utc"] = result["registration"]["captured_at_utc"]
    return result


def legacy_pickup_record(profile, reference, ready_quaternion):
    """Explicit compatibility assumption when no teaching-time scene exists."""
    xy = profile["coarse_xy_m"]
    pose = Pose((xy[0], xy[1], 0.), rotate_quaternion(ready_quaternion, math.radians(profile["yaw_deg"])))
    target = record_target(reference, pose, source="legacy_coarse_hover_not_final_grasp")
    target["tcp_pose"]["position_m"][2] = None
    target["z_not_recorded"] = True  # Runtime always uses the taught clearance.
    record = make_record(reference, approach=target, grasp=deepcopy(target))
    record["migration"] = {"confidence": "assumed_calibration_reference",
                           "warning": "Teaching frame/final grasp not recovered; hover verification required"}
    return record


def legacy_placement_record(profile, reference, ready_quaternion):
    pose = Pose(tuple(profile["place_release_tcp_m"]), rotate_quaternion(
        ready_quaternion, math.radians(profile["place"]["yaw_deg"])))
    record = make_record(reference, release=record_target(reference, pose,
        source="saved_measured_release_TCP"))
    record["migration"] = {"confidence": "assumed_calibration_reference",
        "warning": "Placement teaching frame unavailable; verify release hover"}
    return record


def resolve_profile_target(profile, frame, ready_pose, nominal, *, action, no_cv=False):
    """Single XY/orientation resolver for teaching, previews and execution.

    Legacy pickup fallback is explicit and differential from the original
    calibration frame, never from the current live frame. Legacy placement
    offsets remain board-local; new placement records are independent.
    """
    current = snapshot(frame)
    if action == "pick":
        record = profile.get("pickup_board")
        if record is None:
            reference = frame[3].get("calibration_board_reference")
            if reference is None:
                raise ValueError("legacy pickup lacks a calibration board reference")
            quat = frame[3].get("calibration_ready_quaternion", ready_pose.quaternion_wxyz)
            record = legacy_pickup_record(profile, reference, quat)
        return resolve_record(record, current, kind="grasp" if no_cv else "approach")
    settings = profile.get("place")
    if not settings:
        return list(nominal.position_m[:2]), nominal.quaternion_wxyz
    record = profile.get("placement_board")
    if not record and profile.get("place_release_tcp_m"):
        reference = frame[3].get("calibration_board_reference")
        if reference:
            record = legacy_placement_record(profile, reference,
                frame[3].get("calibration_ready_quaternion", ready_pose.quaternion_wxyz))
    if record:
        return resolve_record(record, current, kind="release")
    _, axes = frame_arrays(current)
    xy = np.asarray(nominal.position_m[:2]) + axes @ np.asarray(settings["offset_board_xy_m"])
    return xy.tolist(), rotate_quaternion(ready_pose.quaternion_wxyz, math.radians(settings["yaw_deg"]))
