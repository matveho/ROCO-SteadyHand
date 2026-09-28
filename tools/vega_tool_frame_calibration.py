"""Offline calibration/analyzer for Vega modeled tip_r vs physical claw center.

No robot, camera, gripper, or DexControl imports are used.

Transform convention:
    T_A_B maps coordinates from frame B into frame A.
    T_base_claw = T_base_tip @ T_tip_claw

The physical claw frame used here has:
    origin: operator-defined physical grasp/claw center
    +Z: from wrist toward fingertips (down when the claw is physically vertical)
    +X: optional jaw-direction reference used to resolve yaw

A fixed T_tip_claw can explain absolute TCP-center offsets and apparent motion
caused by orientation changes. It cannot create translation coupling when the
modeled tip orientation is unchanged:
    Delta p_claw = Delta p_tip       if R_tip_1 == R_tip_2.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ROOT / "configs" / "robots" / "vega_1u_competition_kinematics.urdf"

EXPECTED_RECORDER_TOOL = "tools/vega_tool_frame_record.py"
EXPECTED_RECORDER_MODE = "READ_ONLY_NO_MOTION_COMMANDS"
EXPECTED_ROBOT_NAME = "dm/vgfcb66075ea-1u"
EXPECTED_BASE_FRAME = "vega_1u_base_link"
EXPECTED_WORKING_ARM = "right"
EXPECTED_TCP_FRAME = "tip_r"
EXPECTED_JOINT_NAMES = (
    "R_arm_j1",
    "R_arm_j2",
    "R_arm_j3",
    "R_arm_j4",
    "R_arm_j5",
    "R_arm_j6",
    "R_arm_j7",
)


def validate_right_arm_provenance(data):
    """Reject any calibration record not explicitly produced for right tip_r.

    This is a semantic provenance gate, not a cryptographic authenticity check.
    It prevents old left-arm or provenance-free observation JSON from being
    silently interpreted as right-arm tool-frame evidence.
    """
    if not isinstance(data, dict):
        raise ValueError("tool-frame calibration input must be a JSON object")

    recorder = data.get("recorder")
    if not isinstance(recorder, dict):
        raise ValueError(
            "missing recorder provenance; refusing to interpret observations as right-arm data"
        )

    required = {
        "tool": EXPECTED_RECORDER_TOOL,
        "mode": EXPECTED_RECORDER_MODE,
        "robot_name": EXPECTED_ROBOT_NAME,
        "base_frame": EXPECTED_BASE_FRAME,
        "working_arm": EXPECTED_WORKING_ARM,
        "tcp_frame": EXPECTED_TCP_FRAME,
    }
    for field, expected in required.items():
        if field not in recorder:
            raise ValueError(f"missing recorder provenance field: {field}")
        actual = recorder.get(field)
        if actual != expected:
            raise ValueError(
                f"recorder provenance {field}={actual!r}, expected {expected!r}"
            )

    joint_names = recorder.get("joint_names")
    if not isinstance(joint_names, (list, tuple)):
        raise ValueError("missing recorder provenance field: joint_names")
    if tuple(joint_names) != EXPECTED_JOINT_NAMES:
        raise ValueError(
            "recorder provenance joint_names do not match the exact right-arm "
            f"joint order {EXPECTED_JOINT_NAMES!r}"
        )

    return {
        "tool": EXPECTED_RECORDER_TOOL,
        "mode": EXPECTED_RECORDER_MODE,
        "robot_name": EXPECTED_ROBOT_NAME,
        "base_frame": EXPECTED_BASE_FRAME,
        "working_arm": EXPECTED_WORKING_ARM,
        "tcp_frame": EXPECTED_TCP_FRAME,
        "joint_names": list(EXPECTED_JOINT_NAMES),
    }


def _vec(value, n, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain {n} finite numbers")
    return a


def _unit(value, name):
    a = _vec(value, 3, name)
    norm = float(np.linalg.norm(a))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be nonzero")
    return a / norm


def quat_to_matrix(q):
    w, x, y, z = _vec(q, 4, "quaternion_wxyz")
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm <= 0:
        raise ValueError("quaternion_wxyz must be nonzero")
    w, x, y, z = (v / norm for v in (w, x, y, z))
    return np.asarray([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=float)


def matrix_to_quat(r):
    r = np.asarray(r, dtype=float)
    if r.shape != (3, 3) or not np.all(np.isfinite(r)):
        raise ValueError("rotation must be finite 3x3")
    u, _, vt = np.linalg.svd(r)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    tr = float(np.trace(r))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.asarray([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def rpy_matrix(rpy):
    roll, pitch, yaw = _vec(rpy, 3, "rpy")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=float)
    ry = np.asarray([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=float)
    rz = np.asarray([[cy,-sy,0],[sy,cy,0],[0,0,1]], dtype=float)
    return rz @ ry @ rx


def rotation_angle(r):
    r = np.asarray(r, dtype=float)
    c = max(-1.0, min(1.0, float((np.trace(r) - 1.0) / 2.0)))
    return math.acos(c)


def project_rotation(mats):
    m = np.mean(np.asarray(mats, dtype=float), axis=0)
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def pose_from_record(record):
    pose = record["modeled_tip_pose"]
    p = _vec(pose["position_m"], 3, "modeled_tip_pose.position_m")
    r = quat_to_matrix(pose["quaternion_wxyz"])
    return p, r


def inspect_urdf(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joints = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        origin = joint.find("origin")
        if parent is None or child is None:
            continue
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            if origin.get("xyz"):
                xyz = [float(v) for v in origin.get("xyz").split()]
            if origin.get("rpy"):
                rpy = [float(v) for v in origin.get("rpy").split()]
        joints[joint.get("name")] = {
            "parent": parent.get("link"),
            "child": child.get("link"),
            "xyz_m": xyz,
            "rpy_rad": rpy,
            "rotation": rpy_matrix(rpy),
        }

    wanted = {}
    for name in ("R_ee_j0", "R_ee_fixed", "R_gripper_joint_tip"):
        if name in joints:
            row = dict(joints[name])
            row["rotation"] = row["rotation"].tolist()
            wanted[name] = row

    if "R_gripper_joint_tip" in joints:
        nominal = joints["R_gripper_joint_tip"]
        t_gripper_tip = np.eye(4)
        t_gripper_tip[:3, :3] = nominal["rotation"]
        t_gripper_tip[:3, 3] = nominal["xyz_m"]
        t_tip_gripper = np.linalg.inv(t_gripper_tip)
        nominal_tip_to_gripper = {
            "translation_m": t_tip_gripper[:3, 3].tolist(),
            "quaternion_wxyz": matrix_to_quat(t_tip_gripper[:3, :3]).tolist(),
        }
    else:
        nominal_tip_to_gripper = None

    return {
        "urdf": str(Path(urdf_path)),
        "fixed_joints": wanted,
        "nominal_T_tip_gripper_link": nominal_tip_to_gripper,
    }


def solve_direct_center_offset(observations):
    estimates = []
    labels = []
    for obs in observations:
        if obs.get("physical_claw_center_base_m") is None:
            continue
        p, r = pose_from_record(obs)
        center = _vec(obs["physical_claw_center_base_m"], 3, "physical_claw_center_base_m")
        estimates.append(r.T @ (center - p))
        labels.append(obs.get("label", f"obs_{len(labels)}"))
    if not estimates:
        return None
    a = np.asarray(estimates)
    mean = np.mean(a, axis=0)
    residuals = np.linalg.norm(a - mean, axis=1)
    return {
        "method": "direct_base_center_measurements",
        "translation_tip_to_claw_center_m": mean.tolist(),
        "rms_residual_mm": float(np.sqrt(np.mean(residuals**2)) * 1000.0),
        "max_residual_mm": float(np.max(residuals) * 1000.0),
        "per_observation_translation_m": {
            label: est.tolist() for label, est in zip(labels, estimates)
        },
    }


def solve_pivot_group(observations, group):
    selected = [o for o in observations if o.get("pivot_group") == group]
    if len(selected) < 3:
        raise ValueError(f"pivot group {group!r} needs at least 3 observations")
    rows = []
    rhs = []
    for obs in selected:
        p, r = pose_from_record(obs)
        rows.append(np.hstack([r, -np.eye(3)]))
        rhs.append(-p)
    a = np.vstack(rows)
    b = np.hstack(rhs)
    x, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    if rank < 6:
        raise ValueError(
            f"pivot group {group!r} is rank deficient (rank={rank}); "
            "use rotations about at least two nonparallel axes"
        )
    t = x[:3]
    pivot = x[3:]
    residual_vectors = []
    for obs in selected:
        p, r = pose_from_record(obs)
        residual_vectors.append(p + r @ t - pivot)
    residual_vectors = np.asarray(residual_vectors)
    norms = np.linalg.norm(residual_vectors, axis=1)
    return {
        "method": "same_physical_pivot",
        "pivot_group": group,
        "observation_count": len(selected),
        "translation_tip_to_claw_center_m": t.tolist(),
        "pivot_base_m": pivot.tolist(),
        "rank": int(rank),
        "condition_number": float(singular[0] / singular[-1]),
        "rms_residual_mm": float(np.sqrt(np.mean(norms**2)) * 1000.0),
        "max_residual_mm": float(np.max(norms) * 1000.0),
    }


def solve_axis_correction(observations):
    tip_axes = []
    labels = []
    for obs in observations:
        if obs.get("physical_claw_axis_base") is None:
            continue
        _, r = pose_from_record(obs)
        axis_base = _unit(obs["physical_claw_axis_base"], "physical_claw_axis_base")
        tip_axes.append(r.T @ axis_base)
        labels.append(obs.get("label", f"obs_{len(labels)}"))
    if not tip_axes:
        return None
    a = np.asarray(tip_axes)
    mean = _unit(np.mean(a, axis=0), "mean physical claw axis in tip")
    scatter = [
        math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(mean, _unit(v, "axis")))))))
        for v in a
    ]
    expected = np.asarray([0.0, 0.0, -1.0])
    tilt_deg = math.degrees(
        math.acos(max(-1.0, min(1.0, float(np.dot(mean, expected)))))
    )
    return {
        "physical_claw_axis_in_tip": mean.tolist(),
        "tilt_from_nominal_minus_tip_z_deg": tilt_deg,
        "axis_scatter_rms_deg": float(math.sqrt(np.mean(np.square(scatter)))),
        "axis_observation_count": len(tip_axes),
    }


def solve_full_orientation_correction(observations):
    rotations = []
    labels = []
    for obs in observations:
        if obs.get("physical_claw_axis_base") is None or obs.get("physical_claw_x_axis_base") is None:
            continue
        _, r_bt = pose_from_record(obs)
        z = _unit(obs["physical_claw_axis_base"], "physical_claw_axis_base")
        x_raw = _unit(obs["physical_claw_x_axis_base"], "physical_claw_x_axis_base")
        x = x_raw - z * float(np.dot(z, x_raw))
        x = _unit(x, "physical_claw_x_axis_base projected perpendicular to axis")
        y = np.cross(z, x)
        y = _unit(y, "physical claw y axis")
        x = _unit(np.cross(y, z), "physical claw x axis")
        r_bc = np.column_stack([x, y, z])
        rotations.append(r_bt.T @ r_bc)
        labels.append(obs.get("label", f"obs_{len(labels)}"))
    if not rotations:
        return None
    mean = project_rotation(rotations)
    scatter_deg = [
        math.degrees(rotation_angle(mean.T @ r)) for r in rotations
    ]
    return {
        "rotation_tip_to_physical_claw": mean.tolist(),
        "quaternion_tip_to_physical_claw_wxyz": matrix_to_quat(mean).tolist(),
        "orientation_scatter_rms_deg": float(math.sqrt(np.mean(np.square(scatter_deg)))),
        "orientation_observation_count": len(rotations),
    }


def corrected_vertical_tip_quaternion(rotation_tip_to_claw, yaw_deg=0.0):
    """Return R_base_tip so the measured physical claw frame is top-down.

    Physical claw frame convention: +Z points wrist -> fingertips.
    Desired physical orientation is Rz(yaw) @ Rx(pi), matching the historical
    top-down gripper-link convention.
    """
    r_tc = np.asarray(rotation_tip_to_claw, dtype=float)
    if r_tc.shape != (3, 3):
        raise ValueError("rotation_tip_to_claw must be 3x3")
    yaw = math.radians(float(yaw_deg))
    desired_claw = rpy_matrix((math.pi, 0.0, yaw))
    desired_tip = desired_claw @ r_tc.T
    return matrix_to_quat(desired_tip)


def choose_offset_solution(observations):
    groups = sorted({o.get("pivot_group") for o in observations if o.get("pivot_group")})
    pivots = []
    for group in groups:
        count = sum(o.get("pivot_group") == group for o in observations)
        if count >= 3:
            try:
                pivots.append(solve_pivot_group(observations, group))
            except ValueError as exc:
                pivots.append({"method":"same_physical_pivot","pivot_group":group,"error":str(exc)})
    direct = solve_direct_center_offset(observations)
    candidates = [p for p in pivots if "translation_tip_to_claw_center_m" in p]
    if direct is not None:
        candidates.append(direct)
    if not candidates:
        return None, {"pivot_groups": pivots, "direct": direct}
    best = min(candidates, key=lambda x: x.get("rms_residual_mm", float("inf")))
    return best, {"pivot_groups": pivots, "direct": direct}


def translation_checks(data, observations, offset):
    by_label = {o.get("label"): o for o in observations if o.get("label")}
    out = []
    for check in data.get("translation_checks", []):
        start = by_label[check["from"]]
        end = by_label[check["to"]]
        p0, r0 = pose_from_record(start)
        p1, r1 = pose_from_record(end)
        measured = _vec(
            check["measured_physical_center_delta_m"],
            3,
            "measured_physical_center_delta_m",
        )
        tip_delta = p1 - p0
        orientation_change_deg = math.degrees(rotation_angle(r0.T @ r1))
        if offset is None:
            if orientation_change_deg <= 0.25:
                predicted = tip_delta.copy()
                source = "offset_invariant_same_orientation"
            else:
                predicted = None
                source = "needs_solved_offset_for_orientation_change"
        else:
            t = _vec(
                offset["translation_tip_to_claw_center_m"],
                3,
                "translation_tip_to_claw_center_m",
            )
            predicted = tip_delta + (r1 - r0) @ t
            source = "fixed_rigid_tip_to_center_transform"

        residual = None if predicted is None else measured - predicted
        norm_tip = float(np.linalg.norm(tip_delta))
        slope_deg = None
        horizontal = math.hypot(float(measured[0]), float(measured[1]))
        if horizontal > 1e-9:
            slope_deg = math.degrees(math.atan2(float(measured[2]), horizontal))

        explained = None
        if residual is not None:
            tol = float(check.get("explanation_tolerance_m", 0.003))
            explained = float(np.linalg.norm(residual)) <= tol

        out.append({
            "from": check["from"],
            "to": check["to"],
            "modeled_tip_delta_m": tip_delta.tolist(),
            "modeled_orientation_change_deg": orientation_change_deg,
            "measured_physical_center_delta_m": measured.tolist(),
            "measured_center_path_slope_deg": slope_deg,
            "prediction_source": source,
            "predicted_physical_center_delta_m": None if predicted is None else predicted.tolist(),
            "residual_m": None if residual is None else residual.tolist(),
            "residual_norm_mm": None if residual is None else float(np.linalg.norm(residual) * 1000.0),
            "fixed_tool_transform_explains_within_tolerance": explained,
            "fixed_offset_alone_can_create_coupling": False if orientation_change_deg <= 0.25 else None,
            "modeled_tip_translation_norm_m": norm_tip,
        })
    return out


def analyze(data, urdf_path=DEFAULT_URDF, yaw_deg=0.0):
    provenance = validate_right_arm_provenance(data)
    observations = data.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("input must contain a non-empty observations list")
    labels = [o.get("label") for o in observations]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("every observation requires a unique non-empty label")

    offset, offset_detail = choose_offset_solution(observations)
    axis = solve_axis_correction(observations)
    full_orientation = solve_full_orientation_correction(observations)
    checks = translation_checks(data, observations, offset)

    corrected_quat = None
    if full_orientation is not None:
        corrected_quat = corrected_vertical_tip_quaternion(
            full_orientation["rotation_tip_to_physical_claw"],
            yaw_deg=yaw_deg,
        ).tolist()

    assessment = []
    if axis is None:
        assessment.append(
            "No physical claw-axis observations were supplied, so an orientation correction is not yet identifiable."
        )
    elif axis["tilt_from_nominal_minus_tip_z_deg"] > 2.0:
        assessment.append(
            "Measured physical claw axis disagrees with the nominal -tip-Z assumption; a tool-orientation correction is required."
        )
    else:
        assessment.append(
            "Measured physical claw axis is close to the nominal -tip-Z assumption."
        )

    for row in checks:
        if row["modeled_orientation_change_deg"] <= 0.25 and row["fixed_tool_transform_explains_within_tolerance"] is False:
            assessment.append(
                f"{row['from']}->{row['to']}: same-orientation translation does not match physical center motion; "
                "a fixed tip-to-center offset cannot be the sole cause."
            )
        elif row["fixed_tool_transform_explains_within_tolerance"] is True:
            assessment.append(
                f"{row['from']}->{row['to']}: measured center motion is consistent with one fixed rigid tip-to-center transform."
            )

    if offset is None:
        assessment.append(
            "Physical claw-center offset is not solved; supply three same-pivot poses with nonparallel orientation changes "
            "or surveyed physical center coordinates."
        )

    if full_orientation is None:
        assessment.append(
            "A unique corrected vertical-claw quaternion is withheld because yaw/full claw-frame orientation is not measured."
        )

    result = {
        "schema_version": 1,
        "validated_provenance": provenance,
        "transform_convention": "T_A_B maps B coordinates into A; T_base_claw = T_base_tip @ T_tip_claw",
        "physical_claw_frame_convention": {
            "origin": "operator-defined physical grasp/claw center",
            "z_axis": "wrist toward fingertips; points down when physically vertical",
            "x_axis": "operator-supplied jaw/reference direction; required to resolve yaw",
        },
        "urdf": inspect_urdf(urdf_path),
        "offset_solution": offset,
        "offset_diagnostics": offset_detail,
        "axis_solution": axis,
        "full_orientation_solution": full_orientation,
        "translation_checks": checks,
        "corrected_vertical_tip_quaternion_wxyz": corrected_quat,
        "corrected_vertical_yaw_deg": float(yaw_deg) if corrected_quat is not None else None,
        "assessment": assessment,
    }
    return result


def _print_report(result):
    nominal = result["urdf"]
    print("URDF FIXED TRANSFORMS")
    for name, row in nominal["fixed_joints"].items():
        print(f"  {name}: {row['parent']} -> {row['child']} xyz={tuple(row['xyz_m'])} rpy={tuple(row['rpy_rad'])}")
    if result["offset_solution"] is not None:
        row = result["offset_solution"]
        print("SOLVED PHYSICAL CLAW-CENTER OFFSET IN tip_r (m) =",
              tuple(round(float(v), 6) for v in row["translation_tip_to_claw_center_m"]))
        print("OFFSET FIT RMS (mm) =", round(float(row["rms_residual_mm"]), 3))
    else:
        print("SOLVED PHYSICAL CLAW-CENTER OFFSET = NOT IDENTIFIED")

    axis = result["axis_solution"]
    if axis is not None:
        print("PHYSICAL CLAW AXIS IN tip_r =",
              tuple(round(float(v), 6) for v in axis["physical_claw_axis_in_tip"]))
        print("TILT FROM NOMINAL -tip_r Z (deg) =",
              round(float(axis["tilt_from_nominal_minus_tip_z_deg"]), 3))
    else:
        print("PHYSICAL CLAW AXIS = NOT MEASURED")

    for row in result["translation_checks"]:
        print(
            f"TRANSLATION {row['from']} -> {row['to']}: "
            f"orientation_change={row['modeled_orientation_change_deg']:.3f} deg, "
            f"measured_slope={row['measured_center_path_slope_deg']!r} deg, "
            f"residual_mm={row['residual_norm_mm']!r}, "
            f"fixed_transform_explains={row['fixed_tool_transform_explains_within_tolerance']}"
        )
        if row["fixed_offset_alone_can_create_coupling"] is False:
            print("  INVARIANT: with unchanged modeled tip orientation, a fixed rotated tool offset cannot create X/Z coupling.")

    q = result["corrected_vertical_tip_quaternion_wxyz"]
    if q is None:
        print("CORRECTED VERTICAL tip_r QUATERNION = WITHHELD (insufficient full orientation measurement)")
    else:
        print("CORRECTED VERTICAL tip_r QUATERNION wxyz =",
              tuple(round(float(v), 8) for v in q))

    print("ASSESSMENT")
    for line in result["assessment"]:
        print(" -", line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="operator calibration JSON")
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--yaw-deg", type=float, default=0.0,
                        help="desired physical claw in-plane yaw if full orientation is measured")
    parser.add_argument("--json-output")
    args = parser.parse_args(argv)

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = analyze(data, urdf_path=Path(args.urdf), yaw_deg=args.yaw_deg)
    _print_report(result)
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("WROTE", out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
