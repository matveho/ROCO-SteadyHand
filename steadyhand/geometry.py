"""Rigid-transform helpers used by calibration and perception.

Conventions:
- metres
- quaternions are wxyz
- T_destination_source maps source coordinates into destination coordinates
"""

import math

from .config import check_transform
from .models import Pose


def _mat3_vec(r, v):
    return tuple(sum(r[i][j] * v[j] for j in range(3)) for i in range(3))


def _normalized_quaternion(q):
    values = tuple(float(v) for v in q)
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError("Quaternion must contain four finite wxyz numbers")
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Quaternion norm must be finite and positive")
    return tuple(v / norm for v in values)


def _interpolation_fraction(t):
    t = float(t)
    if not math.isfinite(t) or not 0 <= t <= 1:
        raise ValueError("Interpolation fraction must be finite and in [0, 1]")
    return t


def _rigid_matrix(t):
    try:
        value = [[float(x) for x in row] for row in t]
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected a finite rigid 4x4 transform") from exc
    check_transform(value, "transform")
    return value


def quaternion_to_matrix(q):
    w, x, y, z = _normalized_quaternion(q)
    return (
        (1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)),
        (2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)),
        (2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)),
    )


def matrix_to_quaternion(r):
    """Return a normalized wxyz quaternion from a proper 3x3 rotation."""
    try:
        rows = [[float(v) for v in row] for row in r]
    except (TypeError, ValueError) as exc:
        raise ValueError("Rotation must be a proper 3x3 matrix") from exc
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("Rotation must be a proper 3x3 matrix")
    _rigid_matrix([row + [0.0] for row in rows] + [[0, 0, 0, 1]])
    r = rows
    tr = r[0][0] + r[1][1] + r[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (r[2][1] - r[1][2]) / s
        y = (r[0][2] - r[2][0]) / s
        z = (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2
        w = (r[2][1] - r[1][2]) / s
        x = 0.25 * s
        y = (r[0][1] + r[1][0]) / s
        z = (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2
        w = (r[0][2] - r[2][0]) / s
        x = (r[0][1] + r[1][0]) / s
        y = 0.25 * s
        z = (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2
        w = (r[1][0] - r[0][1]) / s
        x = (r[0][2] + r[2][0]) / s
        y = (r[1][2] + r[2][1]) / s
        z = 0.25 * s
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    q = (w/norm, x/norm, y/norm, z/norm)
    # q and -q represent the same rotation. Prefer a stable sign.
    return tuple(-v for v in q) if q[0] < 0 else q


def pose_to_matrix(pose):
    r = quaternion_to_matrix(pose.quaternion_wxyz)
    x, y, z = pose.position_m
    return [
        [r[0][0], r[0][1], r[0][2], x],
        [r[1][0], r[1][1], r[1][2], y],
        [r[2][0], r[2][1], r[2][2], z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matrix_to_pose(t):
    t = _rigid_matrix(t)
    r = tuple(tuple(float(t[i][j]) for j in range(3)) for i in range(3))
    return Pose(
        position_m=(float(t[0][3]), float(t[1][3]), float(t[2][3])),
        quaternion_wxyz=matrix_to_quaternion(r),
    )


def compose(a, b):
    """Compose transforms: T_a_b @ T_b_c -> T_a_c."""
    return [
        [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
        for i in range(4)
    ]


def invert_rigid(t):
    t = _rigid_matrix(t)
    r = [[float(t[i][j]) for j in range(3)] for i in range(3)]
    p = [float(t[i][3]) for i in range(3)]
    rt = [[r[j][i] for j in range(3)] for i in range(3)]
    pinv = tuple(-v for v in _mat3_vec(rt, p))
    return [
        [rt[0][0], rt[0][1], rt[0][2], pinv[0]],
        [rt[1][0], rt[1][1], rt[1][2], pinv[1]],
        [rt[2][0], rt[2][1], rt[2][2], pinv[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def transform_point(t, point):
    x, y, z = (float(v) for v in point)
    v = (x, y, z, 1.0)
    out = [sum(t[i][k] * v[k] for k in range(4)) for i in range(4)]
    return tuple(out[:3])


def transform_pose(t_destination_source, pose_in_source):
    return matrix_to_pose(compose(t_destination_source, pose_to_matrix(pose_in_source)))


def offset_z(pose, dz_m):
    x, y, z = pose.position_m
    return Pose((x, y, z + float(dz_m)), pose.quaternion_wxyz)


def quaternion_slerp(a, b, t):
    """Shortest-path normalized quaternion interpolation, wxyz."""
    a = _normalized_quaternion(a)
    b = _normalized_quaternion(b)
    t = _interpolation_fraction(t)
    dot = sum(x*y for x, y in zip(a, b))
    if dot < 0:
        b = [-x for x in b]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        q = [x + float(t) * (y - x) for x, y in zip(a, b)]
        n = math.sqrt(sum(x*x for x in q))
        return tuple(x / n for x in q)
    theta = math.acos(dot)
    s = math.sin(theta)
    w0 = math.sin((1.0 - float(t)) * theta) / s
    w1 = math.sin(float(t) * theta) / s
    return tuple(w0*x + w1*y for x, y in zip(a, b))


def quaternion_angle(a, b):
    """Smallest angular distance between two orientations in radians."""
    a = _normalized_quaternion(a)
    b = _normalized_quaternion(b)
    dot = abs(sum(x*y for x, y in zip(a, b)))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def interpolate_pose(a, b, t):
    t = _interpolation_fraction(t)
    return Pose(
        position_m=tuple(
            float(x) + float(t) * (float(y) - float(x))
            for x, y in zip(a.position_m, b.position_m)
        ),
        quaternion_wxyz=quaternion_slerp(
            a.quaternion_wxyz, b.quaternion_wxyz, t
        ),
    )


def pose_distance(a, b):
    dp = math.sqrt(sum((float(x)-float(y))**2 for x, y in zip(a.position_m, b.position_m)))
    return dp, quaternion_angle(a.quaternion_wxyz, b.quaternion_wxyz)
