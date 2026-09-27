"""Rigid-transform helpers used by calibration and perception.

Conventions:
- metres
- quaternions are wxyz
- T_destination_source maps source coordinates into destination coordinates
"""

import math

from .models import Pose


def _mat3_vec(r, v):
    return tuple(sum(r[i][j] * v[j] for j in range(3)) for i in range(3))


def quaternion_to_matrix(q):
    w, x, y, z = (float(v) for v in q)
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm <= 0:
        raise ValueError("Quaternion norm must be positive")
    w, x, y, z = (v / norm for v in (w, x, y, z))
    return (
        (1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)),
        (2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)),
        (2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)),
    )


def matrix_to_quaternion(r):
    """Return a normalized wxyz quaternion from a proper 3x3 rotation."""
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
