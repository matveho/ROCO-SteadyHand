"""Vision geometry for coarse Vega task-board registration.

The head camera is used only for coarse board registration. Fine part centering
is intended to move to the left wrist camera once the board frame is known.

The head transform uses the organizer source-robot camera calibration as a
reference and normalizes it through the organizer URDF head kinematics for the
LIVE head joint angles. The absolute camera calibration is still from a
different Vega serial, so this is deliberately a coarse board-localization
path, not a precision insertion calibration.
"""

from collections import deque
import math


# Organizer source-robot calibration, dm/vg3eb20f25bb-1u, 2026-09-24.
# T_base_camera maps rectified left optical coordinates into Vega base axes.
SOURCE_HEAD_Q_RAD = (
    -0.00017453292093705386,
    -0.00017453292093705386,
    -0.5056219100952148,
)
SOURCE_T_BASE_CAMERA = (
    (-0.0013837130901385743, -0.6362664678189364, 0.7714681245957596, 0.08905748732366152),
    (-0.9999302774818821, -0.008166771632430021, -0.008529010265557607, 0.023881858821925006),
    (0.011727127230930423, -0.7714261375986157, -0.636210805249794, 1.1704476979498903),
    (0.0, 0.0, 0.0, 1.0),
)

# Organizer competition URDF head chain.
DEFAULT_LIFT_M = 0.0
DEFAULT_TORSO_FLIP_RAD = 0.22689280275926285


def detect_white_board_corners(
    rgb,
    *,
    min_value=150,
    max_chroma=65,
    seed_xy_fraction=(0.50, 0.68),
    roi_x_fraction=(0.10, 0.90),
    roi_y_fraction=(0.25, 0.98),
):
    """Return board pixels ordered TL, TR, BR, BL.

    The physical task board is the large, connected, nearly-white component
    near the lower/central image. Dark parts are holes in that component, so
    flood-filling the white surface is much more robust than using every bright
    pixel in the scene.
    """
    import numpy as np

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("rgb must have shape HxWx3")
    image = image[..., :3]
    h, w = image.shape[:2]

    lo = image.min(axis=2)
    hi = image.max(axis=2)
    mask = (lo >= int(min_value)) & ((hi - lo) <= int(max_chroma))

    x0 = int(round(float(roi_x_fraction[0]) * w))
    x1 = int(round(float(roi_x_fraction[1]) * w))
    y0 = int(round(float(roi_y_fraction[0]) * h))
    y1 = int(round(float(roi_y_fraction[1]) * h))
    roi = np.zeros_like(mask, dtype=bool)
    roi[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    mask &= roi

    ys, xs = np.nonzero(mask)
    if len(xs) < 1000:
        raise RuntimeError("Could not find enough white board pixels")

    sx = float(seed_xy_fraction[0]) * w
    sy = float(seed_xy_fraction[1]) * h
    nearest = int(np.argmin((xs - sx) ** 2 + (ys - sy) ** 2))
    start_y, start_x = int(ys[nearest]), int(xs[nearest])

    visited = np.zeros_like(mask, dtype=bool)
    q = deque([(start_y, start_x)])
    visited[start_y, start_x] = True
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    ys, xs = np.nonzero(visited)
    if len(xs) < 5000:
        raise RuntimeError(
            f"White component is too small for the task board ({len(xs)} px)"
        )
    if np.ptp(xs) < 80 or np.ptp(ys) < 80:
        raise RuntimeError("Detected white component does not span a plausible board")

    sums = xs + ys
    diffs = xs - ys
    corners = (
        (int(xs[int(np.argmin(sums))]), int(ys[int(np.argmin(sums))])),  # TL
        (int(xs[int(np.argmax(diffs))]), int(ys[int(np.argmax(diffs))])),  # TR
        (int(xs[int(np.argmax(sums))]), int(ys[int(np.argmax(sums))])),  # BR
        (int(xs[int(np.argmin(diffs))]), int(ys[int(np.argmin(diffs))])),  # BL
    )
    return corners


def head_left_optical_transform(
    head_q_rad,
    *,
    lift_m=DEFAULT_LIFT_M,
    torso_flip_rad=DEFAULT_TORSO_FLIP_RAD,
):
    """Approximate T_base_left_optical for arbitrary live head joints.

    A fixed optical correction is derived from the source calibration at its
    reference head pose, then carried through the organizer URDF head chain.
    """
    import numpy as np

    q = np.asarray(head_q_rad, dtype=float)
    if q.shape != (3,) or not np.all(np.isfinite(q)):
        raise ValueError("head_q_rad must contain 3 finite joint values")

    model_ref = _head_model_transform(SOURCE_HEAD_Q_RAD, lift_m, torso_flip_rad)
    calibrated_ref = np.asarray(SOURCE_T_BASE_CAMERA, dtype=float)
    model_to_optical = np.linalg.inv(model_ref) @ calibrated_ref
    return _head_model_transform(q, lift_m, torso_flip_rad) @ model_to_optical


def pixels_to_horizontal_plane(
    pixels,
    *,
    fx,
    fy,
    cx,
    cy,
    T_base_camera,
    plane_z_m,
):
    """Intersect rectified optical rays with a horizontal base-frame plane."""
    import numpy as np

    tbc = np.asarray(T_base_camera, dtype=float)
    if tbc.shape != (4, 4):
        raise ValueError("T_base_camera must be 4x4")
    origin = tbc[:3, 3]
    rotation = tbc[:3, :3]
    out = []
    for u, v in pixels:
        ray_camera = np.array(
            [(float(u) - cx) / fx, (float(v) - cy) / fy, 1.0],
            dtype=float,
        )
        direction = rotation @ ray_camera
        if abs(float(direction[2])) < 1e-5:
            raise RuntimeError("Camera ray is nearly parallel to board plane")
        scale = (float(plane_z_m) - float(origin[2])) / float(direction[2])
        if scale <= 0:
            raise RuntimeError("Detected board ray intersects plane behind camera")
        point = origin + scale * direction
        point[2] = float(plane_z_m)
        out.append(point)
    return np.asarray(out, dtype=float)


def board_frame_from_corners(corners_base):
    """Return center-frame T_base_board and average width/height.

    Input order is TL, TR, BR, BL. Board +X follows image left->right and
    board +Y follows image top->bottom; +Z is forced upward.
    """
    import numpy as np

    p = np.asarray(corners_base, dtype=float)
    if p.shape != (4, 3) or not np.all(np.isfinite(p)):
        raise ValueError("corners_base must be finite 4x3")

    tl, tr, br, bl = p
    center = p.mean(axis=0)

    xvec = (tr - tl) + (br - bl)
    xvec[2] = 0.0
    xnorm = np.linalg.norm(xvec)
    if xnorm < 1e-6:
        raise RuntimeError("Degenerate board X axis")
    xaxis = xvec / xnorm

    yvec = (bl - tl) + (br - tr)
    yvec[2] = 0.0
    yvec -= xaxis * float(np.dot(yvec, xaxis))
    ynorm = np.linalg.norm(yvec)
    if ynorm < 1e-6:
        raise RuntimeError("Degenerate board Y axis")
    yaxis = yvec / ynorm

    zaxis = np.cross(xaxis, yaxis)
    if zaxis[2] < 0:
        yaxis = -yaxis
        zaxis = -zaxis

    T = np.eye(4)
    T[:3, 0] = xaxis
    T[:3, 1] = yaxis
    T[:3, 2] = zaxis
    T[:3, 3] = center

    width = 0.5 * (
        np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)
    )
    height = 0.5 * (
        np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)
    )
    return T, float(width), float(height)


def _head_model_transform(head_q_rad, lift_m, torso_flip_rad):
    import numpy as np

    q1, q2, q3 = (float(x) for x in head_q_rad)
    M = np.eye(4)
    M = M @ _joint(
        (0.0, 0.0, 0.01), (0.0, 0.0, 1.57079), (0.0, 0.0, 1.0),
        float(lift_m), prismatic=True,
    )
    M = M @ _joint(
        (0.0, -0.06187, 0.786), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        float(torso_flip_rad),
    )
    M = M @ _fixed((0.0, 0.03852, 0.25982), (0.0, 0.0, -1.57079))
    M = M @ _joint(
        (-0.0735, -0.0725, 0.014), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), q1,
    )
    M = M @ _joint(
        (0.0, 0.0725, -0.0034999), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), q2,
    )
    M = M @ _joint(
        (0.0, 0.002, 0.0495), (math.pi, 0.0, 0.0), (0.0, 1.0, 0.0), q3,
    )
    M = M @ _fixed(
        (0.0365, -0.023, -0.0489),
        (1.5707963, -0.0000062, 3.1415863),
    )
    return M


def _fixed(xyz, rpy):
    import numpy as np

    M = np.eye(4)
    M[:3, :3] = _rpy_matrix(*rpy)
    M[:3, 3] = np.asarray(xyz, dtype=float)
    return M


def _joint(xyz, rpy, axis, value, *, prismatic=False):
    import numpy as np

    M = _fixed(xyz, rpy)
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    if prismatic:
        move = np.eye(4)
        move[:3, 3] = axis * float(value)
        return M @ move

    x, y, z = axis
    K = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    a = float(value)
    R = np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)
    rot = np.eye(4)
    rot[:3, :3] = R
    return M @ rot


def _rpy_matrix(roll, pitch, yaw):
    import numpy as np

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    Ry = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    Rz = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    return Rz @ Ry @ Rx
