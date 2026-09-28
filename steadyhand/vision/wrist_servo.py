"""Image-based XY centering for a downward-facing wrist camera.

This module intentionally does not require wrist-camera intrinsics/extrinsics.
For a fixed vertical-claw orientation and approximately fixed Z, two small known
robot XY motions identify the local 2x2 image Jacobian:

    [du, dv]^T = J_px_per_m @ [dx_base, dy_base]^T

The inverse maps pixel centering error directly to a robot-base XY correction.
That makes it suitable for onsite visual servoing even when camera calibration
is incomplete.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PixelJacobian:
    """Local wrist image Jacobian in pixels per base-frame metre."""

    du_dx: float
    du_dy: float
    dv_dx: float
    dv_dy: float

    def __post_init__(self):
        if not all(math.isfinite(v) for row in self.matrix() for v in row):
            raise ValueError("Wrist pixel Jacobian must be finite")

    def condition_number(self):
        # Singular-value ratio of a 2x2 matrix; no numerical dependency needed.
        a, b = self.du_dx, self.du_dy
        c, d = self.dv_dx, self.dv_dy
        det = abs(a * d - b * c)
        trace = a*a + b*b + c*c + d*d
        if det < 1e-12:
            return math.inf
        largest_sq = (trace + math.sqrt(max(0.0, trace*trace - 4*det*det))) / 2
        return largest_sq / det

    def matrix(self):
        return (
            (self.du_dx, self.du_dy),
            (self.dv_dx, self.dv_dy),
        )

    def base_delta_for_pixel_error(
        self,
        error_uv,
        *,
        gain=0.7,
        max_step_m=0.025,
    ):
        """Return (dx,dy) that moves a feature toward image center."""
        eu, ev = (float(x) for x in error_uv)
        if not all(math.isfinite(v) for v in (eu, ev, gain, max_step_m)):
            raise ValueError("Servo error and limits must be finite")
        if not 0 < gain <= 1 or max_step_m <= 0:
            raise ValueError("Servo gain must be in (0,1] and step limit positive")
        a, b = self.du_dx, self.du_dy
        c, d = self.dv_dx, self.dv_dy
        det = a * d - b * c
        if not math.isfinite(det) or abs(det) < 1e-6 or self.condition_number() > 30:
            raise ValueError("Wrist pixel Jacobian is singular or ill-conditioned")

        # Desired image change is -error.
        dx = float(gain) * ((d * (-eu) - b * (-ev)) / det)
        dy = float(gain) * ((-c * (-eu) + a * (-ev)) / det)
        norm = math.hypot(dx, dy)
        limit = float(max_step_m)
        if norm > limit:
            scale = limit / norm
            dx *= scale
            dy *= scale
        return dx, dy


def jacobian_from_probes(center_uv, plus_x_uv, plus_y_uv, step_m):
    """Estimate PixelJacobian from +base-X and +base-Y camera probes."""
    step = float(step_m)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("step_m must be finite and positive")
    u0, v0 = (float(x) for x in center_uv)
    ux, vx = (float(x) for x in plus_x_uv)
    uy, vy = (float(x) for x in plus_y_uv)
    return PixelJacobian(
        du_dx=(ux - u0) / step,
        du_dy=(uy - u0) / step,
        dv_dx=(vx - v0) / step,
        dv_dy=(vy - v0) / step,
    )


def image_center(shape):
    """Return (u,v) center for HxW or HxWxC image shape."""
    if len(shape) < 2:
        raise ValueError("image shape needs H,W")
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError("image dimensions must be positive")
    return ((w - 1) / 2.0, (h - 1) / 2.0)


def pixel_error(feature_uv, image_shape):
    """Feature minus image center; drive this vector to zero."""
    u, v = (float(x) for x in feature_uv)
    cu, cv = image_center(image_shape)
    return (u - cu, v - cv)


def jacobian_from_measured_probes(reference_uv, probe_uvs, reference_xy, probe_xys):
    """Use measured TCP displacement, including cross-axis motion, not commands."""
    import numpy as np

    displacement = (np.asarray(probe_xys, float) - np.asarray(reference_xy, float)).T
    pixels = (np.asarray(probe_uvs, float) - np.asarray(reference_uv, float)).T
    if displacement.shape != (2, 2) or pixels.shape != (2, 2):
        raise ValueError("Need two XY probes and two corresponding pixel observations")
    if not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(pixels)):
        raise ValueError("Probe observations must be finite")
    if np.linalg.svd(displacement, compute_uv=False)[-1] < 0.003:
        raise ValueError("Measured probes are too small or collinear (need >=3 mm)")
    if np.min(np.linalg.norm(pixels, axis=0)) < 3:
        raise ValueError("Feature barely moved: check selected wrist-camera label / tracking")
    matrix = pixels @ np.linalg.inv(displacement)
    result = PixelJacobian(*matrix.ravel())
    if result.condition_number() > 30:
        raise ValueError("Image probes are ill-conditioned; check camera / feature")
    return result


class TemplateTracker:
    """Fixed RGB-image patch, local normalized correlation, no online drift.

    Coordinates are always pixels in the original image. Ambiguous or lost
    matches raise before the controller can command another correction.
    """

    def __init__(self, rgb, feature_uv=None, *, patch_radius=20, search_radius=180,
                 min_score=0.75, min_margin=0.06):
        import cv2
        import numpy as np

        self.cv2 = cv2
        self.np = np
        gray = self._gray(rgb)
        self.shape = gray.shape
        self.radius = int(patch_radius)
        self.search_radius = int(search_radius)
        self.min_score = float(min_score)
        self.min_margin = float(min_margin)
        if self.radius < 4 or self.search_radius <= self.radius:
            raise ValueError("Invalid template/search radius")
        if feature_uv is None:
            mask = np.zeros_like(gray)
            h, w = gray.shape
            mask[h//4:3*h//4, w//4:3*w//4] = 255
            corners = cv2.goodFeaturesToTrack(gray, 100, 0.05, 20, mask=mask)
            if corners is None:
                raise ValueError("No textured board feature found; supply --feature U V")
            center = np.asarray(image_center(gray.shape))
            feature_uv = min(corners.reshape(-1, 2), key=lambda p: np.linalg.norm(p-center))
        if len(feature_uv) != 2 or not np.all(np.isfinite(feature_uv)):
            raise ValueError("Feature needs two finite pixel coordinates")
        u, v = (int(round(float(x))) for x in feature_uv)
        r = self.radius
        if not (r <= u < gray.shape[1]-r and r <= v < gray.shape[0]-r):
            raise ValueError("Feature patch extends beyond image")
        self.template = gray[v-r:v+r+1, u-r:u+r+1].copy()
        if float(self.template.std()) < 5:
            raise ValueError("Feature patch is textureless; select a corner/mark")
        self.uv = (float(u), float(v))
        self.locate(rgb)  # Reject an ambiguous initial patch before probing.

    def _gray(self, rgb):
        array = self.np.asarray(rgb)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != self.np.uint8:
            raise ValueError("Expected uint8 HxWx3 RGB wrist frame")
        return self.cv2.cvtColor(array, self.cv2.COLOR_RGB2GRAY)

    def locate(self, rgb):
        gray = self._gray(rgb)
        if gray.shape != self.shape:
            raise ValueError("Wrist image size changed during servo")
        r, s = self.radius, self.search_radius
        u, v = (int(round(x)) for x in self.uv)
        x0, y0 = max(0, u-s-r), max(0, v-s-r)
        x1, y1 = min(gray.shape[1], u+s+r+1), min(gray.shape[0], v+s+r+1)
        scores = self.cv2.matchTemplate(gray[y0:y1, x0:x1], self.template,
                                        self.cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = self.cv2.minMaxLoc(scores)
        alternatives = scores.copy()
        exclusion = max(4, r//2)
        alternatives[max(0,y-exclusion):y+exclusion+1,
                     max(0,x-exclusion):x+exclusion+1] = -1
        margin = score - float(alternatives.max())
        if not math.isfinite(score) or score < self.min_score or margin < self.min_margin:
            raise RuntimeError(f"Feature lost/ambiguous: score={score:.3f}, margin={margin:.3f}")
        self.uv = (float(x0+x+r), float(y0+y+r))
        return self.uv, float(score)


def run_xy_servo(robot, capture_rgb, *, floor_m, feature_uv=None, goal_uv=None,
                 probe_m=0.012, gain=0.65, max_step_m=0.015, max_radius_m=0.06,
                 tolerance_px=5.0, max_iterations=8, speed_scale=0.45,
                 tracker_factory=TemplateTracker, event=None):
    """Calibrate and center at the current hover pose; never descend or grip.

    capture_rgb must return a fresh post-motion image. Robot motion calls must
    block until reached. Exceptions stop the sequence in place (no blind return).
    """
    from ..executor import move_tcp_segmented
    from ..geometry import quaternion_angle, quaternion_to_matrix
    from ..models import Pose

    def report(kind, **fields):
        if event:
            event(kind, fields)

    values = (floor_m, probe_m, gain, max_step_m, max_radius_m, tolerance_px, speed_scale)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("Servo settings must be finite")
    if not (0.006 <= probe_m <= 0.015 and 0 < max_step_m <= 0.02
            and probe_m <= max_radius_m <= 0.10 and 0 < gain <= 1
            and 0.45 <= speed_scale <= 1 and 0 < tolerance_px
            and 1 <= max_iterations <= 20):
        raise ValueError("Invalid servo limits (probe 6–15 mm, step <=20 mm, radius <=100 mm)")
    origin = robot.get_tcp_pose()
    z = float(origin.position_m[2])
    quat = origin.quaternion_wxyz
    if not all(math.isfinite(v) for v in origin.position_m):
        raise ValueError("Non-finite live TCP pose")
    if z < floor_m + 0.06:
        raise ValueError("Wrist servo requires at least 60 mm clearance above TCP floor")
    if quaternion_to_matrix(quat)[2][2] < math.cos(0.12):
        raise ValueError("TCP is not in the corrected vertical tip_r orientation; use --move-to-board")

    def target(dx, dy):
        return Pose((origin.position_m[0]+dx, origin.position_m[1]+dy, z), quat)

    def check_pose(pose):
        if not all(math.isfinite(v) for v in pose.position_m):
            raise RuntimeError("Non-finite TCP readback")
        if abs(pose.position_m[2]-z) > 0.004 or quaternion_angle(pose.quaternion_wxyz, quat) > 0.025:
            raise RuntimeError("TCP Z/orientation drifted; local image calibration no longer valid")
        if math.dist(pose.position_m[:2], origin.position_m[:2]) > max_radius_m + 0.003:
            raise RuntimeError("Measured TCP left the local servo radius")

    def move(pose, *, label):
        before = robot.get_tcp_pose()
        check_pose(before)
        if math.dist(pose.position_m[:2], origin.position_m[:2]) > max_radius_m + 1e-9:
            raise RuntimeError("Requested XY exceeds local servo radius; coarse approach needed")
        move_tcp_segmented(robot, pose, speed_scale=speed_scale,
                           max_translation_step_m=0.02, max_orientation_step_rad=0.05,
                           min_tcp_z_m=floor_m)
        actual = robot.get_tcp_pose()
        check_pose(actual)
        position_error = math.dist(actual.position_m, pose.position_m)
        orientation_error = quaternion_angle(actual.quaternion_wxyz, pose.quaternion_wxyz)
        report(
            "motion",
            label=label,
            requested_tcp=pose.position_m,
            measured_tcp=actual.position_m,
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
        )
        if position_error > 0.003:
            raise RuntimeError("TCP missed servo waypoint by >3 mm")
        return actual

    rgb = capture_rgb()
    tracker = tracker_factory(rgb, feature_uv)
    uv0 = tracker.uv
    goal = image_center(rgb.shape) if goal_uv is None else tuple(float(v) for v in goal_uv)
    if len(goal) != 2 or not all(math.isfinite(v) for v in goal):
        raise ValueError("Goal pixel must contain two finite coordinates")
    if not (0 <= goal[0] < rgb.shape[1] and 0 <= goal[1] < rgb.shape[0]):
        raise ValueError("Goal pixel is outside image")
    report("reference", feature_uv=uv0, goal_uv=goal, tcp=origin.position_m)

    def observe(label):
        rgb = capture_rgb()
        uv, score = tracker.locate(rgb)
        actual = robot.get_tcp_pose()
        check_pose(actual)
        report(label, feature_uv=uv, score=score, tcp=actual.position_m)
        return uv, actual

    # Probe waypoints are prechecked when using the physical Vega adapter.
    if getattr(robot, "_kinematics", None) is not None:
        seed = robot._read_joint_positions()
        for pose in (target(probe_m, 0), target(0, probe_m), origin):
            robot._kinematics.solve(pose, seed)
    probe_uvs, probe_xys = [], []
    for label, pose in (("probe_x", target(probe_m, 0)), ("probe_y", target(0, probe_m))):
        move(pose, label=label)
        uv, actual = observe(label)
        probe_uvs.append(uv)
        probe_xys.append(actual.position_m[:2])
        move(origin, label=f"return_after_{label}")
        returned_uv, _ = observe("return_reference")
        if math.dist(returned_uv, uv0) > 8:
            raise RuntimeError("Feature did not return within 8 px; check tracking / scene motion")
    jacobian = jacobian_from_measured_probes(uv0, probe_uvs, origin.position_m[:2], probe_xys)
    report("calibrated", jacobian_px_per_m=jacobian.matrix(), condition=jacobian.condition_number())

    previous_error = None
    stalled = 0
    for iteration in range(max_iterations + 1):
        uv, actual = observe("servo_observation")
        error = (uv[0]-goal[0], uv[1]-goal[1])
        magnitude = math.hypot(*error)
        report("error", iteration=iteration, error_px=magnitude)
        if magnitude <= tolerance_px:
            result = {"status": "converged", "iterations": iteration,
                      "error_px": magnitude, "tcp_position_m": actual.position_m,
                      "feature_uv": uv, "goal_uv": goal,
                      "jacobian_px_per_m": jacobian.matrix()}
            report("complete", **result)
            return result
        if previous_error is not None:
            if magnitude > previous_error * 1.3 + 2:
                raise RuntimeError("Pixel error increased; stopping before another correction")
            stalled = stalled+1 if magnitude >= previous_error-1 else 0
            if stalled >= 2:
                raise RuntimeError("Centering stalled; inspect tracking and actual TCP motion")
        if iteration == max_iterations:
            raise RuntimeError(f"Not centered after {max_iterations} corrections: {magnitude:.1f} px")
        dx, dy = jacobian.base_delta_for_pixel_error(error, gain=gain, max_step_m=max_step_m)
        correction_target = Pose(
            (actual.position_m[0]+dx, actual.position_m[1]+dy, z),
            quat,
        )
        report(
            "correction",
            iteration=iteration,
            dx_m=dx,
            dy_m=dy,
            measured_tcp=actual.position_m,
            requested_tcp=correction_target.position_m,
        )
        move(correction_target, label=f"correction_{iteration}")
        previous_error = magnitude
