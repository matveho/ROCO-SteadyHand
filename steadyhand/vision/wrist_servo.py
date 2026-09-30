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
import time


DEFAULT_CENTERING_ITERATIONS = 40
MAX_CENTERING_ITERATIONS = 60


class ServoWaypointError(RuntimeError):
    """Stop visual servoing; a caller may separately validate a new plan."""

    def __init__(self, message, *, position_error_m=None, measured_pose=None):
        super().__init__(message)
        self.position_error_m = position_error_m
        self.measured_pose = measured_pose


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
    if np.min(np.linalg.norm(pixels, axis=0)) < 2:
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

    enable_edge_matching = False

    def __init__(self, rgb, feature_uv=None, *, patch_radius=20, search_radius=180,
                 min_score=0.75, min_margin=0.06):
        import cv2
        import numpy as np

        self.cv2 = cv2
        self.np = np
        array = np.asarray(rgb)
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
        self.template_rgb = array[v-r:v+r+1, u-r:u+r+1].copy()
        self.template = gray[v-r:v+r+1, u-r:u+r+1].copy()
        self.color_strength = self._color_strength(self.template_rgb)
        self.color_preferred = self.color_strength >= 18.0
        if float(self.template.std()) < 5 and not self.color_preferred:
            raise ValueError("Feature patch is textureless; select a corner/mark")
        self.anchor = (float(r), float(r))
        self.uv = (float(u), float(v))
        self.last_tracking_mode = "grayscale"
        self.locate(rgb)  # Reject an ambiguous initial patch before probing.

    @classmethod
    def from_saved_template(
        cls,
        rgb,
        template_rgb,
        template_uv=None,
        *,
        initial_uv=None,
        search_radius=180,
        min_score=0.75,
        min_margin=0.06,
    ):
        """Initialize from a taught part patch and locate it globally once.

        A profile template is deliberately small and can retain visible board
        edges around a partially covered part. The first match is global;
        subsequent frames use the normal bounded local tracker. Ambiguous or
        weak matches still stop before motion.
        """
        import cv2
        import numpy as np

        obj = cls.__new__(cls)
        obj.cv2, obj.np = cv2, np
        obj.radius = 20
        obj.search_radius = int(search_radius)
        obj.min_score = float(min_score)
        obj.min_margin = float(min_margin)
        gray = obj._gray(rgb)
        template_gray = obj._gray(np.asarray(template_rgb, dtype=np.uint8))
        if template_gray.ndim != 2 or min(template_gray.shape) < 8:
            raise ValueError("saved wrist template is too small")
        obj.shape = gray.shape
        obj.template_rgb = np.asarray(template_rgb, dtype=np.uint8).copy()
        if obj.template_rgb.ndim != 3 or obj.template_rgb.shape[2] != 3:
            raise ValueError("saved wrist template must be RGB")
        obj.template = template_gray.copy()
        obj.color_strength = obj._color_strength(obj.template_rgb)
        obj.color_preferred = obj.color_strength >= 18.0
        obj.radius = max(4, min(template_gray.shape) // 2)
        if float(template_gray.std()) < 5 and not obj.color_preferred:
            raise ValueError("Saved feature patch is textureless")
        if any(t > g for t, g in zip(template_gray.shape, gray.shape)):
            raise ValueError("Saved feature patch is larger than wrist image")
        obj.anchor = tuple(float(v) for v in (
            template_uv if template_uv is not None
            else ((template_gray.shape[1] - 1) / 2.0, (template_gray.shape[0] - 1) / 2.0)
        ))
        if (len(obj.anchor) != 2 or not all(math.isfinite(v) for v in obj.anchor)
                or not 0 <= obj.anchor[0] < template_gray.shape[1]
                or not 0 <= obj.anchor[1] < template_gray.shape[0]):
            raise ValueError("Saved feature anchor is outside template")
        obj.last_tracking_mode = "grayscale"
        if initial_uv is not None:
            try:
                point = tuple(float(v) for v in initial_uv)
                if len(point) == 2 and all(math.isfinite(v) for v in point):
                    if (obj.radius <= point[0] < gray.shape[1] - obj.radius
                            and obj.radius <= point[1] < gray.shape[0] - obj.radius):
                        obj.uv = point
                        obj.locate(rgb)
                        return obj
            except (TypeError, ValueError, RuntimeError):
                # The board may have moved farther than the previous wrist
                # view.  Fall back to the global saved-template search.
                pass
        obj._locate_global(rgb)
        return obj

    @staticmethod
    def _color_strength(rgb):
        """Return average per-pixel chroma used to prefer color matching."""
        import numpy as np

        array = np.asarray(rgb, dtype=np.float32)
        if array.ndim != 3 or array.shape[2] != 3:
            return 0.0
        return float(np.mean(np.max(array, axis=2) - np.min(array, axis=2)))

    def _validate_match(self, scores, score, x, y, *, exclusion,
                        min_score=None, min_margin=None):
        if min_score is None:
            min_score = self.min_score
        if min_margin is None:
            min_margin = self.min_margin
        alternatives = scores.copy()
        alternatives[max(0, y-exclusion):y+exclusion+1,
                     max(0, x-exclusion):x+exclusion+1] = -1
        margin = score - float(alternatives.max())
        if not math.isfinite(score) or score < min_score or margin < min_margin:
            raise RuntimeError(
                f"Feature lost/ambiguous: score={score:.3f}, margin={margin:.3f}"
            )
        return float(score), float(margin)

    def _locate_from_map(self, scores, *, x_offset=0, y_offset=0,
                         min_score=None, min_margin=None, label="grayscale"):
        """Locate and validate one score map, retaining the original UV convention."""
        _, score, _, (x, y) = self.cv2.minMaxLoc(scores)
        score, margin = self._validate_match(
            scores, float(score), int(x), int(y),
            exclusion=max(4, min(self.template.shape) // 4),
            min_score=min_score,
            min_margin=min_margin,
        )
        self.uv = (
            float(x_offset + x + self.anchor[0]),
            float(y_offset + y + self.anchor[1]),
        )
        self.last_tracking_mode = label
        return self.uv, score

    def _color_match(self, rgb, *, x_offset=0, y_offset=0):
        """Match RGB/chroma when grayscale texture is weak or repetitive.

        The color path is deliberately a fallback.  It uses the same saved
        patch and bounded search window, but preserves chroma information that
        grayscale matching loses on the slim blue battery and similar parts.
        Its threshold is relaxed only enough to survive camera exposure noise;
        the candidate still needs a measurable local margin.
        """
        import numpy as np

        image = np.asarray(rgb, dtype=np.uint8)
        template = np.asarray(self.template_rgb, dtype=np.uint8)
        if image.ndim != 3 or template.ndim != 3 or image.shape[2] != 3 or template.shape[2] != 3:
            raise RuntimeError("color feature fallback needs RGB images")
        if float(self.template.std()) < 5:
            # CCOEFF is undefined for a constant grayscale patch.  A
            # normalized squared-color distance remains useful for a uniform
            # blue/black battery surface and still produces a bounded score.
            scores = 1.0 - self.cv2.matchTemplate(
                image, template, self.cv2.TM_SQDIFF_NORMED
            )
        else:
            scores = self.cv2.matchTemplate(image, template, self.cv2.TM_CCOEFF_NORMED)
        return self._locate_from_map(
            scores,
            x_offset=x_offset,
            y_offset=y_offset,
            min_score=(self.min_score if getattr(self, "color_preferred", False)
                       else max(0.62, self.min_score - 0.10)),
            min_margin=(max(0.018, self.min_margin * 0.50)
                        if getattr(self, "color_preferred", False)
                        else max(0.012, self.min_margin * 0.25)),
            label="color",
        )

    def _locate_global(self, rgb):
        gray = self._gray(rgb)
        if gray.shape != self.shape:
            raise ValueError("Wrist image size changed during servo")
        scores = self.cv2.matchTemplate(gray, self.template, self.cv2.TM_CCOEFF_NORMED)
        try:
            return self._locate_from_map(scores)
        except RuntimeError as grayscale_error:
            try:
                return self._color_match(rgb)
            except (RuntimeError, ValueError):
                try:
                    return self._edge_match(rgb)
                except (RuntimeError, ValueError):
                    raise grayscale_error

    def _edge_match(self, rgb, *, x_offset=0, y_offset=0):
        """Use Canny edge geometry as a final fallback for low-texture parts."""
        if not self.enable_edge_matching:
            raise RuntimeError("edge fallback is not enabled for this tracker")
        gray = self._gray(rgb)
        template_edges = self.cv2.Canny(self.template, 35, 110)
        image_edges = self.cv2.Canny(gray, 35, 110)
        if int((template_edges > 0).sum()) < 8:
            raise RuntimeError("saved feature has insufficient edge structure")
        scores = self.cv2.matchTemplate(image_edges, template_edges, self.cv2.TM_CCOEFF_NORMED)
        return self._locate_from_map(
            scores, x_offset=x_offset, y_offset=y_offset,
            min_score=max(0.45, self.min_score - 0.25),
            min_margin=max(0.010, self.min_margin * 0.25),
            label="edge",
        )

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
        if getattr(self, "color_preferred", False):
            try:
                return self._color_match(
                    rgb[y0:y1, x0:x1], x_offset=x0, y_offset=y0,
                )
            except (RuntimeError, ValueError):
                pass
        gray_region = gray[y0:y1, x0:x1]
        scores = self.cv2.matchTemplate(gray_region, self.template,
                                         self.cv2.TM_CCOEFF_NORMED)
        try:
            return self._locate_from_map(
                scores, x_offset=x0, y_offset=y0,
                label="grayscale",
            )
        except RuntimeError as grayscale_error:
            try:
                return self._color_match(
                    rgb[y0:y1, x0:x1], x_offset=x0, y_offset=y0,
                )
            except (RuntimeError, ValueError):
                try:
                    return self._edge_match(
                        rgb[y0:y1, x0:x1], x_offset=x0, y_offset=y0,
                    )
                except (RuntimeError, ValueError):
                    raise grayscale_error

    def verify_selected(self, rgb, selected_uv, *, max_error_px=18.0):
        """Verify that an operator's second click is the original feature.

        Teaching uses two annotations: the first creates this tracker's patch,
        and the second is supposed to identify that same patch after the jaws
        have been aligned.  Temporarily searching around the second click with
        the original patch rejects a different mark or a weak/ambiguous click
        before the click can become a saved servo goal.
        """
        import math
        point = tuple(float(v) for v in selected_uv)
        if len(point) != 2 or not all(math.isfinite(v) for v in point):
            raise ValueError("selected feature must contain two finite pixels")
        if not (0 <= point[0] < self.shape[1] and 0 <= point[1] < self.shape[0]):
            raise ValueError("selected feature is outside the wrist image")
        previous = self.uv
        self.uv = point
        try:
            located, score = self.locate(rgb)
        except Exception:
            self.uv = previous
            raise RuntimeError(
                "the second click could not be matched to the original feature; "
                "capture another image and select the same distinctive feature"
            )
        error = math.dist(tuple(located), point)
        if error > float(max_error_px):
            self.uv = previous
            raise RuntimeError(
                "the second click does not identify the original feature "
                f"(match is {error:.1f} px away); choose the same feature"
            )
        # Keep the tracker at the verified observation for the next servo run.
        self.uv = tuple(float(v) for v in located)
        return tuple(float(v) for v in located), float(score), float(error)


def run_xy_servo(robot, capture_rgb, *, floor_m, feature_uv=None, goal_uv=None,
                 probe_m=0.012, gain=0.65, max_step_m=0.015, max_radius_m=0.06,
                 tolerance_px=5.0, max_iterations=DEFAULT_CENTERING_ITERATIONS, speed_scale=0.45,
                 tracker_factory=TemplateTracker, event=None,
                 surface_z=None, reference_quaternion_wxyz=None, checkpoint=None,
                 waypoint_guard=None):
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
            and (0.10 if checkpoint else 0.45) <= speed_scale <= 1 and 0 < tolerance_px
            and isinstance(max_iterations, int) and not isinstance(max_iterations, bool)
            and 1 <= max_iterations <= MAX_CENTERING_ITERATIONS):
        raise ValueError("Invalid servo limits (probe 6–15 mm, step <=20 mm, radius <=100 mm)")
    origin = robot.get_tcp_pose()
    z = float(origin.position_m[2])
    quat = origin.quaternion_wxyz
    if not all(math.isfinite(v) for v in origin.position_m):
        raise ValueError("Non-finite live TCP pose")
    if reference_quaternion_wxyz is not None:
        if quaternion_angle(quat, reference_quaternion_wxyz) > 0.025:
            raise ValueError("TCP does not match the measured ready/yaw orientation")
    elif quaternion_to_matrix(quat)[2][2] < math.cos(0.12):
        raise ValueError("TCP is not in the corrected vertical tip_r orientation; use --move-to-board")

    def commanded_z(x, y):
        if surface_z is None:
            return z
        return z + float(surface_z(x, y)) - float(surface_z(*origin.position_m[:2]))

    def target(dx, dy):
        x, y = origin.position_m[0]+dx, origin.position_m[1]+dy
        return Pose((x, y, commanded_z(x, y)), quat)

    def check_pose(pose):
        if not all(math.isfinite(v) for v in pose.position_m):
            raise RuntimeError("Non-finite TCP readback")
        if abs(pose.position_m[2]-commanded_z(*pose.position_m[:2])) > 0.004 or quaternion_angle(pose.quaternion_wxyz, quat) > 0.025:
            raise RuntimeError("TCP Z/orientation drifted; local image calibration no longer valid")
        if math.dist(pose.position_m[:2], origin.position_m[:2]) > max_radius_m + 0.003:
            raise RuntimeError("Measured TCP left the local servo radius")

    def move(pose, *, label):
        before = robot.get_tcp_pose()
        check_pose(before)
        if math.dist(pose.position_m[:2], origin.position_m[:2]) > max_radius_m + 1e-9:
            raise RuntimeError("Requested XY exceeds local servo radius; coarse approach needed")
        if checkpoint:
            checkpoint(f"before_servo_{label}")
        move_tcp_segmented(robot, pose, speed_scale=speed_scale,
                           max_translation_step_m=0.02, max_orientation_step_rad=0.05,
                           waypoint_guard=waypoint_guard,
                           min_tcp_z_m=None)
        actual = robot.get_tcp_pose()
        check_pose(actual)
        position_error = math.dist(actual.position_m, pose.position_m)
        # Let delayed measured state settle without issuing another command.
        # Keep the existing 8 mm guard; never turn a failed probe into a blind
        # correction or silently pretend it reached the requested position.
        for _ in range(3):
            if position_error <= .008:
                break
            time.sleep(.1)
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
        # The Vega state stream can settle a few millimetres away from an
        # individual 8–10 mm probe while remaining stationary and usable for
        # the measured Jacobian.  The actual pose is used below, so reject
        # only a materially missed local waypoint.
        if position_error > 0.008:
            raise ServoWaypointError(
                f"TCP missed servo waypoint by >8 mm ({label}: {position_error * 1000:.2f} mm "
                "after read-only settling); centering stopped at the measured pose",
                position_error_m=position_error, measured_pose=actual,
            )
        if checkpoint:
            checkpoint(f"after_servo_{label}")
        return actual

    rgb = capture_rgb()
    tracker = tracker_factory(rgb, feature_uv)
    uv0 = tracker.uv
    goal = image_center(rgb.shape) if goal_uv is None else tuple(float(v) for v in goal_uv)
    if len(goal) != 2 or not all(math.isfinite(v) for v in goal):
        raise ValueError("Goal pixel must contain two finite coordinates")
    if not (0 <= goal[0] < rgb.shape[1] and 0 <= goal[1] < rgb.shape[0]):
        raise ValueError("Goal pixel is outside image")
    report(
        "reference",
        feature_uv=uv0,
        goal_uv=goal,
        tcp=origin.position_m,
        max_iterations=max_iterations,
        tracking_mode=getattr(tracker, "last_tracking_mode", "grayscale"),
    )

    def observe(label):
        rgb = capture_rgb()
        uv, score = tracker.locate(rgb)
        actual = robot.get_tcp_pose()
        check_pose(actual)
        report(
            label,
            feature_uv=uv,
            score=score,
            tcp=actual.position_m,
            tracking_mode=getattr(tracker, "last_tracking_mode", "grayscale"),
        )
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
        if math.dist(returned_uv, uv0) > 18:
            raise RuntimeError("Feature did not return within 18 px; check tracking / scene motion")
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
            # Small perspective/nonlinear effects and camera jitter can make a
            # single correction look temporarily worse. Stop only on a clear
            # divergence, while retaining the bounded radius and step gates.
            if magnitude > previous_error * 1.8 + 5:
                raise RuntimeError("Pixel error increased; stopping before another correction")
            stalled = stalled+1 if magnitude >= previous_error-2 else 0
            if stalled >= 3:
                raise RuntimeError("Centering stalled; inspect tracking and actual TCP motion")
        if iteration == max_iterations:
            raise RuntimeError(f"Not centered after {max_iterations} corrections: {magnitude:.1f} px")
        dx, dy = jacobian.base_delta_for_pixel_error(error, gain=gain, max_step_m=max_step_m)
        correction_target = Pose(
            (actual.position_m[0]+dx, actual.position_m[1]+dy,
             commanded_z(actual.position_m[0]+dx, actual.position_m[1]+dy)),
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
