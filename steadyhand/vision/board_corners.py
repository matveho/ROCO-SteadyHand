"""Local white-board corner servoing, with no camera model or runtime probes.

Only observed boundary corners are used. The convex hull nominates candidates;
the original silhouette must support them. Curved fisheye edges are not fitted
as straight lines and hidden corners are never extrapolated.
"""

import math
import copy

import cv2
import numpy as np

from ..geometry import quaternion_angle, quaternion_to_matrix, matrix_to_quaternion
from ..models import Pose
from .wrist_servo import PixelJacobian, jacobian_from_measured_probes


METHOD = "white_board_corners_v1"
TEACH_PROBE_M = .010


class CornerVisualError(RuntimeError):
    """Image evidence is insufficient; no additional correction is authorized."""


def _candidates(gray, saturation, threshold):
    mask = ((gray > threshold) & (saturation < 65)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], mask
    # The bright venue floor can be larger than the board in a fisheye view.
    # It is a crescent/ring around the dark table, not a solid board surface.
    solid = [c for c in contours if cv2.contourArea(c) >= mask.size * .035
             and cv2.contourArea(c) / max(1., cv2.contourArea(cv2.convexHull(c))) >= .75]
    if not solid:
        return [], mask
    contour = max(solid, key=cv2.contourArea)
    # Interior holes/parts do not become candidate board corners.
    mask[:] = 0
    cv2.drawContours(mask, [contour], -1, 255, -1)
    hull = cv2.convexHull(contour)
    polygon = cv2.approxPolyDP(hull, .008 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    result = []
    h, w = mask.shape
    for i, point in enumerate(polygon):
        a, b = polygon[i-1] - point, polygon[(i+1) % len(polygon)] - point
        if min(np.linalg.norm(a), np.linalg.norm(b)) < 35:
            continue
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        angle = math.degrees(math.acos(float(np.clip(a @ b, -1, 1))))
        u, v = point
        if not (40 < angle < 137 and 22 <= u < w-22 and 22 <= v < h-22):
            continue
        # Polygon simplification on a curved edge can shift its vertex along
        # that edge when another corner is occluded. Refine on the *observed*
        # silhouette locally, rather than letting the hull move our landmark.
        peaks = cv2.goodFeaturesToTrack(mask[v-17:v+18, u-17:u+18], 6, .1, 5, blockSize=5)
        if peaks is None:
            continue
        options = []
        for peak in peaks[:, 0]:
            candidate = peak + [u-17, v-17]
            cu, cv = np.rint(candidate).astype(int)
            if not (20 <= cu < w-20 and 20 <= cv < h-20):
                continue
            fraction = float((mask[cv-20:cv+21, cu-20:cu+21] > 0).mean())
            if .12 < fraction < .44 and np.linalg.norm(candidate - point) <= 16:
                options.append((fraction, candidate))
        if not options:
            continue
        candidate = min(options, key=lambda item: item[0])[1]
        refined = np.array([[candidate]], dtype=np.float32)
        cv2.cornerSubPix(gray, refined, (4, 4), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, .05))
        if np.linalg.norm(refined[0, 0] - candidate) > 4:
            continue
        u, v = np.rint(refined[0, 0]).astype(int)
        if not (22 <= u < w-22 and 22 <= v < h-22):
            continue
        # A real convex corner has white material in its wedge. A gripper
        # occlusion is a missing wedge; a hull bridge is not evidence.
        patch = mask[v-20:v+21, u-20:u+21]
        fraction = float((patch > 0).mean())
        if not .12 < fraction < .47:
            continue
        # The hull may bridge over a jaw and nominate the *intersection* of
        # that occlusion with an edge. Both adjacent edges must have visible
        # white material just inside them; a hull bridge through a jaw fails.
        inward = (a + b) / np.linalg.norm(a + b)
        supported = True
        for direction in (a, b):
            samples = np.rint(refined[0, 0] + np.array([8, 14, 20, 26])[:, None]
                               * direction + 6 * inward).astype(int)
            if (np.any(samples < 0) or np.any(samples[:, 0] >= w)
                    or np.any(samples[:, 1] >= h)
                    or np.mean(mask[samples[:, 1], samples[:, 0]] > 0) < .75):
                supported = False
                break
        if not supported:
            continue
        result.append({"uv": refined[0, 0].astype(float).tolist(),
                       "directions": [a.tolist(), b.tolist()], "angle": angle,
                       "signature": (cv2.resize(patch, (17, 17), interpolation=cv2.INTER_AREA)
                                     / 255.).round(3).tolist()})
    return result, mask


def detect_board_corners(rgb):
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise CornerVisualError("Board corner detection needs an RGB uint8 image")
    scale = min(1., 960. / image.shape[1])
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    saturation = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[:, :, 1]
    neutral = gray[(saturation < 65) & (gray > 12)]
    if len(neutral) < gray.size * .1 or float(neutral.std()) < 12:
        raise CornerVisualError("No contrasting white board silhouette")
    threshold = float(cv2.threshold(neutral, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0])
    groups = [_candidates(gray, saturation, t) for t in
              sorted(set(float(np.clip(threshold + d, 40, 235)) for d in (-10, 0, 10, 20)))]
    stable = []
    full_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    for group_index, (candidates, _) in enumerate(groups):
        for item in candidates:
            uv = np.asarray(item["uv"])
            support = [other for i, (group, _) in enumerate(groups) if i != group_index
                       for other in group if np.linalg.norm(np.asarray(other["uv"]) - uv) < 4]
            if not support or any(math.dist(c["uv"], uv / scale) < 12 for c in stable):
                continue
            # Refine in original pixels: half-resolution quantization can be
            # several millimetres at a shallow wrist viewing angle.
            pixel = np.array([[uv / scale]], dtype=np.float32)
            cv2.cornerSubPix(full_gray, pixel, (5, 5), (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, .01))
            if np.linalg.norm(pixel[0, 0] - uv / scale) > 4:
                continue
            stable.append(dict(item, uv=pixel[0, 0].astype(float).tolist()))
    if not stable:
        raise CornerVisualError("No visible, stable board corners; inspect the white-board outline")
    return sorted(stable, key=lambda item: (item["uv"][1], item["uv"][0])), groups[1][1]


def make_reference(rgb, selected_uvs=None):
    corners, _ = detect_board_corners(rgb)
    if selected_uvs is not None:
        selected = []
        for click in selected_uvs:
            nearest = min(corners, key=lambda c: math.dist(c["uv"], click))
            if math.dist(nearest["uv"], click) > 18 or nearest in selected:
                raise CornerVisualError("Click a distinct visible outer board corner, not a part or claw")
            selected.append(nearest)
        corners = selected
    if not 1 <= len(corners) <= 4:
        raise CornerVisualError("Board outline is ambiguous; select 1–4 actual board corners")
    return {"method": METHOD, "image_shape": list(rgb.shape[:2]),
            "corners": [dict(c, id=f"C{i+1}") for i, c in enumerate(corners)]}


def match_corners(rgb, reference):
    if list(rgb.shape[:2]) != reference.get("image_shape"):
        raise CornerVisualError("Board reference image resolution changed")
    found, _ = detect_board_corners(rgb)
    matches, used = {}, set()
    radius = min(100., rgb.shape[1] * .055)
    for taught in reference["corners"]:
        candidates = []
        for index, candidate in enumerate(found):
            distance = math.dist(candidate["uv"], taught["uv"])
            if index in used or distance > radius or abs(candidate["angle"] - taught["angle"]) > 18:
                continue
            old = np.asarray(taught["directions"])
            new = np.asarray(candidate["directions"])
            agreement = max(float(np.min(np.sum(old * new, axis=1))),
                            float(np.min(np.sum(old * new[::-1], axis=1))))
            a, b = np.asarray(taught["signature"]).ravel(), np.asarray(candidate["signature"]).ravel()
            score = float(np.corrcoef(a, b)[0, 1])
            if agreement < math.cos(math.radians(18)) or not math.isfinite(score) or score < .82:
                continue
            candidates.append((distance, index, candidate, score))
        candidates.sort(key=lambda c: c[0])
        if not candidates or (len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 20):
            continue
        _, index, candidate, score = candidates[0]
        used.add(index)
        matches[taught["id"]] = {"uv": candidate["uv"], "score": score}
    if not matches:
        raise CornerVisualError("No taught board corner is visible and uniquely matched")
    return matches


def correction(reference, matches):
    """Per-corner metric correction; don't average fisheye pixel translations."""
    estimates, errors, ids = [], [], []
    for corner in reference["corners"]:
        key = corner["id"]
        if key not in matches:
            continue
        error = np.asarray(matches[key]["uv"]) - corner["uv"]
        jacobian = PixelJacobian(*np.asarray(corner["jacobian_px_per_m"]).ravel())
        # A large limit here returns the raw estimate, before bounding motion.
        delta = jacobian.base_delta_for_pixel_error(error, gain=1., max_step_m=1.)
        estimates.append(delta)
        errors.append(float(np.linalg.norm(error)))
        ids.append(key)
    if not estimates:
        raise CornerVisualError("No corner with a measured motion calibration is visible")
    estimates = np.asarray(estimates)
    consensus = np.median(estimates, axis=0)
    inliers = np.linalg.norm(estimates - consensus, axis=1) <= .004
    if not all(inliers):
        if sum(inliers) < 2 or sum(inliers) <= len(inliers) / 2:
            raise CornerVisualError("Board corners disagree: rotation, occlusion or wrong boundary")
        estimates = estimates[inliers]
        errors = np.asarray(errors)[inliers].tolist()
        ids = [key for key, good in zip(ids, inliers) if good]
        consensus = np.median(estimates, axis=0)
    limit = .010 if len(ids) == 1 else .030
    if np.linalg.norm(consensus) > limit:
        raise CornerVisualError(f"Corner correction exceeds {limit*1000:.0f} mm limit ({len(ids)} corners)")
    return consensus, max(errors), ids


def draw_corners(rgb, reference, matches=None):
    image = np.asarray(rgb).copy()
    for corner in reference["corners"]:
        point = tuple(round(v) for v in corner["uv"])
        cv2.drawMarker(image, point, (0, 255, 0), cv2.MARKER_CROSS, 26, 2)
        cv2.putText(image, corner["id"], (point[0]+12, point[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)
        if matches and corner["id"] in matches:
            actual = tuple(round(v) for v in matches[corner["id"]]["uv"])
            cv2.circle(image, actual, 12, (255, 180, 0), 2)
            cv2.line(image, actual, point, (255, 180, 0), 2)
    return image


def rotate_reference(reference, current_board):
    """Rotate the measured base-XY Jacobians with the board-relative wrist yaw."""
    result = copy.deepcopy(reference)
    old = reference["reference_board"]["board_x_unit_base_xy"]
    new = current_board["board_x_unit_base_xy"]
    angle = math.atan2(new[1], new[0]) - math.atan2(old[1], old[0])
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    for corner in result["corners"]:
        corner["jacobian_px_per_m"] = (np.asarray(corner["jacobian_px_per_m"]) @ rotation.T).tolist()
    rotation3 = np.eye(3)
    rotation3[:2, :2] = rotation
    result["reference_quaternion_wxyz"] = list(matrix_to_quaternion(
        rotation3 @ np.asarray(quaternion_to_matrix(reference["reference_quaternion_wxyz"]))))
    return result


class CornerServo:
    """Uses the session's existing preflight/motion layer, never opens jaws."""

    def __init__(self, robot, move, capture, surface, event=None):
        self.robot, self.move, self.capture, self.surface = robot, move, capture, surface
        self.event = event or (lambda *_: None)
        self.origin = robot.stationary_tcp_pose(settle_timeout_s=2.)
        self.clearance = self.origin.position_m[2] - surface(*self.origin.position_m[:2])
        if self.clearance < .060:
            raise ValueError("Board corner alignment requires >=60 mm hover clearance")

    def pose(self):
        pose = self.robot.stationary_tcp_pose(settle_timeout_s=2.)
        if (abs(pose.position_m[2] - self.surface(*pose.position_m[:2]) - self.clearance) > .004
                or quaternion_angle(pose.quaternion_wxyz, self.origin.quaternion_wxyz) > .025):
            raise RuntimeError("TCP height/orientation drift invalidates board-corner alignment")
        if math.dist(pose.position_m[:2], self.origin.position_m[:2]) > .033:
            raise RuntimeError("TCP left the board-corner alignment radius")
        return pose

    def go(self, xy):
        if math.dist(xy, self.origin.position_m[:2]) > .030001:
            raise CornerVisualError("Board corner alignment reached its 30 mm travel limit")
        before = self.pose()
        target = Pose((*xy, self.surface(*xy) + self.clearance), self.origin.quaternion_wxyz)
        self.move(target)
        actual = self.pose()
        self.event("place_corner_motion", {
            "requested_tcp_m": list(target.position_m), "measured_tcp_m": list(actual.position_m),
            "measured_delta_xy_m": (np.asarray(actual.position_m[:2]) - before.position_m[:2]).tolist(),
            "position_error_m": math.dist(target.position_m, actual.position_m),
            "orientation_error_rad": quaternion_angle(target.quaternion_wxyz, actual.quaternion_wxyz),
        })
        if math.dist(target.position_m, actual.position_m) > .008:
            raise RuntimeError("TCP missed board-corner waypoint by more than 8 mm")
        return actual

    def observe(self, reference):
        first = match_corners(self.capture(), reference)
        second = match_corners(self.capture(), reference)
        common = {key: dict(value, uv=((np.asarray(value["uv"]) + first[key]["uv"]) / 2).tolist())
                  for key, value in second.items() if key in first
                  and math.dist(value["uv"], first[key]["uv"]) <= 2.}
        if not common:
            raise CornerVisualError("Board corner image unstable between fresh frames")
        return common, self.pose()

    def teach(self, selected_uvs=None, anchor_xy=None):
        rgb = self.capture()
        reference = make_reference(rgb, selected_uvs)
        initial, actual = self.observe(reference)
        def record(label, matches, pose):
            self.event("place_corner_measurement", {
                "stage": label, "tcp_position_m": list(pose.position_m),
                "quaternion_wxyz": list(pose.quaternion_wxyz), "matched_corners": matches,
            })
        record("reference", initial, actual)
        # Reference pixel and pose describe the same fresh, stationary view.
        reference["corners"] = [dict(c, uv=initial[c["id"]]["uv"])
                                for c in reference["corners"] if c["id"] in initial]
        self.origin = actual
        probe_matches, probe_poses, returns = [], [], []
        # Six-millimetre probes produced <1 px of scene movement onsite.
        # Use the established 10 mm wrist probe size, but learn from measured
        # displacement and retain the same conditioning/return checks.
        for axis, delta in zip(("x", "y"), ((TEACH_PROBE_M, 0.), (0., TEACH_PROBE_M))):
            self.go(tuple(a+b for a, b in zip(actual.position_m[:2], delta)))
            observation, pose = self.observe(reference)
            record(f"probe_{axis}", observation, pose)
            probe_matches.append(observation)
            probe_poses.append(pose.position_m[:2])
            self.go(actual.position_m[:2])
            returned, pose = self.observe(reference)
            record(f"return_{axis}", returned, pose)
            returns.append((returned, pose))
        valid, rejected = [], []
        def reject(key, reason, **details):
            rejected.append(f"{key}: {reason}")
            self.event("place_corner_rejected", {"corner_id": key, "reason": reason, **details})
        for corner in reference["corners"]:
            key = corner["id"]
            if any(key not in obs for obs in probe_matches + [item[0] for item in returns]):
                reject(key, "corner missing from a probe or return")
                continue
            try:
                jacobian = jacobian_from_measured_probes(corner["uv"],
                    [obs[key]["uv"] for obs in probe_matches], actual.position_m[:2], probe_poses)
            except ValueError as exc:
                reject(key, str(exc),
                       probe_xy_deltas_m=(np.asarray(probe_poses) - actual.position_m[:2]).tolist(),
                       probe_pixel_deltas=(np.asarray([obs[key]["uv"] for obs in probe_matches]) - corner["uv"]).tolist())
                continue
            matrix = np.asarray(jacobian.matrix())
            residuals = [float(np.linalg.norm(np.asarray(obs[key]["uv"]) - corner["uv"] -
                         matrix @ (np.asarray(pose.position_m[:2]) - actual.position_m[:2])))
                         for obs, pose in returns]
            if any(error > 4. for error in residuals):
                reject(key, f"return residual {max(residuals):.2f} px exceeds 4 px",
                       return_residuals_px=residuals, jacobian_px_per_m=matrix.tolist())
                continue
            valid.append(dict(corner, jacobian_px_per_m=matrix.tolist()))
        if not valid:
            raise CornerVisualError("Corner motion/return measurements disagree; "
                                    + "; ".join(rejected) + ". Physical placement remains saved")
        if anchor_xy is not None:
            offset = np.asarray(anchor_xy) - actual.position_m[:2]
            if np.linalg.norm(offset) > .008:
                raise RuntimeError("TCP too far from saved release XY to teach board corners")
            for corner in valid:
                corner["observed_uv"] = list(corner["uv"])
                corner["uv"] = (np.asarray(corner["uv"]) +
                                np.asarray(corner["jacobian_px_per_m"]) @ offset).tolist()
        reference.update(corners=valid, reference_clearance_m=self.clearance,
                         reference_quaternion_wxyz=list(actual.quaternion_wxyz))
        self.event("place_corner_calibrated", reference)
        # Finish at the actual taught alignment, verified on two fresh frames.
        self.align(reference)
        return reference, rgb

    def align(self, reference, expected_quaternion=None):
        if abs(self.clearance - float(reference["reference_clearance_m"])) > .004:
            raise CornerVisualError("Board corner reference was taught at a different hover height")
        if quaternion_angle(self.origin.quaternion_wxyz,
                            expected_quaternion or reference["reference_quaternion_wxyz"]) > .025:
            raise CornerVisualError("Arrival orientation differs from board corner reference")
        previous, stalled = None, 0
        for iteration in range(9):
            matches, pose = self.observe(reference)
            delta, error, ids = correction(reference, matches)
            self.event("place_corner_observation", {"iteration": iteration, "matched_corners": matches,
                "used_corners": ids, "error_px": error, "correction_m": delta.tolist(),
                "estimated_error_m": float(np.linalg.norm(delta))})
            if error <= 2 and np.linalg.norm(delta) <= .001:
                return {"status": "converged", "iterations": iteration, "error_px": error,
                        "estimated_error_m": float(np.linalg.norm(delta)),
                        "corners": ids, "tcp_position_m": list(pose.position_m)}
            if previous is not None:
                stalled = stalled+1 if error >= previous-1 else 0
                if error > previous * 1.6 + 3 or stalled >= 2:
                    raise CornerVisualError("Board corner correction stalled or diverged")
            if iteration == 8:
                break
            step = .6 * delta
            length = float(np.linalg.norm(step))
            if length > .005:
                step *= .005 / length
            xy = np.asarray(pose.position_m[:2]) + step
            if len(ids) == 1 and math.dist(xy, self.origin.position_m[:2]) > .010:
                raise CornerVisualError("Single-corner correction reached its 10 mm travel limit")
            self.go(tuple(xy))
            previous = error
        raise CornerVisualError("Board corner alignment not verified within eight corrections")
