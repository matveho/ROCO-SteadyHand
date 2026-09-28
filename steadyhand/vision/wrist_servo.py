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
        a, b = self.du_dx, self.du_dy
        c, d = self.dv_dx, self.dv_dy
        det = a * d - b * c
        if not math.isfinite(det) or abs(det) < 1e-6:
            raise ValueError("Wrist pixel Jacobian is singular")

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
    return ((w - 1) / 2.0, (h - 1) / 2.0)


def pixel_error(feature_uv, image_shape):
    """Feature minus image center; drive this vector to zero."""
    u, v = (float(x) for x in feature_uv)
    cu, cv = image_center(image_shape)
    return (u - cu, v - cv)
