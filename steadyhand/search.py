"""Small deterministic search patterns for contact-rich insertion."""

def centered_grid(n, extent_xy):
    """Return (dx, dy) offsets in metres, ordered center-out.

    n may be odd or even. extent_xy gives the maximum absolute x/y offset.
    Sorting center-out makes the nominal calibrated pose the first attempt.
    """
    if int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    ex, ey = (float(v) for v in extent_xy)
    if ex < 0 or ey < 0:
        raise ValueError("extent must be non-negative")

    def axis(count, extent):
        if count == 1:
            return [0.0]
        return [-extent + 2 * extent * i / (count - 1) for i in range(count)]

    points = [(x, y) for x in axis(n, ex) for y in axis(n, ey)]
    return sorted(
        points,
        key=lambda p: (
            p[0]*p[0] + p[1]*p[1],
            abs(p[0]) + abs(p[1]),
            p[1],
            p[0],
        ),
    )


def square_spiral(step_m=0.001, rings=3):
    """Discrete square spiral around zero, useful for force-gated probing."""
    step_m = float(step_m)
    if step_m <= 0 or int(rings) != rings or rings < 0:
        raise ValueError("step_m must be > 0 and rings must be a non-negative integer")
    out = [(0.0, 0.0)]
    for r in range(1, rings + 1):
        s = r * step_m
        # perimeter only; no duplicate corners
        for i in range(-r + 1, r + 1):
            out.append((i * step_m, -s))
        for i in range(-r + 1, r + 1):
            out.append((s, i * step_m))
        for i in range(r - 1, -r - 1, -1):
            out.append((i * step_m, s))
        for i in range(r - 1, -r - 1, -1):
            out.append((-s, i * step_m))
    return out
