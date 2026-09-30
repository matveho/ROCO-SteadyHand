"""Confirmation policy for supervised remote arm movements."""

import math

REMOTE_CLEARANCE_M = .040


def needs_low_clearance_confirmation(current, target, surface_z):
    """Check measured start and requested end, never just the nominal hover.

    Missing/non-finite geometry retains the prompt.  This is a TCP clearance
    policy, not a collision model for links, fingers or a held object.
    """
    try:
        for pose in (current, target):
            if pose is None:
                continue
            x, y, z = pose.position_m
            clearance = float(z) - float(surface_z(x, y))
            if not math.isfinite(clearance) or clearance < REMOTE_CLEARANCE_M - 1e-9:
                return True
        return current is None
    except (TypeError, ValueError, IndexError, KeyError, AttributeError):
        return True


def confirm_low_clearance(current, target, surface_z):
    """Minimal confirmation for menu position tests (no cameras or gripper)."""
    if not needs_low_clearance_confirmation(current, target, surface_z):
        return
    print("ARM MOVE below 40 mm clearance (or unknown clearance). NEXT TARGET =",
          target.position_m if target is not None else None, flush=True)
    while True:
        answer = input("Enter to continue / abort: ").strip().lower()
        if not answer:
            return
        if answer in ("abort", "stop", "q", "exit"):
            raise KeyboardInterrupt()
