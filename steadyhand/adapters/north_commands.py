"""Pure preparation of North joint commands. No connection or publication.

Inputs use SDK observation order and radians, not simulator joint order.
These checks do not provide collision checking, a watchdog, or a physical stop.
"""

import math


GROUPS = ("left_arm", "right_arm", "motor")


def _seven(values, label):
    result = tuple(float(v) for v in values)
    if len(result) != 7 or not all(math.isfinite(v) for v in result):
        raise ValueError(f"{label} requires seven finite radians")
    return result


def joint_step(reference, *, arm=None, target=None, limits=None, max_delta_rad=None):
    """Hold all 21 joints, optionally replacing one arm within explicit bounds.

    For a trajectory, reference is the previous accepted target, initialized
    from fresh measured feedback. A caller must separately check tracking error,
    freshness, fault codes, controller mode and elapsed time before publication.
    Hands and chassis are deliberately omitted; they are not zeroed.
    """
    groups = {name: _seven(reference[name], name) for name in GROUPS}
    if arm is None:
        if any(value is not None for value in (target, limits, max_delta_rad)):
            raise ValueError("An arm is required with a target or bounds")
    else:
        if arm not in ("left_arm", "right_arm"):
            raise ValueError("Select left_arm or right_arm")
        if target is None or limits is None or max_delta_rad is None:
            raise ValueError("Arm targets require joint limits and a step bound")
        desired = _seven(target, arm)
        bounds = tuple(tuple(float(v) for v in pair) for pair in limits)
        if len(bounds) != 7 or any(
            len(pair) != 2 or not all(math.isfinite(v) for v in pair)
            or pair[0] >= pair[1] for pair in bounds
        ):
            raise ValueError("Joint limits require seven finite [lower, upper] pairs")
        delta = float(max_delta_rad)
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError("Step bound must be positive and finite")
        for old, new, (low, high) in zip(groups[arm], desired, bounds):
            if not low <= old <= high or not low <= new <= high:
                raise ValueError("Arm reference or target exceeds joint limits")
            if abs(new - old) > delta:
                raise ValueError("Arm target exceeds the step bound")
        groups[arm] = desired
    return {f"/action/{name}/joint_angle": list(groups[name]) for name in GROUPS}


def check_sdk_encoding(client_type, step):
    """Exercise the installed SDK serializer without constructing a client.

    Uses private serializer methods of the inspected SDK; incompatible versions
    must fail this check. No constructor, transport, sender thread or publisher
    is created. The returned summary contains no measured joint values.
    """
    expected = {f"/action/{name}/joint_angle" for name in GROUPS}
    if set(step) != expected:
        raise ValueError("Encoding check requires exactly both arms and body/neck")
    for key, values in step.items():
        _seven(values, key)
    client = client_type.__new__(client_type)
    client.using_eef_control = False
    client.static_lowbody = False
    client.output_text = None
    bundle = client._build_joint_action_bundle(client._prepare_action_dict(step))
    decoded = type(bundle)()
    decoded.ParseFromString(bundle.SerializeToString())
    for name in ("left_arm", "right_arm"):
        actual = tuple(getattr(decoded, name).joint.position)
        wanted = step[f"/action/{name}/joint_angle"]
        if len(actual) != 7 or any(abs(a - b) > 1e-6 for a, b in zip(actual, wanted)):
            raise ValueError(f"SDK changed {name} order, values or length")
    motors = decoded.motor.commands
    if tuple(m.motor_id for m in motors) != tuple(range(1, 8)):
        raise ValueError("SDK body motor IDs differ from 1 through 7")
    if any(abs(m.value - v) > 1e-6 for m, v in zip(motors, step['/action/motor/joint_angle'])):
        raise ValueError("SDK changed body/neck values")
    if any(decoded.HasField(name) for name in ('left_glove', 'right_glove', 'chassis')):
        raise ValueError("SDK unexpectedly added hand or chassis commands")
    return {"serialized_joint_counts": [7, 7, 7], "published": False}
