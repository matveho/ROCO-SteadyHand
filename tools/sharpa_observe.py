#!/usr/bin/env python3
"""Check North observations without publishing commands or saving sensor data."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.sharpa import SharpaAdapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", required=True)
    parser.add_argument("--endpoint", required=True, help="tcp/ROBOT_HOST:7449")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--check-sdk", action="store_true",
                        help="Also serialize an identity target locally; never publish it")
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or not 0 < args.duration <= 60:
        parser.error("duration must be between 0 and 60 seconds")
    adapter = SharpaAdapter({"observation_only": True, "sdk_root": args.sdk_root,
                            "zenoh_endpoint": args.endpoint})
    try:
        adapter.connect()
        deadline = time.monotonic() + args.duration
        checks = 0
        while True:
            observation = adapter.observe()
            checks += 1
            if time.monotonic() >= deadline:
                break
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        encoding = None
        if args.check_sdk:
            sys.path.insert(0, str(Path(args.sdk_root).resolve()))
            from sharpa_north_ces_lite import NorthClient
            from steadyhand.adapters.north_commands import joint_step, check_sdk_encoding
            encoding = check_sdk_encoding(NorthClient, joint_step(observation.extras['joint_groups']))
        print(json.dumps({
            "observation_only": True,
            "readiness_checks": checks,
            "joint_count": len(observation.joint_positions),
            "joint_groups": {k: len(v) for k, v in observation.extras["joint_groups"].items()},
            "cameras": {k: {"encoding": v.format, "bytes": len(v.data)}
                        for k, v in observation.cameras.items()},
            "tactile_force_sensors": len(observation.extras["tactile_force6d"]),
            "mode": observation.extras["mode"],
            "faults": observation.extras["faults"],
            "sdk_encoding": encoding,
            "on_sleep": observation.extras["on_sleep"],
            "note": "Advancing sensor messages verified; motion and calibration remain unverified.",
        }, indent=2))
        return 0
    except (RuntimeError, ValueError, OSError) as error:
        print(f"North observation check failed: {error}", file=sys.stderr)
        return 2
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
