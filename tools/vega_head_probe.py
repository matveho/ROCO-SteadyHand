"""Read one Vega head-camera observation without constructing Robot().

Run ONBOARD the Vega Jetson with the installed system Python after setting
ROBOT_NAME. This subscribes to the existing head-camera service; it does not
send arm/gripper commands.

Example:
    export ROBOT_NAME=<competition robot name>
    /usr/bin/python3 tools/vega_head_probe.py
"""

import os
import sys


def main():
    if not os.environ.get("ROBOT_NAME"):
        print("ROBOT_NAME is not set; ask the onsite engineer for the robot name.", file=sys.stderr)
        return 2

    from dexcontrol.core.config import get_robot_config
    from dexcontrol.sensors.manager import Sensors

    configs = get_robot_config()
    configs.sensors["head_camera"].enabled = True
    sensors = Sensors({"head_camera": configs.sensors["head_camera"]})
    try:
        sensors.wait_for_all_active(timeout=10)
        head = sensors.head_camera.get_obs(
            obs_keys=["left_rgb", "right_rgb", "depth"],
            include_timestamp=True,
        )
        keys = ("left_rgb", "right_rgb", "depth")
        if any(head[k] is None for k in keys):
            raise RuntimeError("Head streams are not ready")

        for key in keys:
            record = head[key]
            data = record["data"]
            print(
                key,
                "shape=", getattr(data, "shape", None),
                "dtype=", getattr(data, "dtype", None),
                "timestamp_ns=", record.get("timestamp_ns"),
            )
        print("camera_info=", sensors.head_camera.get_camera_info())
        return 0
    finally:
        sensors.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
