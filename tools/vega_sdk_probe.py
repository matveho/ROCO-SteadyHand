"""Connection-only DexMate Vega SDK probe.

This tool deliberately does NOT construct SteadyHand IK or CAN grippers.
It uses the installed dexcontrol Robot interface to print the live competition
robot's identity/state so we can fill the onsite config.

WARNING: Robot() may move the head to its default pose. The script refuses to
run without --confirm-head-motion. It sends no arm/gripper commands.
"""

import argparse
import importlib.metadata
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def finite_list(values):
    return [float(x) for x in values]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirm-head-motion", action="store_true")
    args = p.parse_args(argv)

    if not args.confirm_head_motion:
        raise SystemExit(
            "Refusing Robot(): clear the head workspace, have the physical e-stop "
            "ready, then pass --confirm-head-motion."
        )

    print("python =", sys.executable)
    print("python_version =", sys.version.split()[0])
    print("ROBOT_NAME =", os.environ.get("ROBOT_NAME"))
    try:
        print("dexcontrol_version =", importlib.metadata.version("dexcontrol"))
    except importlib.metadata.PackageNotFoundError:
        print("dexcontrol_version = <package metadata unavailable>")

    from dexcontrol.robot import Robot

    robot = Robot()
    try:
        print("robot_type =", type(robot).__name__)
        try:
            print("version_info =", robot.get_version_info())
        except Exception as exc:
            print("version_info_error =", repr(exc))

        for component_name in ("left_arm", "right_arm", "head", "torso", "chassis"):
            try:
                present = bool(robot.has_component(component_name))
            except Exception as exc:
                print(component_name, "presence_error =", repr(exc))
                continue
            print(f"{component_name}.present =", present)
            if not present:
                continue

            component = getattr(robot, component_name)
            if component_name in ("left_arm", "right_arm"):
                for label, fn in (
                    ("joint_names", getattr(component, "get_joint_name", None)),
                    ("joint_pos_rad", getattr(component, "get_joint_pos", None)),
                    ("joint_vel_rad_s", getattr(component, "get_joint_vel", None)),
                    ("timestamp_ns", getattr(component, "get_timestamp_ns", None)),
                ):
                    if fn is None:
                        print(f"{component_name}.{label} = <method unavailable>")
                        continue
                    try:
                        value = fn()
                        if label in ("joint_pos_rad", "joint_vel_rad_s"):
                            value = finite_list(value)
                        elif label == "joint_names":
                            value = list(value)
                        print(f"{component_name}.{label} =", value)
                    except Exception as exc:
                        print(f"{component_name}.{label}_error =", repr(exc))
                try:
                    print(
                        f"{component_name}.joint_limits_rad =",
                        [finite_list(pair) for pair in component.joint_pos_limit],
                    )
                except Exception as exc:
                    print(f"{component_name}.joint_limits_error =", repr(exc))
                try:
                    print(
                        f"{component_name}.wrench_raw =",
                        finite_list(component.wrench_sensor.get_wrench_state()),
                    )
                except Exception as exc:
                    print(f"{component_name}.wrench_error =", repr(exc))

        if robot.has_component("estop"):
            try:
                print("estop_state =", robot.estop.get_state())
            except Exception as exc:
                print("estop_state_error =", repr(exc))
            try:
                robot.estop.show()
            except Exception as exc:
                print("estop_show_error =", repr(exc))

        try:
            print("joint_pos_dict =", robot.get_joint_pos_dict(["left_arm", "right_arm"]))
        except Exception as exc:
            print("joint_pos_dict_error =", repr(exc))

        print("PROBE COMPLETE: no arm/gripper command was intentionally sent.")
        return 0
    finally:
        robot.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
