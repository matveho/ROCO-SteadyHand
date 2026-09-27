"""Wrapper for the competition Vega's third-party CAN gripper library.

The field manual documents /home/dexmate/gripper.py and the Grippers API. This
wrapper does not configure SocketCAN itself; bring can1 up using the robot's
known-good setup before connecting.

Only the documented dual-gripper operations are enabled here. A single-side
mapping should be added only after inspecting the actual onsite library.
"""

import importlib.util
from pathlib import Path


class VegaCanGripper:
    def __init__(self, config):
        self.config = dict(config)
        self._driver = None

    def connect(self):
        if self._driver is not None:
            return
        scope = self.config.get("scope")
        if scope != "both":
            raise ValueError(
                "gripper.scope must be 'both' for the documented wrapper; "
                "single-side control needs the onsite library mapping"
            )

        module = _load_gripper_module(self.config.get("driver_path"))
        self._driver = module.Grippers()

        if self.config.get("home_on_connect", True):
            self._driver.home()
        elif not self.config.get("skip_home_verified", False):
            self.close()
            raise ValueError(
                "Refusing to skip gripper homing without skip_home_verified=true"
            )

    def open(self):
        self._require()
        self._driver.both_open()

    def close_empty(self):
        """Close an empty jaw to its calibrated closed position."""
        self._require()
        self._driver.both_close()

    def grip(self, current_a=None):
        """Close under a current limit, the documented object-pick operation."""
        self._require()
        value = self.config.get("grip_current_a") if current_a is None else current_a
        if value is None:
            raise ValueError("gripper.grip_current_a must be set before gripping")
        self._driver.grip(current=float(value))

    def move_fraction(self, fraction, *, speed=500):
        self._require()
        fraction = float(fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("gripper fraction must be in [0, 1]")
        self._driver.both_move_to(fraction, speed=float(speed))

    def position(self):
        self._require()
        return self._driver.position()

    def status(self):
        self._require()
        return self._driver.status()

    def halt(self):
        if self._driver is not None:
            self._driver.halt()

    def close(self):
        if self._driver is None:
            return
        try:
            self._driver.halt()
        finally:
            self._driver.close_bus()
            self._driver = None

    def release_motors(self):
        """De-energize jaws; destroys the calibrated encoder frame."""
        if self._driver is None:
            return
        try:
            self._driver.halt()
            self._driver.release()
        finally:
            self._driver.close_bus()
            self._driver = None

    def _require(self):
        if self._driver is None:
            raise RuntimeError("Vega CAN gripper is not connected")


def _load_gripper_module(path):
    if not path:
        raise ValueError("gripper.driver_path is required")
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("steadyhand_vega_gripper_driver", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load gripper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Grippers"):
        raise ImportError(f"{path} does not define Grippers")
    return module
