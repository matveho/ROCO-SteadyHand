"""Wrapper for the competition Vega's third-party CAN gripper driver.

Authoritative competition reference:
intelligent-control-lab/dexmate-setup @
4be4d16140b25f730673d664b49c58280b07ef46

The official driver exposes:
    g.left / g.right          -> Motor
    Motor.open()/close()
    Motor.grip(current=...)   -> dict including "gripped"
    Motor.position()
    g.both_open()/both_close()
    g.status()
    g.halt()
    g.release()
    g.close_bus()

CAN is can1 at 1 Mbit/s. Homing establishes the multi-turn reference; release()
destroys that reference, whereas halt() stops motion while preserving it.
"""

import importlib.util
from pathlib import Path


class VegaCanGripper:
    def __init__(self, config):
        self.config = dict(config)
        self._driver = None
        self._last_grip_result = None

    @property
    def scope(self):
        return self.config.get("scope")

    def connect(self):
        if self._driver is not None:
            return
        if self.scope not in ("left", "right", "both"):
            raise ValueError("gripper.scope must be 'left', 'right', or 'both'")

        module = _load_gripper_module(self.config.get("driver_path"))
        self._driver = module.Grippers()

        if self.config.get("home_on_connect", True):
            if self.scope == "both":
                # For a dual-arm operation, both motors are required.
                self._driver.home(require_all=True)
            else:
                self._motor().home()
        elif not self.config.get("skip_home_verified", False):
            self.close()
            raise ValueError(
                "Refusing to skip gripper homing without skip_home_verified=true"
            )

    def open(self):
        self._require()
        if self.scope == "both":
            return self._driver.both_open()
        return self._motor().open()

    def close_empty(self):
        """Position-close an empty jaw. Use grip() when holding an object."""
        self._require()
        if self.scope == "both":
            return self._driver.both_close()
        return self._motor().close()

    def grip(self, current_a=None):
        """Current-limited object grasp on the selected physical gripper."""
        self._require()
        if self.scope == "both":
            raise ValueError(
                "Object grip requires a single gripper scope ('left' or 'right'); "
                "the official driver has per-motor grip(), not a top-level both-grip."
            )
        value = self.config.get("grip_current_a") if current_a is None else current_a
        if value is None:
            raise ValueError("gripper.grip_current_a must be set before gripping")
        self._last_grip_result = self._motor().grip(current=float(value))
        return self._last_grip_result

    def last_grip_result(self):
        return self._last_grip_result

    def move_fraction(self, fraction, *, speed=500):
        self._require()
        fraction = float(fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("gripper fraction must be in [0, 1]")
        if self.scope == "both":
            return self._driver.both_move_to(fraction, speed=float(speed))
        return self._motor().move_to(fraction, speed=float(speed))

    def position(self):
        self._require()
        if self.scope == "both":
            return {
                "left": self._driver.left.position(),
                "right": self._driver.right.position(),
            }
        return self._motor().position()

    def status(self):
        self._require()
        if self.scope == "both":
            return self._driver.status()
        motor = self._motor()
        return {
            "side": self.scope,
            "angle": motor.angle(),
            "current": motor.current(),
            "temp_c": motor.temperature(),
            "volts": motor.voltage(),
            "enabled": motor.enabled(),
            "position": motor.position(),
        }

    def halt(self):
        if self._driver is None:
            return
        if self.scope == "both":
            self._driver.halt()
        else:
            self._motor().halt()

    def close(self):
        if self._driver is None:
            return
        try:
            self.halt()
        finally:
            self._driver.close_bus()
            self._driver = None

    def release_motors(self):
        """De-energize selected jaw(s); this destroys their calibration reference."""
        if self._driver is None:
            return
        try:
            self.halt()
            if self.scope == "both":
                self._driver.release()
            else:
                self._motor().release()
        finally:
            self._driver.close_bus()
            self._driver = None

    def _motor(self):
        self._require()
        return getattr(self._driver, self.scope)

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
