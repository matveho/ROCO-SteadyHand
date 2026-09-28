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
import math
from pathlib import Path
import sys


class VegaCanGripper:
    def __init__(self, config):
        self.config = dict(config)
        self._driver = None
        self._last_grip_result = None

    @property
    def scope(self):
        return self.config.get("scope")

    def validate_config(self):
        """Validate gates/files without executing the third-party driver module."""
        if self.scope != "right":
            raise ValueError("competition gripper.scope must be 'right'")
        path = self.config.get("driver_path")
        if not path:
            raise ValueError("gripper.driver_path is required")
        resolved = _resolve_driver_path(path)
        if not resolved.is_file():
            raise FileNotFoundError(str(resolved))
        _positive(self.config.get("grip_current_a"), "gripper.grip_current_a")
        _positive(self.config.get("grip_speed_dps", 60), "gripper.grip_speed_dps")
        _positive(self.config.get("open_speed_dps", 500), "gripper.open_speed_dps")
        if (not self.config.get("home_on_connect", True)
                and not self.config.get("skip_home_verified", False)):
            raise ValueError("Refusing to skip gripper homing without skip_home_verified=true")

    def connect(self):
        if self._driver is not None:
            return
        self.validate_config()
        module = _load_gripper_module(self.config.get("driver_path"))
        self._driver = module.Grippers()
        try:
            for name in ("home", "both_open", "both_close", "status", "halt", "close_bus"):
                if not callable(getattr(self._driver, name, None)):
                    raise TypeError(f"Onsite Grippers driver does not implement documented {name}()")
            motor = self._motor()
            for name in ("home", "open", "close", "grip", "move_to",
                         "position", "halt", "release"):
                if not callable(getattr(motor, name, None)):
                    raise TypeError(
                        f"Onsite right gripper does not implement {name}()"
                    )
            if self.config.get("home_on_connect", True):
                if self.scope == "both":
                    self._driver.home(require_all=True)
                else:
                    self._motor().home()
        except BaseException:
            try:
                self.close()
            except BaseException as exc:
                print(f"Gripper initialization cleanup failed: {exc}", file=sys.stderr)
            raise

    def open(self):
        self._require()
        speed = int(round(_positive(
            self.config.get("open_speed_dps", 500),
            "gripper.open_speed_dps",
        )))
        if self.scope == "both":
            return self._driver.both_open(speed=speed)
        return self._motor().open(speed=speed)

    def close_empty(self):
        """Position-close an empty jaw. Use grip() when holding an object."""
        self._require()
        if self.scope == "both":
            return self._driver.both_close()
        return self._motor().close()

    def grip(self, current_a=None):
        """Current-limited object grasp on the selected physical gripper."""
        self._require()
        value = self.config.get("grip_current_a") if current_a is None else current_a
        speed = int(round(_positive(
            self.config.get("grip_speed_dps", 60),
            "gripper.grip_speed_dps",
        )))
        self._last_grip_result = self._motor().grip(
            current=_positive(value, "gripper.grip_current_a"),
            speed=speed,
        )
        return self._last_grip_result

    def last_grip_result(self):
        return self._last_grip_result

    def move_fraction(self, fraction, *, speed):
        self._require()
        fraction = float(fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("gripper fraction must be in [0, 1]")
        speed = int(round(_positive(speed, "gripper speed")))
        if self.scope == "both":
            if not callable(getattr(self._driver, "both_move_to", None)):
                raise TypeError("Onsite Grippers driver has no both_move_to()")
            return self._driver.both_move_to(fraction, speed=speed)
        return self._motor().move_to(fraction, speed=speed)

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
            try:
                self._driver.close_bus()
            finally:
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
            try:
                self._driver.close_bus()
            finally:
                self._driver = None

    def _motor(self):
        self._require()
        return getattr(self._driver, self.scope)

    def _require(self):
        if self._driver is None:
            raise RuntimeError("Vega CAN gripper is not connected")


def _resolve_driver_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path


def _load_gripper_module(path):
    if not path:
        raise ValueError("gripper.driver_path is required")
    path = _resolve_driver_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("steadyhand_vega_gripper_driver", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load gripper module from {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses and other runtime decorators resolve __module__ through
    # sys.modules; direct exec_module without registration breaks those drivers.
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
        raise
    if not hasattr(module, "Grippers"):
        raise ImportError(f"{path} does not define Grippers")
    return module


def _positive(value, name):
    if value is None:
        raise ValueError(f"{name} must be measured/verified and set")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return number
