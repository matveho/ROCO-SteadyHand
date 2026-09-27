"""Deterministic skill executor for the RoCo industrial board-assembly task.

For each part the harness supplies a `PartTarget`; this module turns it into
a nine-phase end-effector path and follows it with the harness's Lula IK
controller.

Two properties are deliberate:

  * targets are read from the `PartTarget` of the trial in hand, never from
    `param_config`, so the policy acts on whatever pose the harness reports
    rather than replaying fixed coordinates;
  * the snap phase advances on the harness's `obs.snap_fired` event rather
    than on a fixed step budget, so insertion cost scales with the trial
    instead of being padded for the worst case.

Every timing and tolerance constant is taken from `param_config` at import.
The policy takes no arguments and reads no environment variables.
"""
from __future__ import annotations

import os.path
import sys

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

import numpy as np  # noqa: E402

import param_config as pc  # noqa: E402
from controllers.ee_pose_controller import (  # noqa: E402
    EEPathFollower,
    build_pick_place_phases,
)
from policy_api import EnvInfo, Observation, PartTarget, Policy  # noqa: E402

def _or(cfg, key, default):
    # dict.get with a None-aware fallback: config keys exist but may be None.
    v = cfg.get(key)
    return default if v is None else v


class BoardAssemblyPolicy(Policy):
    def __init__(self, env_info: EnvInfo) -> None:
        super().__init__(env_info)
        L_controller = getattr(env_info, "L_controller", None)
        if L_controller is None:
            raise ValueError("BoardAssemblyPolicy requires env_info.L_controller")
        self._L_controller = L_controller
        self._follower = EEPathFollower(
            L_controller,
            position_tolerance=getattr(pc, "POS_TOL", 0.005),
            orientation_tolerance=getattr(pc, "ORN_TOL", 0.05),
            default_timeout_steps=getattr(pc, "WAYPOINT_TIMEOUT_STEPS", None),
        )
        self._is_first_part = True
        self._last_obs: Observation = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    def _read_snap_fired(self) -> bool:
        return bool(self._last_obs is not None
                    and getattr(self._last_obs, "snap_fired", False))

    def _build_path(self, target: PartTarget, return_home_q):
        cfg = dict(target.extra or {})
        ee_off = np.asarray(_or(cfg, "ee_offset", (0.0, 0.0, 0.0)), dtype=np.float64)
        orn = target.ee_orientation
        orn = None if orn is None else np.asarray(orn, dtype=np.float64)

        pick = None if target.pick_pos is None else np.asarray(target.pick_pos, dtype=np.float64).copy()
        place = None if target.place_pos is None else np.asarray(target.place_pos, dtype=np.float64).copy()

        pick_ee = None if pick is None else pick + ee_off
        place_ee = None if place is None else place + ee_off

        # Snap parts sweep the harness's own XY search pattern; open-release
        # parts do not search at all.
        is_snap = target.release_mode == "snap"
        snap_cfg = cfg.get("snap") or {}
        if is_snap:
            sr = snap_cfg.get("search") or {}
            s_n = int(sr.get("n", 0))
            s_ext = tuple(sr.get("extent_xy", (0.002, 0.002)))
            s_dwell = int(sr.get("dwell_steps", 1))
        else:
            s_n, s_ext, s_dwell = 0, (0.002, 0.002), 1
        return build_pick_place_phases(
            pick_pos=pick_ee,
            pick_orn=orn if pick_ee is not None else None,
            place_pos=place_ee,
            place_orn=orn if place_ee is not None else None,
            init_height=_or(cfg, "init_height", pc.INIT_HEIGHT),
            final_height=_or(cfg, "final_height", getattr(pc, "FINAL_HEIGHT", None)),
            include_close=pc.INCLUDE_CLOSE,
            include_open=pc.INCLUDE_OPEN,
            settle_close_steps=pc.SETTLE_CLOSE,
            settle_open_steps=pc.SETTLE_OPEN,
            settle_hover_place_steps=pc.SETTLE_HOVER_PLACE,
            settle_descend_place_steps=pc.SETTLE_DESCEND_PLACE,
            transit_steps=int(_or(cfg, "transit_steps", pc.TRANSIT_STEPS)),
            descend_pick_steps=pc.DESCEND_PICK_STEPS,
            descend_place_steps=pc.DESCEND_PLACE_STEPS,
            gripper_open_value=target.gripper_open,
            gripper_close_value=target.gripper_close,
            release_mode=target.release_mode,
            snap_advance_when=(self._read_snap_fired if is_snap else None),
            snap_timeout_steps=snap_cfg.get("timeout_steps"),
            snap_search_n=s_n,
            snap_search_extent_xy=(s_ext, s_ext) if isinstance(s_ext, float) else s_ext,
            snap_search_dwell_steps=s_dwell,
            return_home_q=return_home_q,
            return_home_gripper=target.gripper_open,
            return_home_cspace_tol=getattr(pc, "RETURN_HOME_CSPACE_TOL", None),
            return_home_settle_steps=getattr(pc, "RETURN_HOME_SETTLE_STEPS", 20),
        )

    # ------------------------------------------------------------------
    def reset(self, obs: Observation, target: PartTarget) -> None:
        self._last_obs = obs
        return_home_q = None if self._is_first_part else self.env_info.L_arm_init_q
        self._is_first_part = False
        path = self._build_path(target, return_home_q)
        self._follower.reset()
        self._follower.set_path(path)

    def act(self, obs: Observation):
        self._last_obs = obs
        return self._follower.step()

    def is_done(self, obs: Observation) -> bool:
        self._last_obs = obs
        return self._follower.is_done()
