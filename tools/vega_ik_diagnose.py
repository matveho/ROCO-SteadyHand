"""Offline right-arm IK diagnosis. Never imports dexcontrol or connects to Robot.

Compare production IK with a bounded least-squares diagnostic, using the exact
same URDF, target, joint limits, and measured seed. Results are NOT motion plans.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steadyhand.config import load_bundle
from steadyhand.kinematics.pinocchio_arm import PinocchioArmKinematics
from steadyhand.models import Pose
from steadyhand.vega_camera_clear import vertical_claw_tip_quaternion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q', nargs=7, type=float, required=True)
    parser.add_argument('--bounded-restarts', type=int, default=0,
                        help='additional deterministic numerical seeds; no motion')
    args = parser.parse_args()
    import numpy as np
    try:
        from scipy.optimize import least_squares
    except ImportError:
        least_squares = None
    cfg = load_bundle('vega')['robot']
    if cfg['working_arm'] != 'right' or cfg['kinematics']['ee_frame'] != 'tip_r':
        raise ValueError('Requires right arm / tip_r configuration')
    kc = dict(cfg['kinematics'])
    kc.update(joint_limits_rad=cfg['arm_joint_limits_rad']['right'],
              max_iterations=180, position_tolerance_m=0.002,
              orientation_tolerance_rad=0.02, max_seed_delta_rad=2.5)
    kin = PinocchioArmKinematics(cfg['urdf_path'], 'tip_r', kc['right_arm_joint_names'], kc)
    pin = kin.pin
    seed = np.asarray(args.q)
    if not 0 <= args.bounded_restarts <= 20:
        parser.error('--bounded-restarts must be 0..20')
    rng = np.random.default_rng(20260928)
    current = kin.forward(seed)
    print('NO ROBOT CONNECTION OR MOTION', flush=True)
    print('CURRENT', current, flush=True)
    print('SCIPY_AVAILABLE', least_squares is not None, flush=True)
    # Verify the actual LOCAL log-error Jacobian against finite differences.
    desired = kin._root_M_base * pin.SE3(np.eye(3), np.asarray(current.position_m) + [0.01, 0, 0])
    def error(qarm, goal):
        q = kin._full_q(qarm)
        pin.forwardKinematics(kin.model, kin.data, q)
        pin.updateFramePlacements(kin.model, kin.data)
        return pin.log6(kin.data.oMf[kin.frame_id].actInv(goal)).vector.copy()
    q = kin._full_q(seed)
    error(seed, desired)
    relative = kin.data.oMf[kin.frame_id].actInv(desired)
    analytic = kin._local_error_jacobian(q, relative).copy()
    eps = 1e-6
    numeric = np.column_stack([(error(seed + np.eye(7)[i]*eps, desired) -
                                error(seed - np.eye(7)[i]*eps, desired))/(2*eps)
                               for i in range(7)])
    print('JACOBIAN_MAX_ABS_ERROR', float(np.max(np.abs(analytic-numeric))), flush=True)
    for yaw in (0, 90, -90, 180):
        for xyz in ((0.50, 0.0, 0.806), (0.40, -0.20, 0.806), (0.544097, 0.00001, 0.550)):
            target = Pose(xyz, vertical_claw_tip_quaternion(math.radians(yaw)))
            row = {'yaw_deg': yaw, 'target_xyz': xyz}
            try:
                sol = kin.solve(target, seed)
                row['production'] = {'q': sol}
            except RuntimeError as exc:
                row['production'] = str(exc)
            if least_squares is not None:
                from steadyhand.geometry import quaternion_to_matrix, quaternion_angle
                goal = kin._root_M_base * pin.SE3(np.array(quaternion_to_matrix(target.quaternion_wxyz)), np.array(xyz))
                # Keep the existing seed-distance gate as bounds, not just an
                # endpoint check. This diagnostic never relaxes joint limits.
                lower = np.maximum(kin._lower, seed-2.5)
                upper = np.minimum(kin._upper, seed+2.5)
                fit = least_squares(error, seed, args=(goal,), bounds=(lower, upper), max_nfev=400)
                for _ in range(args.bounded_restarts):
                    if np.linalg.norm(fit.fun) < 1e-5:
                        break
                    alternative = least_squares(error, rng.uniform(lower, upper),
                                                args=(goal,), bounds=(lower, upper), max_nfev=250)
                    if np.linalg.norm(alternative.fun) < np.linalg.norm(fit.fun):
                        fit = alternative
                actual = kin.forward(fit.x)
                row['bounded'] = {
                    'position_error_m': float(np.linalg.norm(np.array(actual.position_m)-xyz)),
                    'orientation_error_rad': quaternion_angle(actual.quaternion_wxyz,target.quaternion_wxyz),
                    'max_seed_delta_rad': float(np.max(np.abs(fit.x-seed))),
                    'q': fit.x.tolist(),
                }
            print(json.dumps(row), flush=True)
    print('DONE: IK feasibility only; no collision/path safety is established.', flush=True)


if __name__ == '__main__':
    main()
