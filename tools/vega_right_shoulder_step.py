"""Supervised right-shoulder preparation, not an automatic stow or IK bypass.

One positive R_arm_j1 step only. Other joints hold their measured targets.
Numerical FK screening is not collision checking: operator approval is required.
"""
import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steadyhand.adapters.vega import VegaAdapter
from steadyhand.config import load_bundle
from steadyhand.skill_config import load_vega_skills


def plan_step(kin, seed, delta, floor):
    if not math.isfinite(delta) or not 0 < delta <= 0.5:
        raise ValueError('Require positive shoulder step <= 0.5 rad')
    if len(seed) != 7 or not all(math.isfinite(v) for v in seed):
        raise ValueError('Require seven finite measured right joint positions')
    if not math.isfinite(floor):
        raise ValueError('Require finite provisional floor')
    goal = list(seed)
    goal[0] += delta
    samples = []
    for i in range(101):
        q = list(seed)
        q[0] += delta*i/100
        # FK rejects out-of-limit joint vectors.
        pose = kin.forward(q)
        if not all(math.isfinite(v) for v in pose.position_m):
            raise ValueError('Nonfinite FK along shoulder step')
        if pose.position_m[2] < floor + 0.30:
            raise ValueError('Shoulder preparation must remain at least 0.30 m above provisional floor')
        samples.append(pose.position_m)
    if any(b[2] < a[2]-1e-6 for a,b in zip(samples,samples[1:])):
        raise ValueError('Positive shoulder step would lower TCP; refusing this startup route')
    return goal, samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delta-rad', type=float, default=0.10)
    p.add_argument('--confirm-physical-motion', action='store_true')
    args = p.parse_args()
    if not args.confirm_physical_motion or not sys.stdin.isatty():
        p.error('Requires interactive terminal and --confirm-physical-motion; connection may home HEAD')
    if not math.isfinite(args.delta_rad) or not 0 < args.delta_rad <= 0.5:
        p.error('--delta-rad must be positive and <= 0.5')
    cfg = load_bundle('vega')['robot']
    if cfg['working_arm'] != 'right' or cfg['kinematics']['ee_frame'] != 'tip_r':
        p.error('Requires right arm / tip_r')
    cfg['allow_robot_init_head_motion'] = True
    cfg['auto_clear_software_estop_on_connect'] = True
    floor = float(load_vega_skills()['safety']['min_tcp_z_m'])
    robot = VegaAdapter(cfg)
    try:
        robot.connect()
        seed = robot._read_joint_positions()
        goal, samples = plan_step(robot._kinematics, seed, args.delta_rad, floor)
        print('RIGHT Q BEFORE =', list(seed), flush=True)
        print('RIGHT Q TARGET =', goal, flush=True)
        print('TIP START / END =', samples[0], samples[-1], flush=True)
        print('Sampled XYZ bounds =', [(min(v[i] for v in samples), max(v[i] for v in samples)) for i in range(3)], flush=True)
        print('Only R_arm_j1 changes. No gripper command. Left arm is not commanded.', flush=True)
        print('FK height check is NOT collision clearance. Confirm arm, claw and cable sweep is clear.', flush=True)
        if input('Press Enter for this one step; type anything to cancel: ').strip():
            print('Cancelled before arm motion')
            return 0
        fresh = robot._read_joint_positions()
        if max(abs(a-b) for a,b in zip(seed,fresh)) > 0.01:
            raise RuntimeError('Arm moved since preview; rerun rather than using stale goal')
        robot.move_joints(goal, speed_scale=0.45)
        print('RIGHT Q AFTER =', list(robot._read_joint_positions()), flush=True)
        print('TIP AFTER =', robot.get_tcp_pose(), flush=True)
        print('Completed one step; no automatic return or further motion.', flush=True)
    except BaseException:
        robot.stop()
        raise
    finally:
        robot.close()


if __name__ == '__main__':
    main()
