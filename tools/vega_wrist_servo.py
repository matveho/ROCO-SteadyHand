"""Bounded, manual-feature wrist XY servo at an already-established hover pose.

Run on the robot in an environment containing both wrist_cameras and dexcontrol.
Images are saved as PPM and NPY. Enter the SAME feature's pixel coordinates
from each image. This validates the physical loop before adding a detector.
Image center is a viewing target, NOT a calibrated gripping-pad target.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadyhand.adapters.vega import VegaAdapter
from steadyhand.cameras.vega import VegaWristCameras
from steadyhand.config import load_bundle
from steadyhand.geometry import pose_distance
from steadyhand.models import Pose
from steadyhand.skill_config import load_vega_skills
from steadyhand.vision.wrist_servo import jacobian_from_measured_probes, pixel_error


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left-camera', required=True, choices=('wrist_a', 'wrist_b'),
                   help='Physically verified API label for the LEFT wrist')
    p.add_argument('--output', required=True, help='New directory for this attempt')
    p.add_argument('--probe-m', type=float, default=0.01)
    p.add_argument('--speed-scale', type=float, default=0.45)
    p.add_argument('--confirm-head-motion', action='store_true')
    p.add_argument('--confirm-physical-motion', action='store_true')
    args = p.parse_args(argv)
    if not args.confirm_head_motion or not args.confirm_physical_motion:
        p.error('Both motion confirmations are required; Robot() may home the head')
    if not math.isfinite(args.probe_m) or not 0.005 <= args.probe_m <= 0.015:
        p.error('--probe-m must be 5-15 mm')
    if not math.isfinite(args.speed_scale) or not 0 < args.speed_scale <= 1:
        p.error('--speed-scale must be in (0,1]')
    import numpy as np
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cameras = VegaWristCameras()
    robot = None
    records = []
    last_frame_id = None
    try:
        # Prove acquisition before Robot() can move the head.
        cameras.connect()
        cameras.read(timeout=5, fresh=True)
        cfg = load_bundle('vega')['robot']
        if cfg['working_arm'] != 'left' or cfg['kinematics']['ee_frame'] != 'tip_l':
            raise ValueError('This benchmark requires left arm and tip_l')
        cfg['allow_robot_init_head_motion'] = True
        robot = VegaAdapter(cfg)
        robot.prepare()
        robot.connect()
        origin = robot.get_tcp_pose()
        floor = float(load_vega_skills()['safety']['min_tcp_z_m'])
        if not floor + 0.03 <= origin.position_m[2] <= floor + 0.10:
            raise ValueError('First establish a hover pose 3-10 cm above the task floor')
        # Correct physical vertical family: tip_l +Z is up, gripper points down.
        from steadyhand.geometry import quaternion_to_matrix
        rotation = quaternion_to_matrix(origin.quaternion_wxyz)
        if float(rotation[2][2]) < math.cos(0.12):
            raise ValueError('First establish the corrected vertical tip_l orientation')

        def move_xy(dx, dy):
            if math.hypot(dx, dy) > 0.04:
                raise ValueError('Servo workspace radius exceeds 40 mm')
            target = Pose((origin.position_m[0]+dx, origin.position_m[1]+dy,
                           origin.position_m[2]), origin.quaternion_wxyz)
            robot.move_tcp(target, speed_scale=args.speed_scale)
            actual = robot.get_tcp_pose()
            distance, angle = pose_distance(actual, target)
            if distance > 0.004 or angle > 0.05:
                raise RuntimeError('Measured TCP did not reach the probe/correction')
            return actual

        def measure(label):
            nonlocal last_frame_id
            frame = getattr(cameras.read(timeout=5, fresh=True), args.left_camera)
            if frame.frame_id is None or frame.frame_id == last_frame_id:
                raise RuntimeError('Wrist frame ID missing or unchanged')
            last_frame_id = frame.frame_id
            rgb = np.asarray(frame.rgb)
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ValueError('Expected HxWx3 uint8 RGB wrist frame')
            np.save(output / f'{label}.npy', rgb)
            image = output / f'{label}.ppm'
            with image.open('wb') as f:
                f.write(f'P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n'.encode())
                f.write(rgb.tobytes())
            print(f'View {image.resolve()}; coordinates are u=column, v=row.', flush=True)
            uv = tuple(float(v) for v in input('Same stationary feature u v (Ctrl-C aborts): ').split())
            if len(uv) != 2 or not (0 <= uv[0] < rgb.shape[1] and 0 <= uv[1] < rgb.shape[0]):
                raise ValueError('Feature must be inside image')
            actual = robot.get_tcp_pose()
            records.append(dict(label=label, uv=uv, xyz=actual.position_m,
                                frame_id=frame.frame_id, timestamp_ns=frame.timestamp_ns))
            (output/'measurements.json').write_text(json.dumps(records, indent=2))
            return uv, actual.position_m[:2], rgb.shape

        input('Clear the 40 mm XY region around this hover pose; Enter begins probes: ')
        uv0, xy0, shape = measure('reference')
        move_xy(args.probe_m, 0)
        uvx, xyx, _ = measure('plus_x')
        move_xy(0, 0)
        move_xy(0, args.probe_m)
        uvy, xyy, _ = measure('plus_y')
        jacobian = jacobian_from_measured_probes([uv0, uvx, uvy], [xy0, xyx, xyy])
        (output/'jacobian.json').write_text(json.dumps(jacobian.matrix()))
        move_xy(0, 0)
        previous = None
        for step in range(6):
            uv, xy, shape = measure(f'correction_{step}')
            error = pixel_error(uv, shape)
            norm = math.hypot(*error)
            print(f'Pixel error = {norm:.2f}', flush=True)
            if norm <= 5:
                print('VIEW CENTERED. Gripping-pad alignment is not calibrated.')
                return 0
            if previous is not None and norm > previous * 1.25:
                raise RuntimeError('Pixel error increased; stop and inspect feature/Jacobian')
            if step == 5:
                raise RuntimeError('Correction budget exhausted without convergence')
            dx, dy = jacobian.base_delta_for_pixel_error(error, max_step_m=0.01)
            move_xy(xy[0]-origin.position_m[0]+dx, xy[1]-origin.position_m[1]+dy)
            previous = norm
    except BaseException:
        if robot is not None:
            try:
                robot.stop()
            except BaseException as exc:
                print(f'STOP FAILED: {exc}; use physical e-stop', file=sys.stderr)
        raise
    finally:
        try:
            cameras.close()
        finally:
            if robot is not None:
                robot.close()


if __name__ == '__main__':
    raise SystemExit(main())
