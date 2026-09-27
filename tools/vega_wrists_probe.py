"""Read one pair of Vega wrist-camera frames without arm motion.

Run ONBOARD with /usr/bin/python3. wrist_a/wrist_b are camera-link labels;
do not assume which physical side is left/right until verified onsite.
"""

def main():
    from wrist_cameras import WristCameras

    with WristCameras() as cameras:
        obs = cameras.get_obs(timeout=3, fresh=True)
        for key in ("wrist_a", "wrist_b"):
            record = obs[key]
            rgb = record["rgb"]
            print(
                key,
                "shape=", getattr(rgb, "shape", None),
                "dtype=", getattr(rgb, "dtype", None),
                "frame_id=", record.get("frame_id"),
                "timestamp_ns=", record.get("timestamp_ns"),
                "received_monotonic_ns=", record.get("received_monotonic_ns"),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
