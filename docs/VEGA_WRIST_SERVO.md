# Vega wrist XY servo — competition path

The verified physical LEFT wrist camera is `wrist_b`. The next physical
milestone uses `tools/vega_wrist_fine_center.py`.

This is a fixed-height XY validation only:
- starts from an already-established safe low hover;
- accepts that coarse base-frame XY explicitly on the command line;
- never performs board/global navigation;
- uses only `wrist_b` for feedback;
- discards startup-black frames and rejects stale frame identities;
- calibrates a local 2x2 image Jacobian with reversible +X/+Y probes;
- returns to the reference pose between probes;
- applies bounded XY corrections only;
- never descends, connects the gripper, grips, releases or inserts.

## Preferred physical test

After the preceding board/part coarse-motion benchmark has placed the LEFT TCP
at the desired low hover, use the exact base XY printed by that successful run:

```bash
python3 tools/vega_wrist_fine_center.py \
  --coarse-xy X Y \
  --confirm-physical-motion
```

Replace `X Y` with the successful coarse target. The tool does **not** move to
that XY itself. It verifies that the measured TCP is already within 15 mm of
the supplied XY before allowing any probe.

For a known feature/part pixel in the initial wrist image, add:

```bash
  --feature U V
```

If omitted, the tracker chooses a textured feature near image center. That is
sufficient to validate the servo mechanics but does not prove part identity.

The convergence goal is always **image center**. This is deliberate. Image
center is **not** a jaw-alignment pixel and must not be reused as grasp
calibration. A jaw-alignment goal pixel will be taught separately before the
first battery descent.

## Bounds

Current defaults:
- allowed starting hover: 60–120 mm above the measured task TCP floor;
- +base-X and +base-Y probes: 8 mm;
- each probe returns to the original TCP before the next probe;
- correction gain: 0.65;
- maximum single correction: 10 mm;
- maximum local travel radius: 40 mm;
- maximum corrections: 6;
- convergence: 5 px;
- speed scale: 0.45.

The underlying servo uses measured TCP displacement, not requested displacement,
when fitting the 2x2 Jacobian. It stops on tracking ambiguity, stale feedback,
negligible probe image motion, ill-conditioned Jacobian, TCP Z/orientation
drift, waypoint error, local-radius escape, divergence or stalled convergence.

## Logging

Every run creates `runs/wrist_fine_<UTC>/` unless `--output` is supplied.
It records:
- `arguments.json`;
- `start_state.json` with supplied coarse XY and measured TCP;
- every accepted `wrist_b` RGB frame and metadata;
- `capture_events.jsonl`, including discarded startup-dark frames;
- `events.jsonl` with reference observations, measured probe/return TCPs,
  tracked feature pixels, fitted Jacobian, every correction request and every
  measured post-motion TCP;
- tracked PNG overlays;
- `result.json`.

Success ends with `WRIST_B FINE CENTER PASS`.

For the main agent, return the console lines for:
`START STATE`, `REFERENCE`, `PROBE_X`, both `RETURN_REFERENCE` events,
`PROBE_Y`, `CALIBRATED`, every `CORRECTION`/`MOTION`, and
`COMPLETE`, plus whether the physical motion looked like two small
perpendicular probes followed by smooth bounded centering.

## Existing general tool

`tools/vega_wrist_servo.py` remains useful for capture-only work and its older
combined coarse-navigation path. It is not the preferred next physical
milestone because that path can depend on a saved board registration and board
planner assumptions. The new fine-centering tool intentionally has no such
dependency.

Both tools ultimately use `steadyhand.vision.wrist_servo.run_xy_servo`.
That module now emits explicit post-motion measured TCP records in addition to
feature/Jacobian/correction events.

Offline validation:

```bash
python3 -m unittest discover -s tests -p test_wrist_servo.py -v
```

Synthetic validation is not evidence of physical success.
