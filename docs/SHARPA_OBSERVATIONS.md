# North observations

The Sharpa adapter can subscribe to the organizer's `north_observation` topic.
It creates no command publishers or service queries and does not instantiate
the vendor's action-capable `NorthClient`. All motion and hardware-stop methods
remain unavailable. Closing this observer only disconnects its subscriber.

Use the organizer-provided SDK and its Zenoh/protobuf dependencies:

```bash
python tools/sharpa_observe.py \
  --sdk-root /path/to/sharpa_north_ces_lite_sdk-main \
  --endpoint tcp/ROBOT_HOST:7449 \
  --duration 3
```

Add `--check-sdk` to encode and decode the current arm/body values locally using
the installed SDK serializer. This requires NumPy as well as Zenoh/protobuf.
It creates no action client or publisher and reports `published: false`.
See [the inference integration path](SHARPA_INFERENCE.md) for the command
contract, controller stop limitations and remaining hardware integration.

The command prints a bounded readiness summary. It does not save images, joint
values, passwords, configuration, or trial logs. North's own control application
and camera service must already be running.

The 65 positions follow the organizer's protocol order: left arm (7), left hand
(22), right arm (7), right hand (22), then body/neck (7). Body motor IDs must be
1 through 7 in order. Native joint names are preserved in `extras` for later
URDF mapping verification; dimension checks do not establish that motion mapping.
Arm error codes and body motor error/fault flags are preserved in `extras.faults`;
the observer reports faults rather than pretending to be a motion-ready gate.

All four cameras are required. `RobotObservation.cameras` contains immutable
`CameraFrame` values with encoded image bytes, format and source stamp. Decode
and apply the organizer's RGB preprocessing before using them as policy input.
Optional fingertip force measurements remain absent if the source omits them;
the adapter does not manufacture zero readings.

Readiness requires at least two advancing source stamps for the aggregate,
each joint group, mode, each camera, and each present tactile force sensor.
Both local arrival age and time since source-stamp advancement must remain below
`max_age_s` (default 0.5 seconds). A fresh aggregate containing a frozen component
fails the check. Source clocks are preserved and are not subtracted from the
workstation clock. This does not provide camera/joint time synchronization or
detect a driver that stamps cached hardware data as new.

Passing this check establishes observation transport and structure only. It does
not validate robot motion, e-stop behavior, hand commands, joint limits, image
calibration, camera-to-robot transforms, or the board position. These must be
validated before implementing the command side of the adapter.
