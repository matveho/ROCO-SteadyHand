"""Exercise sequencing and failure logs only. No physics or robot control."""

from .sessions import append_event, write_json
from .skills import PHASES


def dry_run(folder, robot, part, fail_at=None):
    if fail_at is not None and fail_at not in PHASES:
        raise ValueError(f"Unknown failure phase: {fail_at}")
    failed_phase = None
    for phase in PHASES:
        append_event(folder, mode="dry_run", robot=robot, part=part,
                     phase=phase, event="started")
        if phase == fail_at:
            failed_phase = phase
            append_event(folder, mode="dry_run", robot=robot, part=part,
                         phase=phase, event="mock_failure")
            append_event(folder, mode="dry_run", robot=robot, part=part,
                         phase="stop", event="mock_stop")
            break
        append_event(folder, mode="dry_run", robot=robot, part=part,
                     phase=phase, event="mock_completed")
    result = {
        "mode": "dry_run", "robot": robot, "part": part,
        "mock_sequence_completed": failed_phase is None,
        "failed_phase": failed_phase,
        "physical_success": None,
        "note": "No hardware was contacted. No physics or grasp quality was tested.",
    }
    write_json(folder / "result.json", result)
    return result
