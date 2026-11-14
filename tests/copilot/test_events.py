"""T-211: the event detector, replayed over real console telemetry from the scripted stage."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from teleop.copilot.events import CopilotEvent, EventDetector, EventKind

FIXTURE = Path(__file__).parent / "fixtures" / "stage_telemetry.jsonl"


def stage() -> list[dict[str, Any]]:
    return [json.loads(line) for line in FIXTURE.read_text().splitlines() if line]


def run(messages: list[dict[str, Any]], det: EventDetector | None = None) -> list[CopilotEvent]:
    det = det or EventDetector()
    return [e for m in messages for e in det.feed(m)]


def summary(events: list[CopilotEvent]) -> list[tuple[str, int]]:
    return [(e.kind.value, e.bay) for e in events]


STAGE_EVENTS = [
    ("stall", 1),  # beat 2: flap jam, Bay 1 goes straight to RESCUE
    ("rescue_started", 1),  # beat 3: clutch engaged
    ("rescue_released", 1),
    ("validating", 1),  # beat 4: edge fit and sim gate
    ("gate_result", 1),
    ("autonomous_started", 2),  # beat 5: Bay 2 reuses the skill
    ("autonomous_finished", 2),
    ("refused", 2),  # beat 6: pinned bolt is out of distribution
]


def test_scripted_stage_gives_the_six_beats_in_order() -> None:
    events = run(stage())
    assert summary(events) == STAGE_EVENTS
    stall = events[0]
    assert (stall.previous_state, stall.state) == ("NOMINAL", "RESCUE")
    assert stall.telemetry["bay"]["id"] == 1 and stall.telemetry["mode"] == "rescue"
    assert stall.telemetry["bay"]["confidence"] < stall.telemetry["threshold"]
    gate = events[4]
    assert gate.gate_ok is True and gate.telemetry["gate"]["passed"] == 20
    assert [e.sim_t for e in events] == sorted(e.sim_t for e in events)
    assert all(e.wall for e in events)


def test_the_first_message_is_only_a_baseline() -> None:
    msgs = stage()
    mid_rescue = next(i for i, m in enumerate(msgs) if m["mode"] == "rescue")
    events = run(msgs[mid_rescue:])  # connect while Bay 1 is already waiting for rescue
    assert ("stall", 1) not in summary(events)
    assert summary(events)[0] == ("rescue_started", 1)


def test_a_reconnect_does_not_repeat_events() -> None:
    msgs = stage()
    det = EventDetector()
    first = run(msgs[:40], det)
    # On reconnect the console first resends its latest state, then carries on.
    again = run([msgs[39], msgs[39], *msgs[40:]], det)
    assert summary(first + again) == STAGE_EVENTS


def test_a_console_restart_sets_a_new_baseline() -> None:
    msgs = stage()
    det = EventDetector()
    run(msgs, det)
    restarted = run(msgs, det)  # sim_t jumps back to 0
    assert summary(restarted) == STAGE_EVENTS


def test_identical_messages_fire_nothing() -> None:
    msg = stage()[10]
    det = EventDetector()
    assert run([msg] * 20, det) == []


def _with_state(msg: dict[str, Any], bay: int, state: str, sim_t: float) -> dict[str, Any]:
    out = copy.deepcopy(msg)
    out["sim_t"] = sim_t
    for entry in out["bays"]:
        if entry["id"] == bay:
            entry["state"] = state
    return out


def test_replayed_bays_never_fire() -> None:
    base = stage()[0]
    msg = copy.deepcopy(base)
    for entry in msg["bays"]:
        if entry["id"] == 3:
            entry["replayed"] = True
    events = run([msg, _with_state(msg, 3, "STALLED", 1.0)])
    assert events == []


def test_stall_states_and_skill_jams() -> None:
    base = stage()[0]
    seq = [
        base,
        _with_state(base, 1, "STALLED", 1.0),
        _with_state(base, 1, "QUEUED", 1.1),  # same incident: no second stall
        _with_state(base, 1, "RESCUE", 1.2),
        _with_state(base, 1, "NOMINAL", 2.0),
        _with_state(base, 1, "AUTONOMOUS_SKILL", 3.0),
        _with_state(base, 1, "STALLED", 4.0),  # the reused skill jammed too
    ]
    assert summary(run(seq)) == [
        ("stall", 1),
        ("autonomous_started", 1),
        ("autonomous_finished", 1),
        ("stall", 1),
    ]


def test_a_refused_bay_moving_to_rescue_is_not_a_new_stall() -> None:
    base = stage()[0]
    seq = [
        base,
        _with_state(base, 2, "OOD_ESCALATED", 1.0),
        _with_state(base, 2, "RESCUE", 2.0),
    ]
    assert summary(run(seq)) == [("refused", 2)]


def test_a_second_gate_result_fires_again() -> None:
    msgs = stage()
    gate_msg = next(m for m in msgs if (m.get("gate") or {}).get("ok") is True)
    det = EventDetector()
    run(msgs, det)
    failing = copy.deepcopy(gate_msg)
    failing["sim_t"] = msgs[-1]["sim_t"] + 1.0
    failing["gate"] = None
    validating = _with_state(failing, 1, "VALIDATING", failing["sim_t"] + 1.0)
    failed = copy.deepcopy(validating)
    failed["sim_t"] += 5.0
    failed["gate"] = dict(gate_msg["gate"], ok=False, passed=12, fit_ms=9.9)
    events = run([failing, validating, failed], det)
    assert summary(events)[-1] == ("gate_result", 1)
    assert events[-1].gate_ok is False


def test_event_kinds_are_strings() -> None:
    assert {k.value for k in EventKind} == {
        "stall",
        "refused",
        "rescue_started",
        "rescue_released",
        "validating",
        "gate_result",
        "autonomous_started",
        "autonomous_finished",
    }
