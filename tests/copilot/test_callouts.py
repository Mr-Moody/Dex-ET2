"""T-212: call-outs from a fake Gemini (fast, slow, failing), templates and outputs."""

from __future__ import annotations

import io
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from teleop.copilot.callouts import Callout, CalloutService, CopilotOutputs, format_line
from teleop.copilot.events import CopilotEvent, EventDetector, EventKind
from teleop.copilot.frames import StoredFrame
from teleop.copilot.gemini import GeminiHttpError, GeminiTimeoutError, ImagePart, Part, TextPart
from teleop.copilot.prompts import (
    CALLOUT_SCHEMA,
    SAY_MAX_WORDS,
    CalloutText,
    callout_parts,
    parse_callout,
    template_callout,
)
from teleop.copilot.speech import Speaker

FIXTURES = Path(__file__).parent / "fixtures"
JPEG = (FIXTURES / "frame-0-1.bin").read_bytes()[2:]
ANSWER = {"say": "Bay 1 flap jammed. Rescue needed.", "detail": "C=0.24.", "severity": "alarm"}


def stage_events() -> list[CopilotEvent]:
    det = EventDetector()
    lines = (FIXTURES / "stage_telemetry.jsonl").read_text().splitlines()
    return [e for line in lines if line for e in det.feed(json.loads(line))]


def first(kind: EventKind) -> CopilotEvent:
    return next(e for e in stage_events() if e.kind is kind)


def wait_for(cond: Callable[[], bool], timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not cond():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.005)


class FakeModel:
    """A fake ``generate_json``: answers at once, after ``gate`` is set, or raises."""

    def __init__(
        self,
        answer: dict[str, Any] | None = None,
        error: Exception | None = None,
        gate: threading.Event | None = None,
    ) -> None:
        self.answer = answer if answer is not None else ANSWER
        self.error = error
        self.gate = gate
        self.calls: list[tuple[str, list[Part], Mapping[str, Any] | None, float, Any]] = []
        self.finished = threading.Event()

    def generate_json(
        self,
        model: str,
        system: str,
        parts: Sequence[Part],
        schema: Mapping[str, Any] | None,
        timeout_s: float,
        *,
        thinking_level: Any = None,
    ) -> dict[str, Any]:
        self.calls.append((model, list(parts), schema, timeout_s, thinking_level))
        try:
            if self.gate is not None:
                self.gate.wait(5)
            if self.error is not None:
                raise self.error
            return self.answer
        finally:
            self.finished.set()


class FakeSpeaker:
    def __init__(self) -> None:
        self.queue: dict[int, str] = {}
        self.said: list[tuple[str, str, int | None]] = []
        self.cancelled: list[int] = []
        self.speaking: set[int] = set()  # tokens "already being spoken"
        self._next = 1

    def say(self, text: str, *, severity: str = "info", bay: int | None = None) -> int:
        token = self._next
        self._next += 1
        self.said.append((text, severity, bay))
        self.queue[token] = text
        return token

    def replace(self, token: int, text: str, severity: Any = None) -> bool:
        if token in self.speaking or token not in self.queue:
            return False
        self.queue[token] = text
        return True

    def cancel_bay(self, bay: int) -> int:
        self.cancelled.append(bay)
        return 0


def frame(bay: int = 1) -> StoredFrame:
    return StoredFrame(bay, f"bay{bay}/cctv", JPEG, 1, 0.0)


def service(
    model: FakeModel | None,
    speaker: FakeSpeaker | None = None,
    **kwargs: Any,
) -> tuple[CalloutService, list[Callout]]:
    out: list[Callout] = []
    svc = CalloutService(
        model,
        model="gemini-4-hydrogen",
        sink=out.append,
        speaker=speaker,
        frame_for=lambda bay: frame(bay),
        **kwargs,
    )
    return svc, out


# -- templates and prompts ------------------------------------------------------------------------


def test_every_stage_event_has_a_short_template() -> None:
    events = stage_events()
    lines = {e.kind: template_callout(e) for e in events}
    assert set(lines) == set(EventKind)
    assert lines[EventKind.STALL].say == "Bay 1 stalled on a flap jam. Rescue needed."
    assert lines[EventKind.STALL].severity == "alarm"
    assert lines[EventKind.GATE_RESULT].say == "Bay 1 skill passed 20 of 20. Shared with the fleet."
    assert lines[EventKind.REFUSED].severity == "alarm"
    for text in lines.values():
        assert len(text.say.split()) <= SAY_MAX_WORDS
        assert text.detail


def test_failed_gate_and_unfinished_skill_ask_for_attention() -> None:
    gate = first(EventKind.GATE_RESULT)
    failed = CopilotEvent(
        gate.kind,
        gate.bay,
        gate.sim_t,
        gate.wall,
        "VALIDATING",
        "NOMINAL",
        gate.telemetry | {"gate": {"ok": False, "passed": 12, "total": 20}},
    )
    assert template_callout(failed).say == "Bay 1 skill failed the gate 12 of 20. Not shared."
    assert template_callout(failed).severity == "attention"
    fin = first(EventKind.AUTONOMOUS_FINISHED)
    stuck = CopilotEvent(fin.kind, fin.bay, 0.0, "", "AUTONOMOUS_SKILL", "RESCUE", fin.telemetry)
    assert template_callout(stuck).severity == "attention"


def test_callout_parts_carry_the_event_telemetry_and_one_frame() -> None:
    stall = first(EventKind.STALL)
    parts = callout_parts(stall, JPEG)
    assert isinstance(parts[0], TextPart) and isinstance(parts[1], ImagePart)
    assert "stall on bay 1" in parts[0].text and '"confidence"' in parts[0].text
    assert len(callout_parts(stall, None)) == 1


def test_parse_callout_clips_and_validates() -> None:
    fallback = CalloutText("t", "d", "alarm")
    long = " ".join(["word"] * 30)
    parsed = parse_callout({"say": long, "severity": "loud"}, fallback)
    assert len(parsed.say.split()) == SAY_MAX_WORDS
    assert parsed.severity == "alarm" and parsed.detail == "d"
    with pytest.raises(ValueError):
        parse_callout({"detail": "no say"}, fallback)


# -- the service ----------------------------------------------------------------------------------


def test_a_fast_answer_is_spoken_and_the_template_never_is() -> None:
    model, speaker = FakeModel(), FakeSpeaker()
    svc, out = service(model, speaker)
    svc.handle(first(EventKind.STALL))
    wait_for(lambda: len(out) == 1)
    assert out[0].source == "model" and out[0].say == ANSWER["say"]
    assert out[0].model == "gemini-4-hydrogen" and out[0].latency_ms is not None
    assert out[0].frame_sha256 == frame().sha256
    assert speaker.said == [(ANSWER["say"], "alarm", 1)]
    name, parts, schema, timeout_s, thinking = model.calls[0]
    assert name == "gemini-4-hydrogen" and schema == CALLOUT_SCHEMA
    assert timeout_s == 8.0 and thinking == "low"
    assert any(isinstance(p, ImagePart) for p in parts)
    wait_for(lambda: not svc.in_flight(1))
    svc.close()


def test_a_slow_answer_speaks_the_template_first_then_replaces_it() -> None:
    gate = threading.Event()
    model, speaker = FakeModel(gate=gate), FakeSpeaker()
    svc, out = service(model, speaker, template_after_s=0.05)
    svc.handle(first(EventKind.STALL))
    wait_for(lambda: len(out) == 1)
    assert out[0].source == "template" and "slower than" in out[0].note
    assert speaker.said == [("Bay 1 stalled on a flap jam. Rescue needed.", "alarm", 1)]
    gate.set()
    wait_for(lambda: len(out) == 2)
    assert out[1].source == "model" and out[1].spoken is True
    assert speaker.queue[1] == ANSWER["say"]  # swapped in the queue, not said twice
    assert len(speaker.said) == 1
    svc.close()


def test_a_slow_answer_after_the_template_was_spoken_is_only_printed() -> None:
    gate = threading.Event()
    model, speaker = FakeModel(gate=gate), FakeSpeaker()
    svc, out = service(model, speaker, template_after_s=0.05)
    svc.handle(first(EventKind.STALL))
    wait_for(lambda: len(out) == 1)
    speaker.speaking.add(1)
    gate.set()
    wait_for(lambda: len(out) == 2)
    assert out[1].spoken is False and out[1].note == "template already spoken"
    assert format_line(out[1]).count("(update)") == 1
    svc.close()


@pytest.mark.parametrize(
    "error",
    [GeminiHttpError(429, "rate limited"), GeminiTimeoutError("no answer"), ValueError("bad")],
)
def test_model_errors_speak_the_template_at_once(error: Exception) -> None:
    model, speaker = FakeModel(error=error), FakeSpeaker()
    svc, out = service(model, speaker, template_after_s=5.0)
    svc.handle(first(EventKind.STALL))
    wait_for(lambda: len(out) == 1)
    assert out[0].source == "template" and out[0].note.startswith("model failed")
    assert len(speaker.said) == 1
    svc.close()


def test_a_malformed_answer_falls_back_to_the_template() -> None:
    model = FakeModel(answer={"severity": "alarm"})
    svc, out = service(model, FakeSpeaker())
    svc.handle(first(EventKind.STALL))
    wait_for(lambda: len(out) == 1)
    assert out[0].source == "template"
    svc.close()


def test_the_hard_timeout_drops_a_late_answer() -> None:
    gate = threading.Event()
    model, speaker = FakeModel(gate=gate), FakeSpeaker()
    svc, out = service(model, speaker, template_after_s=5.0, hard_timeout_s=0.05)
    svc.handle(first(EventKind.STALL))
    wait_for(lambda: len(out) == 1)
    assert out[0].source == "template" and "timed out" in out[0].note
    assert svc.in_flight(1)  # the request is still running
    gate.set()
    wait_for(lambda: not svc.in_flight(1))
    assert len(out) == 1 and len(speaker.said) == 1
    svc.close()


def test_one_request_per_bay_and_newer_events_supersede() -> None:
    gate = threading.Event()
    model, speaker = FakeModel(gate=gate), FakeSpeaker()
    svc, out = service(model, speaker, template_after_s=5.0)
    stall = first(EventKind.STALL)
    svc.handle(stall)
    wait_for(lambda: len(model.calls) == 1)
    svc.handle(stall)  # bay 1 is still waiting on the model: template now, no second request
    assert len(model.calls) == 1
    assert out[-1].source == "template" and "still running" in out[-1].note
    gate.set()
    wait_for(lambda: not svc.in_flight(1))
    assert len(out) == 1  # the superseded answer is dropped
    assert speaker.cancelled == [1, 1]
    svc.close()


def test_other_bays_are_independent() -> None:
    gate = threading.Event()
    model = FakeModel(gate=gate)
    svc, _ = service(model, FakeSpeaker(), template_after_s=5.0)
    stall = first(EventKind.STALL)
    other = CopilotEvent(
        stall.kind, 2, stall.sim_t, stall.wall, "NOMINAL", "RESCUE", stall.telemetry
    )
    svc.handle(stall)
    svc.handle(other)
    wait_for(lambda: len(model.calls) == 2)
    gate.set()
    svc.close()


def test_templated_events_never_call_the_model() -> None:
    model, speaker = FakeModel(), FakeSpeaker()
    svc, out = service(model, speaker)
    for kind in (EventKind.RESCUE_STARTED, EventKind.VALIDATING, EventKind.AUTONOMOUS_STARTED):
        svc.handle(first(kind))
    assert model.calls == [] and [c.source for c in out] == ["template"] * 3
    svc.close()


def test_without_a_client_every_event_is_a_template() -> None:
    svc, out = service(None, FakeSpeaker())
    for event in stage_events():
        svc.handle(event)
    assert len(out) == len(stage_events()) and {c.source for c in out} == {"template"}


def test_no_images_sends_zero_image_parts() -> None:
    model = FakeModel()
    svc, out = service(model, FakeSpeaker(), no_images=True)
    svc.handle(first(EventKind.STALL))
    wait_for(lambda: len(out) == 1)
    assert not any(isinstance(p, ImagePart) for p in model.calls[0][1])
    svc.close()


def test_the_stage_with_a_real_speaker_queue() -> None:
    """Every stage event reaches the real speaker; alarms come first in its queue."""
    speaker = Speaker(None, tts_model="m", voice="v", start=False, max_waiting=10)
    svc, out = service(None, speaker)  # type: ignore[arg-type]
    for event in stage_events():
        svc.handle(event)
    waiting = speaker.waiting()
    assert waiting[0].startswith("Bay 1 stalled") or waiting[0].startswith("Bay 2 refused")
    assert len(out) == len(stage_events())
    speaker.close()


# -- outputs --------------------------------------------------------------------------------------


def test_outputs_print_a_terminal_line_and_log_jsonl(tmp_path: Path) -> None:
    buf = io.StringIO()
    outputs = CopilotOutputs(buf, tmp_path)
    stall = first(EventKind.STALL)
    outputs.event(stall, frame())
    callout = Callout(1, "stall", "Bay 1 stalled.", "C=0.24", "alarm", "template")
    outputs.callout(callout)
    line = buf.getvalue().splitlines()[0]
    assert "BAY 1 ALARM" in line and line.endswith("Bay 1 stalled.")
    log = outputs.log_path()
    assert log is not None and log.parent == tmp_path
    records = [json.loads(r) for r in log.read_text(encoding="utf-8").splitlines()]
    assert [r["type"] for r in records] == ["event", "callout"]
    assert records[0]["frame_sha256"] == frame().sha256
    raw = log.read_text(encoding="utf-8")
    assert "jpeg" not in raw and "/9j/" not in raw  # hashes only, never image bytes


def test_format_line_matches_the_plan() -> None:
    callout = Callout(
        1, "stall", "Bay 1 stalled on a flap jam. Rescue needed.", "d", "alarm", "template"
    )
    line = format_line(callout, datetime(2026, 11, 14, 12, 1, 7)).splitlines()[0]
    assert line == "[12:01:07] BAY 1 ALARM     Bay 1 stalled on a flap jam. Rescue needed."


def test_a_templated_event_does_not_drop_a_pending_model_answer() -> None:
    """Live run: "clutch engaged" ~1 s after a stall used to discard the stall's model line."""
    gate = threading.Event()
    model, speaker = FakeModel(gate=gate), FakeSpeaker()
    svc, out = service(model, speaker, template_after_s=0.05)
    svc.handle(first(EventKind.STALL))
    wait_for(lambda: len(out) == 1)
    svc.handle(first(EventKind.RESCUE_STARTED))
    gate.set()
    wait_for(lambda: len(out) == 3)
    assert [c.source for c in out] == ["template", "template", "model"]
    assert out[2].event == "stall"
    svc.close()


def test_speech_outcomes_are_logged(tmp_path: Path) -> None:
    from teleop.copilot.speech import Spoken

    outputs = CopilotOutputs(io.StringIO(), tmp_path)
    outputs.spoken(Spoken(3, "Bay 1 stalled.", "system", None, "TTS slower than 4 s"))
    log = outputs.log_path()
    assert log is not None
    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["type"] == "speech" and record["route"] == "system"


def test_a_broken_terminal_does_not_stop_the_log(tmp_path: Path) -> None:
    class Broken(io.StringIO):
        def write(self, s: str) -> int:
            raise BrokenPipeError("terminal closed")

    outputs = CopilotOutputs(Broken(), tmp_path)
    outputs.callout(Callout(1, "stall", "Bay 1 stalled.", "d", "alarm", "template"))
    log = outputs.log_path()
    assert log is not None and '"callout"' in log.read_text(encoding="utf-8")
