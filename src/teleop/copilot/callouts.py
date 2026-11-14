"""Live call-outs: one short spoken line per console event, from the fast model or a template.

For an event worth a model call (:data:`~teleop.copilot.prompts.MODEL_EVENTS`):

1. Ask the fast model (Gemini 4 Hydrogen) on a worker thread, with the bay's telemetry and its
   latest camera frame (downscaled to at most 512 px; none with ``no_images``).
2. If the model has not answered within ``template_after_s`` (1 s), speak the deterministic
   template at once, so the operator always hears something quickly.
3. When the model answers in time, its line is spoken; if the template was already queued, the
   model's text replaces it in the queue. If the template is already being spoken, the model's
   version is only printed and logged.
4. ``hard_timeout_s`` (8 s) is a wall-clock limit: an answer after it is dropped.

At most one model request is in flight per bay. A newer event for a bay supersedes the older one:
its pending answer is dropped and its lines that are still waiting to be spoken are cancelled.
While the older request is still running, the newer event gets the template.

Other events (clutch flips, ``validating``, ``autonomous_started``) are templated straight away.
Advisory only: nothing here talks back to the console.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TextIO

from teleop.copilot.events import CopilotEvent
from teleop.copilot.frames import StoredFrame, downscale_jpeg
from teleop.copilot.gemini import GeminiError, Part, ThinkingLevel
from teleop.copilot.prompts import (
    CALLOUT_SCHEMA,
    CALLOUT_SYSTEM,
    MODEL_EVENTS,
    CalloutText,
    Severity,
    callout_parts,
    parse_callout,
    template_callout,
)
from teleop.copilot.speech import Spoken

logger = logging.getLogger(__name__)

TEMPLATE_AFTER_S = 1.0
HARD_TIMEOUT_S = 8.0
MAX_WORKERS = 4

Clock = Callable[[], int]


class JsonModel(Protocol):
    """What the service needs from :class:`~teleop.copilot.gemini.GeminiClient`."""

    def generate_json(
        self,
        model: str,
        system: str,
        parts: Sequence[Part],
        schema: Mapping[str, Any] | None,
        timeout_s: float,
        *,
        thinking_level: ThinkingLevel | None = None,
    ) -> dict[str, Any]: ...


class SpeechQueue(Protocol):
    """What the service needs from :class:`~teleop.copilot.speech.Speaker`."""

    def say(self, text: str, *, severity: Severity = "info", bay: int | None = None) -> int: ...

    def replace(self, token: int, text: str, severity: Severity | None = None) -> bool: ...

    def cancel_bay(self, bay: int) -> int: ...


@dataclass(frozen=True, slots=True)
class Callout:
    """One call-out as shown, spoken and logged."""

    bay: int
    event: str
    say: str
    detail: str
    severity: Severity
    source: str  # "model" or "template"
    model: str | None = None
    latency_ms: float | None = None
    frame_sha256: str | None = None
    spoken: bool = True  # False for a model refinement of a template already being spoken
    note: str = ""


@dataclass(slots=True)
class _Request:
    event: CopilotEvent
    template: CalloutText
    frame: StoredFrame | None
    started_ns: int
    done: bool = False
    template_emitted: bool = False
    template_token: int | None = None
    timers: list[threading.Timer] = field(default_factory=list[threading.Timer])

    def stop_timers(self) -> None:
        for timer in self.timers:
            timer.cancel()


class CalloutService:
    """Turns :class:`CopilotEvent` objects into call-outs. :meth:`handle` never blocks."""

    def __init__(
        self,
        client: JsonModel | None,
        *,
        model: str,
        sink: Callable[[Callout], None],
        speaker: SpeechQueue | None = None,
        frame_for: Callable[[int], StoredFrame | None] | None = None,
        thinking_level: ThinkingLevel | None = "low",
        no_images: bool = False,
        template_after_s: float = TEMPLATE_AFTER_S,
        hard_timeout_s: float = HARD_TIMEOUT_S,
        clock: Clock = time.monotonic_ns,
    ) -> None:
        self._client = client
        self._model = model
        self._sink = sink
        self._speaker = speaker
        self._frame_for = frame_for
        self._thinking: ThinkingLevel | None = thinking_level
        self._no_images = no_images
        self._template_after_s = template_after_s
        self._hard_timeout_s = hard_timeout_s
        self._clock = clock
        self._lock = threading.RLock()
        self._current: dict[int, _Request] = {}
        self._busy: set[int] = set()
        self._pool = ThreadPoolExecutor(MAX_WORKERS, thread_name_prefix="copilot-callout")
        self._closed = False

    def in_flight(self, bay: int) -> bool:
        with self._lock:
            return bay in self._busy

    def handle(self, event: CopilotEvent) -> None:
        bay = event.bay
        template = template_callout(event)
        frame = self._frame_for(bay) if self._frame_for is not None else None
        sha = frame.sha256 if frame is not None else None
        with self._lock:
            if self._closed:
                return
            uses_model = self._client is not None and event.kind in MODEL_EVENTS
            if uses_model:  # a newer model event supersedes the older one's pending answer;
                # a templated event (say a clutch flip right after a stall) lets it finish
                old = self._current.pop(bay, None)
                if old is not None and not old.done:
                    old.done = True
                    old.stop_timers()
            if self._speaker is not None:
                self._speaker.cancel_bay(bay)  # a newer event supersedes unspoken lines
            if not uses_model:
                self._emit_template(event, template, sha, "")
                return
            if bay in self._busy:
                self._emit_template(event, template, sha, "previous model request still running")
                return
            req = _Request(event, template, frame, self._clock())
            self._current[bay] = req
            self._busy.add(bay)
            req.timers = [
                threading.Timer(self._template_after_s, self._template_due, (req,)),
                threading.Timer(self._hard_timeout_s, self._hard_timeout, (req,)),
            ]
            for timer in req.timers:
                timer.daemon = True
                timer.start()
            self._pool.submit(self._work, req)

    # -- worker and timers ------------------------------------------------------------------

    def _work(self, req: _Request) -> None:
        assert self._client is not None
        try:
            jpeg: bytes | None = None
            if req.frame is not None and not self._no_images:
                jpeg = downscale_jpeg(req.frame.jpeg)
            answer = self._client.generate_json(
                self._model,
                CALLOUT_SYSTEM,
                callout_parts(req.event, jpeg),
                CALLOUT_SCHEMA,
                self._hard_timeout_s,
                thinking_level=self._thinking,
            )
            text = parse_callout(answer, req.template)
        except (GeminiError, ValueError, OSError) as exc:
            self._finish(req, None, str(exc))
        except Exception as exc:  # a call-out must never take the copilot down
            logger.exception("copilot call-out failed")
            self._finish(req, None, f"{type(exc).__name__}: {exc}")
        else:
            self._finish(req, text, "")
        finally:
            with self._lock:
                self._busy.discard(req.event.bay)

    def _finish(self, req: _Request, text: CalloutText | None, error: str) -> None:
        latency_ms = (self._clock() - req.started_ns) / 1e6
        with self._lock:
            if req.done:
                logger.info(
                    "copilot: dropped a late or superseded answer for bay %d after %.0f ms",
                    req.event.bay,
                    latency_ms,
                )
                return
            req.done = True
            req.stop_timers()
            sha = req.frame.sha256 if req.frame is not None else None
            if text is None:
                if not req.template_emitted:
                    self._emit_template(req.event, req.template, sha, f"model failed: {error}")
                else:
                    logger.info("copilot: model failed after the template: %s", error)
                return
            callout = Callout(
                req.event.bay,
                req.event.kind.value,
                text.say,
                text.detail,
                text.severity,
                "model",
                self._model,
                latency_ms,
                sha,
            )
            if not req.template_emitted:
                self._emit(callout)
                return
            replaced = False
            if self._speaker is not None and req.template_token is not None:
                replaced = self._speaker.replace(req.template_token, text.say, text.severity)
            note = "replaced the queued template" if replaced else "template already spoken"
            self._sink(_with(callout, spoken=replaced, note=note))

    def _template_due(self, req: _Request) -> None:
        with self._lock:
            if req.done or req.template_emitted:
                return
            sha = req.frame.sha256 if req.frame is not None else None
            req.template_token = self._emit_template(
                req.event, req.template, sha, f"model slower than {self._template_after_s:g} s"
            )
            req.template_emitted = True

    def _hard_timeout(self, req: _Request) -> None:
        with self._lock:
            if req.done:
                return
            req.done = True
            req.stop_timers()
            if not req.template_emitted:
                sha = req.frame.sha256 if req.frame is not None else None
                note = f"model timed out after {self._hard_timeout_s:g} s"
                self._emit_template(req.event, req.template, sha, note)
            else:
                logger.info("copilot: model timed out for bay %d", req.event.bay)

    # -- output -----------------------------------------------------------------------------

    def _emit_template(
        self, event: CopilotEvent, text: CalloutText, sha: str | None, note: str
    ) -> int | None:
        callout = Callout(
            event.bay,
            event.kind.value,
            text.say,
            text.detail,
            text.severity,
            "template",
            frame_sha256=sha,
            note=note,
        )
        return self._emit(callout)

    def _emit(self, callout: Callout) -> int | None:
        self._sink(callout)
        if self._speaker is None:
            return None
        return self._speaker.say(callout.say, severity=callout.severity, bay=callout.bay)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for req in self._current.values():
                req.done = True
                req.stop_timers()
            self._current.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)


def _with(callout: Callout, **changes: Any) -> Callout:
    return Callout(**(asdict(callout) | changes))


# -- terminal and JSONL outputs -----------------------------------------------------------------


def format_line(callout: Callout, now: datetime | None = None) -> str:
    """``[12:01:07] BAY 1 ALARM      Bay 1 stalled on a flap jam. Rescue needed.``"""
    stamp = (now or datetime.now()).strftime("%H:%M:%S")
    head = f"[{stamp}] BAY {callout.bay} {callout.severity.upper():<9} "
    say = callout.say if callout.spoken else f"(update) {callout.say}"
    source = callout.source
    if callout.latency_ms is not None:
        source += f" {callout.latency_ms / 1000:.1f} s"
    tail = f"    {callout.detail}  [{source}{'; ' + callout.note if callout.note else ''}]"
    return head + say + "\n" + " " * 11 + tail.strip() + "\n"


class CopilotOutputs:
    """Prints call-outs and appends call-outs and events to ``<log_dir>/<date>.jsonl``.

    Records hold frame hashes and model names, never images or API keys. Thread-safe.
    """

    def __init__(self, out: TextIO, log_dir: Path | None) -> None:
        self._out = out
        self._log_dir = log_dir
        self._lock = threading.Lock()

    def log_path(self, now: datetime | None = None) -> Path | None:
        if self._log_dir is None:
            return None
        return self._log_dir / f"{(now or datetime.now()).strftime('%Y-%m-%d')}.jsonl"

    def callout(self, callout: Callout) -> None:
        now = datetime.now()
        with self._lock:
            self._out.write(format_line(callout, now))
            self._out.flush()
            self._append(
                {"type": "callout", "t": now.isoformat(timespec="milliseconds")} | asdict(callout),
                now,
            )

    def event(self, event: CopilotEvent, frame: StoredFrame | None) -> None:
        now = datetime.now()
        record: dict[str, Any] = {
            "type": "event",
            "t": now.isoformat(timespec="milliseconds"),
            "event": event.kind.value,
            "bay": event.bay,
            "sim_t": event.sim_t,
            "wall": event.wall,
            "from": event.previous_state,
            "to": event.state,
            "frame_sha256": frame.sha256 if frame is not None else None,
        }
        with self._lock:
            self._append(record, now)

    def spoken(self, spoken: Spoken) -> None:
        """Log how a line was spoken (Gemini TTS, system voice, dropped) and the TTS time."""
        now = datetime.now()
        record = {"type": "speech", "t": now.isoformat(timespec="milliseconds")} | asdict(spoken)
        with self._lock:
            self._append(record, now)

    def note(self, text: str) -> None:
        with self._lock:
            self._out.write(text + "\n")
            self._out.flush()

    def _append(self, record: dict[str, Any], now: datetime) -> None:
        path = self.log_path(now)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            logger.warning("copilot log write failed: %s", exc)
