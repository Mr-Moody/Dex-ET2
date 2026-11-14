"""Turns successive console telemetry messages into copilot events. Pure: no I/O, no clock.

| Event                 | Trigger                                                              |
|-----------------------|----------------------------------------------------------------------|
| ``stall``             | a bay enters ``STALLED``, ``QUEUED`` or ``RESCUE`` from normal work   |
| ``refused``           | a bay enters ``OOD_ESCALATED``                                       |
| ``rescue_started``    | the rescue's clutch is engaged                                       |
| ``rescue_released``   | the rescue's clutch is released                                      |
| ``validating``        | a bay enters ``VALIDATING``                                          |
| ``gate_result``       | a skill gate finishes (``gate.ok`` becomes true or false)            |
| ``autonomous_started``| a bay starts reusing a fleet skill (``AUTONOMOUS_SKILL``)            |
| ``autonomous_finished``| that bay leaves ``AUTONOMOUS_SKILL``                                |

"Normal work" is ``NOMINAL`` or ``AUTONOMOUS_SKILL``, so a stall in the middle of a rescue or a
refusal is not reported twice. The first message only sets the baseline: connecting (or
reconnecting) while a bay is already stalled fires nothing. If ``sim_t`` jumps backwards the
console restarted, and the next message sets a new baseline. Replayed background bays never
fire events.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

STALL_STATES = frozenset({"STALLED", "QUEUED", "RESCUE"})
WORKING_STATES = frozenset({"NOMINAL", "AUTONOMOUS_SKILL"})
SHARED_FIELDS = ("mode", "queue", "selected_bay", "rescue", "gate", "bolt_clear_cm", "threshold")


class EventKind(StrEnum):
    STALL = "stall"
    REFUSED = "refused"
    RESCUE_STARTED = "rescue_started"
    RESCUE_RELEASED = "rescue_released"
    VALIDATING = "validating"
    GATE_RESULT = "gate_result"
    AUTONOMOUS_STARTED = "autonomous_started"
    AUTONOMOUS_FINISHED = "autonomous_finished"


@dataclass(frozen=True, slots=True)
class CopilotEvent:
    """One event for one bay, with the telemetry needed to explain it."""

    kind: EventKind
    bay: int
    sim_t: float
    wall: str
    previous_state: str | None
    state: str | None
    telemetry: dict[str, Any] = field(repr=False)  # that bay's entry plus the shared fields

    @property
    def gate_ok(self) -> bool | None:
        gate = self.telemetry.get("gate")
        if isinstance(gate, dict):
            ok = cast(dict[str, Any], gate).get("ok")
            return ok if isinstance(ok, bool) else None
        return None


class EventDetector:
    def __init__(self) -> None:
        self._states: dict[int, str] | None = None
        self._clutch: tuple[int, bool] | None = None
        self._gate_key: tuple[Any, ...] | None = None
        self._validating_bay: int | None = None
        self._sim_t: float | None = None

    def reset(self) -> None:
        """Forget everything; the next message sets a new baseline."""
        self._states = None
        self._clutch = None
        self._gate_key = None
        self._validating_bay = None
        self._sim_t = None

    def feed(self, msg: Mapping[str, Any]) -> list[CopilotEvent]:
        """The events this message causes, in a fixed order per message."""
        sim_t = _number(msg.get("sim_t"))
        if sim_t is not None and self._sim_t is not None and sim_t < self._sim_t:
            self.reset()  # the console restarted
        if sim_t is not None:
            self._sim_t = sim_t
        bays = _bays(msg)
        states = {k: str(b.get("state")) for k, b in bays.items()}
        clutch = _clutch(msg)
        gate_key, gate_ok = _gate(msg)
        if self._states is None:  # baseline
            self._states, self._clutch, self._gate_key = states, clutch, gate_key
            self._validating_bay = next((k for k, s in states.items() if s == "VALIDATING"), None)
            return []

        when = (sim_t if sim_t is not None else 0.0, str(msg.get("wall") or ""))
        events: list[CopilotEvent] = []

        def emit(kind: EventKind, bay: int) -> None:
            entry = bays.get(bay, {})
            telemetry = {"bay": dict(entry)} | {k: msg.get(k) for k in SHARED_FIELDS}
            events.append(
                CopilotEvent(
                    kind,
                    bay,
                    when[0],
                    when[1],
                    self._states.get(bay) if self._states is not None else None,
                    states.get(bay),
                    telemetry,
                )
            )

        # Clutch releases come first: in one message a release and the next state can coincide.
        if self._clutch is not None and self._clutch[1] and (clutch != self._clutch):
            emit(EventKind.RESCUE_RELEASED, self._clutch[0])
        for bay in sorted(states):
            if bays[bay].get("replayed") is True:
                continue
            before, now = self._states.get(bay), states[bay]
            if before == now:
                continue
            if before == "AUTONOMOUS_SKILL":
                emit(EventKind.AUTONOMOUS_FINISHED, bay)
            if now in STALL_STATES and (before is None or before in WORKING_STATES):
                emit(EventKind.STALL, bay)
            elif now == "OOD_ESCALATED":
                emit(EventKind.REFUSED, bay)
            elif now == "VALIDATING":
                self._validating_bay = bay
                emit(EventKind.VALIDATING, bay)
            elif now == "AUTONOMOUS_SKILL":
                emit(EventKind.AUTONOMOUS_STARTED, bay)
        if clutch is not None and clutch[1] and clutch != self._clutch:
            emit(EventKind.RESCUE_STARTED, clutch[0])
        if gate_ok is not None and gate_key != self._gate_key and self._validating_bay is not None:
            emit(EventKind.GATE_RESULT, self._validating_bay)
            self._validating_bay = None

        self._states, self._clutch, self._gate_key = states, clutch, gate_key
        return events


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _bays(msg: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    raw = msg.get("bays")
    if not isinstance(raw, list):
        return out
    for item in cast(list[object], raw):
        if isinstance(item, dict):
            entry = cast(dict[str, Any], item)
            bay = entry.get("id")
            if isinstance(bay, int) and isinstance(entry.get("state"), str):
                out[bay] = entry
    return out


def _clutch(msg: Mapping[str, Any]) -> tuple[int, bool] | None:
    rescue = msg.get("rescue")
    if not isinstance(rescue, dict):
        return None
    r = cast(dict[str, Any], rescue)
    bay = r.get("bay")
    return (bay, r.get("clutch") is True) if isinstance(bay, int) else None


def _gate(msg: Mapping[str, Any]) -> tuple[tuple[Any, ...] | None, bool | None]:
    """A key that changes when a new gate result appears, and that result's ``ok``."""
    gate = msg.get("gate")
    if not isinstance(gate, dict):
        return None, None
    g = cast(dict[str, Any], gate)
    ok = g.get("ok")
    if not isinstance(ok, bool):
        return None, None
    return (ok, g.get("fit_ms"), g.get("passed"), g.get("total")), ok
