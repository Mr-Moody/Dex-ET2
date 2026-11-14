"""Prompts, the response schema and the deterministic template text for call-outs.

The template text is built from telemetry alone, so the operator hears something within about a
second of every event even when Gemini is slow or offline. The model's version replaces it when it
arrives in time (see :mod:`teleop.copilot.callouts`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, cast

from teleop.copilot.events import CopilotEvent, EventKind
from teleop.copilot.gemini import ImagePart, Part, TextPart

Severity = Literal["info", "attention", "alarm"]
SEVERITIES: tuple[Severity, ...] = ("info", "attention", "alarm")

SAY_MAX_WORDS = 15
DETAIL_MAX_WORDS = 40
HISTORY_MAX_POINTS = 20

# Events worth a model call. The rest are fully described by telemetry, so they are templated:
# clutch flips (no model on start, per the plan), ``validating`` (its deep review is T-213) and
# ``autonomous_started``.
MODEL_EVENTS = frozenset(
    {
        EventKind.STALL,
        EventKind.REFUSED,
        EventKind.GATE_RESULT,
        EventKind.AUTONOMOUS_FINISHED,
    }
)

CALLOUT_SYSTEM = (
    "You are the call-out voice for the human operator of a robot fleet control room. Each bay "
    "has a robot hand that opens a sprung latch: a stopper flap must be held down while a bolt "
    "slides at least 5.2 cm clear. A bay stalls when its contact confidence C(t) stays below the "
    "0.3 threshold; a human then rescues it with data gloves and a clutch. The rescue becomes a "
    "skill that must pass a randomized simulation gate before the fleet reuses it. A jam whose "
    "Mahalanobis distance from known jams exceeds 3.0 is out of distribution and is refused "
    "(OOD_ESCALATED) instead of being solved with a fleet skill.\n"
    "Rules:\n"
    "- Use only the telemetry and image given. If something is not visible or not in the "
    "telemetry, say 'unclear' rather than guess.\n"
    "- Never suggest bypassing a safety stop, overriding the gate or editing settings.\n"
    f"- 'say' is spoken aloud: at most {SAY_MAX_WORDS} plain words, start with the bay, no "
    "symbols or units the voice cannot read.\n"
    f"- 'detail' is shown on screen: at most {DETAIL_MAX_WORDS} words.\n"
    "- severity: 'alarm' when a human must act now, 'attention' when the operator should look, "
    "'info' otherwise."
)

CALLOUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "say": {"type": "string", "description": f"Spoken line, at most {SAY_MAX_WORDS} words."},
        "detail": {"type": "string", "description": f"At most {DETAIL_MAX_WORDS} words."},
        "severity": {"type": "string", "enum": list(SEVERITIES)},
    },
    "required": ["say", "detail", "severity"],
    "additionalProperties": False,
}

_FAULT_PHRASES = {"flap": "on a flap jam", "pin": "on a pinned bolt"}


@dataclass(frozen=True, slots=True)
class CalloutText:
    say: str
    detail: str
    severity: Severity


def _bay(event: CopilotEvent) -> dict[str, Any]:
    bay = event.telemetry.get("bay")
    return cast(dict[str, Any], bay) if isinstance(bay, dict) else {}


def _gate(event: CopilotEvent) -> dict[str, Any]:
    gate = event.telemetry.get("gate")
    return cast(dict[str, Any], gate) if isinstance(gate, dict) else {}


def _num(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _facts(event: CopilotEvent) -> str:
    """A short on-screen detail from telemetry: confidence, bolt travel, fault."""
    bay = _bay(event)
    bits: list[str] = []
    conf = _num(bay.get("confidence"))
    if conf is not None:
        bits.append(f"C={conf:.2f}")
    bolt = _num(bay.get("x_bolt_cm"))
    if bolt is not None:
        clear = _num(event.telemetry.get("bolt_clear_cm")) or 5.2
        bits.append(f"bolt {bolt:.1f} of {clear:.1f} cm")
    fault = bay.get("fault")
    if isinstance(fault, dict):
        kind = cast(dict[str, Any], fault).get("kind")
        if isinstance(kind, str):
            bits.append(f"fault {kind}")
    return f"{event.previous_state} -> {event.state}" + (f", {', '.join(bits)}" if bits else "")


def template_callout(event: CopilotEvent) -> CalloutText:
    """The deterministic call-out for ``event``, from telemetry only."""
    n = event.bay
    detail = _facts(event)
    match event.kind:
        case EventKind.STALL:
            fault = _bay(event).get("fault")
            kind = cast(dict[str, Any], fault).get("kind") if isinstance(fault, dict) else None
            phrase = _FAULT_PHRASES.get(kind, "") if isinstance(kind, str) else ""
            stalled = f"Bay {n} stalled {phrase}".rstrip()
            return CalloutText(f"{stalled}. Rescue needed.", detail, "alarm")
        case EventKind.REFUSED:
            return CalloutText(
                f"Bay {n} refused. Jam is out of distribution. Inspect before rescue.",
                detail,
                "alarm",
            )
        case EventKind.RESCUE_STARTED:
            return CalloutText(f"Clutch engaged on bay {n}.", detail, "info")
        case EventKind.RESCUE_RELEASED:
            return CalloutText(f"Clutch released on bay {n}.", detail, "info")
        case EventKind.VALIDATING:
            return CalloutText(f"Bay {n} rescue captured. Validating the skill.", detail, "info")
        case EventKind.GATE_RESULT:
            gate = _gate(event)
            passed, total = gate.get("passed"), gate.get("total")
            score = f" {passed} of {total}" if passed is not None and total is not None else ""
            if event.gate_ok:
                return CalloutText(
                    f"Bay {n} skill passed{score}. Shared with the fleet.", detail, "info"
                )
            return CalloutText(
                f"Bay {n} skill failed the gate{score}. Not shared.", detail, "attention"
            )
        case EventKind.AUTONOMOUS_STARTED:
            return CalloutText(f"Bay {n} is reusing a fleet skill.", detail, "info")
        case EventKind.AUTONOMOUS_FINISHED:
            if event.state == "NOMINAL":
                return CalloutText(f"Bay {n} cleared with the fleet skill.", detail, "info")
            return CalloutText(f"Bay {n} fleet skill did not clear the jam.", detail, "attention")


def callout_parts(event: CopilotEvent, jpeg: bytes | None) -> list[Part]:
    """The user parts of a call-out request: the event, the telemetry slice, one frame."""
    telemetry = dict(event.telemetry)
    bay = dict(_bay(event))
    history = bay.get("history")
    if isinstance(history, list):
        bay["history"] = cast(list[Any], history)[-HISTORY_MAX_POINTS:]
    telemetry["bay"] = bay
    text = (
        f"Event: {event.kind.value} on bay {event.bay} "
        f"({event.previous_state} -> {event.state}) at sim time {event.sim_t:.1f} s.\n"
        f"Telemetry:\n{json.dumps(telemetry, separators=(',', ':'), default=str)}\n"
        + (
            "The image is the latest camera frame of this bay."
            if jpeg is not None
            else "No camera frame is available."
        )
    )
    parts: list[Part] = [TextPart(text)]
    if jpeg is not None:
        parts.append(ImagePart(jpeg))
    return parts


def _clip_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(",;:") + "…"


def _severity(value: object, default: Severity) -> Severity:
    for known in SEVERITIES:
        if value == known:
            return known
    return default


def parse_callout(answer: dict[str, Any], fallback: CalloutText) -> CalloutText:
    """A model answer as a call-out. Raises ``ValueError`` if it has no usable ``say``."""
    say = answer.get("say")
    if not isinstance(say, str) or not say.strip():
        raise ValueError("call-out answer has no 'say' text")
    detail = answer.get("detail")
    severity = answer.get("severity")
    return CalloutText(
        _clip_words(say, SAY_MAX_WORDS),
        _clip_words(detail, DETAIL_MAX_WORDS) if isinstance(detail, str) else fallback.detail,
        _severity(severity, fallback.severity),
    )
