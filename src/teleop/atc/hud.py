"""Stage HUD for ``teleop atc``: 4-bay status, live C(t), FIFO queue, rescue, gate and hooks.

Pure formatting. :class:`HudView` is a plain snapshot built by ``teleop.atc.stage`` each tick;
:func:`hud_overlays` turns it into the ``(font, gridpos, left, right)`` tuples that
``mujoco.viewer.Handle.set_texts`` draws over the passive viewer, and :func:`hud_lines` gives the
same content as plain text for headless logs. Text is ASCII only (the MuJoCo overlay font has no
glyphs beyond it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from teleop.atc.types import BaySnapshot, BayState

STATE_LABELS: dict[BayState, str] = {
    BayState.NOMINAL: "NOMINAL",
    BayState.STALLED: "STALLED",
    BayState.QUEUED: "QUEUED",
    BayState.RESCUE: "RESCUE",
    BayState.VALIDATING: "VALIDATING",
    BayState.AUTONOMOUS_SKILL: "AUTO SKILL",
    BayState.OOD_ESCALATED: "OOD - HUMAN",
}

BAR_WIDTH = 10


@dataclass(frozen=True, slots=True)
class RescueStatus:
    """The bay holding the single rescue slot, if any."""

    bay_id: int | None = None
    clutch_engaged: bool = False
    elapsed_s: float | None = None


@dataclass(frozen=True, slots=True)
class GateStatus:
    """Latest UNO Q DMP fit and randomized sim gate result."""

    fit_ms: float | None = None
    passed: int | None = None
    total: int | None = None
    elapsed_s: float | None = None
    ok: bool | None = None
    conditions: str | None = None  # what the gate randomized, shown after the verdict
    worst_failure: str | None = None  # most extreme condition among failed rollouts


@dataclass(frozen=True, slots=True)
class ActionNote:
    """Outcome of one operator key press, so the displays can say what it did.

    ``seq`` increases per press. ``level`` is ``ok`` (done), ``info`` (done, with context) or
    ``warn`` (ignored or not what the operator probably meant).
    """

    seq: int
    key: str
    text: str
    level: str = "ok"


@dataclass(frozen=True, slots=True)
class HudView:
    bays: tuple[BaySnapshot, ...]
    queue: tuple[int, ...]
    background_bays: tuple[int, ...] = ()
    selected_bay: int | None = None
    rescue: RescueStatus = field(default_factory=RescueStatus)
    gate: GateStatus = field(default_factory=GateStatus)
    compliance: str = "SOLANA  no events yet"
    reasoning: str = "NEMOTRON  idle"
    beat: str = ""
    unoq: str = ""
    actions: tuple[ActionNote, ...] = ()  # recent key outcomes, oldest first
    # (bay, kind, phase): kind "flap" | "pin"; phase "armed" (next cycle) | "active" (this cycle)
    faults: tuple[tuple[int, str, str], ...] = ()


# --------------------------------------------------------------------------- pieces


def confidence_bar(confidence: float, width: int = BAR_WIDTH) -> str:
    """``[|||||||...] 0.70``: C(t) clamped to [0, 1]."""
    c = min(max(confidence, 0.0), 1.0)
    filled = round(c * width)
    return "[" + "|" * filled + "." * (width - filled) + f"] {c:.2f}"


def _state_text(snap: BaySnapshot) -> str:
    label = STATE_LABELS[snap.state]
    if snap.queue_position is not None:
        label += f" #{snap.queue_position + 1}"
    return label


def _match_text(snap: BaySnapshot) -> str:
    m = snap.last_match
    if m is None:
        return ""
    if m.is_ood:
        return "  OOD"
    return f"  {m.signature_id} d={m.distance:.1f}"


def format_bay_row(snap: BaySnapshot) -> tuple[str, str]:
    skill = f"  {snap.active_skill_id}" if snap.active_skill_id else ""
    pose = f"  pose ±{snap.pose_sigma_mm:.0f} mm" if snap.pose_sigma_mm is not None else ""
    touch = (
        f"  touch {snap.touch_correction_mm:.1f} mm" if snap.touch_correction_mm is not None else ""
    )
    value = (
        f"{_state_text(snap):<14} C {confidence_bar(snap.confidence)}  "
        f"x {snap.x_bolt_m * 100:4.1f} cm{pose}{touch}{_match_text(snap)}{skill}"
    )
    return f"BAY {snap.bay_id}", value


def format_bay_table(view: HudView) -> tuple[str, str]:
    """Left (labels) and right (values) columns of the 4-bay table, newline-separated."""
    rows = [format_bay_row(s) for s in view.bays]
    rows += [(f"BAY {b}", "BACKGROUND  kinematic cycle") for b in view.background_bays]
    rows.append(("QUEUE", " > ".join(str(b) for b in view.queue) or "empty"))
    rows.append(("SELECTED", f"BAY {view.selected_bay}" if view.selected_bay else "none"))
    return "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows)


def _clock(seconds: float) -> str:
    minutes, sec = divmod(max(seconds, 0.0), 60.0)
    return f"{int(minutes):02d}:{sec:04.1f}"


def format_rescue(rescue: RescueStatus) -> str:
    if rescue.bay_id is None:
        return "RESCUE  none"
    clutch = "ENGAGED" if rescue.clutch_engaged else "HOLD CLUTCH"
    timer = _clock(rescue.elapsed_s) if rescue.elapsed_s is not None else "--:--.-"
    return f"RESCUE BAY {rescue.bay_id} | {clutch} | {timer}"


def format_gate(gate: GateStatus) -> str:
    fit = f"{gate.fit_ms:.1f} ms" if gate.fit_ms is not None else "--"
    if gate.passed is None or gate.total is None:
        return f"UNO Q DMP fit: {fit} | Sim Gate: --"
    verdict = "PASS" if gate.ok else "FAIL"
    elapsed = f" ({gate.elapsed_s:.2f} s)" if gate.elapsed_s is not None else ""
    if gate.conditions:
        label = f"Skill gate {gate.passed}/{gate.total} {verdict} ({gate.conditions})"
    else:
        label = f"Sim Gate: {gate.passed}/{gate.total} {verdict}"
    return f"UNO Q DMP fit: {fit} | {label}{elapsed}"


# --------------------------------------------------------------------------- whole HUD


def hud_lines(view: HudView) -> list[str]:
    """The HUD as plain text lines (headless logging, tests)."""
    labels, values = format_bay_table(view)
    lines = [
        f"{lab:<9} {val}" for lab, val in zip(labels.split("\n"), values.split("\n"), strict=True)
    ]
    lines += [format_rescue(view.rescue), format_gate(view.gate), view.compliance, view.reasoning]
    if view.beat:
        lines.append(f"BEAT {view.beat}")
    if view.actions:
        last = view.actions[-1]
        lines.append(f"KEY {last.key}: {last.text}")
    if view.unoq:
        lines.append(view.unoq)
    return lines


def hud_overlays(view: HudView) -> list[tuple[int, int, str, str]]:
    """``viewer.set_texts`` overlays: one panel per screen corner, big font for the bay table."""
    import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]

    mj: Any = _mujoco
    font, grid = mj.mjtFont, mj.mjtGridPos
    labels, values = format_bay_table(view)
    status = "\n".join([format_rescue(view.rescue), format_gate(view.gate)])
    hooks = "\n".join([view.compliance, view.reasoning])
    last_key = f"{view.actions[-1].key}: {view.actions[-1].text}" if view.actions else ""
    stage = "\n".join(
        x for x in (f"BEAT {view.beat}" if view.beat else "", last_key, view.unoq) if x
    )
    keys = "1-4 bay | SPACE clutch | F/G/P faults | R reset"
    return [
        (int(font.mjFONT_BIG), int(grid.mjGRID_TOPLEFT), labels, values),
        (int(font.mjFONT_NORMAL), int(grid.mjGRID_TOPRIGHT), status, ""),
        (int(font.mjFONT_NORMAL), int(grid.mjGRID_BOTTOMLEFT), hooks, ""),
        (int(font.mjFONT_NORMAL), int(grid.mjGRID_BOTTOMRIGHT), stage or "ATC", keys),
    ]


__all__ = [
    "ActionNote",
    "GateStatus",
    "HudView",
    "RescueStatus",
    "confidence_bar",
    "format_bay_row",
    "format_bay_table",
    "format_gate",
    "format_rescue",
    "hud_lines",
    "hud_overlays",
]
