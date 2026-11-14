"""Turns the stage ``HudView`` into the console's telemetry JSON (10 Hz).

Adds what the HUD does not keep: a per-bay C(t) history for the sparklines, a timestamped
event log derived from state transitions, the gate result and the Solana/reasoner lines,
background bays (replayed canned cycle, ``replayed: true``) and absent bays (``STANDBY``).
``layout_id`` increments whenever the feed layout changes, so the page can drop frames that
were rendered for the previous layout.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

from teleop.atc.hud import STATE_LABELS, HudView
from teleop.atc.types import BOLT_CLEARED_M, N_BAYS, BaySnapshot, BayState
from teleop.console.layout import FeedSpec

THRESHOLD = 0.3
HISTORY_LEN = 150  # 15 s at 10 Hz
EVENT_LOG_LEN = 50
ACTION_LEN = 8  # key outcomes sent per message; the page toasts the ones it has not seen
STANDBY = "STANDBY"

_TRANSITIONS: dict[BayState, tuple[str, str]] = {
    BayState.STALLED: ("alert", "Bay {k} stalled · C(t) {c:.2f}"),
    BayState.QUEUED: ("warn", "Bay {k} queued #{pos}"),
    BayState.RESCUE: ("info", "Operator took control of Bay {k}"),
    BayState.VALIDATING: ("info", "Bay {k}: validating learned skill"),
    BayState.AUTONOMOUS_SKILL: ("ok", "Bay {k}: replaying learned skill"),
    BayState.OOD_ESCALATED: ("alert", "Bay {k}: stall refused, no validated skill is close enough"),
    BayState.NOMINAL: ("ok", "Bay {k} back to nominal"),
}


def _finite_or_none(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _scrub(obj: Any) -> Any:
    """Replace non-finite floats with ``None``: the browser's ``JSON.parse`` rejects NaN."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]  # pyright: ignore[reportUnknownVariableType]
    return obj


class TelemetryAdapter:
    def __init__(
        self,
        bays_present: Sequence[int],
        background: Sequence[int],
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.bays_present = frozenset(bays_present)
        self.background = frozenset(background) & self.bays_present
        self._clock = clock
        self._history: dict[int, deque[float]] = {
            k: deque(maxlen=HISTORY_LEN) for k in range(1, N_BAYS + 1)
        }
        self._events: deque[dict[str, Any]] = deque(maxlen=EVENT_LOG_LEN)
        self._states: dict[int, BayState] = {}
        self._gate_key: tuple[Any, ...] | None = None
        self._touch: dict[int, float | None] = {}
        self._lines: dict[str, str] = {}
        self._feeds: tuple[FeedSpec, ...] | None = None
        self._action_seq = 0
        self.layout_id = 0

    # ------------------------------------------------------------------ events

    def _log(self, sim_t: float, wall: str, bay: int | None, level: str, text: str) -> None:
        self._events.append(
            {"sim_t": round(sim_t, 3), "wall": wall, "bay": bay, "level": level, "text": text}
        )

    def _track_state(self, snap: BaySnapshot, sim_t: float, wall: str) -> None:
        prev = self._states.get(snap.bay_id, BayState.NOMINAL)
        self._states[snap.bay_id] = snap.state
        if snap.state == prev:
            return
        level, template = _TRANSITIONS[snap.state]
        pos = (snap.queue_position or 0) + 1
        text = template.format(k=snap.bay_id, c=snap.confidence, pos=pos)
        self._log(sim_t, wall, snap.bay_id, level, text)

    def _track_touch(self, snap: BaySnapshot, sim_t: float, wall: str) -> None:
        mm = snap.touch_correction_mm
        prev = self._touch.get(snap.bay_id)
        self._touch[snap.bay_id] = mm
        if mm is not None and mm != prev:
            text = f"Bay {snap.bay_id} touched latch: pose corrected {mm:.1f} mm"
            self._log(sim_t, wall, snap.bay_id, "info", text)

    def _track_gate(self, view: HudView, sim_t: float, wall: str) -> None:
        g = view.gate
        key = (g.ok, g.passed, g.total, g.elapsed_s)
        if g.ok is None or key == self._gate_key:
            self._gate_key = key if g.ok is not None else self._gate_key
            return
        self._gate_key = key
        verdict = "passed" if g.ok else "FAILED"
        secs = _finite_or_none(g.elapsed_s)
        elapsed = f"{secs:.2f}" if secs is not None else "?"
        detail = f" ({g.conditions})" if g.conditions else ""
        text = f"Skill gate {g.passed}/{g.total} {verdict}{detail} in {elapsed} s"
        if not g.ok and g.worst_failure:
            text += f", {g.worst_failure}"
        self._log(sim_t, wall, None, "ok" if g.ok else "alert", text)

    def _track_actions(self, view: HudView, sim_t: float, wall: str) -> None:
        for note in view.actions:
            if note.seq > self._action_seq:
                level = "warn" if note.level == "warn" else "info"
                self._log(sim_t, wall, None, level, f"[{note.key}] {note.text}")
        if view.actions:
            self._action_seq = max(self._action_seq, view.actions[-1].seq)

    def _track_line(self, name: str, line: str, sim_t: float, wall: str) -> None:
        prev = self._lines.get(name)
        self._lines[name] = line
        if prev is not None and line != prev:
            self._log(sim_t, wall, None, "info", line)

    # ------------------------------------------------------------------ build

    def _bay_entry(
        self,
        k: int,
        snaps: Mapping[int, BaySnapshot],
        background_confidence: Mapping[int, float],
        faults: Mapping[int, dict[str, str]],
    ) -> dict[str, Any]:
        hist = self._history[k]
        if k in snaps:
            s = snaps[k]
            hist.append(round(s.confidence, 3))
            return {
                "id": k,
                "state": s.state.value,
                "label": STATE_LABELS[s.state],
                "confidence": round(s.confidence, 3),
                "history": list(hist),
                "queue_position": s.queue_position,
                "skill": s.active_skill_id,
                "background": False,
                "replayed": False,
                "x_bolt_cm": round(s.x_bolt_m * 100.0, 2),
                "fault": faults.get(k),
                "pose_sigma_mm": _finite_or_none(s.pose_sigma_mm),
                "pose_sigma_yaw_deg": _finite_or_none(s.pose_sigma_yaw_deg),
                "touch_correction_mm": _finite_or_none(s.touch_correction_mm),
            }
        if k in self.background:
            raw = float(background_confidence.get(k, 1.0))
            c = round(min(max(raw, 0.0), 1.0), 3) if math.isfinite(raw) else 0.0
            hist.append(c)
            return {
                "id": k,
                "state": BayState.NOMINAL.value,
                "label": STATE_LABELS[BayState.NOMINAL],
                "confidence": c,
                "history": list(hist),
                "queue_position": None,
                "skill": None,
                "background": True,
                "replayed": True,
                "x_bolt_cm": None,
                "fault": None,
                "pose_sigma_mm": None,
                "pose_sigma_yaw_deg": None,
                "touch_correction_mm": None,
            }
        return {
            "id": k,
            "state": STANDBY,
            "label": STANDBY,
            "confidence": 0.0,
            "history": [],
            "queue_position": None,
            "skill": None,
            "background": k in self.background,
            "replayed": False,
            "x_bolt_cm": None,
            "fault": None,
            "pose_sigma_mm": None,
            "pose_sigma_yaw_deg": None,
            "touch_correction_mm": None,
        }

    def build(
        self,
        view: HudView,
        sim_t: float,
        feeds: Sequence[FeedSpec],
        background_confidence: Mapping[int, float],
        *,
        view_state: Any = None,
        dummy_status: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """One telemetry message. Call at the telemetry rate: each call adds a history sample."""
        wall = self._clock().strftime("%H:%M:%S")
        feeds_t = tuple(feeds)
        topo = tuple((f.slot, f.camera, f.bay, f.role) for f in feeds_t)
        prev_topo = (
            None
            if self._feeds is None
            else tuple((f.slot, f.camera, f.bay, f.role) for f in self._feeds)
        )
        if prev_topo is not None and topo != prev_topo:
            self.layout_id += 1
        self._feeds = feeds_t

        snaps = {s.bay_id: s for s in view.bays if s.bay_id in self.bays_present}
        for s in snaps.values():
            self._track_state(s, sim_t, wall)
            self._track_touch(s, sim_t, wall)
        self._track_gate(view, sim_t, wall)
        self._track_actions(view, sim_t, wall)
        self._track_line("compliance", view.compliance, sim_t, wall)
        self._track_line("reasoning", view.reasoning, sim_t, wall)

        faults = {bay: {"kind": kind, "phase": phase} for bay, kind, phase in view.faults}
        g = view.gate
        gate = None
        if any(v is not None for v in (g.fit_ms, g.passed, g.total, g.ok)):
            gate = {
                "fit_ms": g.fit_ms,
                "passed": g.passed,
                "total": g.total,
                "elapsed_s": g.elapsed_s,
                "ok": g.ok,
                "conditions": g.conditions,
                "worst_failure": g.worst_failure,
            }
        r = view.rescue
        rescue = (
            None
            if r.bay_id is None
            else {"bay": r.bay_id, "clutch": r.clutch_engaged, "elapsed_s": r.elapsed_s}
        )
        mains = [f for f in feeds_t if f.role == "main"]
        if not mains:
            mode = "grid"
        elif r.bay_id is not None and any(f.bay == r.bay_id for f in mains):
            mode = "rescue"
        else:
            mode = "pin"
        return _scrub(
            {
                "type": "telemetry",
                "layout_id": self.layout_id,
                "sim_t": round(sim_t, 3),
                "wall": wall,
                "threshold": THRESHOLD,
                "mode": mode,
                "view": view_state.to_dict() if hasattr(view_state, "to_dict") else None,
                "dummy": dict(dummy_status) if dummy_status is not None else None,
                "feeds": [
                    {"slot": f.slot, "camera": f.camera, "bay": f.bay, "role": f.role}
                    for f in feeds_t
                ],
                "bays": [
                    self._bay_entry(k, snaps, background_confidence, faults)
                    for k in range(1, N_BAYS + 1)
                ],
                "bolt_clear_cm": round(BOLT_CLEARED_M * 100.0, 2),
                "actions": [
                    {"seq": a.seq, "key": a.key, "text": a.text, "level": a.level}
                    for a in view.actions[-ACTION_LEN:]
                ],
                "queue": list(view.queue),
                "selected_bay": view.selected_bay,
                "rescue": rescue,
                "gate": gate,
                "events": list(self._events),
                "compliance": view.compliance,
                "reasoning": view.reasoning,
                "beat": view.beat,
                "unoq": view.unoq,
            }
        )
