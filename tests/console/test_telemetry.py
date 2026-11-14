"""HudView -> console telemetry JSON (wire protocol in the fleet ops console plan)."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime

from teleop.atc.hud import ActionNote, GateStatus, HudView, RescueStatus
from teleop.atc.types import BaySnapshot, BayState
from teleop.console.layout import feed_layout
from teleop.console.telemetry import EVENT_LOG_LEN, HISTORY_LEN, THRESHOLD, TelemetryAdapter

SKU = "EV-LATCH-01"
GRID = feed_layout([1, 2, 3, 4], None)


def _clock() -> datetime:
    return datetime(2026, 11, 15, 10, 2, 31)


def _snap(
    bay: int, state: BayState = BayState.NOMINAL, c: float = 0.95, **kw: object
) -> BaySnapshot:
    return BaySnapshot(bay, state, c, 0.0, 0.5, SKU, **kw)  # type: ignore[arg-type]


def _view(*snaps: BaySnapshot, **kw: object) -> HudView:
    base = HudView(
        bays=snaps or (_snap(1), _snap(2)),
        queue=(),
        background_bays=(3, 4),
        compliance="SOLANA  no events yet",
        reasoning="NEMOTRON  idle",
    )
    return replace(base, **kw)  # type: ignore[arg-type]


def _adapter(
    present: tuple[int, ...] = (1, 2, 3, 4), bg: tuple[int, ...] = (3, 4)
) -> TelemetryAdapter:
    return TelemetryAdapter(present, bg, clock=_clock)


def _bay(msg: dict[str, object], k: int) -> dict[str, object]:
    bays = msg["bays"]
    assert isinstance(bays, list)
    return next(b for b in bays if b["id"] == k)  # type: ignore[index]


def test_message_shape() -> None:
    msg = _adapter().build(_view(), 1.5, GRID, {3: 0.91, 4: 0.97})
    assert msg["type"] == "telemetry"
    assert msg["wall"] == "10:02:31" and msg["sim_t"] == 1.5
    assert msg["threshold"] == THRESHOLD == 0.3
    assert msg["mode"] == "grid"
    assert msg["feeds"] == [
        {"slot": 0, "camera": "bay1/cctv", "bay": 1, "role": "tile"},
        {"slot": 1, "camera": "bay2/cctv", "bay": 2, "role": "tile"},
        {"slot": 2, "camera": "bay3/cctv", "bay": 3, "role": "tile"},
        {"slot": 3, "camera": "bay4/cctv", "bay": 4, "role": "tile"},
    ]
    assert [b["id"] for b in msg["bays"]] == [1, 2, 3, 4]  # type: ignore[index, union-attr]
    b1 = _bay(msg, 1)
    assert b1["state"] == "NOMINAL" and b1["label"] == "NOMINAL" and b1["confidence"] == 0.95
    assert b1["background"] is False and b1["replayed"] is False
    assert msg["rescue"] is None and msg["gate"] is None
    assert msg["compliance"] == "SOLANA  no events yet"


def test_output_is_json_serialisable() -> None:
    view = _view(
        _snap(1, BayState.RESCUE, 0.12),
        _snap(2, BayState.QUEUED, 0.25, queue_position=0),
        queue=(2,),
        selected_bay=1,
        rescue=RescueStatus(1, True, 3.2),
        gate=GateStatus(12.0, 3, 3, 0.41, True),
    )
    json.dumps(_adapter().build(view, 2.0, feed_layout([1, 2, 3, 4], 1), {3: 0.9, 4: 0.9}))


def test_rescue_fields() -> None:
    view = _view(
        _snap(1, BayState.RESCUE, 0.12), _snap(2), selected_bay=1, rescue=RescueStatus(1, True, 3.2)
    )
    msg = _adapter().build(view, 2.0, feed_layout([1, 2, 3, 4], 1), {})
    assert msg["mode"] == "rescue"
    assert msg["rescue"] == {"bay": 1, "clutch": True, "elapsed_s": 3.2}
    assert msg["selected_bay"] == 1


def test_gate_fields() -> None:
    view = _view(gate=GateStatus(12.0, 3, 3, 0.41, True))
    msg = _adapter().build(view, 2.0, GRID, {})
    assert msg["gate"] == {
        "fit_ms": 12.0,
        "passed": 3,
        "total": 3,
        "elapsed_s": 0.41,
        "ok": True,
        "conditions": None,
        "worst_failure": None,
    }


def test_stall_transition_logs_alert() -> None:
    a = _adapter()
    a.build(_view(), 1.0, GRID, {})
    msg = a.build(_view(_snap(1, BayState.STALLED, 0.21), _snap(2)), 1.1, GRID, {})
    events = msg["events"]
    assert events == [
        {
            "sim_t": 1.1,
            "wall": "10:02:31",
            "bay": 1,
            "level": "alert",
            "text": "Bay 1 stalled · C(t) 0.21",
        }
    ]


def test_transition_texts() -> None:
    a = _adapter()
    a.build(_view(), 1.0, GRID, {})
    steps = [
        (_snap(1, BayState.QUEUED, 0.2, queue_position=0), "warn", "Bay 1 queued #1"),
        (_snap(1, BayState.RESCUE, 0.2), "info", "Operator took control of Bay 1"),
        (_snap(1, BayState.VALIDATING, 0.9), "info", "Bay 1: validating learned skill"),
        (_snap(1, BayState.AUTONOMOUS_SKILL, 0.9), "ok", "Bay 1: replaying learned skill"),
        (
            _snap(1, BayState.OOD_ESCALATED, 0.2),
            "alert",
            "Bay 1: stall refused, no validated skill is close enough",
        ),
        (_snap(1), "ok", "Bay 1 back to nominal"),
    ]
    for i, (snap, level, text) in enumerate(steps):
        msg = a.build(_view(snap, _snap(2)), 2.0 + i, GRID, {})
        last = msg["events"][-1]  # type: ignore[index]
        assert (last["level"], last["text"]) == (level, text)
    assert len(msg["events"]) == len(steps)  # type: ignore[arg-type]


def test_gate_and_hook_lines_log_events() -> None:
    a = _adapter()
    a.build(_view(), 1.0, GRID, {})
    msg = a.build(_view(gate=GateStatus(12.0, 3, 3, 0.41, True)), 1.1, GRID, {})
    assert msg["events"][-1]["text"] == "Skill gate 3/3 passed in 0.41 s"  # type: ignore[index]
    assert msg["events"][-1]["level"] == "ok"  # type: ignore[index]
    msg = a.build(_view(gate=GateStatus(12.0, 1, 3, 0.4, False)), 1.2, GRID, {})
    assert msg["events"][-1]["text"] == "Skill gate 1/3 FAILED in 0.40 s"  # type: ignore[index]
    assert msg["events"][-1]["level"] == "alert"  # type: ignore[index]
    msg = a.build(_view(compliance="SOLANA  STALL_ALERT bay 1 tx 4f2a"), 1.3, GRID, {})
    assert msg["events"][-1] == {  # type: ignore[index]
        "sim_t": 1.3,
        "wall": "10:02:31",
        "bay": None,
        "level": "info",
        "text": "SOLANA  STALL_ALERT bay 1 tx 4f2a",
    }


def test_event_log_is_bounded() -> None:
    a = _adapter()
    for i in range(EVENT_LOG_LEN + 20):
        state = BayState.STALLED if i % 2 else BayState.NOMINAL
        msg = a.build(_view(_snap(1, state, 0.5), _snap(2)), float(i), GRID, {})
    assert len(msg["events"]) == EVENT_LOG_LEN  # type: ignore[arg-type]


def test_history_is_bounded() -> None:
    a = _adapter()
    for i in range(HISTORY_LEN + 50):
        msg = a.build(_view(), float(i), GRID, {3: 0.9})
    assert len(_bay(msg, 1)["history"]) == HISTORY_LEN  # type: ignore[arg-type]
    assert len(_bay(msg, 3)["history"]) == HISTORY_LEN  # type: ignore[arg-type]


def test_absent_bays_are_standby() -> None:
    msg = _adapter(present=(1, 2), bg=()).build(_view(), 1.0, feed_layout([1, 2], None), {})
    for k in (3, 4):
        b = _bay(msg, k)
        assert b["state"] == "STANDBY" and b["label"] == "STANDBY"
        assert b["confidence"] == 0.0 and b["history"] == []


def test_background_bays_marked_replayed() -> None:
    msg = _adapter().build(_view(), 1.0, GRID, {3: 0.91})
    b3 = _bay(msg, 3)
    assert b3["state"] == "NOMINAL" and b3["replayed"] is True and b3["background"] is True
    assert b3["confidence"] == 0.91
    assert _bay(msg, 4)["confidence"] == 1.0  # no sample yet


def test_layout_id_bumps_on_layout_change() -> None:
    a = _adapter()
    first = a.build(_view(), 1.0, GRID, {})["layout_id"]
    assert a.build(_view(), 1.1, GRID, {})["layout_id"] == first
    rescue = a.build(_view(), 1.2, feed_layout([1, 2, 3, 4], 1), {})["layout_id"]
    assert rescue == first + 1  # type: ignore[operator]
    assert a.build(_view(), 1.3, GRID, {})["layout_id"] == first + 2  # type: ignore[operator]


def test_non_finite_values_never_reach_the_wire() -> None:
    # stage.py records a missing gate time as NaN; the browser's JSON.parse rejects NaN.
    a = _adapter()
    view = _view(gate=GateStatus(12.0, 3, 3, float("nan"), True))
    msg = a.build(view, float("nan"), GRID, {3: float("nan"), 4: float("inf")})
    json.dumps(msg, allow_nan=False)
    assert msg["gate"]["elapsed_s"] is None  # type: ignore[index]
    assert _bay(msg, 3)["confidence"] == 0.0
    assert msg["events"][-1]["text"] == "Skill gate 3/3 passed in ? s"  # type: ignore[index]


def test_key_outcomes_are_sent_and_logged_once() -> None:
    a = _adapter()
    note1 = ActionNote(1, "F", "Bay 1 flap jam armed: Bay 1 stalls on its next cycle", "ok")
    note2 = ActionNote(2, "SPACE", "Clutch engaged, but no bay is waiting for rescue", "warn")
    msg = a.build(_view(actions=(note1,)), 1.0, GRID, {})
    assert msg["actions"] == [{"seq": 1, "key": "F", "text": note1.text, "level": "ok"}]
    msg = a.build(_view(actions=(note1, note2)), 2.0, GRID, {})
    assert [x["seq"] for x in msg["actions"]] == [1, 2]  # type: ignore[union-attr]
    msg = a.build(_view(actions=(note1, note2)), 3.0, GRID, {})  # no new press
    logged = [(e["level"], e["text"]) for e in msg["events"]]  # type: ignore[union-attr]
    assert logged == [("info", f"[F] {note1.text}"), ("warn", f"[SPACE] {note2.text}")]


def test_armed_faults_and_bolt_travel_per_bay() -> None:
    a = _adapter()
    snap1 = BaySnapshot(1, BayState.NOMINAL, 0.9, 0.0351, 0.5, SKU)
    msg = a.build(_view(snap1, _snap(2), faults=((2, "pin", "armed"),)), 1.0, GRID, {})
    assert msg["bolt_clear_cm"] == 5.2
    assert _bay(msg, 1)["x_bolt_cm"] == 3.51 and _bay(msg, 1)["fault"] is None
    assert _bay(msg, 2)["fault"] == {"kind": "pin", "phase": "armed"}
    assert _bay(msg, 3)["x_bolt_cm"] is None  # background replay has no live bolt


def test_telemetry_bay_carries_pose_estimate_fields() -> None:
    snap = BaySnapshot(
        1,
        BayState.NOMINAL,
        0.9,
        0.0,
        0.0,
        SKU,
        pose_sigma_mm=3.0,
        pose_sigma_yaw_deg=1.5,
        touch_correction_mm=4.2,
    )
    entry = _adapter()._bay_entry(1, {1: snap}, {}, {}, {})
    assert entry["pose_sigma_yaw_deg"] == 1.5
    assert entry["pose_sigma_mm"] == 3.0 and entry["touch_correction_mm"] == 4.2


def test_touch_and_gate_lines_reach_the_event_log() -> None:
    a = _adapter()
    a.build(_view(_snap(1)), 1.0, GRID, {})
    touched = BaySnapshot(1, BayState.NOMINAL, 0.9, 0.0, 0.0, SKU, touch_correction_mm=3.1)
    a.build(_view(touched), 2.0, GRID, {})
    gate = GateStatus(
        passed=17,
        total=20,
        ok=False,
        elapsed_s=2.0,
        conditions="randomized pose, friction, flap spring and rest angle",
        worst_failure="failed at +23% friction, 2.8 mm y error",
    )
    msg = a.build(_view(touched, gate=gate), 3.0, GRID, {})
    texts = [e["text"] for e in msg["events"]]  # type: ignore[index]
    assert "Bay 1 touched latch: pose corrected 3.1 mm" in texts
    assert any("17/20" in t and "failed at +23% friction, 2.8 mm y error" in t for t in texts)
    assert msg["gate"]["conditions"] == "randomized pose, friction, flap spring and rest angle"  # type: ignore[index]


def test_telemetry_carries_compliance_detail() -> None:
    detail = {
        "kind": "STALL_ALERT",
        "bay_id": 1,
        "digest": "4f2a9c1e0b2a3d4e",
        "state": "anchored",
        "tx_signature": "sig123",
        "explorer_url": "https://explorer.solana.com/tx/sig123?cluster=devnet",
    }
    view = _view(_snap(1), compliance="Audit Trail STALL_ALERT bay 1", compliance_detail=detail)
    a = _adapter()
    msg = a.build(view, 1.0, GRID, {})
    assert msg["compliance"] == "Audit Trail STALL_ALERT bay 1"
    assert msg["compliance_detail"] == detail
