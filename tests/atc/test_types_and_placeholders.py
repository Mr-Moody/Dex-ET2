"""Shared ATC contracts and the offline Solana placeholder and the triage reasoner (T-100)."""

from __future__ import annotations

import dataclasses
import json
import math
import threading
import time
from pathlib import Path

import pytest

from teleop.atc.types import (
    BaySnapshot,
    BayState,
    ConsoleInputState,
    LatchPose,
    SkillPrimitive,
    StallMatch,
    StallSignature,
)
from teleop.cli import COMMANDS
from teleop.compliance import (
    ComplianceEvent,
    ComplianceSink,
    SolanaCompliancePlaceholder,
    make_event,
)
from teleop.reasoning import (
    FleetReasoner,
    ReasoningContext,
    ReasoningVerdict,
    TriageReasoner,
)
from teleop.sim.hands import load_hand

# ---------------------------------------------------------------- shared entrypoints


def test_cli_registers_atc_and_unoq():
    assert COMMANDS["atc"] == "teleop.commands.atc"
    assert COMMANDS["unoq"] == "teleop.commands.unoq"


def test_leap_maps_all_16_actuators_by_name():
    spec = load_hand("leap")
    names = [fm.actuator for fm in spec.flex]
    assert len(names) == 16 == len(set(names))
    for added in ("th_axl_act", "if_rot_act", "mf_rot_act", "rf_rot_act"):
        assert added in names
    rot = {fm.actuator: fm for fm in spec.flex if fm.actuator.endswith("_rot_act")}
    for fm in rot.values():
        assert fm.axis == "z"
        assert fm.range is not None and fm.range[0] == -fm.range[1]  # centred: relaxed -> 0


# ---------------------------------------------------------------- atc.types


def test_bay_state_values_match_spec():
    assert [s.value for s in BayState] == [
        "NOMINAL",
        "STALLED",
        "QUEUED",
        "RESCUE",
        "VALIDATING",
        "AUTONOMOUS_SKILL",
        "OOD_ESCALATED",
    ]
    assert BayState("RESCUE") is BayState.RESCUE


def test_latch_pose_world_pose_is_xyzw_yaw():
    pose = LatchPose(position=(0.5, 0.1, 0.02), yaw_rad=math.radians(90.0))
    p = pose.as_pose()
    assert p.position == (0.5, 0.1, 0.02)
    x, y, z, w = p.orientation
    assert (x, y) == (0.0, 0.0)
    assert z == pytest.approx(math.sqrt(0.5)) and w == pytest.approx(math.sqrt(0.5))


def test_latch_pose_offset_composes_shift_and_rotation():
    base = LatchPose(position=(0.5, 0.0, 0.02), yaw_rad=0.0)
    moved = base.offset(dx=0.08, dy=0.09, dyaw_rad=math.radians(35.0))
    assert moved.position == pytest.approx((0.58, 0.09, 0.02))
    assert moved.yaw_rad == pytest.approx(math.radians(35.0))


def test_stall_signature_vector_roundtrip():
    sig = StallSignature(
        x_bolt_m=0.035, tau_ext_inf_mean_nm=4.1, tau_ext_inf_max_nm=7.9, wrist_tau_delta_nm=2.2
    )
    assert sig.as_vector() == (0.035, 4.1, 7.9, 2.2)
    assert StallSignature.from_vector(sig.as_vector()) == sig
    with pytest.raises(ValueError):
        StallSignature.from_vector((1.0, 2.0, 3.0))


def test_stall_match_ood_has_no_signature():
    StallMatch(signature_id="Jam_Sig_1", distance=1.2, is_ood=False)
    StallMatch(signature_id=None, distance=7.0, is_ood=True)
    with pytest.raises(ValueError):
        StallMatch(signature_id=None, distance=1.0, is_ood=False)


def _skill() -> SkillPrimitive:
    leap = tuple(fm.actuator for fm in load_hand("leap").flex)
    k, d = 30, 6 + len(leap)
    return SkillPrimitive(
        skill_id="ev_latch_v1/Jam_Sig_1/1",
        part_sku_id="ev_latch_v1",
        signature_id="Jam_Sig_1",
        duration_s=4.2,
        alpha_s=4.0,
        actuator_names=leap,
        goal_orientation=(0.0, 0.0, 0.0, 1.0),
        y0=tuple(0.0 for _ in range(d)),
        y_goal=tuple(1.0 for _ in range(d)),
        weights=tuple(tuple(float(i * d + j) for j in range(d)) for i in range(k)),
        tau_demo=tuple(0.1 * i for i in range(k)),
        prototype=StallSignature(0.035, 4.0, 8.0, 2.0),
        fit_ms=11.4,
    )


def test_skill_primitive_shapes_and_json_roundtrip():
    skill = _skill()
    assert skill.n_kernels == 30 and skill.n_dims == 22
    wire = json.loads(json.dumps(skill.to_dict()))
    assert SkillPrimitive.from_dict(wire) == skill


def test_skill_primitive_rejects_bad_shapes():
    skill = _skill()
    with pytest.raises(ValueError):
        dataclasses.replace(skill, y0=skill.y0[:-1])
    with pytest.raises(ValueError):
        dataclasses.replace(skill, tau_demo=skill.tau_demo[:-1])
    with pytest.raises(ValueError):
        dataclasses.replace(skill, weights=(*skill.weights[:-1], skill.weights[-1][:-1]))


def test_bay_snapshot_and_console_state_are_frozen_and_validated():
    snap = BaySnapshot(
        bay_id=2,
        state=BayState.AUTONOMOUS_SKILL,
        confidence=0.91,
        x_bolt_m=0.02,
        tau_ext_inf_nm=1.5,
        part_sku_id="ev_latch_v1",
        queue_position=None,
        active_skill_id="ev_latch_v1/Jam_Sig_1/1",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.confidence = 0.1  # type: ignore[misc]
    with pytest.raises(ValueError):
        dataclasses.replace(snap, bay_id=5)
    with pytest.raises(ValueError):
        dataclasses.replace(snap, confidence=1.5)

    console = ConsoleInputState(selected_bay=1, clutch=True, estop=False, fault_rig_latched=True)
    assert console.timestamp_ns == 0
    with pytest.raises(ValueError):
        ConsoleInputState(selected_bay=0)


# ---------------------------------------------------------------- compliance


def test_make_event_hashes_canonical_payload():
    a = make_event("STALL_ALERT", 1, "ev_latch_v1", "stall", {"x": 0.035, "c": 0.2}, 123)
    b = make_event("STALL_ALERT", 1, "ev_latch_v1", "stall", {"c": 0.2, "x": 0.035}, 123)
    c = make_event("STALL_ALERT", 1, "ev_latch_v1", "stall", {"c": 0.2, "x": 0.036}, 123)
    assert isinstance(a, ComplianceEvent)
    assert len(a.payload_sha256) == 64
    assert a.payload_sha256 == b.payload_sha256 != c.payload_sha256
    assert make_event("OOD_REFUSAL", 1, "p", "s").timestamp_ns > 0


def test_solana_placeholder_ring_buffer_and_jsonl(tmp_path: Path):
    out = tmp_path / "audit" / "compliance.jsonl"
    sink = SolanaCompliancePlaceholder(capacity=3, jsonl_path=out)
    assert isinstance(sink, ComplianceSink)
    events = [make_event("STALL_ALERT", 1, "p", f"e{i}", {"i": i}, i + 1) for i in range(5)]
    for e in events:
        sink.emit(e)
    assert [e.summary for e in sink.events()] == ["e2", "e3", "e4"]
    assert sink.latest() == events[-1]
    sink.close()
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [line["summary"] for line in lines] == [f"e{i}" for i in range(5)]
    assert lines[0]["payload_sha256"] == events[0].payload_sha256
    sink.emit(events[0])  # after close: dropped, never raises


def test_solana_placeholder_emit_is_non_blocking_and_thread_safe(tmp_path: Path):
    sink = SolanaCompliancePlaceholder(capacity=10_000, jsonl_path=tmp_path / "a.jsonl")
    event = make_event("RESCUE_LOGGED", 1, "p", "s", {}, 1)

    def spam() -> None:
        for _ in range(500):
            sink.emit(event)

    threads = [threading.Thread(target=spam) for _ in range(4)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    per_emit_ms = (time.perf_counter() - t0) * 1000.0 / 2000
    assert per_emit_ms < 0.5
    assert len(sink.events()) == 2000
    sink.close()
    assert len((tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()) == 2000


def test_solana_placeholder_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "https://api.devnet.solana.com")
    monkeypatch.delenv("SOLANA_KEYPAIR_PATH", raising=False)
    sink = SolanaCompliancePlaceholder.from_env(jsonl_path=None)
    assert sink.rpc_url == "https://api.devnet.solana.com"
    assert sink.keypair_path is None
    sink.close()


# ---------------------------------------------------------------- reasoning


def _ctx(**kw: object) -> ReasoningContext:
    base: dict[str, object] = {
        "event_kind": "STALL_ALERT",
        "bay_id": 1,
        "part_sku_id": "ev_latch_v1",
        "confidence": 0.21,
        "x_bolt_m": 0.035,
        "tau_ext_inf_nm": 7.9,
        "matched_signature_id": None,
        "mahalanobis_distance": None,
        "is_ood": True,
    }
    base.update(kw)
    return ReasoningContext(**base)  # type: ignore[arg-type]


def test_triage_reasoner_is_deterministic_and_per_bay():
    r = TriageReasoner()
    assert isinstance(r, FleetReasoner)
    assert r.latest_verdict(1) is None
    r.submit(_ctx())
    v1 = r.latest_verdict(1)
    assert isinstance(v1, ReasoningVerdict)
    assert "Bay 1" in v1.headline and "3.5 cm" in v1.headline
    assert v1.model == "triage"
    assert v1.latency_ms >= 0.0
    r.submit(_ctx())
    v2 = r.latest_verdict(1)
    assert v2 is not None and dataclasses.replace(v2, latency_ms=0.0) == dataclasses.replace(
        v1, latency_ms=0.0
    )
    assert r.latest_verdict(2) is None
    r.close()


def test_triage_reasoner_explains_ood_refusal_and_validation():
    r = TriageReasoner()
    r.submit(_ctx(event_kind="OOD_REFUSAL", bay_id=2, x_bolt_m=0.0, mahalanobis_distance=7.4))
    ood = r.latest_verdict(2)
    assert ood is not None
    assert "0.0 cm" in ood.headline
    assert "7.40" in ood.compliance_rationale
    assert "refus" in ood.compliance_rationale.lower()

    r.submit(
        _ctx(
            event_kind="VALIDATION_GATE",
            bay_id=1,
            is_ood=False,
            matched_signature_id="Jam_Sig_1",
            rollout_passed=3,
            rollout_total=3,
        )
    )
    gate = r.latest_verdict(1)
    assert gate is not None and "3/3" in gate.headline
    assert r.model == "triage"


def test_triage_gate_verdict_uses_the_gate_threshold():
    r = TriageReasoner()
    for passed, word in ((18, "PASS"), (17, "FAIL")):
        r.submit(
            _ctx(
                event_kind="VALIDATION_GATE",
                bay_id=1,
                is_ood=False,
                matched_signature_id="Jam_Sig_1",
                rollout_passed=passed,
                rollout_total=20,
                rollout_threshold=18,
            )
        )
        gate = r.latest_verdict(1)
        assert gate is not None and f"{passed}/20 {word}" in gate.headline
