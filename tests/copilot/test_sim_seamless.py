"""Tests for seamless sim + Copilot integration across audio, non-blocking gate, and OOD audit."""

from __future__ import annotations

import contextlib
import threading
import time
from concurrent.futures import Future
from typing import Any

import numpy as np
import pytest

from teleop.atc.coordinator import AtcCoordinator, CoordinatorConfig
from teleop.atc.hud import GateStatus, HudView, format_audit, hud_lines
from teleop.atc.stage import AudioCues, NullSpeaker, StageController
from teleop.atc.types import BayState, ConsoleInputState
from teleop.commands.atc import AtcConfig, _EpisodeTagRouter, _setup_copilot
from teleop.compliance.protocol import ComplianceEvent, make_event
from teleop.console.server import ConsoleServer
from teleop.copilot.config import CopilotConfig
from teleop.copilot.gate import GraspGate, GraspGateVerdict, GraspTelemetrySummary, PoseSnapshot
from teleop.unoq.client import UnoQClient


class NullComplianceSink:
    def emit(self, event: ComplianceEvent) -> None:
        del event

    def close(self) -> None:
        pass


class NullReasoner:
    def submit(self, ctx: Any) -> None:
        del ctx

    def latest_verdict(self, bay_id: int) -> None:
        del bay_id
        return None

    def close(self) -> None:
        pass


class _SlowGraspGate:
    """Fake GraspGateRunner whose VLM future resolves only when explicitly triggered."""

    def __init__(self) -> None:
        self.source = "gemini"
        self.model = "gemini-4-hydrogen"
        self.submitted: list[GraspTelemetrySummary] = []
        self.ood_audits: list[dict[str, Any]] = []
        self.aborted_audits: list[dict[str, Any]] = []
        self.pending_future: Future[GraspGateVerdict] = Future()

    def submit(
        self,
        summary: GraspTelemetrySummary,
        snapshots: Any,
        *,
        model: Any = None,
        audit_snapshots: Any = (),
        episode_uid: str | None = None,
    ) -> Future[GraspGateVerdict]:
        del snapshots, model, audit_snapshots, episode_uid
        self.submitted.append(summary)
        return self.pending_future

    def submit_audit_only(
        self,
        *,
        bay_id: int,
        skill_id: str,
        snapshots: Any,
        model: Any = None,
        timeline: Any = None,
        episode_uid: str | None = None,
    ) -> None:
        del snapshots, model, episode_uid
        info = {"bay_id": bay_id, "skill_id": skill_id, "timeline": dict(timeline or {})}
        if skill_id == "ood_refused":
            self.ood_audits.append(info)
        else:
            self.aborted_audits.append(info)

    def close(self) -> None:
        pass


def test_audio_cues_suppresses_validation_gate_when_vlm_already_spoke() -> None:
    spoken: list[str] = []

    class RecSpeaker:
        def say(self, text: str) -> None:
            spoken.append(text)

        def close(self) -> None:
            pass

    cues = AudioCues(RecSpeaker(), enabled=True, skip_validation_gate=True)
    ev_vlm = make_event(
        "VALIDATION_GATE",
        1,
        "ev",
        "gate passed",
        {"passed": True, "vlm_source": "gemini"},
    )
    assert cues.on_event(ev_vlm) is None
    assert spoken == []

    # Physics-only fallback still speaks the validation cue
    ev_phys = make_event(
        "VALIDATION_GATE",
        1,
        "ev",
        "gate passed",
        {"passed": True, "vlm_source": "physics_only"},
    )
    assert cues.on_event(ev_phys) is not None
    assert len(spoken) == 1


def test_setup_copilot_local_edge_wires_speaker_and_never_uses_system_speaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("COPILOT_LOCAL_URL", "http://127.0.0.1:11434/v1")

    unoq = UnoQClient.connect_or_mock("127.0.0.1", 0, 0, connect_timeout_s=0.01)
    controller = StageController.build(
        unoq,
        NullComplianceSink(),
        NullReasoner(),
        audio=AudioCues(NullSpeaker(), enabled=False),
    )
    cfg = AtcConfig(
        gemini=False,
        audio=True,
        headless=False,
        display="viewer",
    )
    with contextlib.ExitStack() as stack:
        stack.callback(controller.close)
        stack.callback(unoq.close)
        _setup_copilot(cfg, controller, stack, _EpisodeTagRouter(), speak=True)
        gate = controller.coordinator.grasp_gate
        assert isinstance(gate, GraspGate)
        assert gate.source == "local"
        assert gate.speaker is not None
        assert controller.audio is not None
        assert controller.audio.enabled is True
        assert controller.audio.skip_validation_gate is True


def test_stage_gate_status_resets_and_shows_early_vlm_verdict() -> None:
    unoq = UnoQClient.connect_or_mock("127.0.0.1", 0, 0, connect_timeout_s=0.01)
    controller = StageController.build(unoq, NullComplianceSink(), NullReasoner())
    try:
        slow_gate = _SlowGraspGate()
        controller.coordinator.grasp_gate = slow_gate  # type: ignore[assignment]
        controller.gate = GateStatus(
            fit_ms=2.0, passed=20, total=20, ok=True, vlm_verdict="approve"
        )

        # When a new rescue starts validating, stale ok=True must clear immediately to pending
        ev = make_event("RESCUE_LOGGED", 1, "ev", "logged", {"autonomous": False, "fit_ms": 1.8})
        controller._on_event(ev)
        assert controller.gate.ok is None
        assert controller.gate.passed is None
        assert controller.gate.vlm_verdict == "pending"
        assert controller.gate.vlm_source == "gemini"

        # When Hydrogen finishes early (~4 s), VLM fields populate while ok stays None
        early = GraspGateVerdict(
            verdict="approve",
            say="Clean hook.",
            reason="Flap depressed and bolt cleared.",
            latency_ms=3400.0,
            model="gemini-4-hydrogen",
            source="gemini",
        )
        controller.on_grasp_verdict(early)
        assert controller.gate.ok is None
        assert controller.gate.vlm_verdict == "approve"
        assert controller.gate.vlm_latency_ms == 3400.0
    finally:
        controller.close()
        unoq.close()


def test_coordinator_does_not_freeze_60hz_loop_when_vlm_exceeds_deadline() -> None:
    unoq = UnoQClient.connect_or_mock("127.0.0.1", 0, 0, connect_timeout_s=0.01)
    slow_gate = _SlowGraspGate()
    co = AtcCoordinator.build(
        unoq,
        NullComplianceSink(),
        NullReasoner(),
        config=CoordinatorConfig(gate_deadline_s=0.05),
        grasp_gate=slow_gate,
    )
    co.nonblocking_gate = True
    try:
        # Inject an active gate on Bay 1 whose sim rollout is done, but VLM is still running
        from teleop.atc.coordinator import _Gate
        from teleop.atc.validator import ValidationReport

        bay1 = co._bays[1]
        bay1.state = BayState.VALIDATING
        sim_fut: Future[ValidationReport] = Future()
        sim_fut.set_result(ValidationReport((), 0.01, 0))
        skill = unoq.fit_dmp(
            "ev_latch_v1",
            _dummy_sig(),
            np.array([0.0, 0.5, 1.0]),
            np.tile(np.array([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]), (3, 1)),
            np.zeros((3, 6)),
            np.zeros(3),
        )
        bay1.gate = _Gate(
            skill, sim_fut, deadline=0.02, grasp=slow_gate.pending_future, max_wait_s=5.0
        )

        # Stepping past deadline (0.02 s) must NOT block while pending_future is unresolved
        t0 = time.perf_counter()
        for _ in range(5):
            co.tick(ConsoleInputState())
        dt = time.perf_counter() - t0
        assert dt < 0.5
        assert co.state(1) == BayState.VALIDATING

        # Once VLM resolves, the very next tick applies the gate cleanly
        slow_gate.pending_future.set_result(
            GraspGateVerdict(
                "approve", "Approved.", "Clean", model="gemini-4-hydrogen", source="gemini"
            )
        )
        co.tick(ConsoleInputState())
        assert co.state(1) == BayState.NOMINAL
        assert skill.skill_id in co.library
    finally:
        co.close()
        unoq.close()


def _dummy_sig() -> Any:
    from teleop.atc.types import StallSignature

    return StallSignature(0.01, 0.4, 0.6, 0.2)


def test_console_server_publish_copilot_broadcasts_in_process() -> None:
    from websockets.sync.client import connect

    srv = ConsoleServer(port=0)
    srv.start()
    try:
        with connect(f"ws://127.0.0.1:{srv.port}/ws", open_timeout=5) as ws:
            srv.publish_copilot(
                {
                    "type": "copilot",
                    "bay": 1,
                    "severity": "alarm",
                    "say": "Bay 1 stalled on a flap jam.",
                    "detail": "C=0.18",
                    "source": "model",
                }
            )
            import json

            msg = json.loads(ws.recv(timeout=5))
            assert msg["type"] == "copilot"
            assert msg["bay"] == 1
            assert msg["say"] == "Bay 1 stalled on a flap jam."
            assert msg["seq"] == 1
    finally:
        srv.stop()


def test_hud_formats_argon_audit_line_for_viewer_and_logs() -> None:
    audit = {
        "status": "complete",
        "bay": 1,
        "verdict": "approve",
        "dataset_tag": "approve",
        "operator_note": "Clean hook and HV clearance.",
    }
    line = format_audit(audit)
    assert "AUDIT bay 1: APPROVE [ACT: approve]" in line
    view = HudView(bays=(), queue=(), audit=audit)
    assert any("AUDIT bay 1" in row for row in hud_lines(view))


def test_speaker_precache_cues_async_warms_pcm_cache_without_blocking() -> None:
    from teleop.copilot.speech import Speaker

    done = threading.Event()

    class FakeStreamer:
        def stream_pcm(self, text: str, **_: Any) -> Any:
            del text
            yield b"\x00\x00" * 120
            done.set()

    spk = Speaker(FakeStreamer(), start=False)
    try:
        thread = spk.precache_cues_async(["Bay 1 stalled."])
        assert thread is not None
        assert done.wait(timeout=3.0)
        thread.join(timeout=3.0)
        assert "Bay 1 stalled." in spk._pcm_cache
    finally:
        spk.close()


def test_grasp_gate_submit_audit_only_renders_and_submits_without_speaking() -> None:
    submitted: list[dict[str, Any]] = []
    spoken: list[str] = []

    class RecAudit:
        def submit(self, **kwargs: Any) -> str:
            submitted.append(kwargs)
            return "bay2-001"

    class RecSpeaker:
        def say(self, text: str, **_: Any) -> int:
            spoken.append(text)
            return 1

    gate = GraspGate(
        None,
        cfg=CopilotConfig(no_images=True),
        speaker=RecSpeaker(),
        audit_reviewer=RecAudit(),
    )
    try:
        snap = PoseSnapshot("ood-refusal", 1.2, 0.008, 0.55, *(np.zeros(1) for _ in range(5)))
        fut = gate.submit_audit_only(
            bay_id=2,
            skill_id="ood_refused",
            snapshots=(snap,),
            timeline={"event": "OOD_REFUSAL"},
        )
        assert fut is not None
        assert fut.result(timeout=3.0) == "bay2-001"
        assert len(submitted) == 1
        assert submitted[0]["bay"] == 2
        assert submitted[0]["skill_id"] == "ood_refused"
        assert spoken == []
    finally:
        gate.close()
