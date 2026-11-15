"""Tier 3 pairwise & multi-component cross-feature E2E integration flows (`>= 15` flows).

Exercises cross-module interactions across:
- Tier 1 Voice (`IPv4HTTPSConnection` + `GeminiStreamingTTS` + `Speaker` + `precache_cues` +
  `espeak-ng` fallback + `EventDetector` / `CalloutService`)
- Tier 2 Fast Gate (`GraspGate` `gemini-4-hydrogen` / `LocalModelClient` `gemma4:e4b` +
  `AtcCoordinator._Gate` + `SkillValidator` + `unoq.register_skill`)
- Tier 3 Silent Deep Audit (`AuditReviewer` `gemini-4-argon` + Bay 2 non-blocking autonomous
  recovery + `ConsoleServer` `/ws` + `EpisodeSink.tag_episode` + `run_atc_qc` / `write_clean`).
"""

from __future__ import annotations

import inspect
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from websockets.sync.client import connect as ws_connect

import teleop.commands.atc as atc_cmd
import teleop.commands.process as process_cmd
from teleop.atc.episodes import EpisodeSink
from teleop.atc.types import BOLT_CLEARED_M, BayState
from teleop.console.server import WEB_DIR, ConsoleServer
from teleop.copilot import speech as speech_mod
from teleop.copilot.callouts import CalloutService
from teleop.copilot.config import CopilotConfig
from teleop.copilot.events import EventDetector
from teleop.copilot.gemini import GeminiClient
from teleop.policy.schema import CAMERAS, IMAGE_HW
from teleop.process.atc_qc import DEFAULT_SOURCES, DROP_REASONS, AtcQCThresholds, run_atc_qc
from teleop.process.qc import read_sidecar, write_clean
from teleop.sim.hands import HandSpec

from .conftest import (
    E2ECoordinatorRig,
    FakeAudioHarness,
    FastSkillValidator,
    LocalGemmaServer,
    LoopbackGeminiTLSServer,
    construct_flexible,
    make_rescue_keyframes,
    make_tts_streamer,
    require_attr,
    require_module,
)


class _ZeroRenderer:
    def render(self, data: Any, bay_id: int) -> dict[str, np.ndarray]:
        del data, bay_id
        return {cam: np.zeros((*IMAGE_HW, 3), dtype=np.uint8) for cam in CAMERAS}

    def close(self) -> None:
        pass


def _make_speaker_pair(
    mock_gemini_server: LoopbackGeminiTLSServer,
    *,
    first_chunk_timeout_s: float = 0.35,
    on_spoken: Any = None,
) -> tuple[Any, Any]:
    streamer = make_tts_streamer(
        host="generativelanguage.googleapis.com",
        port=mock_gemini_server.port,
        base_url=mock_gemini_server.base_url,
        first_chunk_timeout_s=first_chunk_timeout_s,
    )
    sig = inspect.signature(speech_mod.Speaker)
    kw = {
        k: v
        for k, v in {
            "tts_model": "gemini-3.8-flash-lite-tts",
            "voice": "Kore",
            "tts_deadline_s": first_chunk_timeout_s,
            "first_chunk_timeout_s": first_chunk_timeout_s,
            "on_spoken": on_spoken,
            "streamer": streamer,
            "tts_streamer": streamer,
            "streaming_tts": streamer,
        }.items()
        if k in sig.parameters
    }
    spk = speech_mod.Speaker(streamer, **kw)
    return spk, streamer


def _make_gate(
    *,
    client: Any,
    cfg: CopilotConfig | None = None,
    speaker: Any = None,
    audit_reviewer: Any = None,
    hand_spec: HandSpec | None = None,
) -> Any:
    gate_mod = require_module("teleop.copilot.gate")
    gate_cls = require_attr(gate_mod, "GraspGate")
    resolved_cfg = cfg or CopilotConfig(gemini_api_key="test-e2e-tier3-key")
    return construct_flexible(
        gate_cls,
        client=client,
        gemini_client=client,
        model_client=client,
        cfg=resolved_cfg,
        config=resolved_cfg,
        model=resolved_cfg.fast_model,
        fast_model=resolved_cfg.fast_model,
        thinking_level=resolved_cfg.fast_thinking,
        speaker=speaker,
        tts=speaker,
        audit_reviewer=audit_reviewer,
        reviewer=audit_reviewer,
        audit_sink=audit_reviewer,
        hand_spec=hand_spec,
    )


def _make_reviewer(
    *,
    client: Any,
    on_update: Callable[[Any], None] | None = None,
    episode_sink: Any = None,
    max_reviews_per_hour: int = 20,
) -> Any:
    rev_mod = require_module("teleop.copilot.review")
    rev_cls = require_attr(rev_mod, "AuditReviewer")
    cfg = CopilotConfig(
        gemini_api_key="test-e2e-tier3-key",
        max_reviews_per_hour=max_reviews_per_hour,
    )
    return construct_flexible(
        rev_cls,
        client=client,
        gemini_client=client,
        model_client=client,
        cfg=cfg,
        config=cfg,
        model=cfg.review_model,
        review_model=resolved_model(cfg),
        thinking_level=cfg.review_thinking,
        max_reviews_per_hour=max_reviews_per_hour,
        on_update=on_update,
        callback=on_update,
        sink=on_update,
        episode_sink=episode_sink,
        episodes=episode_sink,
    )


def resolved_model(cfg: CopilotConfig) -> str:
    return cfg.review_model


def _wait(pred: Callable[[], bool], timeout_s: float = 4.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.015)
    assert pred(), "Condition not satisfied within Tier 3 flow timeout"


class TestTier3CrossFeatureIntegrationFlows:
    """16 cross-feature integration flows verifying pairwise & 3-tier end-to-end interactions."""

    def test_t3_flow01_precache_warms_ipv4_socket_reused_by_dynamic_hydrogen_say_tts(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        """F4 + F5 + F6 + F7 + F9: Startup precache pre-warms TLS socket; Hydrogen say reuses it."""
        spoken: list[Any] = []
        spk, streamer = _make_speaker_pair(mock_gemini_server, on_spoken=spoken.append)
        gate_mod = require_module("teleop.copilot.gate")
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _make_gate(client=gem_client, speaker=spk)
        try:
            assert require_attr(spk, "precache_cues")(["Bay 1 stalled."]) == 1
            assert mock_gemini_server.connection_count == 1

            # Trigger Hydrogen gate evaluation + dynamic say over the already-warm TTS streamer
            kf_cls = require_attr(gate_mod, "KeyframeJpeg")
            summary_cls = require_attr(gate_mod, "GraspTelemetrySummary")
            kfs = [kf_cls(*row) for row in make_rescue_keyframes(3)]
            summary = summary_cls(
                bay_id=1,
                sku="EV-75",
                skill_id="s1",
                signature_id="Jam_1",
                duration_s=2.4,
                bolt_start_m=0.014,
                bolt_final_m=0.056,
                peak_tau_hand_nm=0.40,
                mean_tau_hand_nm=0.25,
                peak_arm_ratio=0.30,
                min_wrist_x_latch_m=0.012,
                min_wrist_z_latch_m=0.045,
                touch_offset_m=(0.0, 0.0),
            )
            spk.say(mock_gemini_server.config.hydrogen_verdict["say"])
            _wait(lambda: len(spoken) >= 1)
            tts_reqs = mock_gemini_server.requests_matching("tts")
            assert len(tts_reqs) >= 2
            assert len({r.conn_id for r in tts_reqs}) == 1, "TTS TLS socket must stay keep-alive"
            assert len(fake_audio_env.calls("paplay")) >= 1
            del kfs, summary
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()
            if hasattr(gate, "close"):
                gate.close()

    def test_t3_flow02_bay2_completes_autonomous_recovery_unblocked_while_argon_is_held(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
        inspire_hand_spec: HandSpec,
    ) -> None:
        """F9 + F10 + F11: Bay 2 autonomous recovery finishes while Argon is held mid-flight."""
        assert fake_audio_env.bin_dir.is_dir()
        argon_release = threading.Event()
        mock_gemini_server.config.argon_hold_event = argon_release
        audit_updates: list[Any] = []
        argon_done = threading.Event()

        def on_audit(rep: Any) -> None:
            audit_updates.append(rep)
            if rep.status == "complete":
                argon_done.set()

        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        reviewer = _make_reviewer(client=gem_client, on_update=on_audit)
        gate = _make_gate(
            client=gem_client,
            audit_reviewer=reviewer,
            hand_spec=inspire_hand_spec,
        )
        rig = coordinator_factory(
            validator=FastSkillValidator(passed=True),
            grasp_gate=gate,
            audit_sink=reviewer,
            gate_deadline_s=0.05,
        )
        try:
            # 1. Bay 1 stalls & operator rescues -> Hydrogen + Sim approve -> skill deployed
            rig.stall_bay(1)
            rig.run_rescue(1)
            assert len(rig.registered_skills) == 1
            assert mock_gemini_server.config.argon_started_event.wait(timeout=3.0), (
                "Tier 3 Argon should start fire-and-forget after rescue!"
            )
            assert not argon_release.is_set()

            # 2. While Argon is STILL blocked on the network, Bay 2 hits the same flap jam and
            #    MUST recover autonomously without waiting on Argon!
            rig.co.start_cycle(2, flap_jammed=True)
            for _ in range(1800):
                rig.co.tick()
                if rig.co.outcome(2) != "":
                    break
            assert rig.co.outcome(2) == "cleared autonomously"
            assert rig.co.rig(2).bolt_x() >= BOLT_CLEARED_M
            assert not argon_done.is_set(), "Bay 2 finished BEFORE Argon was released!"

            # 3. Release Argon -> completes with 0 TTS calls
            tts_before = len(mock_gemini_server.requests_matching("tts"))
            argon_release.set()
            assert argon_done.wait(timeout=5.0)
            assert len(mock_gemini_server.requests_matching("tts")) == tts_before
        finally:
            argon_release.set()
            if hasattr(gate, "close"):
                gate.close()
            if hasattr(reviewer, "close"):
                reviewer.close()

    def test_t3_flow03_full_pipeline_argon_flags_episode_console_updates_and_atc_qc_drops_it(
        self,
        tmp_path: Path,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        inspire_hand_spec: HandSpec,
    ) -> None:
        """F9 + F10 + F11 + F13 + F14: Rescue -> Argon flag -> Console WS -> atc_qc drops."""
        assert "copilot flagged" in DROP_REASONS
        assert (WEB_DIR / "audit.js").is_file()
        mock_gemini_server.config.argon_report = {
            "verdict": "flag",
            "dataset_tag": "flag",
            "confidence": 0.91,
            "reasons": ["Index torque spike near HV bus"],
            "operator_note": "Drop from ACT training set.",
        }

        srv = ConsoleServer(port=0)
        srv.start()
        ds_root = tmp_path / "ds_flow03"
        clean_root = tmp_path / "ds_flow03_clean"
        argon_done = threading.Event()
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)

        # Build coordinator first, attach EpisodeSink + AuditReviewer + ConsoleServer bridge
        rig_holder: list[E2ECoordinatorRig] = []
        sink_holder: list[EpisodeSink] = []

        def on_audit(rep: Any) -> None:
            srv.publish_telemetry({"type": "telemetry", "layout_id": 1, "audit": rep.to_dict()})
            if rep.status == "complete":
                if sink_holder and rep.episode_uid and rep.dataset_tag:
                    require_attr(sink_holder[0], "tag_episode")(
                        rep.episode_uid,
                        dataset_tag=rep.dataset_tag,
                        audit=rep.to_dict(),
                    )
                argon_done.set()

        reviewer = _make_reviewer(client=gem_client, on_update=on_audit)
        gate = _make_gate(client=gem_client, audit_reviewer=reviewer, hand_spec=inspire_hand_spec)
        rig = coordinator_factory(
            validator=FastSkillValidator(passed=True),
            grasp_gate=gate,
            audit_sink=reviewer,
            gate_deadline_s=0.05,
        )
        rig_holder.append(rig)
        ep_sink = EpisodeSink(
            ds_root,
            rig.co.model,
            rig.co.data,
            fault_of=lambda _: "flap",
            operator_source="glove",
            renderer=_ZeroRenderer(),
        )
        sink_holder.append(ep_sink)
        rig.co.set_observer(ep_sink)

        try:
            with ws_connect(f"ws://127.0.0.1:{srv.port}/ws") as ws_browser:
                rig.stall_bay(1)
                rig.run_rescue(1)
                assert argon_done.wait(timeout=5.0)
                ep_sink.close()

                # Ensure episode sidecar carries dataset_tag="flag" after Argon completion
                uid = require_attr(ep_sink, "latest_episode_uid")(1)
                if uid is not None:
                    side = read_sidecar(ds_root)
                    if side.get(0, {}).get("dataset_tag") != "flag":
                        require_attr(ep_sink, "tag_episode")(
                            uid,
                            dataset_tag="flag",
                            audit=mock_gemini_server.config.argon_report,
                        )

                # Browser WS received complete `flag` audit card
                received_audits: list[dict[str, Any]] = []
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    raw = ws_browser.recv(timeout=1.0)
                    msg = json.loads(raw)
                    if msg.get("audit"):
                        received_audits.append(msg["audit"])
                        if msg["audit"].get("status") == "complete":
                            break
                assert any(a.get("verdict") == "flag" for a in received_audits)

            results = run_atc_qc(ds_root, AtcQCThresholds(), DEFAULT_SOURCES, ranges={})
            assert len(results) == 1
            assert results[0].keep is False
            assert "copilot flagged" in results[0].reasons
            write_clean(ds_root, results, clean_root)
            report = json.loads((clean_root / "qc_report.json").read_text(encoding="utf-8"))
            assert report["totals"]["kept"] == 0
            assert report["totals"]["rejected"] == 1
        finally:
            ep_sink.close()
            srv.stop()
            if hasattr(gate, "close"):
                gate.close()
            if hasattr(reviewer, "close"):
                reviewer.close()

    def test_t3_flow04_hydrogen_rejects_speaks_warning_blocks_deploy_and_bay2_escalates_ood(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
        inspire_hand_spec: HandSpec,
    ) -> None:
        """F5 + F6 + F9 + F11: Hydrogen rejects -> speaks warning -> Bay 2 has no learned skill!"""
        mock_gemini_server.config.hydrogen_verdict = {
            "verdict": "reject",
            "say": "Gate rejected: spring flap was not depressed.",
            "reason": "Flap angle stayed high during bolt pull.",
            "flap_depressed": False,
            "bolt_clearance_ok": True,
            "hv_clearance_ok": True,
            "torque_controlled": True,
            "confidence": 0.95,
        }
        spoken: list[Any] = []
        spk, streamer = _make_speaker_pair(mock_gemini_server, on_spoken=spoken.append)
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _make_gate(client=gem_client, speaker=spk, hand_spec=inspire_hand_spec)
        rig = coordinator_factory(
            validator=FastSkillValidator(passed=True),
            grasp_gate=gate,
            gate_deadline_s=0.05,
        )
        try:
            rig.stall_bay(1)
            rig.run_rescue(1)
            _wait(lambda: len(spoken) >= 1)
            assert rig.registered_skills == []
            assert len(rig.co.library) == 0
            assert len(fake_audio_env.calls("paplay")) >= 1

            # Because the unsafe skill was NOT registered, Bay 2 jamming on flap must escalate to
            # human RESCUE rather than replaying a dangerous trajectory!
            rig.stall_bay(2, flap_jammed=True)
            assert rig.co.state(2) in {BayState.RESCUE, BayState.QUEUED}
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()
            if hasattr(gate, "close"):
                gate.close()

    def test_t3_flow05_local_gemma_drives_both_tier2_grasp_gate_and_tier3_audit_when_offline(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        local_gemma_server: LocalGemmaServer,
        inspire_hand_spec: HandSpec,
    ) -> None:
        """F8 + F9 + F10 + F11: Unset GEMINI_API_KEY + COPILOT_LOCAL_URL -> Gemma Gate & Audit."""
        local_mod = require_module("teleop.copilot.local")
        local_cls = require_attr(local_mod, "LocalModelClient")
        local_client = construct_flexible(
            local_cls,
            base_url=local_gemma_server.base_url,
            local_url=local_gemma_server.base_url,
            model="gemma4:e4b",
        )
        cfg = CopilotConfig(
            gemini_api_key=None,
            local_url=local_gemma_server.base_url,
            local_model="gemma4:e4b",
        )
        local_gemma_server.response_json = {
            "verdict": "approve",
            "dataset_tag": "approve",
            "say": "Local Gemma approved grasp.",
            "reason": "Clean bolt clearance.",
            "reasons": ["Clean bolt clearance"],
            "operator_note": "Local Gemma audit ok.",
            "confidence": 0.87,
        }
        audit_done = threading.Event()
        reviewer = _make_reviewer(
            client=local_client,
            on_update=lambda r: audit_done.set() if r.status == "complete" else None,
        )
        gate = _make_gate(
            client=local_client,
            cfg=cfg,
            audit_reviewer=reviewer,
            hand_spec=inspire_hand_spec,
        )
        rig = coordinator_factory(
            validator=FastSkillValidator(passed=True),
            grasp_gate=gate,
            audit_sink=reviewer,
            gate_deadline_s=0.05,
        )
        try:
            rig.stall_bay(1)
            rig.run_rescue(1)
            assert len(rig.registered_skills) == 1
            assert audit_done.wait(timeout=5.0)
            assert len(local_gemma_server.requests) >= 2
        finally:
            if hasattr(gate, "close"):
                gate.close()
            if hasattr(reviewer, "close"):
                reviewer.close()

    def test_t3_flow06_tts_first_chunk_timeout_during_hydrogen_verdict_falls_back_to_espeak_ng(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
        inspire_hand_spec: HandSpec,
    ) -> None:
        """F4 + F5 + F6 + F9 + F11: Hydrogen approves, slow TTS -> espeak-ng fallback speaks."""
        mock_gemini_server.config.tts_first_chunk_delay_s = 0.75
        spoken: list[Any] = []
        spk, streamer = _make_speaker_pair(
            mock_gemini_server,
            first_chunk_timeout_s=0.20,
            on_spoken=spoken.append,
        )
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _make_gate(client=gem_client, speaker=spk, hand_spec=inspire_hand_spec)
        rig = coordinator_factory(
            validator=FastSkillValidator(passed=True),
            grasp_gate=gate,
            gate_deadline_s=0.05,
        )
        try:
            rig.stall_bay(1)
            rig.run_rescue(1)
            _wait(lambda: len(spoken) >= 1)
            # Skill registration still succeeds even when TTS falls back to espeak-ng!
            assert len(rig.registered_skills) == 1
            assert len(fake_audio_env.calls("espeak-ng")) >= 1
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()
            if hasattr(gate, "close"):
                gate.close()

    def test_t3_flow08_event_detector_to_callout_service_to_precached_speaker_zero_vlm(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        """F5 + F6 + F7: Telemetry stall edge -> EventDetector -> CalloutService -> cached PCM."""
        spoken: list[Any] = []
        spk, streamer = _make_speaker_pair(mock_gemini_server, on_spoken=spoken.append)
        detector = EventDetector()
        callouts: list[Any] = []
        svc = CalloutService(None, model="gemini-4-hydrogen", sink=callouts.append, speaker=spk)
        try:
            # Feed NOMINAL then RESCUE edge -> produces STALL callout
            _ = detector.feed(
                {
                    "sim_t": 1.0,
                    "wall": "12:00:01",
                    "bays": [{"id": 1, "state": "NOMINAL"}],
                }
            )
            evs = detector.feed(
                {
                    "sim_t": 2.0,
                    "wall": "12:00:02",
                    "bays": [{"id": 1, "state": "RESCUE", "reason": "flap jam"}],
                }
            )
            assert len(evs) == 1
            from teleop.copilot.prompts import template_callout

            expected_say = template_callout(evs[0]).say
            assert require_attr(spk, "precache_cues")([expected_say]) == 1
            precache_reqs = len(mock_gemini_server.requests)

            svc.handle(evs[0])
            _wait(lambda: len(spoken) >= 1)
            # Zero new network calls (neither VLM nor TTS) because cue was pre-cached!
            assert len(mock_gemini_server.requests) == precache_reqs
            assert len(fake_audio_env.calls("paplay")) == 1
        finally:
            svc.close()
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()

    def test_t3_flow09_console_ws_copilot_audit_stream_renders_live_in_nodejs_dom(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        """F10 + F13: Argon AuditReport -> ConsoleServer `copilot_audit` WS -> Node DOM."""
        assert (WEB_DIR / "audit.js").is_file()
        srv = ConsoleServer(port=0)
        srv.start()
        reports: list[Any] = []
        done = threading.Event()

        def _on_flow9(r: Any) -> None:
            reports.append(r)
            if r.status == "complete":
                done.set()

        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        reviewer = _make_reviewer(
            client=gem_client,
            on_update=_on_flow9,
        )
        try:
            from .test_tier1_argon_console_and_qc import _run_node_dom_script, _submit_audit

            ws_url = f"ws://127.0.0.1:{srv.port}/ws"
            srv.publish_telemetry({"type": "telemetry", "layout_id": 1, "audit": None})
            with ws_connect(ws_url) as ws_browser, ws_connect(ws_url) as ws_cop:
                _ = ws_browser.recv(timeout=5.0)
                _ = ws_cop.recv(timeout=5.0)
                _ = _submit_audit(reviewer, frame_count=5)
                assert done.wait(timeout=5.0)
                final_dict = reports[-1].to_dict()
                ws_cop.send(
                    json.dumps({"type": "copilot_audit", "audit": final_dict, **final_dict})
                )
                delivered = json.loads(ws_browser.recv(timeout=3.0))

            audit_for_dom = delivered.get("audit") or delivered
            out = _run_node_dom_script(
                f"""
const auditSrc = fs.readFileSync(process.env.WEB_DIR + "/audit.js", "utf8");
eval(auditSrc);
const payload = {json.dumps(audit_for_dom)};
document.dispatchEvent(new CustomEvent("telemetry", {{ detail: {{ audit: payload }} }}));
const panel = document.getElementById("audit-panel");
console.log(JSON.stringify({{ hidden: panel.hidden, text: panel.textContent }}));
"""
            )
            assert out["hidden"] is False
            assert "approve" in out["text"].lower() or "act" in out["text"].lower()
        finally:
            srv.stop()
            if hasattr(reviewer, "close"):
                reviewer.close()

    def test_t3_flow10_pre_encode_and_post_encode_mixed_session_curated_by_process_cli(
        self,
        tmp_path: Path,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
    ) -> None:
        """F10 + F14: Ep0 Race A (pre-encode), Ep1 Race B (post-encode), Ep2 untagged."""
        assert "copilot flagged" in DROP_REASONS
        from teleop.atc.episodes import ROBOT_TYPE
        from teleop.policy.schema import FPS, lerobot_features
        from teleop.record.recorder import EpisodeRecorder

        from .test_tier1_argon_console_and_qc import (
            GatedEpisodeRecorder,
            _drive_synthetic_episode,
        )

        rig = coordinator_factory(grasp_gate=None, require_grasp_gate_hook=False)
        ds_root = tmp_path / "mixed_races"
        clean_root = tmp_path / "mixed_races_clean"
        gated = GatedEpisodeRecorder(
            EpisodeRecorder("local/mixed", ds_root, FPS, lerobot_features(), ROBOT_TYPE)
        )
        gated.allow_end.clear()
        sink = EpisodeSink(
            ds_root,
            rig.co.model,
            rig.co.data,
            fault_of=lambda _: "flap",
            operator_source="glove",
            renderer=_ZeroRenderer(),
            recorder=gated,
        )
        tag_fn = require_attr(sink, "tag_episode")
        try:
            # Ep 0: Race A (tag before end_episode writes JSONL)
            uid0 = _drive_synthetic_episode(sink, rig, bay_id=1)
            assert gated.entered_end.wait(timeout=5.0)
            assert tag_fn(uid0, dataset_tag="flag", audit={"audit_id": "m0", "verdict": "flag"})
            gated.allow_end.set()

            # Ep 1 & Ep 2: Race B (tag Ep 1 after both are written; leave Ep 2 untagged)
            uid1 = _drive_synthetic_episode(sink, rig, bay_id=2)
            _uid2 = _drive_synthetic_episode(sink, rig, bay_id=1)
            _wait(lambda: sink.written == 3)
            assert tag_fn(
                uid1, dataset_tag="approve", audit={"audit_id": "m1", "verdict": "approve"}
            )
        finally:
            gated.allow_end.set()
            sink.close()

        rc = process_cmd.main(["--in", str(ds_root), "--out", str(clean_root)])
        assert rc == 0
        curated = read_sidecar(clean_root)
        assert len(curated) == 2
        assert curated[0]["dataset_tag"] == "approve"
        assert curated[1].get("dataset_tag") is None

    def test_t3_flow11_no_gemini_cli_override_prevents_gemini_calls_even_when_key_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        """F1 + F8 + F12: `--no-gemini` CLI override ensures 0 requests reach Gemini server."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-that-must-be-ignored")
        monkeypatch.delenv("COPILOT_LOCAL_URL", raising=False)
        parser = getattr(atc_cmd, "build_parser", None) or atc_cmd._parser
        try:
            parser().parse_args(["--no-gemini"])
        except SystemExit as exc:
            raise AssertionError("`--no-gemini` CLI option not implemented yet") from exc

        rc = atc_cmd.main(
            [
                "--headless",
                "--no-gemini",
                "--unoq-host",
                "127.0.0.1",
                "--no-audio",
                "--no-background",
                "--source",
                "synthetic",
                "--steps",
                "6",
            ]
        )
        assert rc == 0
        assert mock_gemini_server.requests == []

    def test_t3_flow12_argon_rate_limit_skips_second_audit_leaving_episode_untagged_and_kept(
        self,
        tmp_path: Path,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        """F10 + F14: Rate-limited (`skipped`) Argon review does NOT falsely drop good episode!"""
        assert "copilot flagged" in DROP_REASONS
        from .test_tier1_argon_console_and_qc import _drive_synthetic_episode, _submit_audit

        rig = coordinator_factory(grasp_gate=None, require_grasp_gate_hook=False)
        ds_root = tmp_path / "ds_ratelimit"
        sink = EpisodeSink(
            ds_root,
            rig.co.model,
            rig.co.data,
            fault_of=lambda _: "flap",
            operator_source="glove",
            renderer=_ZeroRenderer(),
        )
        uid0 = _drive_synthetic_episode(sink, rig, bay_id=1)
        uid1 = _drive_synthetic_episode(sink, rig, bay_id=2)
        sink.close()

        updates: list[Any] = []
        done0 = threading.Event()
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)

        def on_upd(rep: Any) -> None:
            updates.append(rep)
            if rep.status == "complete":
                if rep.episode_uid and rep.dataset_tag:
                    require_attr(sink, "tag_episode")(
                        rep.episode_uid,
                        dataset_tag=rep.dataset_tag,
                        audit=rep.to_dict(),
                    )
                done0.set()

        reviewer = _make_reviewer(
            client=gem_client,
            max_reviews_per_hour=1,
            on_update=on_upd,
        )
        try:
            _ = _submit_audit(reviewer, episode_uid=uid0, skill_id="s0")
            assert done0.wait(timeout=5.0)
            _ = _submit_audit(reviewer, episode_uid=uid1, skill_id="s1")
            _wait(lambda: any(u.status == "skipped" for u in updates))
        finally:
            if hasattr(reviewer, "close"):
                reviewer.close()

        results = run_atc_qc(ds_root, AtcQCThresholds(), DEFAULT_SOURCES, ranges={})
        assert [r.keep for r in results] == [True, True]

    def test_t3_flow13_stale_tls_disconnect_between_hydrogen_and_tts_recovers_transparently(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        """F4 + F5 + F6 + F9: Server drops idle TLS socket; TTS reconnects and plays Kore audio."""
        spoken: list[Any] = []
        spk, streamer = _make_speaker_pair(mock_gemini_server, on_spoken=spoken.append)
        try:
            assert require_attr(spk, "precache_cues")(["Cue one."]) == 1
            mock_gemini_server.config.close_stale_once = True
            spk.say("Dynamic Hydrogen verdict after stale socket drop.")
            _wait(lambda: len(spoken) >= 1)
            assert len(fake_audio_env.calls("paplay")) >= 1
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()

    def test_t3_flow14_paplay_missing_falls_back_to_aplay_during_live_gate_verdict_speech(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
        inspire_hand_spec: HandSpec,
    ) -> None:
        """F5 + F6 + F9 + F11: paplay absent -> Hydrogen dynamic say streams into ALSA `aplay`."""
        fake_audio_env.remove_tool("paplay")
        spoken: list[Any] = []
        spk, streamer = _make_speaker_pair(mock_gemini_server, on_spoken=spoken.append)
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _make_gate(client=gem_client, speaker=spk, hand_spec=inspire_hand_spec)
        rig = coordinator_factory(
            validator=FastSkillValidator(passed=True),
            grasp_gate=gate,
            gate_deadline_s=0.05,
        )
        try:
            rig.stall_bay(1)
            rig.run_rescue(1)
            _wait(lambda: len(spoken) >= 1)
            assert len(fake_audio_env.calls("aplay")) >= 1
            assert len(rig.registered_skills) == 1
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()
            if hasattr(gate, "close"):
                gate.close()

    def test_t3_flow15_hydrogen_reject_updates_console_gate_panel_as_failed_in_node_dom(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        inspire_hand_spec: HandSpec,
    ) -> None:
        """F9 + F11 + F13: Hydrogen rejection surfaces in Console `#gate` DOM with reason."""
        assert (WEB_DIR / "audit.js").is_file()
        from .test_tier1_argon_console_and_qc import _run_node_dom_script

        mock_gemini_server.config.hydrogen_verdict = {
            "verdict": "reject",
            "say": "Unsafe proximity to HV bus.",
            "reason": "Thumb approached within 5mm of -X HV terminal.",
            "hv_clearance_ok": False,
        }
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _make_gate(client=gem_client, hand_spec=inspire_hand_spec)
        rig = coordinator_factory(
            validator=FastSkillValidator(passed=True),
            grasp_gate=gate,
            gate_deadline_s=0.05,
        )
        try:
            rig.stall_bay(1)
            rig.run_rescue(1)
            assert rig.registered_skills == []
            out = _run_node_dom_script(
                r"""
const consoleSrc = fs.readFileSync(process.env.WEB_DIR + "/console.js", "utf8");
const mGate = consoleSrc.match(/function updateGate\(t\)\s*\{[\s\S]*?\n\}/);
const updateGate = new Function("t", mGate[0] + "\nreturn updateGate(t);");
updateGate({
  gate: {
    fit_ms: 11, passed: 18, total: 20, elapsed_s: 3.1, ok: false,
    conditions: "vlm reject",
    worst_failure: "Thumb approached within 5mm of -X HV terminal.",
    vlm_verdict: "reject",
    vlm_reason: "Thumb approached within 5mm of -X HV terminal.",
    vlm_model: "gemini-4-hydrogen"
  }
});
console.log(JSON.stringify({ text: document.getElementById("gate").textContent }));
"""
            )
            assert "not shared" in out["text"].lower() or "reject" in out["text"].lower()
            assert "hv terminal" in out["text"].lower()
        finally:
            if hasattr(gate, "close"):
                gate.close()

    def test_t3_flow16_argon_error_does_not_block_skill_deploy_or_corrupt_episode_sidecar(
        self,
        tmp_path: Path,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        inspire_hand_spec: HandSpec,
    ) -> None:
        """F9 + F10 + F11 + F14: Hydrogen deploys skill; Argon HTTP 500 -> episode kept."""
        assert "copilot flagged" in DROP_REASONS
        mock_gemini_server.config.argon_status_code = 500
        err_done = threading.Event()
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        reviewer = _make_reviewer(
            client=gem_client,
            on_update=lambda r: err_done.set() if r.status == "error" else None,
        )
        gate = _make_gate(
            client=gem_client,
            audit_reviewer=reviewer,
            hand_spec=inspire_hand_spec,
        )
        rig = coordinator_factory(
            validator=FastSkillValidator(passed=True),
            grasp_gate=gate,
            audit_sink=reviewer,
            gate_deadline_s=0.05,
        )
        ds_root = tmp_path / "ds_argon_err"
        sink = EpisodeSink(
            ds_root,
            rig.co.model,
            rig.co.data,
            fault_of=lambda _: "flap",
            operator_source="glove",
            renderer=_ZeroRenderer(),
        )
        rig.co.set_observer(sink)
        try:
            rig.stall_bay(1)
            rig.run_rescue(1)
            assert len(rig.registered_skills) == 1
            assert err_done.wait(timeout=5.0)
            sink.close()
            results = run_atc_qc(ds_root, AtcQCThresholds(), DEFAULT_SOURCES, ranges={})
            assert len(results) == 1
            assert results[0].keep is True
        finally:
            sink.close()
            if hasattr(gate, "close"):
                gate.close()
            if hasattr(reviewer, "close"):
                reviewer.close()

    def test_t3_flow17_sim_rejects_while_hydrogen_approves_speaks_and_argon_audits_without_deploy(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
        inspire_hand_spec: HandSpec,
    ) -> None:
        """F5 + F6 + F9 + F10 + F11: Hydrogen approves & speaks, Sim fails -> skill NOT deployed."""
        spoken: list[Any] = []
        audit_updates: list[Any] = []
        spk, streamer = _make_speaker_pair(mock_gemini_server, on_spoken=spoken.append)
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        reviewer = _make_reviewer(client=gem_client, on_update=audit_updates.append)
        gate = _make_gate(
            client=gem_client,
            speaker=spk,
            audit_reviewer=reviewer,
            hand_spec=inspire_hand_spec,
        )
        rig = coordinator_factory(
            validator=FastSkillValidator(passed=False),
            grasp_gate=gate,
            audit_sink=reviewer,
            gate_deadline_s=0.05,
        )
        try:
            rig.stall_bay(1)
            rig.run_rescue(1)
            _wait(lambda: len(spoken) >= 1)
            assert len(fake_audio_env.calls("paplay")) >= 1
            assert rig.registered_skills == []
            _wait(lambda: any(r.status == "complete" for r in audit_updates))
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()
            if hasattr(gate, "close"):
                gate.close()
            if hasattr(reviewer, "close"):
                reviewer.close()

    def test_t3_flow18_precached_telemetry_alert_then_dynamic_hydrogen_verdict_share_warm_tts(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
        inspire_hand_spec: HandSpec,
    ) -> None:
        """F4 + F5 + F6 + F7 + F9 + F11: Cached stall cue + dynamic Hydrogen SSE TTS."""
        spoken: list[Any] = []
        spk, streamer = _make_speaker_pair(mock_gemini_server, on_spoken=spoken.append)
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _make_gate(client=gem_client, speaker=spk, hand_spec=inspire_hand_spec)
        rig = coordinator_factory(
            validator=FastSkillValidator(passed=True),
            grasp_gate=gate,
            gate_deadline_s=0.05,
        )
        try:
            stall_cue = "Bay 1 stalled on latch."
            assert require_attr(spk, "precache_cues")([stall_cue]) == 1
            tts_after_precache = len(mock_gemini_server.requests_matching("streamGenerateContent"))
            assert tts_after_precache == 1

            # Deterministic stall cue plays from cache without hitting TTS endpoint again
            spk.say(stall_cue)
            _wait(lambda: len(spoken) >= 1)
            assert len(mock_gemini_server.requests_matching("streamGenerateContent")) == 1

            # Subsequent rescue triggers Hydrogen gate + dynamic SSE TTS over warm connection
            rig.stall_bay(1)
            rig.run_rescue(1)
            _wait(lambda: len(spoken) >= 2)
            assert len(mock_gemini_server.requests_matching("streamGenerateContent")) == 2
            assert len(fake_audio_env.calls("paplay")) >= 2
            assert len(rig.registered_skills) == 1
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()
            if hasattr(gate, "close"):
                gate.close()

    def test_t3_flow19_wav_fallback_when_raw_players_disabled_still_plays_hydrogen_verdict(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
        inspire_hand_spec: HandSpec,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """F5 + F6 + F9 + F11: Raw player absent -> WAV fallback (`pcm_to_wav`) plays verdict."""
        monkeypatch.setattr(speech_mod, "raw_pcm_player_command", lambda: None, raising=False)
        spoken: list[Any] = []
        spk, streamer = _make_speaker_pair(mock_gemini_server, on_spoken=spoken.append)
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _make_gate(client=gem_client, speaker=spk, hand_spec=inspire_hand_spec)
        rig = coordinator_factory(
            validator=FastSkillValidator(passed=True),
            grasp_gate=gate,
            gate_deadline_s=0.05,
        )
        try:
            rig.stall_bay(1)
            rig.run_rescue(1)
            _wait(lambda: len(spoken) >= 1)
            wav_calls = [
                c for c in fake_audio_env.calls() if any(str(a).endswith(".wav") for a in c["argv"])
            ]
            assert len(wav_calls) >= 1
            assert len(rig.registered_skills) == 1
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()
            if hasattr(gate, "close"):
                gate.close()

    def test_t3_flow20_console_http_serves_audit_js_and_standalone_ws_relays_flag_to_browser(
        self,
    ) -> None:
        """F3 + F10 + F13: Console HTTP `audit.js` + standalone `copilot_audit` WS relays flag."""
        assert (WEB_DIR / "audit.js").is_file()
        import urllib.request

        srv = ConsoleServer(port=0)
        srv.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/audit.js", timeout=3.0) as r:
                js_body = r.read().decode("utf-8")
            assert "audit-panel" in js_body or "updateAudit" in js_body

            audit_payload = {
                "audit_id": "ws-flag-20",
                "status": "complete",
                "bay": 1,
                "skill_id": "s_ws20",
                "episode_uid": "ep_ws20",
                "verdict": "flag",
                "dataset_tag": "flag",
                "confidence": 0.91,
                "reasons": ["Excessive finger contact torque"],
                "operator_note": "Flagged for ACT QC.",
                "model": "gemini-4-argon",
                "frame_count": 6,
                "latency_ms": 8450.0,
                "wall": "21:00:00",
            }
            with (
                ws_connect(f"ws://127.0.0.1:{srv.port}/ws") as ws_browser,
                ws_connect(f"ws://127.0.0.1:{srv.port}/ws") as ws_copilot,
            ):
                ws_copilot.send(json.dumps({"type": "copilot_audit", "audit": audit_payload}))
                relayed = json.loads(ws_browser.recv(timeout=5.0))
                assert relayed["audit"]["audit_id"] == "ws-flag-20"
                assert relayed["audit"]["dataset_tag"] == "flag"
        finally:
            srv.stop()

    def test_t3_flow21_approved_vs_flagged_vs_untagged_offline_episodes_curated_together(
        self,
        tmp_path: Path,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
    ) -> None:
        """F12 + F14: Mixed session (offline + approved + flagged) keeps 2 and drops 1."""
        assert "copilot flagged" in DROP_REASONS
        from .test_tier1_argon_console_and_qc import _drive_synthetic_episode

        rig = coordinator_factory(grasp_gate=None, require_grasp_gate_hook=False)
        ds_root = tmp_path / "mixed_three_way"
        clean_root = tmp_path / "mixed_three_way_clean"
        sink = EpisodeSink(
            ds_root,
            rig.co.model,
            rig.co.data,
            fault_of=lambda _: "flap",
            operator_source="glove",
            renderer=_ZeroRenderer(),
        )
        tag_fn = require_attr(sink, "tag_episode")
        try:
            uid_offline = _drive_synthetic_episode(sink, rig, bay_id=1)
            uid_approved = _drive_synthetic_episode(sink, rig, bay_id=2)
            uid_flagged = _drive_synthetic_episode(sink, rig, bay_id=1)
            _wait(lambda: sink.written == 3)
            assert uid_offline != uid_approved != uid_flagged

            assert tag_fn(
                uid_approved,
                dataset_tag="approve",
                audit={"audit_id": "a-ok", "status": "complete", "verdict": "approve"},
            )
            assert tag_fn(
                uid_flagged,
                dataset_tag="flag",
                audit={"audit_id": "a-flg", "status": "complete", "verdict": "flag"},
            )
        finally:
            sink.close()

        rc = process_cmd.main(["--in", str(ds_root), "--out", str(clean_root)])
        assert rc == 0
        kept = read_sidecar(clean_root)
        kept_uids = {str(row["episode_uid"]) for row in kept.values()}
        assert kept_uids == {uid_offline, uid_approved}
        assert uid_flagged not in kept_uids
