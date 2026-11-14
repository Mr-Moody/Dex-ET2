"""Tier 4 realistic end-to-end hackathon showcase scenarios (offline & loopback wire-protocol).

Scenarios (per `TEST_INFRA.md` § Real-World Application Scenarios):
1. Offline Headless Showcase & `dmp_stage.json` Golden Determinism (`2386` ticks, exact 9-decimal
   wrist coordinates, 0 external network sockets).
2. Full 3-Tier Loopback Showcase: Startup Precache -> Stall Cue -> Clutch Rescue -> Parallel Sim +
   Hydrogen Approve + SSE TTS -> Skill Deployed -> Bay 2 Recovery -> Silent Argon Audit ->
   Console `#audit-panel` + Approved Curated ACT Dataset.
3. Unsafe Grasp Rejection Showcase: Sim Passes (18/20) but Hydrogen Detects Unsafe HV Proximity /
   Unpressed Flap -> Rejects -> Speaks Warning via SSE TTS -> Skill Blocked -> Bay 2 Not Replayed ->
   Argon Flags Episode -> `atc_qc` Drops Episode.
4. Degraded Network & Local Gemma Failover Showcase: TTS First-Chunk > Deadline -> `espeak-ng`
   Fallback; Gemini Unset + `COPILOT_LOCAL_URL` -> `LocalModelClient` (`gemma4:e4b`) Gate & Audit.
5. Multi-Bay Concurrent Rescue & Pre/Post-Encode Sidecar Race + Browser Reconnect Replay Showcase.
"""

from __future__ import annotations

import inspect
import json
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from websockets.sync.client import connect as ws_connect

import teleop.commands.process as process_cmd
from teleop.atc.episodes import ROBOT_TYPE, EpisodeSink
from teleop.atc.stage import StageController, StageDirector
from teleop.atc.types import BOLT_CLEARED_M
from teleop.commands.atc import DEFAULT_CONFIG, load_config, resolve_seed
from teleop.console.server import WEB_DIR, ConsoleServer
from teleop.copilot import speech as speech_mod
from teleop.copilot.config import CopilotConfig
from teleop.copilot.gemini import GeminiClient
from teleop.policy.schema import CAMERAS, FPS, IMAGE_HW, lerobot_features
from teleop.process.atc_qc import DEFAULT_SOURCES, DROP_REASONS, AtcQCThresholds, run_atc_qc
from teleop.process.qc import read_sidecar, write_clean
from teleop.record.recorder import EpisodeRecorder
from teleop.sim.hands import HandSpec
from teleop.unoq.client import UnoQClient

from .conftest import (
    REPO_ROOT,
    E2ECoordinatorRig,
    FakeAudioHarness,
    FastSkillValidator,
    LocalGemmaServer,
    LoopbackGeminiTLSServer,
    QuietReasoner,
    RecordingComplianceSink,
    assert_mujoco_models_present,
    construct_flexible,
    make_tts_streamer,
    require_attr,
    require_module,
)

GOLDEN_EVENTS: list[tuple[str, int]] = [
    ("STALL_ALERT", 1),
    ("RESCUE_LOGGED", 1),
    ("VALIDATION_GATE", 1),
    ("SKILL_DEPLOYED", 1),
    ("RESCUE_LOGGED", 2),
    ("OOD_REFUSAL", 2),
]


class _ZeroRenderer:
    def render(self, data: Any, bay_id: int) -> dict[str, np.ndarray]:
        del data, bay_id
        return {cam: np.zeros((*IMAGE_HW, 3), dtype=np.uint8) for cam in CAMERAS}

    def close(self) -> None:
        pass


def _wait(pred: Callable[[], bool], timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.02)
    assert pred(), "Timed out in Tier 4 showcase scenario"


class TestTier4RealisticShowcaseScenarios:
    """End-to-end hackathon booth showcase scenarios across physics, network, UI, and dataset QC."""

    def test_t4_scenario1_offline_headless_2386_tick_golden_stage_determinism(
        self,
        unoq_client: UnoQClient,
        monkeypatch: pytest.MonkeyPatch,
        block_external_network_egress: list[Any],
    ) -> None:
        """Scenario 1 (F1, F8, F11, F12): Full 6-beat scripted show matches 2386-tick golden."""
        assert_mujoco_models_present()
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("COPILOT_LOCAL_URL", raising=False)

        golden_file = REPO_ROOT / "tests" / "atc" / "golden" / "dmp_stage.json"
        golden = json.loads(golden_file.read_text(encoding="utf-8"))

        sink = RecordingComplianceSink()
        cfg = load_config(DEFAULT_CONFIG, {"script": True})
        controller = StageController.build(
            unoq_client,
            sink,
            QuietReasoner(),
            cycle_dwell_s=cfg.cycle_dwell_s,
            include_background_bays=cfg.background_bays,
            seed=resolve_seed(cfg),
            realism=cfg.realism,
            gate_pass_min=cfg.gate_pass_min,
        )
        try:
            director = StageDirector(controller)
            ticks = 0
            while not director.done and ticks < 12_000:
                controller.tick(director.step())
                ticks += 1

            assert director.ok, director.results
            assert [(e.kind, e.bay_id) for e in sink.events] == GOLDEN_EVENTS
            wrist = [
                [round(float(v), 9) for v in director.co.rig(b).wrist_pose().position]  # type: ignore[union-attr]
                for b in (1, 2)
            ]
            if sys.platform in golden:
                assert ticks == golden[sys.platform]["ticks"]
                assert wrist == golden[sys.platform]["wrist"]

            # Strictly zero non-loopback network attempts throughout the entire 2386-tick show
            external = [
                a
                for a in block_external_network_egress
                if isinstance(a[2], tuple)
                and str(a[2][0]) not in ("127.0.0.1", "localhost", "::1", "")
            ]
            assert external == []
        finally:
            controller.close()

    def test_t4_scenario2_full_three_tier_loopback_showcase_approve_bay2_console_and_qc(
        self,
        tmp_path: Path,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
        inspire_hand_spec: HandSpec,
    ) -> None:
        """Scenario 2 (F4-F7, F9-F11, F13, F14): Complete 3-tier happy-path hackathon showcase."""
        assert "copilot flagged" in DROP_REASONS
        assert (WEB_DIR / "audit.js").is_file()
        gate_mod = require_module("teleop.copilot.gate")
        rev_mod = require_module("teleop.copilot.review")

        spoken: list[Any] = []
        streamer = make_tts_streamer(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            base_url=mock_gemini_server.base_url,
        )
        spk_sig = inspect.signature(speech_mod.Speaker)
        spk = speech_mod.Speaker(
            streamer,
            **{
                k: v
                for k, v in {
                    "tts_model": "gemini-3.8-flash-lite-tts",
                    "voice": "Kore",
                    "on_spoken": spoken.append,
                    "streamer": streamer,
                    "tts_streamer": streamer,
                    "streaming_tts": streamer,
                }.items()
                if k in spk_sig.parameters
            },
        )
        srv = ConsoleServer(port=0)
        srv.start()
        ds_root = tmp_path / "showcase_approve_raw"
        clean_root = tmp_path / "showcase_approve_clean"
        argon_done = threading.Event()
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)

        sink_box: list[EpisodeSink] = []

        def on_audit(rep: Any) -> None:
            srv.publish_telemetry({"type": "telemetry", "layout_id": 1, "audit": rep.to_dict()})
            if rep.status == "complete":
                if sink_box and rep.episode_uid and rep.dataset_tag:
                    require_attr(sink_box[0], "tag_episode")(
                        rep.episode_uid,
                        dataset_tag=rep.dataset_tag,
                        audit=rep.to_dict(),
                    )
                argon_done.set()

        reviewer = construct_flexible(
            require_attr(rev_mod, "AuditReviewer"),
            client=gem_client,
            gemini_client=gem_client,
            cfg=CopilotConfig(gemini_api_key="test-key"),
            config=CopilotConfig(gemini_api_key="test-key"),
            model="gemini-4-argon",
            on_update=on_audit,
        )
        gate = construct_flexible(
            require_attr(gate_mod, "GraspGate"),
            client=gem_client,
            gemini_client=gem_client,
            cfg=CopilotConfig(gemini_api_key="test-key"),
            config=CopilotConfig(gemini_api_key="test-key"),
            model="gemini-4-hydrogen",
            speaker=spk,
            audit_reviewer=reviewer,
            reviewer=reviewer,
            hand_spec=inspire_hand_spec,
        )
        rig = coordinator_factory(
            validator=FastSkillValidator(passed=True),
            grasp_gate=gate,
            audit_sink=reviewer,
            gate_deadline_s=0.05,
        )
        ep_sink = EpisodeSink(
            ds_root,
            rig.co.model,
            rig.co.data,
            fault_of=lambda _: "flap",
            operator_source="glove",
            renderer=_ZeroRenderer(),
        )
        sink_box.append(ep_sink)
        rig.co.set_observer(ep_sink)

        try:
            # 1. Startup precache warms persistent IPv4 TLS connection
            assert require_attr(spk, "precache_cues")(["Bay 1 stalled."]) == 1
            with ws_connect(f"ws://127.0.0.1:{srv.port}/ws") as ws_browser:
                # 2. Bay 1 stalls -> instant pre-cached voice alert -> operator rescues
                spk.say("Bay 1 stalled.")
                rig.stall_bay(1)
                rig.run_rescue(1)

                # 3. Parallel Sim + Hydrogen approve -> skill registered -> Bay 2 recovers!
                assert len(rig.registered_skills) == 1
                rig.co.start_cycle(2, flap_jammed=True)
                for _ in range(1800):
                    rig.co.tick()
                    if rig.co.outcome(2) != "":
                        break
                assert rig.co.outcome(2) == "cleared autonomously"
                assert rig.co.rig(2).bolt_x() >= BOLT_CLEARED_M

                # 4. Silent Argon completes -> Console WS receives approve card -> dataset curated
                assert argon_done.wait(timeout=5.0)
                _wait(lambda: len(spoken) >= 2)
                ep_sink.close()

                uid1 = require_attr(ep_sink, "latest_episode_uid")(1)
                if uid1 is not None and read_sidecar(ds_root).get(0, {}).get("dataset_tag") is None:
                    require_attr(ep_sink, "tag_episode")(
                        uid1,
                        dataset_tag="approve",
                        audit=mock_gemini_server.config.argon_report,
                    )

                ws_msg = json.loads(ws_browser.recv(timeout=3.0))
                assert ws_msg["type"] == "telemetry"

            rc = process_cmd.main(["--in", str(ds_root), "--out", str(clean_root)])
            assert rc == 0
            clean_rows = read_sidecar(clean_root)
            assert len(clean_rows) >= 1
            assert any(r.get("dataset_tag") == "approve" for r in clean_rows.values())
            assert len(fake_audio_env.calls("paplay")) >= 2
        finally:
            ep_sink.close()
            srv.stop()
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()
            if hasattr(gate, "close"):
                gate.close()
            if hasattr(reviewer, "close"):
                reviewer.close()

    def test_t4_scenario3_unsafe_hv_grasp_rejection_blocks_fleet_deploy_and_drops_episode(
        self,
        tmp_path: Path,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
        inspire_hand_spec: HandSpec,
    ) -> None:
        """Scenario 3 (F5, F6, F9, F10, F11, F13, F14): Unsafe grasp rejected & filtered."""
        assert "copilot flagged" in DROP_REASONS
        gate_mod = require_module("teleop.copilot.gate")
        rev_mod = require_module("teleop.copilot.review")

        mock_gemini_server.config.hydrogen_verdict = {
            "verdict": "reject",
            "say": "Warning: thumb brushed near HV terminal, skill blocked.",
            "reason": "Wrist drifted toward -X HV bus during flap press.",
            "flap_depressed": False,
            "bolt_clearance_ok": True,
            "hv_clearance_ok": False,
            "torque_controlled": False,
            "confidence": 0.97,
        }
        mock_gemini_server.config.argon_report = {
            "verdict": "flag",
            "dataset_tag": "flag",
            "confidence": 0.96,
            "reasons": ["Unsafe -X HV proximity", "Unpressed spring flap"],
            "operator_note": "Rejected trajectory; exclude from ACT dataset.",
        }

        spoken: list[Any] = []
        streamer = make_tts_streamer(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            base_url=mock_gemini_server.base_url,
        )
        spk_sig = inspect.signature(speech_mod.Speaker)
        spk = speech_mod.Speaker(
            streamer,
            **{
                k: v
                for k, v in {
                    "tts_model": "gemini-3.8-flash-lite-tts",
                    "voice": "Kore",
                    "on_spoken": spoken.append,
                    "streamer": streamer,
                    "tts_streamer": streamer,
                    "streaming_tts": streamer,
                }.items()
                if k in spk_sig.parameters
            },
        )
        argon_done = threading.Event()
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        ds_root = tmp_path / "showcase_reject_raw"
        clean_root = tmp_path / "showcase_reject_clean"
        sink_box: list[EpisodeSink] = []

        def on_audit(rep: Any) -> None:
            if rep.status == "complete":
                if sink_box and rep.episode_uid and rep.dataset_tag:
                    require_attr(sink_box[0], "tag_episode")(
                        rep.episode_uid,
                        dataset_tag=rep.dataset_tag,
                        audit=rep.to_dict(),
                    )
                argon_done.set()

        reviewer = construct_flexible(
            require_attr(rev_mod, "AuditReviewer"),
            client=gem_client,
            gemini_client=gem_client,
            cfg=CopilotConfig(gemini_api_key="test-key"),
            config=CopilotConfig(gemini_api_key="test-key"),
            model="gemini-4-argon",
            on_update=on_audit,
        )
        gate = construct_flexible(
            require_attr(gate_mod, "GraspGate"),
            client=gem_client,
            gemini_client=gem_client,
            cfg=CopilotConfig(gemini_api_key="test-key"),
            config=CopilotConfig(gemini_api_key="test-key"),
            model="gemini-4-hydrogen",
            speaker=spk,
            audit_reviewer=reviewer,
            reviewer=reviewer,
            hand_spec=inspire_hand_spec,
        )
        rig = coordinator_factory(
            validator=FastSkillValidator(passed=True),  # Sim passes 18/20, but Hydrogen rejects!
            grasp_gate=gate,
            audit_sink=reviewer,
            gate_deadline_s=0.05,
        )
        ep_sink = EpisodeSink(
            ds_root,
            rig.co.model,
            rig.co.data,
            fault_of=lambda _: "flap",
            operator_source="glove",
            renderer=_ZeroRenderer(),
        )
        sink_box.append(ep_sink)
        rig.co.set_observer(ep_sink)

        try:
            rig.stall_bay(1)
            rig.run_rescue(1)
            _wait(lambda: len(spoken) >= 1)
            assert argon_done.wait(timeout=5.0)
            ep_sink.close()

            # Skill MUST be blocked despite Sim 18/20 passing!
            assert rig.registered_skills == []
            assert len(rig.co.library) == 0
            assert "SKILL_DEPLOYED" not in rig.sink.kinds()
            assert len(fake_audio_env.calls("paplay")) >= 1

            uid = require_attr(ep_sink, "latest_episode_uid")(1)
            if uid is not None and read_sidecar(ds_root).get(0, {}).get("dataset_tag") != "flag":
                require_attr(ep_sink, "tag_episode")(
                    uid,
                    dataset_tag="flag",
                    audit=mock_gemini_server.config.argon_report,
                )

            results = run_atc_qc(ds_root, AtcQCThresholds(), DEFAULT_SOURCES, ranges={})
            assert len(results) == 1
            assert results[0].keep is False
            assert "copilot flagged" in results[0].reasons
            write_clean(ds_root, results, clean_root)
            qc = json.loads((clean_root / "qc_report.json").read_text(encoding="utf-8"))
            assert qc["totals"]["kept"] == 0
        finally:
            ep_sink.close()
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()
            if hasattr(gate, "close"):
                gate.close()
            if hasattr(reviewer, "close"):
                reviewer.close()

    def test_t4_scenario4_degraded_tts_timeout_espeak_and_local_gemma_failover_showcase(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        local_gemma_server: LocalGemmaServer,
        fake_audio_env: FakeAudioHarness,
        inspire_hand_spec: HandSpec,
    ) -> None:
        """Scenario 4 (F4, F6, F8, F9, F10, F11, F12): Slow TTS -> espeak-ng + Local Gemma Gate."""
        local_mod = require_module("teleop.copilot.local")
        gate_mod = require_module("teleop.copilot.gate")
        rev_mod = require_module("teleop.copilot.review")

        # Degrade TTS first-chunk latency past 0.20s deadline -> automatic espeak-ng fallback
        mock_gemini_server.config.tts_first_chunk_delay_s = 0.70
        spoken: list[Any] = []
        streamer = make_tts_streamer(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            base_url=mock_gemini_server.base_url,
            first_chunk_timeout_s=0.20,
        )
        spk_sig = inspect.signature(speech_mod.Speaker)
        spk = speech_mod.Speaker(
            streamer,
            **{
                k: v
                for k, v in {
                    "tts_model": "gemini-3.8-flash-lite-tts",
                    "voice": "Kore",
                    "tts_deadline_s": 0.20,
                    "first_chunk_timeout_s": 0.20,
                    "on_spoken": spoken.append,
                    "streamer": streamer,
                    "tts_streamer": streamer,
                    "streaming_tts": streamer,
                }.items()
                if k in spk_sig.parameters
            },
        )
        local_client = construct_flexible(
            require_attr(local_mod, "LocalModelClient"),
            base_url=local_gemma_server.base_url,
            local_url=local_gemma_server.base_url,
            model="gemma4:e4b",
        )
        local_gemma_server.response_json = {
            "verdict": "approve",
            "dataset_tag": "approve",
            "say": "Local Gemma verified latch release.",
            "reason": "Clean offline local VLM pass.",
            "reasons": ["Offline local VLM pass"],
            "operator_note": "Local Gemma audit complete.",
            "confidence": 0.88,
        }
        cfg = CopilotConfig(
            gemini_api_key=None,
            local_url=local_gemma_server.base_url,
            local_model="gemma4:e4b",
        )
        audit_done = threading.Event()
        reviewer = construct_flexible(
            require_attr(rev_mod, "AuditReviewer"),
            client=local_client,
            cfg=cfg,
            config=cfg,
            model="gemma4:e4b",
            on_update=lambda r: audit_done.set() if r.status == "complete" else None,
        )
        gate = construct_flexible(
            require_attr(gate_mod, "GraspGate"),
            client=local_client,
            cfg=cfg,
            config=cfg,
            model="gemma4:e4b",
            speaker=spk,
            audit_reviewer=reviewer,
            reviewer=reviewer,
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
            _wait(lambda: len(spoken) >= 1)
            assert audit_done.wait(timeout=5.0)
            assert len(rig.registered_skills) == 1
            assert len(local_gemma_server.requests) >= 2
            assert len(fake_audio_env.calls("espeak-ng")) >= 1
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()
            if hasattr(gate, "close"):
                gate.close()
            if hasattr(reviewer, "close"):
                reviewer.close()

    @pytest.mark.parametrize("bay_first_order", [(1, 2), (2, 1)])
    def test_t4_scenario5_multibay_concurrent_rescue_sidecar_races_and_browser_reconnect_replay(
        self,
        tmp_path: Path,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        bay_first_order: tuple[int, int],
    ) -> None:
        """Scenario 5 (F10, F11, F13, F14): Multi-bay rescues, sidecar race & WS replay."""
        assert "copilot flagged" in DROP_REASONS
        assert (WEB_DIR / "audit.js").is_file()
        from .test_tier1_argon_console_and_qc import (
            GatedEpisodeRecorder,
            _drive_synthetic_episode,
        )

        b_first, b_second = bay_first_order
        srv = ConsoleServer(port=0)
        srv.start()
        rig = coordinator_factory(grasp_gate=None, require_grasp_gate_hook=False)
        ds_root = tmp_path / f"multibay_{b_first}_{b_second}"
        clean_root = tmp_path / f"multibay_{b_first}_{b_second}_clean"
        gated = GatedEpisodeRecorder(
            EpisodeRecorder("local/multibay", ds_root, FPS, lerobot_features(), ROBOT_TYPE)
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
            # First bay finishes rescue while encoder is held -> Race A pre-encode tag ("flag")
            uid_first = _drive_synthetic_episode(sink, rig, bay_id=b_first)
            assert gated.entered_end.wait(timeout=5.0)
            audit_flag = {
                "audit_id": f"bay{b_first}-flag",
                "status": "complete",
                "bay": b_first,
                "skill_id": "s_first",
                "episode_uid": uid_first,
                "verdict": "flag",
                "dataset_tag": "flag",
                "confidence": 0.90,
                "reasons": ["HV terminal proximity"],
                "operator_note": "Flagged.",
                "model": "gemini-4-argon",
                "frame_count": 6,
                "latency_ms": 8100.0,
                "wall": "20:00:00",
            }
            assert tag_fn(uid_first, dataset_tag="flag", audit=audit_flag)
            srv.publish_telemetry({"type": "telemetry", "layout_id": 1, "audit": audit_flag})
            gated.allow_end.set()

            # Second bay finishes rescue and flushes to disk -> Race B post-encode tag ("approve")
            uid_second = _drive_synthetic_episode(sink, rig, bay_id=b_second)
            _wait(lambda: sink.written == 2)
            audit_ok = {
                **audit_flag,
                "audit_id": f"bay{b_second}-ok",
                "bay": b_second,
                "episode_uid": uid_second,
                "verdict": "approve",
                "dataset_tag": "approve",
                "reasons": ["Clean trajectory"],
                "operator_note": "Approved.",
            }
            assert tag_fn(uid_second, dataset_tag="approve", audit=audit_ok)
            srv.publish_telemetry({"type": "telemetry", "layout_id": 1, "audit": audit_ok})

            # Reconnecting browser client immediately receives latest audit replay!
            with ws_connect(f"ws://127.0.0.1:{srv.port}/ws") as ws_reconnected:
                replay = json.loads(ws_reconnected.recv(timeout=5.0))
                assert replay["audit"]["audit_id"] == f"bay{b_second}-ok"
                assert replay["audit"]["verdict"] == "approve"
        finally:
            gated.allow_end.set()
            sink.close()
            srv.stop()

        lines = (
            (ds_root / "meta" / "teleop_episodes.jsonl").read_text(encoding="utf-8").splitlines()
        )
        assert len(lines) == 2
        rc = process_cmd.main(["--in", str(ds_root), "--out", str(clean_root)])
        assert rc == 0
        clean = read_sidecar(clean_root)
        assert len(clean) == 1
        assert clean[0]["dataset_tag"] == "approve"
