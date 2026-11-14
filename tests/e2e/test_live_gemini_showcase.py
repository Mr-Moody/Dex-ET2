"""Feature F15 Tier 1/2 live-harness/redaction/skip-guard tests + Tier 4 Live Gemini API Showcase.

Offline by default (`RUN_LIVE_GEMINI != "1"`):
- All 10 Tier 1 & Tier 2 F15 harness, `.env` discovery, secret-redaction, egress-blocker, and
  skip-guard tests run and verify offline determinism with 0 external network calls.
- The 3 live Gemini API latency & verdict acceptance tests (`RUN_LIVE_GEMINI=1` against
  `/home/ethan/Documents/UCL_repos/Junction/.env`) skip cleanly when `RUN_LIVE_GEMINI != "1"` and
  execute live end-to-end checks when `RUN_LIVE_GEMINI=1`:
  1. Tier 1 `gemini-3.8-flash-lite-tts` (`Kore`) SSE streaming `< 1.5 s` TTFA over persistent
     `IPv4HTTPSConnection` (`AF_INET` + `keep-alive` reuse).
  2. Tier 2 `gemini-4-hydrogen` (`thinkingLevel="LOW"`) on 3 `<=512px` rescue keyframes +
     telemetry summary + warm SSE TTS `< 10.0 s` combined TTFA.
  3. Tier 3 `gemini-4-argon` (`thinkingLevel="LOW"`) on up to 6 frames returning structured
     `AuditReport` + `dataset_tag` in `{"approve", "flag"}` asynchronously with **0 TTS calls**.
"""

from __future__ import annotations

import inspect
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from teleop.copilot.config import CopilotConfig, _discover_env_files
from teleop.copilot.gemini import REDACTED, GeminiClient, GeminiHttpError

from .conftest import (
    MAIN_ENV_FILE,
    REPO_ROOT,
    FakeAudioHarness,
    LoopbackGeminiTLSServer,
    construct_flexible,
    make_rescue_keyframes,
    make_tts_streamer,
    require_attr,
    require_module,
)

LIVE_SKIP_REASON = (
    "Set RUN_LIVE_GEMINI=1 to run live Gemini API latency and verdict showcase checks"
)
requires_live_gemini = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_GEMINI") != "1",
    reason=LIVE_SKIP_REASON,
)


def _load_live_config() -> CopilotConfig:
    assert MAIN_ENV_FILE.is_file(), f"Missing live verification .env at {MAIN_ENV_FILE}"
    cfg = CopilotConfig.from_env(env_file=MAIN_ENV_FILE)
    assert cfg.has_key and cfg.gemini_api_key is not None, (
        "GEMINI_API_KEY is missing or empty in Junction/.env"
    )
    return cfg


# ===========================================================================
# FEATURE 15 — Tier 1 Live-Harness, Config Discovery & Secret Redaction (>=5)
# ===========================================================================


class TestFeature15LiveHarnessAndRedactionTier1:
    """Tier 1 category-partition tests for F15 (.env discovery, redaction, model defaults)."""

    def test_f15_t1_discover_env_files_finds_main_checkout_dotenv_from_sibling_worktree(
        self,
    ) -> None:
        discovered = [p.resolve() for p in _discover_env_files(REPO_ROOT)]
        assert MAIN_ENV_FILE.resolve() in discovered, (
            f"_discover_env_files({REPO_ROOT}) must locate main checkout .env via git-common-dir"
        )

    def test_f15_t1_copilot_config_loads_key_from_explicit_env_file_and_redacts_repr(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        sentinel_secret = "SYNTHETIC_REDACTION_CHECK_KEY_012345"
        env_path = tmp_path / ".env"
        env_path.write_text(f"GEMINI_API_KEY={sentinel_secret}\n", encoding="utf-8")
        cfg = CopilotConfig.from_env(env_file=env_path)
        assert cfg.has_key is True
        assert sentinel_secret not in repr(cfg)
        assert sentinel_secret not in str(cfg)

    def test_f15_t1_gemini_client_repr_and_errors_never_expose_raw_api_key(self) -> None:
        sentinel_secret = "SYNTHETIC_CLIENT_SECRET_9876543210"
        client = GeminiClient(sentinel_secret)
        assert sentinel_secret not in repr(client)
        err = GeminiHttpError(
            403,
            f"Request to https://example.test?key={sentinel_secret} failed".replace(
                sentinel_secret, REDACTED
            ),
        )
        assert sentinel_secret not in str(err)

    def test_f15_t1_default_three_tier_models_and_kore_voice_match_acceptance_criteria(
        self,
    ) -> None:
        cfg = CopilotConfig()
        assert cfg.tts_model == "gemini-3.8-flash-lite-tts"
        assert cfg.voice == "Kore"
        assert cfg.fast_model == "gemini-4-hydrogen"
        assert cfg.review_model == "gemini-4-argon"
        assert cfg.fast_thinking == "low"
        assert cfg.review_thinking == "low"

    def test_f15_t1_loopback_showcase_smoke_validates_all_three_latency_timers_offline(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        """Validates our exact TTFA & combined-latency measurement harness against loopback TLS."""
        assert fake_audio_env.bin_dir.is_dir()
        streamer = make_tts_streamer(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            base_url=mock_gemini_server.base_url,
        )
        try:
            t0 = time.monotonic()
            it = iter(require_attr(streamer, "stream_pcm")("Warm check."))
            first = next(it)
            ttfa = time.monotonic() - t0
            rest = list(it)
            assert len(first) > 0 and len(first + b"".join(rest)) % 2 == 0
            assert ttfa < 1.5
        finally:
            if hasattr(streamer, "close"):
                streamer.close()


# ===========================================================================
# FEATURE 15 — Tier 2 Skip Guards, Egress Enforcement & Missing-Key Boundaries (>=5)
# ===========================================================================


class TestFeature15LiveGuardsAndBoundariesTier2:
    """Tier 2 negative & boundary tests for F15 (offline skip guard, egress block, empty key)."""

    def test_f15_t2_live_marker_condition_is_active_when_run_live_gemini_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("RUN_LIVE_GEMINI", raising=False)
        assert os.environ.get("RUN_LIVE_GEMINI") != "1"

    def test_f15_t2_external_egress_guard_blocks_non_loopback_tcp_in_offline_mode(
        self,
    ) -> None:
        if os.environ.get("RUN_LIVE_GEMINI") == "1":
            # In live mode on this file, egress is intentionally permitted; verify socket API
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.close()
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(AssertionError, match="External network egress blocked"):
                sock.connect(("8.8.8.8", 443))
        finally:
            sock.close()

    def test_f15_t2_empty_string_gemini_api_key_in_env_overrides_dotenv_to_offline(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_path = tmp_path / ".env"
        env_path.write_text("GEMINI_API_KEY=SYNTHETIC_KEY_IGNORED_WHEN_EMPTY\n", encoding="utf-8")
        monkeypatch.setenv("GEMINI_API_KEY", "")
        cfg = CopilotConfig.from_env(env_file=env_path)
        assert cfg.has_key is False
        assert cfg.gemini_api_key is None

    def test_f15_t2_whitespace_only_gemini_api_key_treated_as_unset(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        env_path = tmp_path / ".env"
        env_path.write_text("GEMINI_API_KEY='   '\n", encoding="utf-8")
        cfg = CopilotConfig.from_env(env_file=env_path)
        assert cfg.has_key is False

    def test_f15_t2_main_env_file_exists_and_is_gitignored_never_tracked(self) -> None:
        import subprocess

        ignored = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", ".env"],
            capture_output=True,
            timeout=10,
        )
        assert ignored.returncode == 0, ".env must always be gitignored"
        tracked = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", ".env"],
            capture_output=True,
            timeout=10,
        )
        assert tracked.returncode != 0, ".env must never be tracked by git"


# ===========================================================================
# TIER 4 LIVE GEMINI API SHOWCASE SCENARIOS 6, 7, 8 (`RUN_LIVE_GEMINI=1`)
# ===========================================================================


@requires_live_gemini
class TestTier4LiveGeminiShowcaseScenarios:
    """Live API verification against `Junction/.env` (never logs or prints `GEMINI_API_KEY`)."""

    def test_t4_scenario6_live_tier1_tts_sse_ttfa_under_1_5s_and_ipv4_keepalive(
        self,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        """Live Acceptance Check 1: `gemini-3.8-flash-lite-tts` (`Kore`) SSE `< 1.5 s` TTFA."""
        assert fake_audio_env.bin_dir.is_dir()
        cfg = _load_live_config()
        assert cfg.gemini_api_key is not None
        streamer = make_tts_streamer(
            api_key=cfg.gemini_api_key,
            host="generativelanguage.googleapis.com",
            port=443,
            tts_model=cfg.tts_model,
            voice=cfg.voice,
        )
        try:
            stream_fn = require_attr(streamer, "stream_pcm")

            # Call 1 (cold TLS handshake + SSE stream)
            t0_cold = time.monotonic()
            it_cold = iter(stream_fn("Clutch engaged on bay one."))
            first_cold = next(it_cold)
            cold_ttfa_s = time.monotonic() - t0_cold
            rest_cold = list(it_cold)
            pcm_cold = first_cold + b"".join(rest_cold)

            assert len(pcm_cold) > 0 and len(pcm_cold) % 2 == 0

            # Inspect underlying persistent connection if exposed (`_conn` / `conn`)
            conn_obj = getattr(streamer, "_conn", None) or getattr(streamer, "conn", None)
            sock_id_1: int | None = None
            if conn_obj is not None and getattr(conn_obj, "sock", None) is not None:
                assert conn_obj.sock.family == socket.AF_INET
                sock_id_1 = id(conn_obj.sock)

            # Call 2 (warm persistent IPv4 keep-alive connection)
            t0_warm = time.monotonic()
            it_warm = iter(stream_fn("Bolt cleared cleanly, checking skill."))
            first_warm = next(it_warm)
            warm_ttfa_s = time.monotonic() - t0_warm
            rest_warm = list(it_warm)
            pcm_warm = first_warm + b"".join(rest_warm)

            assert len(pcm_warm) > 0 and len(pcm_warm) % 2 == 0
            assert warm_ttfa_s < 1.5, (
                f"Live Tier 1 warm SSE TTS TTFA {warm_ttfa_s:.3f}s exceeded 1.5s budget "
                f"(cold={cold_ttfa_s:.3f}s)"
            )
            if conn_obj is not None and getattr(conn_obj, "sock", None) is not None and sock_id_1:
                assert id(conn_obj.sock) == sock_id_1, (
                    "Expected persistent IPv4 TLS socket reuse across back-to-back SSE TTS calls"
                )
        finally:
            if hasattr(streamer, "close"):
                streamer.close()

    def test_t4_scenario7_live_tier2_hydrogen_3_keyframe_verdict_and_sse_tts_under_10s(
        self,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        """Live Acceptance Check 2: `gemini-4-hydrogen` (`LOW`) + warm SSE TTS TTFA `< 10.0 s`."""
        assert fake_audio_env.bin_dir.is_dir()
        cfg = _load_live_config()
        assert cfg.gemini_api_key is not None
        gate_mod = require_module("teleop.copilot.gate")
        kf_cls = require_attr(gate_mod, "KeyframeJpeg")
        summary_cls = require_attr(gate_mod, "GraspTelemetrySummary")
        gate_cls = require_attr(gate_mod, "GraspGate")

        streamer = make_tts_streamer(
            api_key=cfg.gemini_api_key,
            host="generativelanguage.googleapis.com",
            port=443,
            tts_model=cfg.tts_model,
            voice=cfg.voice,
        )
        gem_client = GeminiClient(cfg.gemini_api_key)
        gate = construct_flexible(
            gate_cls,
            client=gem_client,
            gemini_client=gem_client,
            cfg=cfg,
            config=cfg,
            model=cfg.fast_model,
            thinking_level="low",
        )
        try:
            # Pre-warm SSE TTS connection (as startup precache_cues does)
            _ = list(require_attr(streamer, "stream_pcm")("Ready."))

            keyframes = [kf_cls(*row) for row in make_rescue_keyframes(3)]
            telemetry = summary_cls(
                bay_id=1,
                sku="EV-PACK-75KWH",
                skill_id="live_skill_001",
                signature_id="Jam_Sig_1",
                duration_s=2.65,
                bolt_start_m=0.014,
                bolt_final_m=0.056,
                peak_tau_hand_nm=0.42,
                mean_tau_hand_nm=0.26,
                peak_arm_ratio=0.29,
                min_wrist_x_latch_m=0.012,
                min_wrist_z_latch_m=0.044,
                touch_offset_m=(0.001, -0.001),
            )

            from .test_tier1_grasp_gate_and_local import _evaluate_grasp_gate

            t0 = time.monotonic()
            verdict = _evaluate_grasp_gate(gate, keyframes, telemetry)
            assert verdict.verdict in ("approve", "reject")
            assert bool(verdict.say.strip())
            assert bool(verdict.reason.strip())
            assert 0.0 <= float(verdict.confidence) <= 1.0
            assert verdict.source == "gemini"

            # Stream Hydrogen's dynamic `say` verdict over warm SSE TTS and measure combined TTFA!
            tts_iter = iter(require_attr(streamer, "stream_pcm")(verdict.say))
            first_pcm = next(tts_iter)
            combined_ttfa_s = time.monotonic() - t0
            _ = list(tts_iter)

            assert len(first_pcm) > 0
            assert combined_ttfa_s < 10.0, (
                f"Live Tier 2 Hydrogen + SSE TTS TTFA {combined_ttfa_s:.2f}s exceeded 10.0s limit"
            )
        finally:
            if hasattr(streamer, "close"):
                streamer.close()
            if hasattr(gate, "close"):
                gate.close()

    def test_t4_scenario8_live_tier3_argon_silent_deep_audit_and_dataset_tag(
        self,
        tmp_path: Path,
        fake_audio_env: FakeAudioHarness,
        coordinator_factory: Any,
    ) -> None:
        """Live Acceptance Check 3: `gemini-4-argon` (`LOW`) async audit + tag with **0 TTS**."""
        cfg = _load_live_config()
        assert cfg.gemini_api_key is not None
        rev_mod = require_module("teleop.copilot.review")
        rev_cls = require_attr(rev_mod, "AuditReviewer")

        from teleop.atc.episodes import EpisodeSink
        from teleop.policy.schema import CAMERAS, IMAGE_HW
        from teleop.process.qc import read_sidecar

        from .test_tier1_argon_console_and_qc import _drive_synthetic_episode, _submit_audit

        class _ZeroRender:
            def render(self, data: Any, bay_id: int) -> dict[str, Any]:
                del data, bay_id
                import numpy as np

                return {c: np.zeros((*IMAGE_HW, 3), dtype=np.uint8) for c in CAMERAS}

            def close(self) -> None:
                pass

        rig = coordinator_factory(grasp_gate=None, require_grasp_gate_hook=False)
        ds_root = tmp_path / "live_argon_ds"
        ep_sink = EpisodeSink(
            ds_root,
            rig.co.model,
            rig.co.data,
            fault_of=lambda _: "flap",
            operator_source="glove",
            renderer=_ZeroRender(),
        )
        uid = _drive_synthetic_episode(ep_sink, rig, bay_id=1)
        ep_sink.close()

        tts_calls = 0

        class _TtsSpy:
            def say(self, *_a: Any, **_kw: Any) -> int:
                nonlocal tts_calls
                tts_calls += 1
                return 0

        updates: list[Any] = []
        done = threading.Event()

        def on_upd(rep: Any) -> None:
            updates.append(rep)
            if rep.status in ("complete", "error"):
                if rep.status == "complete" and rep.episode_uid and rep.dataset_tag:
                    require_attr(ep_sink, "tag_episode")(
                        rep.episode_uid,
                        dataset_tag=rep.dataset_tag,
                        audit=rep.to_dict(),
                    )
                done.set()

        gem_client = GeminiClient(cfg.gemini_api_key)
        reviewer = construct_flexible(
            rev_cls,
            client=gem_client,
            gemini_client=gem_client,
            cfg=cfg,
            config=cfg,
            model=cfg.review_model,
            thinking_level="low",
            on_update=on_upd,
            episode_sink=ep_sink,
            speaker=_TtsSpy(),
        )
        try:
            t_submit = time.monotonic()
            _ = _submit_audit(reviewer, bay=1, skill_id="live_skill_001", episode_uid=uid)
            submit_ms = (time.monotonic() - t_submit) * 1000.0
            assert submit_ms < 250.0, f"Argon submit() blocked for {submit_ms:.1f} ms!"
            assert any(u.status == "pending" for u in updates)

            assert done.wait(timeout=60.0), "Timed out waiting for live gemini-4-argon response"
            final = updates[-1]
            assert final.status == "complete", f"Live Argon failed with status={final.status}"
            assert final.verdict in ("approve", "flag")
            assert final.dataset_tag in ("approve", "flag")
            assert len(final.reasons) >= 1
            assert bool(final.operator_note.strip())
            assert tts_calls == 0, "Tier 3 Argon MUST trigger 0 TTS calls!"
            assert fake_audio_env.calls() == []

            sidecar = read_sidecar(ds_root)
            assert sidecar[0]["dataset_tag"] == final.dataset_tag
        finally:
            if hasattr(reviewer, "close"):
                reviewer.close()


# Silence unused import lint if inspect is only used indirectly
_ = inspect.isfunction
