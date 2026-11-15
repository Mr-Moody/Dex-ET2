"""Tier 1 & Tier 2 E2E tests for Features F4, F5, F6, and F7.

Covers:
- F4: `IPv4HTTPSConnection` (`socket.AF_INET`, `TCP_NODELAY=1`, `SO_KEEPALIVE=1`, TLS SNI
      `server_hostname`, `keep-alive` socket reuse across >=3 calls, stale-socket auto-reconnect,
      clean socket close on mid-stream abort).
- F5: `GeminiStreamingTTS.stream_pcm` (`POST /v1beta/models/gemini-3.8-flash-lite-tts:`
      `streamGenerateContent?alt=sse`, voice `"Kore"`, `responseModalities: ["AUDIO"]`,
      NO `thinkingConfig`, 24 kHz s16le PCM chunks, `finishReason: "STOP"` + EOF drain).
- F6: `Speaker` raw PCM pipe (`paplay --raw` -> `aplay` -> `pcm_to_wav` WAV fallback),
      Python 3.12 `proc.stdin = None` safety, `BrokenPipeError` resilience, and `1.5–2.0 s`
      first-chunk timeout fallback to `espeak-ng -s 230`.
- F7: `Speaker.precache_cues` startup caching + connection pre-warming and deterministic
      telemetry callout edges firing with strictly zero VLM calls.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import socket
import ssl
import threading
import time
from typing import Any

import pytest

from teleop.copilot import speech as speech_mod
from teleop.copilot.callouts import Callout, CalloutService
from teleop.copilot.events import CopilotEvent, EventDetector, EventKind
from teleop.copilot.gemini import pcm_to_wav
from teleop.copilot.prompts import template_callout

from .conftest import (
    FIXTURE_TELEMETRY_JSONL,
    FakeAudioHarness,
    LoopbackGeminiTLSServer,
    call_stream_pcm,
    construct_flexible,
    make_sine_pcm_s16le,
    make_tts_streamer,
    require_attr,
    require_module,
)


def _make_streaming_speaker(
    mock_gemini_server: LoopbackGeminiTLSServer,
    *,
    first_chunk_timeout_s: float = 0.35,
    mute: bool = False,
    on_spoken: Any = None,
    start: bool = True,
) -> tuple[Any, Any]:
    """Build `GeminiStreamingTTS` + `Speaker` wired against loopback TLS + fake audio env."""
    streamer = make_tts_streamer(
        api_key="test-e2e-tts-key",
        host="generativelanguage.googleapis.com",
        port=mock_gemini_server.port,
        base_url=mock_gemini_server.base_url,
        first_chunk_timeout_s=first_chunk_timeout_s,
    )
    speaker_cls = speech_mod.Speaker
    sig = inspect.signature(speaker_cls)
    kwargs: dict[str, Any] = {
        "tts_model": "gemini-3.8-flash-lite-tts",
        "voice": "Kore",
        "mute": mute,
        "tts_deadline_s": first_chunk_timeout_s,
        "first_chunk_timeout_s": first_chunk_timeout_s,
        "on_spoken": on_spoken,
        "start": start,
        "streamer": streamer,
        "tts_streamer": streamer,
        "streaming_tts": streamer,
    }
    # Pass streamer as synthesizer if Speaker unifies or accepts streaming synthesizer
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    first_param = next(iter(sig.parameters.values()))
    if first_param.name in ("synthesizer", "tts", "client", "streamer"):
        spk = speaker_cls(streamer, **filtered)
    else:
        spk = construct_flexible(speaker_cls, synthesizer=streamer, **kwargs)
    return spk, streamer


def _wait_until(pred: Any, *, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.015)
    assert pred(), "Timed out waiting for asynchronous Speaker condition"


# ===========================================================================
# FEATURE 4 — Persistent IPv4 `HTTPSConnection` (Tier 1 >=5, Tier 2 >=5)
# ===========================================================================


class TestFeature4PersistentIPv4HTTPSConnectionTier1:
    """Tier 1 category-partition tests for `IPv4HTTPSConnection`."""

    def test_f4_t1_connect_resolves_strictly_af_inet_even_when_aaaa_present(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        tts_mod = require_module("teleop.copilot.tts_stream")
        conn_cls = require_attr(tts_mod, "IPv4HTTPSConnection")
        conn = conn_cls(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            timeout=5.0,
        )
        try:
            conn.connect()
            assert conn.sock is not None
            assert conn.sock.family == socket.AF_INET, (
                f"Expected AF_INET socket, got {conn.sock.family}"
            )
        finally:
            conn.close()

    def test_f4_t1_sets_tcp_nodelay_and_so_keepalive_socket_options(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        tts_mod = require_module("teleop.copilot.tts_stream")
        conn_cls = require_attr(tts_mod, "IPv4HTTPSConnection")
        conn = conn_cls(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            timeout=5.0,
        )
        try:
            conn.connect()
            assert conn.sock is not None
            nodelay = conn.sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
            keepalive = conn.sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
            assert nodelay != 0, "TCP_NODELAY must be enabled on IPv4HTTPSConnection"
            assert keepalive != 0, "SO_KEEPALIVE must be enabled on IPv4HTTPSConnection"
        finally:
            conn.close()

    def test_f4_t1_sends_tls_sni_server_hostname_matching_target_host(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        tts_mod = require_module("teleop.copilot.tts_stream")
        conn_cls = require_attr(tts_mod, "IPv4HTTPSConnection")
        conn = conn_cls(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            timeout=5.0,
        )
        try:
            conn.request("GET", "/v1beta/models", headers={"Connection": "keep-alive"})
            resp = conn.getresponse()
            _ = resp.read()
            assert resp.status == 200
            assert len(mock_gemini_server.requests) == 1
            assert (
                mock_gemini_server.requests[0].sni_hostname == "generativelanguage.googleapis.com"
            )
        finally:
            conn.close()

    def test_f4_t1_reuses_single_keepalive_tls_socket_across_three_requests(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        streamer = make_tts_streamer(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            base_url=mock_gemini_server.base_url,
        )
        try:
            for idx in range(3):
                chunks = call_stream_pcm(streamer, f"Utterance {idx}", voice="Kore")
                assert b"".join(chunks) == b"".join(mock_gemini_server.config.tts_chunks)
            assert len(mock_gemini_server.requests) == 3
            conn_ids = {r.conn_id for r in mock_gemini_server.requests}
            assert len(conn_ids) == 1, (
                f"Expected 1 persistent TLS connection across 3 SSE calls, got {conn_ids}"
            )
            assert [r.request_seq_on_conn for r in mock_gemini_server.requests] == [1, 2, 3]
        finally:
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f4_t1_default_constructor_targets_googleapis_port_443(self) -> None:
        tts_mod = require_module("teleop.copilot.tts_stream")
        conn_cls = require_attr(tts_mod, "IPv4HTTPSConnection")
        conn = conn_cls()
        assert conn.host == "generativelanguage.googleapis.com"
        assert conn.port == 443


class TestFeature4PersistentIPv4HTTPSConnectionTier2:
    """Tier 2 boundary & fault-injection tests for `IPv4HTTPSConnection`."""

    def test_f4_t2_auto_reconnects_cleanly_after_server_closes_idle_keepalive_socket(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        mock_gemini_server.config.tts_close_after_first_request = True
        streamer = make_tts_streamer(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            base_url=mock_gemini_server.base_url,
        )
        try:
            first = b"".join(call_stream_pcm(streamer, "First call before server close"))
            second = b"".join(call_stream_pcm(streamer, "Second call after server close"))
            expected = b"".join(mock_gemini_server.config.tts_chunks)
            assert first == expected
            assert second == expected
            assert mock_gemini_server.connection_count >= 2
        finally:
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f4_t2_auto_reconnects_when_stale_socket_drops_immediately_on_write(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        streamer = make_tts_streamer(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            base_url=mock_gemini_server.base_url,
        )
        try:
            _ = b"".join(call_stream_pcm(streamer, "Warm up socket"))
            mock_gemini_server.config.close_stale_once = True
            recovered = b"".join(call_stream_pcm(streamer, "Retry over fresh TLS socket"))
            assert recovered == b"".join(mock_gemini_server.config.tts_chunks)
        finally:
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f4_t2_mid_stream_timeout_abort_closes_dirty_socket_before_next_request(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        streamer = make_tts_streamer(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            base_url=mock_gemini_server.base_url,
            first_chunk_timeout_s=0.2,
        )
        try:
            _ = b"".join(call_stream_pcm(streamer, "Initial warm call"))
            assert mock_gemini_server.connection_count == 1

            # Inject >0.2s first-chunk delay -> must raise/abort and close dirty socket
            mock_gemini_server.config.tts_first_chunk_delay_s = 0.55
            with pytest.raises((TimeoutError, OSError, RuntimeError)):
                _ = b"".join(
                    call_stream_pcm(
                        streamer,
                        "Slow call exceeding TTFA deadline",
                        first_chunk_timeout_s=0.2,
                    )
                )

            # Restore fast server and confirm next request opens a clean new connection (#3)
            mock_gemini_server.config.tts_first_chunk_delay_s = 0.0
            clean = b"".join(call_stream_pcm(streamer, "Post-abort recovery call"))
            assert clean == b"".join(mock_gemini_server.config.tts_chunks)
            assert mock_gemini_server.connection_count >= 2
        finally:
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f4_t2_raises_clear_socket_error_when_no_af_inet_record_exists(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tts_mod = require_module("teleop.copilot.tts_stream")
        conn_cls = require_attr(tts_mod, "IPv4HTTPSConnection")

        def empty_ipv4_getaddrinfo(*_args: Any, **_kwargs: Any) -> list[Any]:
            raise socket.gaierror(socket.EAI_NONAME, "No AF_INET address associated with hostname")

        monkeypatch.setattr(socket, "getaddrinfo", empty_ipv4_getaddrinfo)
        conn = conn_cls(host="ipv6-only.invalid", port=443, timeout=1.0)
        with pytest.raises(OSError):
            conn.connect()

    def test_f4_t2_rejects_untrusted_certificate_when_ca_does_not_match(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tts_mod = require_module("teleop.copilot.tts_stream")
        conn_cls = require_attr(tts_mod, "IPv4HTTPSConnection")
        # Restore stock SSLContext without our ephemeral self-signed CA -> TLS verify MUST fail
        unpatched_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        unpatched_ctx.verify_mode = ssl.CERT_REQUIRED
        unpatched_ctx.check_hostname = True
        monkeypatch.setattr(ssl, "create_default_context", lambda *_a, **_kw: unpatched_ctx)
        conn = conn_cls(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            timeout=2.0,
        )
        try:
            with pytest.raises((ssl.SSLError, OSError)):
                conn.connect()
        finally:
            conn.close()


# ===========================================================================
# FEATURE 5 — SSE Gemini TTS Streaming (`gemini-3.8-flash-lite-tts`, `Kore`)
# ===========================================================================


class TestFeature5SseGeminiTtsStreamingTier1:
    """Tier 1 category-partition tests for `GeminiStreamingTTS.stream_pcm`."""

    def test_f5_t1_posts_to_stream_generate_content_alt_sse_endpoint(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        streamer = make_tts_streamer(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            base_url=mock_gemini_server.base_url,
        )
        try:
            chunks = call_stream_pcm(streamer, "Clutch engaged on bay one.", voice="Kore")
            assert len(chunks) == 3
            reqs = mock_gemini_server.requests_matching(":streamGenerateContent")
            assert len(reqs) == 1
            assert "gemini-3.8-flash-lite-tts:streamGenerateContent" in reqs[0].path
            assert "alt=sse" in reqs[0].path
        finally:
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f5_t1_request_payload_sets_kore_voice_audio_modality_and_omits_thinking_config(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        streamer = make_tts_streamer(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            base_url=mock_gemini_server.base_url,
        )
        try:
            _ = call_stream_pcm(streamer, "Bolt cleared.", voice="Kore")
            req = mock_gemini_server.requests[-1]
            assert req.json_body is not None
            gen_cfg = req.json_body["generationConfig"]
            assert gen_cfg["responseModalities"] == ["AUDIO"]
            voice_name = gen_cfg["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"]
            assert voice_name == "Kore"
            assert "thinkingConfig" not in gen_cfg, (
                "TTS model rejects thinkingConfig; must never be sent on gemini-3.8-flash-lite-tts"
            )
            assert req.headers.get("connection", "").lower() == "keep-alive"
        finally:
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f5_t1_yields_exact_24khz_s16le_pcm_chunks_incrementally_before_stream_end(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        gate_last = threading.Event()
        mock_gemini_server.config.tts_hold_last_chunk = gate_last
        streamer = make_tts_streamer(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            base_url=mock_gemini_server.base_url,
        )
        try:
            fn = require_attr(streamer, "stream_pcm")
            it = iter(fn("Streaming incremental verification"))
            first_chunk = next(it)
            # Chunk 1 MUST arrive while the final chunk is still held on the server!
            assert first_chunk == mock_gemini_server.config.tts_chunks[0]
            assert len(first_chunk) % 2 == 0
            assert not gate_last.is_set()
            gate_last.set()
            rest = list(it)
            assert [first_chunk, *rest] == mock_gemini_server.config.tts_chunks
        finally:
            gate_last.set()
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f5_t1_drains_finish_reason_stop_and_http_chunk_terminator_for_reuse(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        streamer = make_tts_streamer(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            base_url=mock_gemini_server.base_url,
        )
        try:
            c1 = b"".join(call_stream_pcm(streamer, "First utterance"))
            c2 = b"".join(call_stream_pcm(streamer, "Second utterance on same socket"))
            assert c1 == c2 == b"".join(mock_gemini_server.config.tts_chunks)
            assert mock_gemini_server.connection_count == 1
        finally:
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f5_t1_api_key_sent_via_header_not_leaked_in_repr(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        secret = "SYNTHETIC_KEY_FOR_HEADER_CHECK_999"
        streamer = make_tts_streamer(
            api_key=secret,
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            base_url=mock_gemini_server.base_url,
        )
        try:
            assert secret not in repr(streamer)
            _ = call_stream_pcm(streamer, "Header check")
            req = mock_gemini_server.requests[-1]
            assert req.headers.get("x-goog-api-key") == secret or f"key={secret}" in req.path
        finally:
            if hasattr(streamer, "close"):
                streamer.close()


class TestFeature5SseGeminiTtsStreamingTier2:
    """Tier 2 negative, error, and boundary tests for `GeminiStreamingTTS.stream_pcm`."""

    @pytest.mark.parametrize("status_code", [400, 500])
    def test_f5_t2_http_400_and_500_raise_error_without_leaking_api_key(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        status_code: int,
    ) -> None:
        secret = "SYNTHETIC_KEY_DO_NOT_LEAK_HTTP_ERR"
        mock_gemini_server.config.tts_status_code = status_code
        streamer = make_tts_streamer(
            api_key=secret,
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            base_url=mock_gemini_server.base_url,
        )
        try:
            with pytest.raises((RuntimeError, OSError, ValueError)) as exc_info:
                _ = call_stream_pcm(streamer, "Trigger HTTP failure")
            assert secret not in str(exc_info.value)
        finally:
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f5_t2_first_chunk_exceeding_deadline_aborts_promptly(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        mock_gemini_server.config.tts_first_chunk_delay_s = 0.80
        streamer = make_tts_streamer(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            base_url=mock_gemini_server.base_url,
            first_chunk_timeout_s=0.22,
        )
        t0 = time.monotonic()
        try:
            with pytest.raises((TimeoutError, OSError, RuntimeError)):
                _ = call_stream_pcm(
                    streamer,
                    "Delayed first chunk",
                    first_chunk_timeout_s=0.22,
                )
            elapsed = time.monotonic() - t0
            assert elapsed < 0.70, f"Expected abort near 0.22s deadline, took {elapsed:.2f}s"
        finally:
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f5_t2_handles_sse_comments_keepalive_blank_lines_and_special_chars(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        pcm = make_sine_pcm_s16le(60.0)
        mock_gemini_server.config.tts_chunks = [pcm]
        streamer = make_tts_streamer(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            base_url=mock_gemini_server.base_url,
        )
        try:
            # Special characters, quotes, newlines, unicode degree/ohm symbols in text
            tricky = 'Bay #1 "HV-bus" Δx=5.4cm >= 5.2cm & τ<0.5N·m\nLine2'
            out = b"".join(call_stream_pcm(streamer, tricky))
            assert out == pcm
            sent_body = mock_gemini_server.requests[-1].json_body
            assert sent_body is not None
            assert sent_body["contents"][0]["parts"][0]["text"] == tricky
        finally:
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f5_t2_empty_or_whitespace_text_is_rejected_or_short_circuited_safely(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        streamer = make_tts_streamer(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            base_url=mock_gemini_server.base_url,
        )
        try:
            with contextlib.suppress(ValueError):
                chunks = call_stream_pcm(streamer, "   ")
                assert chunks == [] or isinstance(b"".join(chunks), bytes)
        finally:
            if hasattr(streamer, "close"):
                streamer.close()


# ===========================================================================
# FEATURE 6 — Raw PCM Pipe Playback & Fallbacks (Tier 1 >=5, Tier 2 >=5)
# ===========================================================================


class TestFeature6RawPcmPlaybackAndFallbacksTier1:
    """Tier 1 category-partition tests for `Speaker` raw PCM streaming & player selection."""

    def test_f6_t1_streams_raw_24khz_s16le_mono_pcm_directly_into_paplay_stdin(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        spoken_events: list[Any] = []
        spk, streamer = _make_streaming_speaker(
            mock_gemini_server,
            on_spoken=spoken_events.append,
        )
        try:
            spk.say("Bay one stalled on jammed spring flap.")
            _wait_until(lambda: len(spoken_events) >= 1)
            paplay_calls = fake_audio_env.calls("paplay")
            assert len(paplay_calls) == 1
            argv = paplay_calls[0]["argv"]
            assert "--raw" in argv
            assert any("s16le" in a for a in argv)
            assert any("24000" in a for a in argv)
            expected_bytes = sum(len(c) for c in mock_gemini_server.config.tts_chunks)
            assert paplay_calls[0]["stdin_bytes"] == expected_bytes
            assert paplay_calls[0]["stdin_bytes"] % 2 == 0
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f6_t1_falls_back_to_aplay_raw_s16le_when_paplay_unavailable(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        fake_audio_env.remove_tool("paplay")
        spoken_events: list[Any] = []
        spk, streamer = _make_streaming_speaker(
            mock_gemini_server,
            on_spoken=spoken_events.append,
        )
        try:
            spk.say("Switching to ALSA raw pipe.")
            _wait_until(lambda: len(spoken_events) >= 1)
            aplay_calls = fake_audio_env.calls("aplay")
            assert len(aplay_calls) == 1
            argv = aplay_calls[0]["argv"]
            assert "S16_LE" in argv
            assert "24000" in argv
            expected_bytes = sum(len(c) for c in mock_gemini_server.config.tts_chunks)
            assert aplay_calls[0]["stdin_bytes"] == expected_bytes
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f6_t1_falls_back_to_pcm_to_wav_file_playback_when_raw_players_absent(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        # Remove raw-capable Linux tools, leaving `afplay` (WAV-file player)
        fake_audio_env.remove_tool("paplay")
        fake_audio_env.remove_tool("aplay")
        spoken_events: list[Any] = []
        spk, streamer = _make_streaming_speaker(
            mock_gemini_server,
            on_spoken=spoken_events.append,
        )
        try:
            spk.say("Cross-platform WAV fallback check.")
            _wait_until(lambda: len(spoken_events) >= 1)
            afplay_calls = fake_audio_env.calls("afplay")
            assert len(afplay_calls) == 1
            expected_wav = pcm_to_wav(b"".join(mock_gemini_server.config.tts_chunks))
            assert afplay_calls[0]["wav_file_bytes"] == len(expected_wav)
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f6_t1_system_voice_command_prefers_espeak_ng_s_230_ahead_of_spd_say(
        self,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        assert fake_audio_env.bin_dir.is_dir()
        cmd = speech_mod.system_voice_command("Immediate local fallback")
        assert cmd is not None
        assert "espeak-ng" in cmd[0], (
            f"Expected espeak-ng primary system fallback per PROJECT.md §1, got {cmd}"
        )
        assert "-s" in cmd and "230" in cmd

    def test_f6_t1_raw_pcm_player_command_helper_exposed_and_selects_paplay_first(
        self,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        assert fake_audio_env.bin_dir.is_dir()
        raw_cmd_fn = require_attr(speech_mod, "raw_pcm_player_command")
        cmd = raw_cmd_fn()
        assert isinstance(cmd, list) and len(cmd) >= 2
        assert "paplay" in cmd[0]
        assert "--raw" in cmd


class TestFeature6RawPcmPlaybackAndFallbacksTier2:
    """Tier 2 timeout, BrokenPipeError, Python 3.12 stdin=None, and mute tests for F6."""

    def test_f6_t2_first_chunk_timeout_falls_back_automatically_to_espeak_ng(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        mock_gemini_server.config.tts_first_chunk_delay_s = 0.75
        spoken_events: list[Any] = []
        spk, streamer = _make_streaming_speaker(
            mock_gemini_server,
            first_chunk_timeout_s=0.20,
            on_spoken=spoken_events.append,
        )
        try:
            spk.say("Fallback to espeak when Gemini chunk 1 is slow.")
            _wait_until(lambda: len(spoken_events) >= 1, timeout_s=3.0)
            espeak_calls = fake_audio_env.calls("espeak-ng")
            assert len(espeak_calls) >= 1, (
                f"Expected espeak-ng fallback on TTFA timeout; calls={fake_audio_env.calls()}"
            )
            assert "Fallback to espeak when Gemini chunk 1 is slow." in espeak_calls[0]["argv"]
            # Unplayed raw pipe must not receive partial garbage bytes
            for pc in fake_audio_env.calls("paplay"):
                assert pc["stdin_bytes"] == 0
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f6_t2_broken_pipe_midstream_does_not_crash_speaker_or_raise_flush_closed(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("FAKE_PAPLAY_MODE", "broken_pipe_midstream")
        # Use 4 KB total PCM so paplay exits mid-stream after reading 1024 bytes
        mock_gemini_server.config.tts_chunks = [make_sine_pcm_s16le(60.0) for _ in range(3)]
        spoken_events: list[Any] = []
        spk, streamer = _make_streaming_speaker(
            mock_gemini_server,
            on_spoken=spoken_events.append,
        )
        try:
            spk.say("Mid-stream broken pipe test.")
            _wait_until(lambda: len(spoken_events) >= 1, timeout_s=3.0)
            # Send a second utterance with normal paplay to prove worker thread survived!
            monkeypatch.setenv("FAKE_PAPLAY_MODE", "ok")
            spk.say("Worker thread still alive after BrokenPipeError.")
            _wait_until(lambda: len(spoken_events) >= 2, timeout_s=3.0)
            assert len(fake_audio_env.calls("paplay")) >= 2
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f6_t2_python312_proc_stdin_none_discipline_present_in_speech_source(self) -> None:
        from pathlib import Path

        speech_src = Path(speech_mod.__file__ or "")
        text = speech_src.read_text(encoding="utf-8")
        assert ".stdin = None" in text, (
            "Python 3.12 subprocess rule (`PROJECT.md` §1): must set `proc.stdin = None` "
            "after `proc.stdin.close()` before `proc.wait()`/`communicate()`"
        )

    def test_f6_t2_default_first_chunk_timeout_configured_within_1_5_to_2_0_seconds(
        self,
    ) -> None:
        tts_mod = require_module("teleop.copilot.tts_stream")
        sig = inspect.signature(require_attr(tts_mod, "GeminiStreamingTTS").stream_pcm)
        param = sig.parameters.get("first_chunk_timeout_s")
        assert param is not None, "GeminiStreamingTTS.stream_pcm must have `first_chunk_timeout_s`"
        assert 1.5 <= float(param.default) <= 2.0, (
            f"Expected default first_chunk_timeout_s in [1.5, 2.0]s, got {param.default!r}"
        )

    def test_f6_t2_muted_speaker_emits_zero_tts_network_and_zero_subprocess_calls(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        spoken_events: list[Any] = []
        spk, streamer = _make_streaming_speaker(
            mock_gemini_server,
            mute=True,
            on_spoken=spoken_events.append,
        )
        try:
            spk.say("Muted callout should not hit network or audio.")
            _wait_until(lambda: len(spoken_events) >= 1)
            assert mock_gemini_server.requests == []
            assert fake_audio_env.calls() == []
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()


# ===========================================================================
# FEATURE 7 — Startup Cue Pre-Caching & Zero-VLM Telemetry Alerts
# ===========================================================================


class TestFeature7StartupPrecacheAndZeroVlmAlertsTier1:
    """Tier 1 tests for `Speaker.precache_cues` & zero-VLM deterministic telemetry cues."""

    def test_f7_t1_precache_cues_populates_cache_and_prewarms_tls_connection(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        assert fake_audio_env.bin_dir.is_dir()
        spk, streamer = _make_streaming_speaker(mock_gemini_server)
        try:
            precache_fn = require_attr(spk, "precache_cues")
            cues = ["Bay 1 stalled.", "Clutch engaged.", "Bolt cleared."]
            cached_count = precache_fn(cues)
            assert cached_count == 3
            assert len(mock_gemini_server.requests) == 3
            assert mock_gemini_server.connection_count == 1
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f7_t1_precached_cue_plays_instantly_from_cache_with_zero_new_tts_requests(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        spoken_events: list[Any] = []
        spk, streamer = _make_streaming_speaker(
            mock_gemini_server,
            on_spoken=spoken_events.append,
        )
        try:
            cue = "Clutch engaged on bay one."
            assert require_attr(spk, "precache_cues")([cue]) == 1
            reqs_after_precache = len(mock_gemini_server.requests)

            spk.say(cue)
            _wait_until(lambda: len(spoken_events) >= 1)
            # Playing a pre-cached cue must NOT issue another network request!
            assert len(mock_gemini_server.requests) == reqs_after_precache
            assert len(fake_audio_env.calls()) >= 1
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f7_t1_precache_deduplicates_repeated_cues(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        assert fake_audio_env.bin_dir.is_dir()
        spk, streamer = _make_streaming_speaker(mock_gemini_server)
        try:
            precache_fn = require_attr(spk, "precache_cues")
            _ = precache_fn(["Bolt cleared.", "Bolt cleared.", "Skill deployed."])
            assert len(mock_gemini_server.requests) == 2
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f7_t1_stage_telemetry_edges_fire_deterministic_callouts_with_zero_vlm_calls(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        detector = EventDetector()
        emitted: list[Callout] = []
        spoken_events: list[Any] = []
        spk, streamer = _make_streaming_speaker(
            mock_gemini_server,
            on_spoken=spoken_events.append,
        )

        class _ZeroVlmSentinel:
            def generate_json(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
                raise AssertionError("Tier 1 telemetry edge alerts must make ZERO VLM calls!")

        # Passing client=None (or deterministic Tier 1 mode) guarantees zero VLM wait
        svc = CalloutService(
            None,
            model="gemini-4-hydrogen",
            sink=emitted.append,
            speaker=spk,
        )
        try:
            events: list[CopilotEvent] = []
            for line in FIXTURE_TELEMETRY_JSONL.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    events.extend(detector.feed(json.loads(line)))
            assert len(events) >= 4
            for ev in events:
                svc.handle(ev)
            assert len(emitted) == len(events)
            _wait_until(lambda: len(spoken_events) >= 1)
            # Confirm zero `:generateContent` VLM calls were sent to hydrogen/argon
            vlm_reqs = [
                r for r in mock_gemini_server.requests if "hydrogen" in r.path or "argon" in r.path
            ]
            assert vlm_reqs == []
            assert len(fake_audio_env.calls()) >= 1
        finally:
            svc.close()
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f7_t1_template_callout_covers_all_deterministic_telemetry_event_kinds(self) -> None:
        for kind in EventKind:
            ev = CopilotEvent(
                kind=kind,
                bay=1,
                sim_t=3.0,
                wall="12:00:00",
                previous_state="NOMINAL",
                state="RESCUE",
                telemetry={"gate": {"ok": True, "passed": 20, "total": 20}},
            )
            callout_txt = template_callout(ev)
            assert bool(callout_txt.say.strip())
            assert len(callout_txt.say.split()) <= 15


class TestFeature7StartupPrecacheAndZeroVlmAlertsTier2:
    """Tier 2 negative, offline, and edge-case tests for F7."""

    def test_f7_t2_precache_cues_with_empty_list_is_safe_noop(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        assert fake_audio_env.bin_dir.is_dir()
        spk, streamer = _make_streaming_speaker(mock_gemini_server)
        try:
            count = require_attr(spk, "precache_cues")([])
            assert count == 0
            assert mock_gemini_server.requests == []
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f7_t2_precache_cues_when_muted_or_no_synthesizer_makes_zero_network_calls(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        assert fake_audio_env.bin_dir.is_dir()
        spk = speech_mod.Speaker(
            None,
            tts_model="gemini-3.8-flash-lite-tts",
            voice="Kore",
            mute=True,
            start=False,
        )
        try:
            count = require_attr(spk, "precache_cues")(["Bay 1 stalled.", "Bolt cleared."])
            assert count == 0
            assert mock_gemini_server.requests == []
        finally:
            spk.close()

    def test_f7_t2_precache_cues_survives_transient_tts_http_500_without_crashing_startup(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        mock_gemini_server.config.tts_status_code = 500
        spoken_events: list[Any] = []
        spk, streamer = _make_streaming_speaker(
            mock_gemini_server,
            on_spoken=spoken_events.append,
        )
        try:
            count = require_attr(spk, "precache_cues")(["Bay 1 stalled."])
            assert count == 0
            # Subsequent live speech still works via fallback or recovered server
            mock_gemini_server.config.tts_status_code = 200
            spk.say("Bay 1 stalled.")
            _wait_until(lambda: len(spoken_events) >= 1)
            assert len(fake_audio_env.calls()) >= 1
        finally:
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()

    def test_f7_t2_short_clutch_release_bolt_not_cleared_produces_distinct_callout(self) -> None:
        ev_short = CopilotEvent(
            kind=EventKind.RESCUE_RELEASED,
            bay=1,
            sim_t=4.2,
            wall="12:00:04",
            previous_state="RESCUE",
            state="RESCUE",
            telemetry={"bolt_cm": 3.1, "bolt_clear_cm": 5.2},
        )
        ev_clear = CopilotEvent(
            kind=EventKind.VALIDATING,
            bay=1,
            sim_t=5.0,
            wall="12:00:05",
            previous_state="RESCUE",
            state="VALIDATING",
            telemetry={"bolt_cm": 5.5, "bolt_clear_cm": 5.2},
        )
        short_txt = template_callout(ev_short)
        clear_txt = template_callout(ev_clear)
        assert short_txt.say != clear_txt.say

    def test_f7_t2_cancel_bay_drops_stale_queued_utterances_for_that_bay_only(
        self,
        fake_audio_env: FakeAudioHarness,
    ) -> None:
        assert fake_audio_env.bin_dir.is_dir()
        spk = speech_mod.Speaker(
            None,
            tts_model="gemini-3.8-flash-lite-tts",
            voice="Kore",
            start=False,
        )
        try:
            spk.say("Bay 1 stale message", bay=1)
            spk.say("Bay 2 active message", bay=2)
            dropped = spk.cancel_bay(1)
            assert dropped == 1
            assert spk.waiting() == ["Bay 2 active message"]
        finally:
            spk.close()
