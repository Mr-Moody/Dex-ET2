"""T-210: copilot config, Gemini client and probe, all offline through a fake transport."""

from __future__ import annotations

import ast
import base64
import io
import json
import wave
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from teleop.copilot import (
    CopilotConfig,
    GeminiClient,
    GeminiError,
    GeminiHttpError,
    GeminiMalformedJsonError,
    GeminiTimeoutError,
    ImagePart,
    TextPart,
    Transport,
)
from teleop.copilot.__main__ import CALLOUT_SCHEMA, main, run_probe, sample_jpeg
from teleop.copilot.config import _discover_env_files  # pyright: ignore[reportPrivateUsage]
from teleop.copilot.gemini import DEFAULT_BASE_URL, UrllibTransport, pcm_to_wav, wav_duration_ms

KEY = "AIzaSy-canary-key-0123456789"  # canary: must never show up in any error or output
COPILOT_SRC = Path(__file__).resolve().parents[2] / "src" / "teleop" / "copilot"


@dataclass
class Sent:
    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None
    timeout_s: float

    @property
    def json(self) -> dict[str, Any]:
        assert self.body is not None
        return json.loads(self.body)


Reply = tuple[int, bytes] | BaseException


@dataclass
class FakeTransport:
    replies: list[Reply] = field(default_factory=lambda: [])
    sent: list[Sent] = field(default_factory=lambda: [])

    def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
    ) -> tuple[int, bytes]:
        self.sent.append(Sent(method, url, dict(headers), body, timeout_s))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def ok(doc: Any) -> tuple[int, bytes]:
    return 200, json.dumps(doc).encode()


def answer(*parts: dict[str, Any], finish: str = "STOP") -> tuple[int, bytes]:
    return ok(
        {
            "candidates": [
                {"content": {"role": "model", "parts": list(parts)}, "finishReason": finish}
            ]
        }
    )


def json_answer(value: Any) -> tuple[int, bytes]:
    return answer({"text": json.dumps(value)})


def audio_answer(pcm: bytes, mime: str = "audio/L16;codec=pcm;rate=24000") -> tuple[int, bytes]:
    data = base64.b64encode(pcm).decode()
    return answer({"inlineData": {"mimeType": mime, "data": data}})


def client_with(*replies: Reply, no_images: bool = False) -> tuple[GeminiClient, FakeTransport]:
    fake = FakeTransport(list(replies))
    return GeminiClient(KEY, transport=fake, no_images=no_images), fake


CALLOUT = {"say": "Bay 1 stalled.", "detail": "Flap jam.", "severity": "alarm"}


# -- config ---------------------------------------------------------------------------------------


def test_config_defaults_with_an_empty_env() -> None:
    cfg = CopilotConfig.from_env({})
    assert cfg.gemini_api_key is None and not cfg.has_key
    assert cfg.fast_model == "gemini-4-hydrogen"
    assert cfg.review_model == "gemini-4-argon"
    assert cfg.tts_model == "gemini-3.8-flash-lite-tts"
    assert cfg.voice == "Kore"
    assert cfg.local_url is None and cfg.local_model == "gemma4:e4b"
    assert cfg.console_url == "ws://127.0.0.1:8765/ws"
    assert cfg.max_reviews_per_hour == 20
    assert not cfg.no_images and not cfg.mute


def test_config_reads_every_variable() -> None:
    cfg = CopilotConfig.from_env(
        {
            "GEMINI_API_KEY": KEY,
            "COPILOT_FAST_MODEL": "gemini-3.8-flash",
            "COPILOT_REVIEW_MODEL": "gemini-4-hydrogen",
            "COPILOT_TTS_MODEL": "gemini-3.8-flash-tts",
            "COPILOT_VOICE": "Charon",
            "COPILOT_LOCAL_URL": "http://127.0.0.1:11434/v1",
            "COPILOT_LOCAL_MODEL": "gemma4:12b",
            "COPILOT_CONSOLE_URL": "ws://127.0.0.1:9000/ws",
            "COPILOT_MAX_REVIEWS_PER_HOUR": "5",
        },
        no_images=True,
        mute=True,
    )
    assert cfg.gemini_api_key == KEY
    assert (cfg.fast_model, cfg.review_model, cfg.tts_model, cfg.voice) == (
        "gemini-3.8-flash",
        "gemini-4-hydrogen",
        "gemini-3.8-flash-tts",
        "Charon",
    )
    assert cfg.local_url == "http://127.0.0.1:11434/v1" and cfg.local_model == "gemma4:12b"
    assert cfg.console_url == "ws://127.0.0.1:9000/ws"
    assert cfg.max_reviews_per_hour == 5 and cfg.no_images and cfg.mute


def test_config_blank_values_mean_unset() -> None:
    cfg = CopilotConfig.from_env({"GEMINI_API_KEY": "  ", "COPILOT_FAST_MODEL": ""})
    assert not cfg.has_key and cfg.fast_model == "gemini-4-hydrogen"


@pytest.mark.parametrize("raw", ["0", "-3", "many"])
def test_config_rejects_a_bad_review_cap(raw: str) -> None:
    with pytest.raises(ValueError, match="COPILOT_MAX_REVIEWS_PER_HOUR"):
        CopilotConfig.from_env({"COPILOT_MAX_REVIEWS_PER_HOUR": raw})


def test_config_repr_never_shows_the_key() -> None:
    cfg = CopilotConfig.from_env({"GEMINI_API_KEY": KEY})
    assert KEY not in repr(cfg) and KEY not in str(cfg)


def test_config_env_mapping_ignores_the_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    assert not CopilotConfig.from_env({}).has_key


def test_config_env_file_layers_under_the_process_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=from-file\nCOPILOT_VOICE=Puck\nUNRELATED=1\n")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("COPILOT_VOICE", "Charon")
    cfg = CopilotConfig.from_env(env_file=env_file)
    assert cfg.gemini_api_key == "from-file"
    assert cfg.voice == "Charon"  # the process environment wins over .env


def test_env_discovery_finds_the_worktree_and_main_checkout(tmp_path: Path) -> None:
    import subprocess

    main_repo = tmp_path / "main"
    main_repo.mkdir()

    def git(*args: str, cwd: Path = main_repo) -> None:
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)

    git("init", "-q")
    git(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "x",
    )
    git("worktree", "add", "-q", str(tmp_path / "wt"))
    files = _discover_env_files(tmp_path / "wt")
    assert files == [(tmp_path / "wt" / ".env").resolve(), (main_repo / ".env").resolve()]
    assert _discover_env_files(main_repo) == [(main_repo / ".env").resolve()]


# -- client: requests ---------------------------------------------------------------------------


def test_generate_json_request_shape() -> None:
    client, fake = client_with(json_answer(CALLOUT))
    jpeg = sample_jpeg()
    got = client.generate_json(
        "gemini-4-hydrogen",
        "You assist an operator.",
        ["telemetry here", TextPart("more"), ImagePart(jpeg)],
        CALLOUT_SCHEMA,
        8.0,
        thinking_level="low",
    )
    assert got == CALLOUT
    (sent,) = fake.sent
    assert sent.method == "POST"
    assert sent.url == f"{DEFAULT_BASE_URL}/models/gemini-4-hydrogen:generateContent"
    assert sent.headers["x-goog-api-key"] == KEY
    assert sent.headers["Content-Type"] == "application/json"
    assert KEY not in sent.url and KEY.encode() not in (sent.body or b"")
    assert sent.timeout_s == 8.0
    body = sent.json
    assert body["systemInstruction"] == {"parts": [{"text": "You assist an operator."}]}
    (content,) = body["contents"]
    assert content["role"] == "user"
    assert content["parts"][:2] == [{"text": "telemetry here"}, {"text": "more"}]
    inline = content["parts"][2]["inlineData"]
    assert inline["mimeType"] == "image/jpeg"
    assert base64.b64decode(inline["data"]) == jpeg
    gen = body["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert gen["responseJsonSchema"] == CALLOUT_SCHEMA
    assert "responseSchema" not in gen
    assert gen["thinkingConfig"] == {"thinkingLevel": "LOW"}
    assert "thinkingBudget" not in json.dumps(body)


def test_generate_json_without_thinking_or_schema() -> None:
    client, fake = client_with(json_answer({"a": 1}))
    assert client.generate_json("models/gemini-3.8-flash", "", ["hi"], None, 5.0) == {"a": 1}
    body = fake.sent[0].json
    assert fake.sent[0].url.endswith("/models/gemini-3.8-flash:generateContent")
    assert "systemInstruction" not in body
    assert body["generationConfig"] == {"responseMimeType": "application/json"}


def test_openapi_schema_goes_in_response_schema() -> None:
    client, fake = client_with(json_answer({"say": "x"}))
    schema = {"type": "OBJECT", "properties": {"say": {"type": "STRING"}}}
    client.generate_json("gemini-4-hydrogen", "s", ["p"], schema, 5.0, thinking_level="high")
    gen = fake.sent[0].json["generationConfig"]
    assert gen["responseSchema"] == schema and "responseJsonSchema" not in gen
    assert gen["thinkingConfig"] == {"thinkingLevel": "HIGH"}


def test_no_images_sends_zero_image_bytes() -> None:
    client, fake = client_with(json_answer(CALLOUT), no_images=True)
    jpeg = sample_jpeg()
    client.generate_json("gemini-4-hydrogen", "s", ["t", ImagePart(jpeg)], CALLOUT_SCHEMA, 5.0)
    body = fake.sent[0].body or b""
    assert b"inlineData" not in body and base64.b64encode(jpeg) not in body
    assert fake.sent[0].json["contents"][0]["parts"] == [{"text": "t"}]


def test_no_images_with_only_images_refuses_to_send() -> None:
    client, fake = client_with(no_images=True)
    with pytest.raises(ValueError, match="at least one part"):
        client.generate_json("gemini-4-hydrogen", "s", [ImagePart(b"x")], None, 5.0)
    assert fake.sent == []


@pytest.mark.parametrize("bad", ["gemini/../x", "a b", "", "models/"])
def test_bad_model_ids_are_rejected_before_sending(bad: str) -> None:
    client, fake = client_with()
    with pytest.raises(ValueError, match="invalid model id"):
        client.generate_json(bad, "s", ["p"], None, 5.0)
    assert fake.sent == []


def test_bad_thinking_level_and_timeout_are_rejected() -> None:
    client, _ = client_with()
    with pytest.raises(ValueError, match="thinking_level"):
        client.generate_json("m", "s", ["p"], None, 5.0, thinking_level="minimal")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timeout_s"):
        client.generate_json("m", "s", ["p"], None, 0.0)


def test_client_needs_a_key_and_hides_it() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiClient("  ", transport=FakeTransport())
    client, _ = client_with()
    assert KEY not in repr(client)


def test_list_models_follows_pages() -> None:
    page1 = {
        "models": [
            {
                "name": "models/gemini-4-hydrogen",
                "displayName": "Gemini 4 Hydrogen",
                "supportedGenerationMethods": ["generateContent", "countTokens"],
                "inputTokenLimit": 1048576,
            },
            {"name": "models/text-embedding-005", "supportedGenerationMethods": ["embedContent"]},
        ],
        "nextPageToken": "tok 2",
    }
    page2 = {"models": [{"name": "models/gemini-4-argon"}, {"bogus": True}]}
    client, fake = client_with(ok(page1), ok(page2))
    models = client.list_models()
    assert [m.name for m in models] == [
        "gemini-4-hydrogen",
        "text-embedding-005",
        "gemini-4-argon",
    ]
    assert models[0].can_generate and models[0].display_name == "Gemini 4 Hydrogen"
    assert models[0].input_token_limit == 1048576
    assert not models[1].can_generate and not models[2].can_generate
    assert [s.method for s in fake.sent] == ["GET", "GET"]
    assert fake.sent[0].url == f"{DEFAULT_BASE_URL}/models?pageSize=100"
    assert fake.sent[1].url == f"{DEFAULT_BASE_URL}/models?pageSize=100&pageToken=tok+2"
    assert all(s.body is None and s.headers["x-goog-api-key"] == KEY for s in fake.sent)


# -- client: answers ----------------------------------------------------------------------------


def test_thought_parts_and_fences_are_stripped() -> None:
    client, _ = client_with(
        answer(
            {"text": "Let me think about the flap...", "thought": True},
            {"text": "```json\n" + json.dumps(CALLOUT) + "\n```", "thoughtSignature": "c2ln"},
        )
    )
    assert client.generate_json("m", "s", ["p"], CALLOUT_SCHEMA, 5.0) == CALLOUT


def test_answer_split_over_parts_is_joined() -> None:
    text = json.dumps(CALLOUT)
    client, _ = client_with(answer({"text": text[:10]}, {"text": text[10:]}))
    assert client.generate_json("m", "s", ["p"], None, 5.0) == CALLOUT


@pytest.mark.parametrize(
    ("reply", "match"),
    [
        ((200, b"not json"), "not JSON"),
        (ok([1, 2]), "JSON object"),
        (ok({"candidates": []}), "no candidates"),
        (answer(finish="MAX_TOKENS"), "MAX_TOKENS"),
        (answer({"text": "sure, here you go"}), "not JSON"),
        (answer({"text": "[1, 2]"}), "JSON object"),
    ],
)
def test_malformed_answers(reply: Reply, match: str) -> None:
    client, _ = client_with(reply)
    with pytest.raises(GeminiMalformedJsonError, match=match):
        client.generate_json("m", "s", ["p"], None, 5.0)


def test_blocked_prompt_is_an_error() -> None:
    client, _ = client_with(ok({"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(GeminiError, match="blocked: SAFETY"):
        client.generate_json("m", "s", ["p"], None, 5.0)


# -- errors and the key -------------------------------------------------------------------------


def _api_error(status: int, name: str, message: str) -> tuple[int, bytes]:
    doc = {"error": {"code": status, "message": message, "status": name}}
    return status, json.dumps(doc).encode()


@pytest.mark.parametrize(
    ("status", "name"),
    [
        (400, "INVALID_ARGUMENT"),
        (401, "UNAUTHENTICATED"),
        (403, "PERMISSION_DENIED"),
        (404, "NOT_FOUND"),
        (429, "RESOURCE_EXHAUSTED"),
        (500, "INTERNAL"),
        (503, "UNAVAILABLE"),
    ],
)
def test_http_errors_map_and_never_leak_the_key(status: int, name: str) -> None:
    client, _ = client_with(_api_error(status, name, f"API key {KEY} rejected"))
    with pytest.raises(GeminiHttpError) as info:
        client.generate_json("m", "s", ["p"], None, 5.0)
    err = info.value
    assert err.status_code == status
    assert name in err.detail and "[redacted]" in err.detail
    assert KEY not in str(err) and KEY not in repr(err) and KEY not in err.detail


def test_http_error_with_a_plain_body_is_truncated() -> None:
    client, _ = client_with((502, b"<html>" + b"x" * 1000 + b"</html>"))
    with pytest.raises(GeminiHttpError) as info:
        client.list_models()
    assert info.value.status_code == 502 and len(info.value.detail) <= 300


def test_timeout_maps_and_never_leaks_the_key() -> None:
    client, _ = client_with(TimeoutError(f"timed out with {KEY}"))
    with pytest.raises(GeminiTimeoutError) as info:
        client.synthesize("hello", "gemini-3.8-flash-lite-tts", "Kore", 4.0)
    assert "4 s" in str(info.value) and KEY not in str(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_network_error_maps_and_never_leaks_the_key() -> None:
    client, _ = client_with(ConnectionRefusedError(f"refused {KEY}"))
    with pytest.raises(GeminiError) as info:
        client.list_models()
    assert not isinstance(info.value, GeminiTimeoutError)
    assert KEY not in str(info.value) and "network error" in str(info.value)


def test_default_transport_is_a_transport() -> None:
    assert isinstance(UrllibTransport(), Transport)
    assert isinstance(FakeTransport(), Transport)


def test_urllib_transport_does_not_follow_redirects() -> None:
    """A redirect must never carry the key header to another host."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            seen.append(self.headers.get("x-goog-api-key"))
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{server.server_address[1]}/elsewhere")
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = GeminiClient(KEY, base_url=f"http://127.0.0.1:{server.server_port}")
        with pytest.raises(GeminiHttpError) as info:
            client.list_models(timeout_s=5.0)
    finally:
        server.shutdown()
        server.server_close()
    assert info.value.status_code == 302
    assert seen == [KEY]  # one request only: the redirect target was never contacted


# -- speech -------------------------------------------------------------------------------------


def test_synthesize_request_and_wav() -> None:
    pcm = (b"\x01\x00\xff\x7f") * 2400  # 4800 samples = 200 ms at 24 kHz
    client, fake = client_with(audio_answer(pcm))
    wav = client.synthesize("Bay one stalled.", "gemini-3.8-flash-lite-tts", "Kore", 4.0)
    body = fake.sent[0].json
    assert fake.sent[0].url.endswith("/models/gemini-3.8-flash-lite-tts:generateContent")
    assert body["contents"] == [{"role": "user", "parts": [{"text": "Bay one stalled."}]}]
    gen = body["generationConfig"]
    assert gen["responseModalities"] == ["AUDIO"]
    assert gen["speechConfig"] == {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}
    assert "systemInstruction" not in body
    with wave.open(io.BytesIO(wav), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24_000)
        assert w.readframes(w.getnframes()) == pcm
    assert wav_duration_ms(wav) == pytest.approx(200.0)


def test_synthesize_uses_the_rate_in_the_mime_type() -> None:
    client, _ = client_with(audio_answer(b"\x00\x00" * 1600, "audio/L16;codec=pcm;rate=16000"))
    wav = client.synthesize("x", "tts", "Kore", 4.0)
    assert wav_duration_ms(wav) == pytest.approx(100.0)


def test_synthesize_passes_a_wav_through() -> None:
    wav = pcm_to_wav(b"\x00\x00" * 24)
    client, _ = client_with(audio_answer(wav, "audio/wav"))
    assert client.synthesize("x", "tts", "Kore", 4.0) == wav


@pytest.mark.parametrize(
    ("reply", "match"),
    [
        (answer({"text": "I cannot speak"}), "no audio"),
        (audio_answer(b"\x00\x00\x00"), "odd byte count"),
        (answer({"inlineData": {"mimeType": "audio/L16", "data": "@@not base64@@"}}), "base64"),
    ],
)
def test_synthesize_bad_audio(reply: Reply, match: str) -> None:
    client, _ = client_with(reply)
    with pytest.raises(GeminiMalformedJsonError, match=match):
        client.synthesize("x", "tts", "Kore", 4.0)


def test_synthesize_rejects_empty_text() -> None:
    client, fake = client_with()
    with pytest.raises(ValueError, match="non-empty"):
        client.synthesize("  ", "tts", "Kore", 4.0)
    assert fake.sent == []


# -- probe --------------------------------------------------------------------------------------

MODELS_PAGE = {
    "models": [
        {"name": f"models/{n}", "supportedGenerationMethods": ["generateContent"]}
        for n in (
            "gemini-4-hydrogen",
            "gemini-4-argon",
            "gemini-3.8-flash",
            "gemini-3.8-flash-lite-tts",
        )
    ]
}
REVIEW = {
    "verdict": "approve",
    "reasons": ["bolt cleared"],
    "operator_note": "ok",
    "confidence": 0.8,
}


def _clock() -> Any:
    ticks = iter(range(0, 10**12, 250_000_000))  # every reading is 250 ms after the last
    return lambda: next(ticks)


def test_probe_report_and_recommendations() -> None:
    cfg = CopilotConfig.from_env({"GEMINI_API_KEY": KEY})
    client, fake = client_with(
        ok(MODELS_PAGE),
        json_answer(CALLOUT),  # fast model
        json_answer(CALLOUT),  # comparison on gemini-3.8-flash
        json_answer(REVIEW),
        audio_answer(b"\x00\x00" * 24_000),  # 1 s of audio
    )
    out = io.StringIO()
    assert run_probe(cfg, client, out, clock=_clock()) == 0
    report = out.getvalue()
    assert "| `gemini-4-hydrogen` | yes | yes |" in report
    assert "| `gemini-3.8-flash-tts` | no | no |" in report
    assert "| call-out | `gemini-4-hydrogen` | ok | 250 |" in report
    assert "| review | `gemini-4-argon` | ok |" in report
    assert "1000 ms of audio" in report
    assert "COPILOT_FAST_MODEL=gemini-4-hydrogen" in report
    assert "COPILOT_REVIEW_MODEL=gemini-4-argon" in report
    assert "COPILOT_TTS_MODEL=gemini-3.8-flash-lite-tts" in report
    assert KEY not in report
    thinking = [s.json["generationConfig"].get("thinkingConfig") for s in fake.sent[1:4]]
    assert thinking == [
        {"thinkingLevel": "LOW"},
        {"thinkingLevel": "LOW"},
        {"thinkingLevel": "HIGH"},
    ]
    review_parts = fake.sent[3].json["contents"][0]["parts"]
    assert sum("inlineData" in p for p in review_parts) == 3


def test_probe_reports_failures_and_fallbacks_without_the_key() -> None:
    cfg = CopilotConfig.from_env({"GEMINI_API_KEY": KEY}, no_images=True)
    client, fake = client_with(
        _api_error(403, "PERMISSION_DENIED", f"key {KEY} has no access to models"),  # listing
        TimeoutError("slow"),  # fast model
        json_answer(CALLOUT),  # comparison
        _api_error(429, "RESOURCE_EXHAUSTED", "Argon needs a paid tier"),
        _api_error(404, "NOT_FOUND", "no such model"),  # flash-lite TTS
        audio_answer(b"\x00\x00" * 2400),  # gemini-3.8-flash-tts fallback
        no_images=True,
    )
    out = io.StringIO()
    assert run_probe(cfg, client, out, clock=_clock()) == 1  # the review failed
    report = out.getvalue()
    assert KEY not in report
    assert "Listing failed: HTTP 403" in report and "[redacted]" in report
    assert "| call-out | `gemini-4-hydrogen` | FAILED | - |" in report
    assert "COPILOT_FAST_MODEL=gemini-3.8-flash" in report
    assert "COPILOT_REVIEW_MODEL=gemini-3.8-flash  # gemini-4-argon failed" in report
    assert "| speech (fallback) | `gemini-3.8-flash-tts` | ok |" in report
    assert "COPILOT_TTS_MODEL=gemini-3.8-flash-tts" in report
    assert all(b"inlineData" not in (s.body or b"") for s in fake.sent)
    assert fake.replies == []


def test_main_without_a_key_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    empty = tmp_path / ".env"
    empty.write_text("")
    assert main(["probe", "--env-file", str(empty)]) == 2


def test_main_applies_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"GEMINI_API_KEY={KEY}\n")
    seen: list[CopilotConfig] = []

    def factory(cfg: CopilotConfig) -> GeminiClient:
        seen.append(cfg)
        client, _ = client_with(
            ok({"models": []}),
            json_answer(CALLOUT),
            json_answer(REVIEW),
            audio_answer(b"\x00\x00" * 240),
            no_images=cfg.no_images,
        )
        return client

    out = io.StringIO()
    code = main(
        [
            "probe",
            "--env-file",
            str(env_file),
            "--no-images",
            "--fast-model",
            "gemini-3.8-flash",
            "--review-model",
            "gemini-4-hydrogen",
            "--voice",
            "Charon",
        ],
        client_factory=factory,
        out=out,
    )
    assert code == 0
    (cfg,) = seen
    assert cfg.no_images and cfg.fast_model == "gemini-3.8-flash"
    assert cfg.review_model == "gemini-4-hydrogen" and cfg.voice == "Charon"
    assert KEY not in out.getvalue()


# -- advisory-only guard --------------------------------------------------------------------------


def test_copilot_never_imports_the_coordinator_or_sends_keys() -> None:
    files = sorted(COPILOT_SRC.glob("*.py"))
    assert files, "copilot package not found"
    for path in files:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = []
            for name in names:
                assert not name.startswith("teleop.atc"), f"{path.name} imports {name}"
            if isinstance(node, ast.Constant):
                assert node.value != "key", f"{path.name}:{node.lineno} has a 'key' message"
            if isinstance(node, ast.Attribute):
                assert node.attr not in ("send", "send_text"), f"{path.name}:{node.lineno} sends"
