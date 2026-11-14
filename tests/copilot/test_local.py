"""T-214: OpenAI-compatible local Gemma client and call-out/review fallback rules."""

from __future__ import annotations

import base64
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from teleop.copilot import (
    REVIEW_UNAVAILABLE,
    CopilotConfig,
    CopilotModels,
    GeminiError,
    GeminiHttpError,
    GeminiMalformedJsonError,
    GeminiTimeoutError,
    ImagePart,
    LocalModelClient,
    ReviewUnavailableError,
    TextPart,
)
from teleop.copilot.callouts import Callout, CalloutService
from teleop.copilot.events import CopilotEvent, EventDetector, EventKind
from teleop.copilot.frames import StoredFrame
from teleop.copilot.gemini import Part
from teleop.copilot.local import chat_completions_url
from teleop.copilot.prompts import CALLOUT_SCHEMA

FIXTURES = Path(__file__).parent / "fixtures"
JPEG = (FIXTURES / "frame-0-1.bin").read_bytes()[2:]
TOKEN = "sk-local-canary-token-987654"
CALLOUT = {"say": "Bay 1 flap jammed. Rescue needed.", "detail": "C=0.24.", "severity": "alarm"}
LOCAL_CALLOUT = {
    "say": "Bay 1 jammed locally. Rescue needed.",
    "detail": "Local Gemma saw C=0.24.",
    "severity": "alarm",
}
REVIEW = {
    "verdict": "approve",
    "reasons": ["Bolt cleared 5.6 cm."],
    "operator_note": "Clean grasp.",
    "confidence": 0.9,
}


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


def chat_ok(content: Any, finish: str = "stop") -> tuple[int, bytes]:
    doc = {
        "id": "chatcmpl-1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
            }
        ],
    }
    return 200, json.dumps(doc).encode()


def json_reply(value: Any) -> tuple[int, bytes]:
    return chat_ok(json.dumps(value))


class FakeModel:
    def __init__(
        self,
        answer: dict[str, Any] | None = None,
        error: Exception | None = None,
        gate: threading.Event | None = None,
    ) -> None:
        self.answer = answer if answer is not None else CALLOUT
        self.error = error
        self.gate = gate
        self.calls: list[tuple[str, list[Part], Mapping[str, Any] | None, float, Any]] = []

    def generate_json(
        self,
        model: str,
        system: str,
        parts: Sequence[Part],
        schema: Mapping[str, Any] | None,
        timeout_s: float,
        *,
        thinking_level: Any = None,
    ) -> dict[str, Any]:
        self.calls.append((model, list(parts), schema, timeout_s, thinking_level))
        if self.gate is not None:
            self.gate.wait(5)
        if self.error is not None:
            raise self.error
        return self.answer


class FakeSpeaker:
    def __init__(self) -> None:
        self.queue: dict[int, str] = {}
        self.said: list[tuple[str, str, int | None]] = []
        self._next = 1

    def say(self, text: str, *, severity: str = "info", bay: int | None = None) -> int:
        token = self._next
        self._next += 1
        self.said.append((text, severity, bay))
        self.queue[token] = text
        return token

    def replace(self, token: int, text: str, severity: Any = None) -> bool:
        if token not in self.queue:
            return False
        self.queue[token] = text
        return True

    def cancel_bay(self, bay: int) -> int:
        return 0


def first_stall() -> CopilotEvent:
    det = EventDetector()
    lines = (FIXTURES / "stage_telemetry.jsonl").read_text().splitlines()
    events = [e for line in lines if line for e in det.feed(json.loads(line))]
    return next(e for e in events if e.kind is EventKind.STALL)


def wait_for(cond: Callable[[], bool], timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not cond():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.005)


# -- URL & config ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434/v1/chat/completions"),
        ("http://127.0.0.1:11434/", "http://127.0.0.1:11434/v1/chat/completions"),
        ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/v1/chat/completions"),
        ("http://127.0.0.1:8080/v1/chat/completions", "http://127.0.0.1:8080/v1/chat/completions"),
        ("https://local.example/api/v1/", "https://local.example/api/v1/chat/completions"),
    ],
)
def test_chat_completions_url_normalisation(raw: str, expected: str) -> None:
    assert chat_completions_url(raw) == expected


@pytest.mark.parametrize(
    "bad", ["", "   ", "ftp://127.0.0.1:11434", "127.0.0.1:11434", "http://x?a=1"]
)
def test_chat_completions_url_rejects_bad_urls(bad: str) -> None:
    with pytest.raises(ValueError):
        chat_completions_url(bad)


def test_from_config_follows_local_url_and_no_images() -> None:
    assert LocalModelClient.from_config(CopilotConfig.from_env({})) is None
    cfg = CopilotConfig.from_env(
        {"COPILOT_LOCAL_URL": "http://127.0.0.1:11434", "COPILOT_LOCAL_MODEL": "gemma4:12b"},
        no_images=True,
    )
    client = LocalModelClient.from_config(cfg)
    assert client is not None
    assert client.endpoint == "http://127.0.0.1:11434/v1/chat/completions"
    assert client.no_images is True
    assert cfg.local_model == "gemma4:12b"


# -- OpenAI-compatible request & response ---------------------------------------------------------


def test_generate_json_sends_openai_chat_completions_with_image_and_schema() -> None:
    fake = FakeTransport([json_reply(LOCAL_CALLOUT)])
    client = LocalModelClient("http://127.0.0.1:11434", transport=fake)
    answer = client.generate_json(
        "gemma4:e4b",
        "System prompt.",
        [TextPart("Bay 1 telemetry"), ImagePart(b"\xff\xd8\xff\xd9")],
        CALLOUT_SCHEMA,
        5.0,
        thinking_level="low",
    )
    assert answer == LOCAL_CALLOUT
    (sent,) = fake.sent
    assert sent.method == "POST"
    assert sent.url == "http://127.0.0.1:11434/v1/chat/completions"
    assert "Authorization" not in sent.headers
    body = sent.json
    assert body["model"] == "gemma4:e4b"
    assert body["stream"] is False
    assert body["messages"][0] == {"role": "system", "content": "System prompt."}
    user_content = body["messages"][1]["content"]
    assert user_content[0] == {"type": "text", "text": "Bay 1 telemetry"}
    b64 = base64.b64encode(b"\xff\xd8\xff\xd9").decode("ascii")
    assert user_content[1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
    }
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "copilot_response", "strict": True, "schema": CALLOUT_SCHEMA},
    }


def test_no_images_sends_plain_text_and_zero_image_bytes() -> None:
    fake = FakeTransport([json_reply(LOCAL_CALLOUT)])
    client = LocalModelClient("http://127.0.0.1:11434/v1", transport=fake, no_images=True)
    client.generate_json(
        "gemma4:e4b",
        "",
        ["First line", ImagePart(JPEG), TextPart("Second line")],
        None,
        4.0,
    )
    (sent,) = fake.sent
    body = sent.json
    assert len(body["messages"]) == 1
    assert body["messages"][0] == {"role": "user", "content": "First line\nSecond line"}
    assert body["response_format"] == {"type": "json_object"}
    assert b"image_url" not in (sent.body or b"")


def test_generate_json_strips_think_tags_fences_and_part_arrays() -> None:
    fenced = f"<think>reasoning...</think>\n```json\n{json.dumps(LOCAL_CALLOUT)}\n```"
    blocks = [
        {"type": "thinking", "text": "internal"},
        {"type": "text", "text": json.dumps(LOCAL_CALLOUT)},
    ]
    fake = FakeTransport([chat_ok(fenced), chat_ok(blocks)])
    client = LocalModelClient("http://127.0.0.1:11434", transport=fake)
    assert client.generate_json("gemma4:e4b", "", ["hi"], None, 3.0) == LOCAL_CALLOUT
    assert client.generate_json("gemma4:e4b", "", ["hi"], None, 3.0) == LOCAL_CALLOUT


def test_http_timeout_network_and_malformed_errors_and_token_scrubbing() -> None:
    fake = FakeTransport(
        [
            (503, json.dumps({"error": {"message": f"model loading ({TOKEN})"}}).encode()),
            TimeoutError(f"timed out for {TOKEN}"),
            OSError(f"connection refused ({TOKEN})"),
            chat_ok("not json at all"),
            (200, b"{"),
        ]
    )
    client = LocalModelClient("http://127.0.0.1:11434", api_token=TOKEN, transport=fake)
    assert TOKEN not in repr(client)

    with pytest.raises(GeminiHttpError) as http_exc:
        client.generate_json("gemma4:e4b", "", ["hi"], None, 2.0)
    assert http_exc.value.status_code == 503 and TOKEN not in str(http_exc.value)
    assert fake.sent[0].headers["Authorization"] == f"Bearer {TOKEN}"

    with pytest.raises(GeminiTimeoutError) as to_exc:
        client.generate_json("gemma4:e4b", "", ["hi"], None, 2.0)
    assert TOKEN not in str(to_exc.value)

    with pytest.raises(GeminiError) as net_exc:
        client.generate_json("gemma4:e4b", "", ["hi"], None, 2.0)
    assert TOKEN not in str(net_exc.value)

    with pytest.raises(GeminiMalformedJsonError):
        client.generate_json("gemma4:e4b", "", ["hi"], None, 2.0)

    with pytest.raises(GeminiMalformedJsonError):
        client.generate_json("gemma4:e4b", "", ["hi"], None, 2.0)


# -- Fallback order: fast Gemini -> local Gemma -> template ---------------------------------------


def _service(
    gemini: FakeModel | None,
    local: FakeModel | None,
    speaker: FakeSpeaker | None = None,
    **kwargs: Any,
) -> tuple[CalloutService, list[Callout]]:
    out: list[Callout] = []
    svc = CalloutService(
        gemini,
        model="gemini-4-hydrogen",
        local_client=local,
        local_model="gemma4:e4b",
        sink=out.append,
        speaker=speaker,
        frame_for=lambda bay: StoredFrame(bay, f"bay{bay}/cctv", JPEG, 1, 0.0),
        **kwargs,
    )
    return svc, out


def test_callout_prefers_fast_gemini_without_touching_local() -> None:
    gemini, local, speaker = FakeModel(CALLOUT), FakeModel(LOCAL_CALLOUT), FakeSpeaker()
    svc, out = _service(gemini, local, speaker, template_after_s=5.0)
    svc.handle(first_stall())
    wait_for(lambda: len(out) == 1)
    assert out[0].source == "model" and out[0].model == "gemini-4-hydrogen"
    assert out[0].say == CALLOUT["say"]
    assert local.calls == []
    svc.close()


def test_callout_falls_back_to_local_gemma_when_gemini_fails() -> None:
    gemini = FakeModel(error=OSError("network unreachable"))
    local = FakeModel(LOCAL_CALLOUT)
    speaker = FakeSpeaker()
    svc, out = _service(gemini, local, speaker, template_after_s=5.0)
    svc.handle(first_stall())
    wait_for(lambda: len(out) == 1)
    assert out[0].source == "local"
    assert out[0].model == "gemma4:e4b"
    assert out[0].say == LOCAL_CALLOUT["say"]
    assert "gemini failed" in out[0].note
    assert len(gemini.calls) == 1 and len(local.calls) == 1
    assert speaker.said == [(LOCAL_CALLOUT["say"], "alarm", 1)]
    svc.close()


def test_callout_uses_local_gemma_when_gemini_key_is_unset() -> None:
    local, speaker = FakeModel(LOCAL_CALLOUT), FakeSpeaker()
    svc, out = _service(None, local, speaker, template_after_s=5.0)
    svc.handle(first_stall())
    wait_for(lambda: len(out) == 1)
    assert out[0].source == "local" and out[0].model == "gemma4:e4b"
    assert out[0].say == LOCAL_CALLOUT["say"]
    assert out[0].note == ""
    svc.close()


def test_callout_falls_back_to_template_when_both_gemini_and_local_fail() -> None:
    gemini = FakeModel(error=GeminiHttpError(503, "overloaded"))
    local = FakeModel(error=OSError("connection refused"))
    speaker = FakeSpeaker()
    svc, out = _service(gemini, local, speaker, template_after_s=5.0)
    svc.handle(first_stall())
    wait_for(lambda: len(out) == 1)
    assert out[0].source == "template"
    assert "HTTP 503" in out[0].note and "local:" in out[0].note
    assert len(speaker.said) == 1
    svc.close()


# -- Reviews never fall back to local -------------------------------------------------------------


def test_reviews_succeed_on_gemini_and_never_fall_back_to_local() -> None:
    gemini_ok = FakeModel(REVIEW)
    local = FakeModel(REVIEW)
    router = CopilotModels(gemini_ok, local)
    assert router.generate_review("sys", ["review"], None, 30.0) == REVIEW
    assert local.calls == []

    gemini_down = FakeModel(error=OSError("network down"))
    router_down = CopilotModels(gemini_down, local)
    with pytest.raises(ReviewUnavailableError) as exc_down:
        router_down.generate_review("sys", ["review"], None, 30.0)
    assert REVIEW_UNAVAILABLE in str(exc_down.value)
    assert local.calls == []  # local Gemma is healthy, yet never called for a review

    router_no_key = CopilotModels(None, local)
    with pytest.raises(ReviewUnavailableError) as exc_no_key:
        router_no_key.generate_review("sys", ["review"], None, 30.0)
    assert str(exc_no_key.value).startswith(REVIEW_UNAVAILABLE)
    assert local.calls == []
