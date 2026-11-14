"""OpenAI-compatible local Gemma client and role fallback rules (``T-214``).

Connects to any local OpenAI-compatible ``/v1/chat/completions`` server (Ollama, ``llama-server``,
vLLM, LM Studio) configured with ``COPILOT_LOCAL_URL`` and ``COPILOT_LOCAL_MODEL``. It implements
the same :meth:`~LocalModelClient.generate_json` contract as
:class:`~teleop.copilot.gemini.GeminiClient`, over the same injectable
:class:`~teleop.copilot.gemini.Transport` so tests stay offline.

Fallback rules:

- **Call-outs:** fast Gemini model -> local Gemma -> deterministic template.
- **Reviews:** Gemini review model only. Reviews never fall back to local; when Gemini is unset or
  unreachable they report ``"review unavailable"`` (:class:`ReviewUnavailableError`).
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Protocol, cast
from urllib.parse import urlsplit

from teleop.copilot.config import (
    DEFAULT_FAST_MODEL,
    DEFAULT_FAST_THINKING,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_REVIEW_MODEL,
    DEFAULT_REVIEW_THINKING,
    CopilotConfig,
)
from teleop.copilot.gemini import (
    MAX_DETAIL_CHARS,
    REDACTED,
    THINKING_LEVELS,
    USER_AGENT,
    GeminiError,
    GeminiHttpError,
    GeminiMalformedJsonError,
    GeminiTimeoutError,
    Part,
    TextPart,
    ThinkingLevel,
    Transport,
    UrllibTransport,
)

REVIEW_UNAVAILABLE = "review unavailable"

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


class ReviewUnavailableError(GeminiError):
    """Raised when the Gemini review model is unset or unreachable.

    Reviews never fall back to the local Gemma model.
    """

    def __init__(self, reason: str = "") -> None:
        detail = f"{REVIEW_UNAVAILABLE}: {reason}" if reason else REVIEW_UNAVAILABLE
        super().__init__(detail)
        self.reason = reason


class _JsonClient(Protocol):
    def generate_json(
        self,
        model: str,
        system: str,
        parts: Sequence[Part],
        schema: Mapping[str, Any] | None,
        timeout_s: float,
        *,
        thinking_level: ThinkingLevel | None = None,
    ) -> dict[str, Any]: ...


def chat_completions_url(base_url: str) -> str:
    """Normalise an OpenAI-compatible server URL to its ``/chat/completions`` endpoint.

    Accepts ``http://127.0.0.1:11434`` (Ollama root), ``http://127.0.0.1:11434/v1``, or a full
    ``.../chat/completions`` URL.
    """
    cleaned = base_url.strip().rstrip("/")
    if not cleaned:
        raise ValueError("COPILOT_LOCAL_URL is empty")
    parsed = urlsplit(cleaned)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"COPILOT_LOCAL_URL must start with http:// or https://, got {base_url!r}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"COPILOT_LOCAL_URL must not contain query or fragment: {base_url!r}")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return f"{cleaned}/v1/chat/completions"


class LocalModelClient:
    """Calls a local OpenAI-compatible ``/v1/chat/completions`` server for structured JSON."""

    def __init__(
        self,
        base_url: str,
        *,
        api_token: str | None = None,
        transport: Transport | None = None,
        no_images: bool = False,
    ) -> None:
        self._endpoint = chat_completions_url(base_url)
        self._token = api_token.strip() if api_token and api_token.strip() else None
        self._transport: Transport = transport if transport is not None else UrllibTransport()
        self.no_images = no_images

    def __repr__(self) -> str:
        return f"LocalModelClient(endpoint={self._endpoint!r}, no_images={self.no_images})"

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @classmethod
    def from_config(
        cls,
        cfg: CopilotConfig,
        *,
        transport: Transport | None = None,
    ) -> LocalModelClient | None:
        """Build a local client when ``COPILOT_LOCAL_URL`` is set, else ``None``."""
        if not cfg.local_url:
            return None
        return cls(cfg.local_url, transport=transport, no_images=cfg.no_images)

    def generate_json(
        self,
        model: str,
        system: str,
        parts: Sequence[Part],
        schema: Mapping[str, Any] | None,
        timeout_s: float,
        *,
        thinking_level: ThinkingLevel | None = None,
    ) -> dict[str, Any]:
        """Ask ``model`` on the local server for one JSON object following ``schema``."""
        model_id = model.strip()
        if not model_id:
            raise ValueError("local model name must not be empty")
        if thinking_level is not None and thinking_level not in THINKING_LEVELS:
            raise ValueError(f"thinking_level must be one of {THINKING_LEVELS}")
        content = self.build_content(parts)
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        response_format: dict[str, Any]
        if schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "copilot_response",
                    "strict": True,
                    "schema": dict(schema),
                },
            }
        else:
            response_format = {"type": "json_object"}
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "response_format": response_format,
            "stream": False,
        }
        doc = self._call(payload, timeout_s)
        return _parse_json_answer(_choice_text(doc))

    def build_content(self, parts: Iterable[Part]) -> str | list[dict[str, Any]]:
        """OpenAI ``messages[].content`` for ``parts``, honouring ``no_images``."""
        texts: list[str] = []
        blocks: list[dict[str, Any]] = []
        has_image = False
        for part in parts:
            if isinstance(part, str):
                texts.append(part)
                blocks.append({"type": "text", "text": part})
            elif isinstance(part, TextPart):
                texts.append(part.text)
                blocks.append({"type": "text", "text": part.text})
            elif not self.no_images:
                has_image = True
                data = base64.b64encode(part.jpeg_bytes).decode("ascii")
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{part.mime_type};base64,{data}"},
                    }
                )
        if not blocks:
            raise ValueError("generate_json needs at least one part left to send")
        if not has_image:
            return "\n".join(texts)
        return blocks

    def _scrub(self, text: str) -> str:
        if not self._token:
            return text
        return text.replace(self._token, REDACTED)

    def _call(self, payload: Mapping[str, Any], timeout_s: float) -> dict[str, Any]:
        if not timeout_s > 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")
        headers: dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        try:
            status, raw = self._transport.request("POST", self._endpoint, headers, body, timeout_s)
        except TimeoutError as exc:
            raise GeminiTimeoutError(
                self._scrub(f"no answer within {timeout_s:g} s ({exc})")
            ) from None
        except OSError as exc:
            raise GeminiError(self._scrub(f"network error: {exc}")) from None
        if not 200 <= status < 300:
            raise GeminiHttpError(status, self._scrub(_error_detail(raw)))
        try:
            doc: object = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise GeminiMalformedJsonError("response body is not JSON") from None
        if not isinstance(doc, dict):
            raise GeminiMalformedJsonError(f"expected a JSON object, got {type(doc).__name__}")
        return cast(dict[str, Any], doc)


class CopilotModels:
    """Role-aware model router implementing the ``T-214`` fallback policy.

    - Call-outs try Gemini first, then local Gemma (callers fall back to templates if both fail).
    - Reviews try Gemini only and raise :class:`ReviewUnavailableError` (``"review unavailable"``)
      without ever calling the local server.
    """

    def __init__(
        self,
        gemini: _JsonClient | None,
        local: _JsonClient | None = None,
        *,
        fast_model: str = DEFAULT_FAST_MODEL,
        local_model: str = DEFAULT_LOCAL_MODEL,
        review_model: str = DEFAULT_REVIEW_MODEL,
        fast_thinking: ThinkingLevel | None = DEFAULT_FAST_THINKING,
        review_thinking: ThinkingLevel | None = DEFAULT_REVIEW_THINKING,
    ) -> None:
        self._gemini = gemini
        self._local = local
        self._fast_model = fast_model
        self._local_model = local_model
        self._review_model = review_model
        self._fast_thinking: ThinkingLevel | None = fast_thinking
        self._review_thinking: ThinkingLevel | None = review_thinking

    def generate_callout(
        self,
        system: str,
        parts: Sequence[Part],
        schema: Mapping[str, Any] | None,
        timeout_s: float,
    ) -> tuple[dict[str, Any], Literal["model", "local"], str]:
        """Return ``(answer, source, model_used)`` for a call-out."""
        primary_err: Exception | None = None
        if self._gemini is not None:
            try:
                answer = self._gemini.generate_json(
                    self._fast_model,
                    system,
                    parts,
                    schema,
                    timeout_s,
                    thinking_level=self._fast_thinking,
                )
                return answer, "model", self._fast_model
            except (GeminiError, ValueError, OSError) as exc:
                primary_err = exc
        if self._local is not None:
            answer = self._local.generate_json(
                self._local_model,
                system,
                parts,
                schema,
                timeout_s,
            )
            return answer, "local", self._local_model
        if primary_err is not None:
            raise primary_err
        raise GeminiError("no call-out model configured")

    def generate_review(
        self,
        system: str,
        parts: Sequence[Part],
        schema: Mapping[str, Any] | None,
        timeout_s: float,
    ) -> dict[str, Any]:
        """Call the Gemini review model only; never falls back to local Gemma."""
        if self._gemini is None:
            raise ReviewUnavailableError("GEMINI_API_KEY is not set")
        try:
            return self._gemini.generate_json(
                self._review_model,
                system,
                parts,
                schema,
                timeout_s,
                thinking_level=self._review_thinking,
            )
        except (GeminiError, OSError) as exc:
            raise ReviewUnavailableError(str(exc)) from exc


def _error_detail(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace").strip()
    try:
        doc: object = json.loads(text)
    except ValueError:
        doc = None
    if isinstance(doc, dict):
        err = cast(dict[str, Any], doc).get("error")
        if isinstance(err, dict):
            err_doc = cast(dict[str, Any], err)
            code = err_doc.get("type", err_doc.get("code"))
            message = err_doc.get("message")
            pieces = [str(p) for p in (code, message) if isinstance(p, str) and p]
            if pieces:
                text = ": ".join(pieces)
        elif isinstance(err, str) and err:
            text = err
    text = " ".join(text.split()) or "(empty body)"
    if len(text) > MAX_DETAIL_CHARS:
        text = text[: MAX_DETAIL_CHARS - 3] + "..."
    return text


def _choice_text(doc: Mapping[str, Any]) -> str:
    raw_choices = doc.get("choices")
    if not isinstance(raw_choices, list) or not raw_choices:
        raise GeminiMalformedJsonError("response has no choices")
    first = cast(list[object], raw_choices)[0]
    if not isinstance(first, dict):
        raise GeminiMalformedJsonError("choices[0] is not an object")
    first_doc = cast(dict[str, Any], first)
    finish = first_doc.get("finish_reason")
    message = first_doc.get("message")
    if not isinstance(message, dict):
        raise GeminiMalformedJsonError("choices[0].message is missing")
    content = cast(dict[str, Any], message).get("content")
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        pieces: list[str] = []
        for item in cast(list[object], content):
            if isinstance(item, dict):
                block = cast(dict[str, Any], item)
                kind = block.get("type")
                if kind in ("thinking", "reasoning"):
                    continue
                part_text = block.get("text")
                if isinstance(part_text, str):
                    pieces.append(part_text)
        text = "".join(pieces)
    if not text.strip():
        raise GeminiMalformedJsonError(f"answer has no text (finish_reason={finish})")
    return text


def _parse_json_answer(text: str) -> dict[str, Any]:
    cleaned = _THINK_RE.sub("", text).strip()
    fenced = _FENCE_RE.match(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        value: object = json.loads(cleaned)
    except ValueError:
        raise GeminiMalformedJsonError(f"model answer is not JSON: {cleaned[:120]!r}") from None
    if not isinstance(value, dict):
        raise GeminiMalformedJsonError(
            f"model answer must be a JSON object, got {type(value).__name__}"
        )
    return cast(dict[str, Any], value)
