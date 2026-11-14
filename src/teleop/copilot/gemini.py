"""Stdlib Gemini API client: structured JSON answers, model listing and speech (TTS).

REST endpoint ``https://generativelanguage.googleapis.com/v1beta``. The key goes only in the
``x-goog-api-key`` header (never in a URL), and every error message is scrubbed of it.

All HTTP goes through an injectable :class:`Transport`, so tests use a fake and never touch the
network. The default :class:`UrllibTransport` refuses redirects, because urllib would otherwise
copy the key header to whatever host a redirect names.

Field names follow the Gemini REST ``generateContent`` reference (camelCase):
``systemInstruction``, ``contents[].parts[].inlineData``, ``generationConfig.responseMimeType``,
``responseJsonSchema`` / ``responseSchema``, ``thinkingConfig.thinkingLevel`` and, for speech,
``responseModalities`` plus ``speechConfig.voiceConfig.prebuiltVoiceConfig.voiceName``.
``python -m teleop.copilot probe`` is the live check that the key and models accept them.
"""

from __future__ import annotations

import base64
import io
import json
import re
import wave
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast, runtime_checkable
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
USER_AGENT = "teleop-copilot/0.1"
MAX_DETAIL_CHARS = 300
REDACTED = "[redacted]"

# Gemini 4 takes low/medium/high (and xhigh on Hydrogen). "minimal" and thinkingBudget are HTTP
# 400 on Gemini 4, so neither is offered. Sent in the enum's canonical upper case.
ThinkingLevel = Literal["low", "medium", "high", "xhigh"]
THINKING_LEVELS: tuple[ThinkingLevel, ...] = ("low", "medium", "high", "xhigh")

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_RATE_RE = re.compile(r"rate=(\d+)")
_OPENAPI_TYPES = frozenset({"STRING", "NUMBER", "INTEGER", "BOOLEAN", "ARRAY", "OBJECT"})

TTS_SAMPLE_RATE = 24_000  # Gemini TTS returns 24 kHz, 16-bit, mono PCM


class GeminiError(RuntimeError):
    """Any Gemini failure. The message never contains the API key."""


class GeminiHttpError(GeminiError):
    """The API answered with an HTTP error status."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class GeminiTimeoutError(GeminiError):
    """No complete answer within the timeout."""


class GeminiMalformedJsonError(GeminiError):
    """The response (or the model's answer inside it) was not the JSON that was asked for."""


@dataclass(frozen=True, slots=True)
class TextPart:
    text: str


@dataclass(frozen=True, slots=True)
class ImagePart:
    """One encoded image, sent as base64 ``inlineData``. ``repr`` hides the bytes."""

    jpeg_bytes: bytes = field(repr=False)
    mime_type: str = "image/jpeg"


Part = TextPart | ImagePart | str


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """One entry of ``GET /models``, with the ``models/`` prefix removed from ``name``."""

    name: str
    display_name: str = ""
    methods: tuple[str, ...] = ()
    input_token_limit: int | None = None
    output_token_limit: int | None = None

    @property
    def can_generate(self) -> bool:
        return "generateContent" in self.methods


@runtime_checkable
class Transport(Protocol):
    """Sends one HTTP request and returns ``(status, body)``, HTTP error statuses included.

    Raises :class:`TimeoutError` when the timeout expires and :class:`OSError` for other network
    failures. It must not follow redirects.
    """

    def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
    ) -> tuple[int, bytes]: ...


class _NoRedirect(urlrequest.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None  # urllib then raises HTTPError with the 3xx status


class UrllibTransport:
    """The real transport, on ``urllib.request``. The timeout applies to each socket operation."""

    def __init__(self) -> None:
        self._opener = urlrequest.build_opener(_NoRedirect())

    def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
    ) -> tuple[int, bytes]:
        req = urlrequest.Request(url, data=body, headers=dict(headers), method=method)
        try:
            with self._opener.open(req, timeout=timeout_s) as resp:
                return int(resp.status), cast(bytes, resp.read())
        except HTTPError as exc:
            try:
                data = exc.read()
            except OSError:
                data = b""
            return exc.code, data
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):  # socket.timeout is an alias
                raise TimeoutError(str(exc.reason)) from None
            raise OSError(str(exc.reason)) from None


class GeminiClient:
    """Calls the Gemini REST API. Thread-safe as long as the transport is."""

    def __init__(
        self,
        api_key: str,
        *,
        transport: Transport | None = None,
        base_url: str = DEFAULT_BASE_URL,
        no_images: bool = False,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("GEMINI_API_KEY is missing; put it in .env (see docs/copilot.md)")
        self._key = api_key.strip()
        self._transport: Transport = transport if transport is not None else UrllibTransport()
        self._base_url = base_url.rstrip("/")
        self.no_images = no_images

    def __repr__(self) -> str:
        return f"GeminiClient(base_url={self._base_url!r}, no_images={self.no_images})"

    # -- public API ---------------------------------------------------------------------------

    def list_models(self, timeout_s: float = 10.0) -> list[ModelInfo]:
        """Every model the key can see (all pages)."""
        models: list[ModelInfo] = []
        token: str | None = None
        for _ in range(50):  # page guard
            query = {"pageSize": "100"} | ({"pageToken": token} if token else {})
            doc = self._call("GET", f"{self._base_url}/models?{urlencode(query)}", None, timeout_s)
            for raw in _list(doc.get("models")):
                info = _model_info(raw)
                if info is not None:
                    models.append(info)
            next_token = doc.get("nextPageToken")
            if not isinstance(next_token, str) or not next_token:
                break
            token = next_token
        return models

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
        """Ask ``model`` for one JSON object that follows ``schema``.

        ``schema`` is a JSON Schema (lower-case ``"type": "object"``), sent as
        ``responseJsonSchema``. A schema in Gemini's OpenAPI form (upper-case ``"OBJECT"``) is sent
        as ``responseSchema`` instead. With ``no_images`` set, image parts are dropped here, so no
        image byte leaves the machine.
        """
        content_parts = self.build_parts(parts)
        if not content_parts:
            raise ValueError("generate_json needs at least one part left to send")
        generation: dict[str, Any] = {"responseMimeType": "application/json"}
        if schema is not None:
            generation[_schema_field(schema)] = dict(schema)
        if thinking_level is not None:
            if thinking_level not in THINKING_LEVELS:
                raise ValueError(f"thinking_level must be one of {THINKING_LEVELS}")
            generation["thinkingConfig"] = {"thinkingLevel": thinking_level.upper()}
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": content_parts}],
            "generationConfig": generation,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        doc = self._call("POST", self._generate_url(model), payload, timeout_s)
        return _parse_json_answer(_answer_text(doc))

    def synthesize(self, text: str, model: str, voice: str, timeout_s: float) -> bytes:
        """Speak ``text`` with a prebuilt ``voice``; returns a complete WAV file."""
        if not text.strip():
            raise ValueError("synthesize needs non-empty text")
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
            },
        }
        doc = self._call("POST", self._generate_url(model), payload, timeout_s)
        return _audio_to_wav(doc)

    def build_parts(self, parts: Iterable[Part]) -> list[dict[str, Any]]:
        """The REST ``parts`` list for ``parts``, honouring ``no_images``."""
        out: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, str):
                out.append({"text": part})
            elif isinstance(part, TextPart):
                out.append({"text": part.text})
            elif not self.no_images:
                data = base64.b64encode(part.jpeg_bytes).decode("ascii")
                out.append({"inlineData": {"mimeType": part.mime_type, "data": data}})
        return out

    # -- internals ----------------------------------------------------------------------------

    def _generate_url(self, model: str) -> str:
        model_id = model.removeprefix("models/")
        if not _MODEL_ID_RE.match(model_id):
            raise ValueError(f"invalid model id: {model!r}")
        return f"{self._base_url}/models/{quote(model_id)}:generateContent"

    def _scrub(self, text: str) -> str:
        return text.replace(self._key, REDACTED)

    def _call(
        self, method: str, url: str, payload: Mapping[str, Any] | None, timeout_s: float
    ) -> dict[str, Any]:
        if not timeout_s > 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")
        headers = {
            "x-goog-api-key": self._key,
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
        body: bytes | None = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        try:
            status, raw = self._transport.request(method, url, headers, body, timeout_s)
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


def _schema_field(schema: Mapping[str, Any]) -> str:
    kind = schema.get("type")
    if isinstance(kind, str) and kind in _OPENAPI_TYPES:
        return "responseSchema"
    return "responseJsonSchema"


def _list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items = cast(list[object], value)
    return [cast(dict[str, Any], v) for v in items if isinstance(v, dict)]


def _model_info(raw: Mapping[str, Any]) -> ModelInfo | None:
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        return None
    methods_raw = raw.get("supportedGenerationMethods")
    methods = (
        tuple(str(m) for m in cast(list[object], methods_raw))
        if isinstance(methods_raw, list)
        else ()
    )
    display = raw.get("displayName")
    in_limit = raw.get("inputTokenLimit")
    out_limit = raw.get("outputTokenLimit")
    return ModelInfo(
        name=name.removeprefix("models/"),
        display_name=display if isinstance(display, str) else "",
        methods=methods,
        input_token_limit=in_limit if isinstance(in_limit, int) else None,
        output_token_limit=out_limit if isinstance(out_limit, int) else None,
    )


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
            status = err_doc.get("status")
            message = err_doc.get("message")
            pieces = [str(p) for p in (status, message) if isinstance(p, str) and p]
            if pieces:
                text = ": ".join(pieces)
    text = " ".join(text.split()) or "(empty body)"
    if len(text) > MAX_DETAIL_CHARS:
        text = text[: MAX_DETAIL_CHARS - 3] + "..."
    return text


def _first_candidate_parts(doc: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    feedback = doc.get("promptFeedback")
    if isinstance(feedback, dict):
        reason = cast(dict[str, Any], feedback).get("blockReason")
        if isinstance(reason, str) and reason:
            raise GeminiError(f"prompt blocked: {reason}")
    candidates = _list(doc.get("candidates"))
    if not candidates:
        raise GeminiMalformedJsonError("response has no candidates")
    first = candidates[0]
    finish = first.get("finishReason")
    content = first.get("content")
    parts = _list(cast(dict[str, Any], content).get("parts")) if isinstance(content, dict) else []
    return parts, finish if isinstance(finish, str) else None


def _answer_text(doc: Mapping[str, Any]) -> str:
    """Join the answer's text parts, skipping thought parts (``"thought": true``)."""
    parts, finish = _first_candidate_parts(doc)
    text = "".join(
        str(p["text"])
        for p in parts
        if isinstance(p.get("text"), str) and p.get("thought") is not True
    )
    if not text.strip():
        raise GeminiMalformedJsonError(f"answer has no text (finishReason={finish})")
    return text


def _parse_json_answer(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = _FENCE_RE.match(cleaned)
    if fenced:
        cleaned = fenced.group(1)
    try:
        value: object = json.loads(cleaned)
    except ValueError:
        raise GeminiMalformedJsonError(f"model answer is not JSON: {cleaned[:120]!r}") from None
    if not isinstance(value, dict):
        raise GeminiMalformedJsonError(
            f"model answer must be a JSON object, got {type(value).__name__}"
        )
    return cast(dict[str, Any], value)


def _audio_to_wav(doc: Mapping[str, Any]) -> bytes:
    parts, finish = _first_candidate_parts(doc)
    chunks: list[bytes] = []
    mime = ""
    for part in parts:
        inline = part.get("inlineData", part.get("inline_data"))
        if not isinstance(inline, dict):
            continue
        inline_doc = cast(dict[str, Any], inline)
        data = inline_doc.get("data")
        if not isinstance(data, str):
            continue
        try:
            chunks.append(base64.b64decode(data, validate=True))
        except ValueError:
            raise GeminiMalformedJsonError("audio part is not valid base64") from None
        part_mime = inline_doc.get("mimeType", inline_doc.get("mime_type"))
        if isinstance(part_mime, str) and not mime:
            mime = part_mime
    audio = b"".join(chunks)
    if not audio:
        raise GeminiMalformedJsonError(f"response has no audio (finishReason={finish})")
    if audio[:4] == b"RIFF":  # already a WAV file
        return audio
    if len(audio) % 2:
        raise GeminiMalformedJsonError("16-bit PCM audio has an odd byte count")
    rate_match = _RATE_RE.search(mime)
    rate = int(rate_match.group(1)) if rate_match else TTS_SAMPLE_RATE
    return pcm_to_wav(audio, rate)


def pcm_to_wav(pcm: bytes, sample_rate: int = TTS_SAMPLE_RATE) -> bytes:
    """Wrap 16-bit little-endian mono PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_duration_ms(wav_bytes: bytes) -> float:
    """Length of a WAV file in milliseconds."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return 1000.0 * w.getnframes() / w.getframerate()
