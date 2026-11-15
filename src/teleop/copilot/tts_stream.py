"""Tier 1 SSE-streamed Gemini TTS over a persistent IPv4 ``HTTPSConnection``.

Streams ``gemini-3.8-flash-lite-tts`` (voice ``Kore``) via
``POST /v1beta/models/{model}:streamGenerateContent?alt=sse`` over a persistent IPv4 TLS 1.3
connection (``socket.AF_INET``, ``TCP_NODELAY=1``, ``SO_KEEPALIVE=1``, SNI ``server_hostname``),
yielding raw 24 kHz 16-bit signed little-endian mono PCM chunks as SSE events arrive.
"""

from __future__ import annotations

import base64
import contextlib
import http.client
import json
import re
import ssl
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, cast
from urllib.parse import quote, urlsplit

from teleop.copilot.config import DEFAULT_TTS_MODEL, DEFAULT_VOICE
from teleop.copilot.gemini import (
    GEMINI_HOST,
    REDACTED,
    USER_AGENT,
    GeminiError,
    GeminiHttpError,
    GeminiMalformedJsonError,
    GeminiTimeoutError,
    IPv4HTTPSConnection,
    error_detail,
)

DEFAULT_PATH_PREFIX = "/v1beta"
DEFAULT_FIRST_CHUNK_TIMEOUT_S = 1.8
DEFAULT_TOTAL_TIMEOUT_S = 8.0

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class GeminiStreamingTTS:
    """Thread-safe SSE streaming client for ``gemini-3.8-flash-lite-tts`` (24 kHz s16le PCM)."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_TTS_MODEL,
        voice: str = DEFAULT_VOICE,
        host: str = GEMINI_HOST,
        port: int = 443,
        base_url: str | None = None,
        timeout_s: float = 15.0,
        first_chunk_timeout_s: float | None = None,
        connection_factory: Callable[[], http.client.HTTPSConnection] | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("GEMINI_API_KEY is missing; put it in .env (see docs/copilot.md)")
        self._api_key = api_key.strip()
        self.model = model
        self.voice = voice
        resolved_host = host
        resolved_port = port
        path_prefix = DEFAULT_PATH_PREFIX
        if base_url is not None and base_url.strip():
            parsed = urlsplit(base_url.strip())
            if parsed.hostname and host == GEMINI_HOST:
                resolved_host = parsed.hostname
            if parsed.port is not None and port == 443:
                resolved_port = int(parsed.port)
            if parsed.path and parsed.path != "/":
                path_prefix = "/" + parsed.path.strip("/")
        self.host = resolved_host
        self.port = resolved_port
        self._path_prefix = path_prefix
        self._timeout_s = timeout_s
        self._init_first_chunk_timeout = first_chunk_timeout_s
        self._connection_factory = connection_factory
        self._conn: http.client.HTTPSConnection | None = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"GeminiStreamingTTS(host={self.host!r}, port={self.port}, "
            f"model={self.model!r}, voice={self.voice!r})"
        )

    def _scrub(self, text: str) -> str:
        return text.replace(self._api_key, REDACTED)

    def _make_connection(self) -> http.client.HTTPSConnection:
        if self._connection_factory is not None:
            return self._connection_factory()
        return IPv4HTTPSConnection(host=self.host, port=self.port, timeout=self._timeout_s)

    def _ensure_conn(self) -> http.client.HTTPSConnection:
        if self._conn is None:
            self._conn = self._make_connection()
        return self._conn

    def _reset_conn(self) -> None:
        conn = self._conn
        self._conn = None
        if conn is not None:
            with contextlib.suppress(OSError, http.client.HTTPException):
                conn.close()

    def warm(self) -> bool:
        """Pre-establish the persistent IPv4 TLS connection without sending an HTTP request."""
        with self._lock:
            conn = self._ensure_conn()
            try:
                if getattr(conn, "sock", None) is None:
                    conn.connect()
                return True
            except OSError:
                self._reset_conn()
                return False

    def close(self) -> None:
        """Close the persistent TLS connection if open."""
        with self._lock:
            self._reset_conn()

    def _stream_path(self) -> str:
        model_id = self.model.removeprefix("models/")
        if not _MODEL_ID_RE.match(model_id):
            raise ValueError(f"invalid model id: {self.model!r}")
        return f"{self._path_prefix}/models/{quote(model_id)}:streamGenerateContent?alt=sse"

    def _open_response(
        self,
        path: str,
        body: bytes,
        headers: dict[str, str],
        first_timeout_s: float,
    ) -> tuple[http.client.HTTPSConnection, http.client.HTTPResponse]:
        for attempt in range(2):
            conn = self._ensure_conn()
            reused_existing_sock = getattr(conn, "sock", None) is not None
            conn.timeout = first_timeout_s
            if conn.sock is not None:
                conn.sock.settimeout(first_timeout_s)
            try:
                conn.request("POST", path, body=body, headers=headers)
                resp = conn.getresponse()
                return conn, resp
            except TimeoutError as exc:
                self._reset_conn()
                raise GeminiTimeoutError(
                    self._scrub(f"TTS first-chunk timeout exceeded ({first_timeout_s:g} s): {exc}")
                ) from None
            except (
                http.client.HTTPException,
                ssl.SSLError,
                BrokenPipeError,
                ConnectionResetError,
                OSError,
            ) as exc:
                self._reset_conn()
                if reused_existing_sock and attempt == 0:
                    continue
                raise GeminiError(self._scrub(f"TTS connection failed: {exc}")) from None
        raise GeminiError("TTS connection retry exhausted")

    def stream_pcm(
        self,
        text: str,
        *,
        voice: str | None = None,
        first_chunk_timeout_s: float = DEFAULT_FIRST_CHUNK_TIMEOUT_S,
        total_timeout_s: float = DEFAULT_TOTAL_TIMEOUT_S,
    ) -> Iterator[bytes]:
        """Stream raw 24 kHz 16-bit signed LE mono PCM chunks for ``text`` via SSE."""
        if not text or not text.strip():
            return
        effective_first_timeout = (
            self._init_first_chunk_timeout
            if (
                first_chunk_timeout_s == DEFAULT_FIRST_CHUNK_TIMEOUT_S
                and self._init_first_chunk_timeout is not None
            )
            else first_chunk_timeout_s
        )
        if effective_first_timeout <= 0 or total_timeout_s <= 0:
            raise ValueError("TTS timeouts must be positive")

        path = self._stream_path()
        chosen_voice = voice or self.voice
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": chosen_voice},
                    }
                },
            },
        }
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Connection": "keep-alive",
            "User-Agent": USER_AGENT,
        }

        with self._lock:
            t_start = time.monotonic()
            first_deadline = t_start + effective_first_timeout
            total_deadline = t_start + max(total_timeout_s, effective_first_timeout)

            conn, resp = self._open_response(path, body, headers, effective_first_timeout)
            server_wants_close = bool(
                resp.will_close or (resp.getheader("Connection") or "").lower() == "close"
            )
            if not 200 <= resp.status < 300:
                try:
                    err_raw = resp.read()
                except OSError:
                    err_raw = b""
                if server_wants_close:
                    self._reset_conn()
                raise GeminiHttpError(resp.status, self._scrub(error_detail(err_raw)))

            got_first_chunk = False
            stream_cleanly_drained = False
            try:
                while True:
                    now = time.monotonic()
                    deadline = total_deadline if got_first_chunk else first_deadline
                    remaining = deadline - now
                    if remaining <= 0:
                        self._reset_conn()
                        kind = "total" if got_first_chunk else "first-chunk"
                        raise GeminiTimeoutError(f"TTS {kind} timeout exceeded")
                    if conn.sock is not None:
                        conn.sock.settimeout(max(0.01, remaining))
                    try:
                        raw_line = resp.readline()
                    except TimeoutError:
                        self._reset_conn()
                        kind = "total" if got_first_chunk else "first-chunk"
                        raise GeminiTimeoutError(f"TTS {kind} timeout exceeded") from None
                    except (OSError, http.client.HTTPException) as exc:
                        self._reset_conn()
                        raise GeminiError(self._scrub(f"TTS stream read failed: {exc}")) from None

                    if raw_line == b"":
                        stream_cleanly_drained = True
                        break

                    line = raw_line.strip()
                    if not line or line.startswith(b":"):
                        continue
                    if not line.startswith(b"data:"):
                        continue

                    data_str = line[5:].strip().decode("utf-8", errors="replace")
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        ev_obj: object = json.loads(data_str)
                    except ValueError as exc:
                        self._reset_conn()
                        raise GeminiMalformedJsonError(
                            f"SSE event is not valid JSON: {data_str[:120]!r}"
                        ) from exc
                    if not isinstance(ev_obj, dict):
                        continue
                    for pcm_chunk in _extract_pcm_chunks(cast(dict[str, Any], ev_obj)):
                        if pcm_chunk:
                            got_first_chunk = True
                            yield pcm_chunk
            finally:
                if not stream_cleanly_drained or server_wants_close:
                    self._reset_conn()
                elif conn.sock is not None:
                    with contextlib.suppress(OSError):
                        conn.sock.settimeout(self._timeout_s)


def _extract_pcm_chunks(event_doc: dict[str, Any]) -> list[bytes]:
    candidates = event_doc.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return []
    first: object = cast(list[object], candidates)[0]
    if not isinstance(first, dict):
        return []
    content = cast(dict[str, Any], first).get("content")
    if not isinstance(content, dict):
        return []
    parts = cast(dict[str, Any], content).get("parts")
    if not isinstance(parts, list):
        return []
    out: list[bytes] = []
    for raw_part in cast(list[object], parts):
        if not isinstance(raw_part, dict):
            continue
        part = cast(dict[str, Any], raw_part)
        inline = part.get("inlineData", part.get("inline_data"))
        if not isinstance(inline, dict):
            continue
        b64_data = cast(dict[str, Any], inline).get("data")
        if not isinstance(b64_data, str) or not b64_data:
            continue
        try:
            pcm = base64.b64decode(b64_data, validate=True)
        except ValueError as exc:
            raise GeminiMalformedJsonError("TTS SSE audio chunk is not valid base64") from exc
        if len(pcm) % 2 != 0:
            raise GeminiMalformedJsonError("TTS 16-bit PCM chunk has odd byte count")
        if pcm:
            out.append(pcm)
    return out
