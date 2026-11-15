"""Shared opaque-box E2E fixtures, loopback wire servers, and dual-track contract helpers."""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import http.server
import importlib
import importlib.util
import inspect
import json
import math
import os
import shutil
import socket
import ssl
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from teleop.atc.coordinator import (
    ROUTINE,
    AtcCoordinator,
    CoordinatorConfig,
    OperatorInput,
)
from teleop.atc.stage import RESCUE_PRESS, RESCUE_WAYPOINTS
from teleop.atc.types import (
    BOLT_CLEARED_M,
    BayState,
    ConsoleInputState,
    SkillPrimitive,
)
from teleop.atc.validator import (
    CONTROL_DT,
    GATE_PASS_MIN,
    RolloutConditions,
    RolloutResult,
    ValidationReport,
)
from teleop.compliance.protocol import ComplianceEvent
from teleop.copilot.frames import downscale_jpeg
from teleop.reasoning.protocol import ReasoningContext, ReasoningVerdict
from teleop.sim.bays import PANDA_XML
from teleop.sim.hands import HandSpec, load_hand
from teleop.types import Pose
from teleop.unoq.client import UnoQClient
from teleop.unoq.dmp import quat_to_mat
from teleop.unoq.server import UnoQServer

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_FRAME_BIN = REPO_ROOT / "tests" / "copilot" / "fixtures" / "frame-0-1.bin"
FIXTURE_TELEMETRY_JSONL = REPO_ROOT / "tests" / "copilot" / "fixtures" / "stage_telemetry.jsonl"


def _resolve_main_env_file() -> Path:
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            common = Path(proc.stdout.strip())
            if common.name == ".git":
                return common.parent.resolve() / ".env"
    except (OSError, subprocess.SubprocessError):
        pass
    return REPO_ROOT / ".env"


MAIN_ENV_FILE = _resolve_main_env_file()


# ---------------------------------------------------------------------------
# Dual-track contract loaders (NEVER pytest.importorskip!)
# ---------------------------------------------------------------------------


def require_module(name: str) -> ModuleType:
    """Import a required M1-M4 module, raising AssertionError (never skipping!) if missing."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise AssertionError(f"Required M1-M4 module {name!r} is not implemented yet") from exc


def require_attr(target: ModuleType | object, attr: str) -> Any:
    """Fetch a required class/function/attribute, raising AssertionError if missing."""
    if not hasattr(target, attr):
        label = getattr(target, "__name__", type(target).__name__)
        raise AssertionError(f"Required M1-M4 symbol {attr!r} missing on {label}")
    return getattr(target, attr)


def construct_flexible(cls: Any, /, **candidates: Any) -> Any:
    """Instantiate ``cls`` matching accepted parameter names while preserving public contracts."""
    sig = inspect.signature(cls)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs: dict[str, Any] = {}
    pos_args: list[Any] = []
    for name, param in params.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            if name in candidates:
                if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                    pos_args.append(candidates[name])
                else:
                    kwargs[name] = candidates[name]
        elif param.kind == inspect.Parameter.KEYWORD_ONLY and name in candidates:
            kwargs[name] = candidates[name]
    if accepts_var_kw:
        for k, v in candidates.items():
            if k not in kwargs:
                kwargs[k] = v
    return cls(*pos_args, **kwargs)


def make_tts_streamer(
    *,
    api_key: str = "test-e2e-key",
    host: str = "generativelanguage.googleapis.com",
    port: int = 443,
    base_url: str | None = None,
    tts_model: str = "gemini-3.8-flash-lite-tts",
    voice: str = "Kore",
    timeout_s: float = 15.0,
    first_chunk_timeout_s: float = 1.8,
) -> Any:
    """Construct ``GeminiStreamingTTS`` resiliently across minor constructor kwarg variations."""
    tts_mod = require_module("teleop.copilot.tts_stream")
    cls = require_attr(tts_mod, "GeminiStreamingTTS")
    conn_cls = require_attr(tts_mod, "IPv4HTTPSConnection")
    sig = inspect.signature(cls)
    params = sig.parameters
    resolved_base = base_url or f"https://{host}:{port}/v1beta"

    candidates: dict[str, Any] = {
        "api_key": api_key,
        "key": api_key,
        "host": host,
        "port": port,
        "base_url": resolved_base,
        "tts_model": tts_model,
        "model": tts_model,
        "voice": voice,
        "timeout": timeout_s,
        "timeout_s": timeout_s,
        "first_chunk_timeout_s": first_chunk_timeout_s,
    }
    if "conn_factory" in params:
        candidates["conn_factory"] = lambda: conn_cls(host=host, port=port, timeout=timeout_s)
    if "connection_factory" in params:
        candidates["connection_factory"] = lambda: conn_cls(host=host, port=port, timeout=timeout_s)
    if "connection" in params:
        candidates["connection"] = conn_cls(host=host, port=port, timeout=timeout_s)
    if "cfg" in params or "config" in params:
        from teleop.copilot.config import CopilotConfig

        cfg = CopilotConfig(
            gemini_api_key=api_key,
            tts_model=tts_model,
            voice=voice,
        )
        candidates["cfg"] = cfg
        candidates["config"] = cfg
    return construct_flexible(cls, **candidates)


def call_stream_pcm(
    streamer: Any,
    text: str,
    *,
    voice: str = "Kore",
    first_chunk_timeout_s: float | None = None,
    total_timeout_s: float | None = None,
) -> list[bytes]:
    """Call ``streamer.stream_pcm(text, ...)`` and collect yielded PCM byte chunks."""
    fn = require_attr(streamer, "stream_pcm")
    sig = inspect.signature(fn)
    kwargs: dict[str, Any] = {}
    if "voice" in sig.parameters:
        kwargs["voice"] = voice
    if first_chunk_timeout_s is not None:
        if "first_chunk_timeout_s" in sig.parameters:
            kwargs["first_chunk_timeout_s"] = first_chunk_timeout_s
        elif "deadline_s" in sig.parameters:
            kwargs["deadline_s"] = first_chunk_timeout_s
    if total_timeout_s is not None:
        if "total_timeout_s" in sig.parameters:
            kwargs["total_timeout_s"] = total_timeout_s
        elif "timeout_s" in sig.parameters:
            kwargs["timeout_s"] = total_timeout_s
    return list(fn(text, **kwargs))


# ---------------------------------------------------------------------------
# Non-loopback network egress guard (autouse unless RUN_LIVE_GEMINI=1)
# ---------------------------------------------------------------------------

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "0.0.0.0", "::", ""})


def _is_loopback_target(address: Any) -> bool:
    if isinstance(address, str | bytes):
        return True  # AF_UNIX path
    if isinstance(address, tuple) and len(address) >= 1:
        host = str(address[0]).strip().lower()
        if host in _LOOPBACK_HOSTS or host.startswith("127."):
            return True
    return False


@pytest.fixture(autouse=True)
def block_external_network_egress(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> Iterator[list[Any]]:
    """Block every non-loopback socket connection unless RUN_LIVE_GEMINI=1 on live tests."""
    live_enabled = os.environ.get("RUN_LIVE_GEMINI") == "1" and (
        "test_live_gemini_showcase" in request.node.nodeid
    )
    attempts: list[Any] = []
    if live_enabled:
        yield attempts
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sendto = socket.socket.sendto

    def guarded_connect(sock: socket.socket, address: Any) -> Any:
        attempts.append(("connect", sock.family, address))
        if sock.family == socket.AF_UNIX or _is_loopback_target(address):
            return real_connect(sock, address)
        raise AssertionError(
            f"External network egress blocked in offline E2E mode: connect({address!r})"
        )

    def guarded_connect_ex(sock: socket.socket, address: Any) -> int:
        attempts.append(("connect_ex", sock.family, address))
        if sock.family == socket.AF_UNIX or _is_loopback_target(address):
            return real_connect_ex(sock, address)
        raise AssertionError(
            f"External network egress blocked in offline E2E mode: connect_ex({address!r})"
        )

    def guarded_sendto(sock: socket.socket, data: Any, *args: Any) -> int:
        address = args[-1] if args else None
        attempts.append(("sendto", sock.family, address))
        if address is None or sock.family == socket.AF_UNIX or _is_loopback_target(address):
            return real_sendto(sock, data, *args)
        raise AssertionError(
            f"External network egress blocked in offline E2E mode: sendto({address!r})"
        )

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket.socket, "sendto", guarded_sendto)
    yield attempts


# ---------------------------------------------------------------------------
# Synthetic <=512px rescue JPEGs & PCM generators
# ---------------------------------------------------------------------------


def make_sine_pcm_s16le(
    duration_ms: float = 120.0,
    *,
    sample_rate: int = 24_000,
    freq_hz: float = 440.0,
) -> bytes:
    """Generate deterministic 24 kHz 16-bit signed LE mono PCM audio."""
    n_samples = max(int(round(sample_rate * (duration_ms / 1000.0))), 2)
    buf = bytearray(n_samples * 2)
    for i in range(n_samples):
        val = int(12_000 * math.sin(2.0 * math.pi * freq_hz * (i / sample_rate)))
        struct.pack_into("<h", buf, i * 2, val)
    return bytes(buf)


def load_rescue_jpeg(max_side_px: int = 512) -> bytes:
    """Load real stage camera JPEG from fixtures and downscale to <= max_side_px."""
    raw = FIXTURE_FRAME_BIN.read_bytes()[2:]
    return downscale_jpeg(raw, max_side_px=max_side_px, quality=85)


def make_rescue_keyframes(count: int = 3) -> list[tuple[str, float, float, float, bytes]]:
    """Build (label, sim_t, bolt_x_m, tau_hand_nm, jpeg_bytes) tuples with <=512px JPEGs."""
    jpeg = load_rescue_jpeg(max_side_px=512)
    labels_3 = [
        ("stall", 2.10, 0.014, 0.41, jpeg),
        ("mid-hold", 3.45, 0.036, 0.38, jpeg),
        ("bolt-clear", 4.80, 0.056, 0.29, jpeg),
    ]
    if count <= 3:
        return labels_3[:count]
    extra = [
        ("engage", 2.40, 0.015, 0.22, jpeg),
        ("flap-press", 2.95, 0.022, 0.44, jpeg),
        ("post-clear", 5.05, 0.058, 0.19, jpeg),
    ]
    combined = [labels_3[0], extra[0], extra[1], labels_3[1], labels_3[2], extra[2]]
    return combined[:count]


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Parse (width, height) from a baseline/progressive JPEG byte stream."""
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        raise ValueError("Not a valid JPEG header")
    idx = 2
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while idx + 4 <= len(data):
        if data[idx] != 0xFF:
            idx += 1
            continue
        while idx < len(data) and data[idx] == 0xFF:
            idx += 1
        if idx >= len(data):
            break
        marker = data[idx]
        idx += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if idx + 2 > len(data):
            break
        seg_len = int.from_bytes(data[idx : idx + 2], "big")
        if seg_len < 2 or idx + seg_len > len(data):
            break
        if marker in sof_markers and seg_len >= 7:
            height = int.from_bytes(data[idx + 3 : idx + 5], "big")
            width = int.from_bytes(data[idx + 5 : idx + 7], "big")
            return width, height
        idx += seg_len
    raise ValueError("No SOF marker found in JPEG")


def contains_key_recursive(obj: Any, target_key: str) -> bool:
    """Return True if ``target_key`` appears anywhere inside nested dicts/lists."""
    if isinstance(obj, Mapping):
        if target_key in obj:
            return True
        return any(contains_key_recursive(v, target_key) for v in obj.values())
    if isinstance(obj, list | tuple):
        return any(contains_key_recursive(v, target_key) for v in obj)
    return False


# ---------------------------------------------------------------------------
# Self-signed loopback IPv4 TLS HTTP/1.1 Keep-Alive Gemini Mock Server
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordedHttpRequest:
    conn_id: int
    request_seq_on_conn: int
    method: str
    path: str
    headers: dict[str, str]
    raw_body: bytes
    json_body: dict[str, Any] | None
    sni_hostname: str | None
    peer_ip: str


@dataclass
class MockGeminiConfig:
    """Mutable behavior controls for the loopback TLS Gemini mock server."""

    tts_chunks: list[bytes] = field(
        default_factory=lambda: [
            make_sine_pcm_s16le(40.0, freq_hz=440.0),
            make_sine_pcm_s16le(40.0, freq_hz=554.0),
            make_sine_pcm_s16le(40.0, freq_hz=659.0),
        ]
    )
    tts_first_chunk_delay_s: float = 0.0
    tts_inter_chunk_delay_s: float = 0.0
    tts_hold_last_chunk: threading.Event | None = None
    tts_status_code: int = 200
    tts_error_body: dict[str, Any] = field(
        default_factory=lambda: {"error": {"code": 500, "message": "synthetic tts failure"}}
    )
    tts_close_after_first_request: bool = False
    tts_omit_stop_reason: bool = False
    tts_received_event: threading.Event = field(default_factory=threading.Event)

    hydrogen_verdict: dict[str, Any] = field(
        default_factory=lambda: {
            "verdict": "approve",
            "say": "Flap depressed cleanly and bolt cleared safely.",
            "reason": "Thumb held spring flap while index hooked bolt clear of -X HV bus.",
            "flap_depressed": True,
            "bolt_clearance_ok": True,
            "hv_clearance_ok": True,
            "torque_controlled": True,
            "confidence": 0.94,
        }
    )
    hydrogen_status_code: int = 200
    hydrogen_malformed_text: str | None = None
    hydrogen_include_thought_signature: bool = True
    hydrogen_barrier: threading.Barrier | None = None
    hydrogen_hold_event: threading.Event | None = None
    hydrogen_started_event: threading.Event = field(default_factory=threading.Event)

    argon_report: dict[str, Any] = field(
        default_factory=lambda: {
            "verdict": "approve",
            "dataset_tag": "approve",
            "confidence": 0.92,
            "reasons": [
                "Spring flap fully depressed prior to +X slide",
                "Safe clearance from -X HV terminals maintained throughout",
            ],
            "operator_note": "Clean recovery trajectory suitable for ACT training.",
        }
    )
    argon_status_code: int = 200
    argon_malformed_text: str | None = None
    argon_hold_event: threading.Event | None = None
    argon_started_event: threading.Event = field(default_factory=threading.Event)

    close_stale_once: bool = False
    custom_responder: Callable[[RecordedHttpRequest], tuple[int, dict[str, Any]] | None] | None = (
        None
    )


class LoopbackGeminiTLSServer:
    """HTTP/1.1 Keep-Alive TLS server emulating Gemini TTS SSE, Hydrogen, and Argon."""

    def __init__(self, cert_pem: Path, key_pem: Path) -> None:
        self.cert_pem = cert_pem
        self.key_pem = key_pem
        self.config = MockGeminiConfig()
        self._lock = threading.Lock()
        self.requests: list[RecordedHttpRequest] = []
        self.connection_ids: list[int] = []
        self._next_conn_id = 0
        self._stale_triggered = False
        self._sni_by_ssl_id: dict[int, str | None] = {}

        server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_ctx.load_cert_chain(certfile=str(cert_pem), keyfile=str(key_pem))

        def _sni_cb(ssl_sock: ssl.SSLObject, server_name: str, _ctx: ssl.SSLContext) -> int | None:
            with self._lock:
                self._sni_by_ssl_id[id(ssl_sock)] = server_name
            return None

        server_ctx.sni_callback = _sni_cb
        self._server_ctx = server_ctx

        outer = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def handle(self) -> None:
                with outer._lock:
                    outer._next_conn_id += 1
                    self._conn_id = outer._next_conn_id
                    outer.connection_ids.append(self._conn_id)
                self._req_seq = 0
                super().handle()

            def _record(self, method: str) -> RecordedHttpRequest:
                self._req_seq += 1
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length > 0 else b""
                parsed: dict[str, Any] | None = None
                if raw:
                    with contextlib.suppress(ValueError, UnicodeDecodeError):
                        loaded = json.loads(raw.decode("utf-8"))
                        if isinstance(loaded, dict):
                            parsed = loaded
                with outer._lock:
                    sni = outer._sni_by_ssl_id.get(id(self.connection))
                hdrs = {k.lower(): v for k, v in self.headers.items()}
                rec = RecordedHttpRequest(
                    conn_id=self._conn_id,
                    request_seq_on_conn=self._req_seq,
                    method=method,
                    path=self.path,
                    headers=hdrs,
                    raw_body=raw,
                    json_body=parsed,
                    sni_hostname=sni,
                    peer_ip=str(self.client_address[0]),
                )
                with outer._lock:
                    outer.requests.append(rec)
                return rec

            def _send_json(
                self, status: int, payload: Mapping[str, Any], *, close: bool = False
            ) -> None:
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close" if close else "keep-alive")
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
                if close:
                    self.close_connection = True

            def _write_chunk(self, data: bytes) -> None:
                header = f"{len(data):x}\r\n".encode("ascii")
                self.wfile.write(header + data + b"\r\n")
                self.wfile.flush()

            def do_GET(self) -> None:
                rec = self._record("GET")
                if "/models" in rec.path:
                    self._send_json(
                        200,
                        {
                            "models": [
                                {
                                    "name": "models/gemini-3.8-flash-lite-tts",
                                    "supportedGenerationMethods": [
                                        "generateContent",
                                        "streamGenerateContent",
                                    ],
                                },
                                {
                                    "name": "models/gemini-4-hydrogen",
                                    "supportedGenerationMethods": ["generateContent"],
                                },
                                {
                                    "name": "models/gemini-4-argon",
                                    "supportedGenerationMethods": ["generateContent"],
                                },
                            ]
                        },
                    )
                    return
                self._send_json(404, {"error": {"code": 404, "message": "not found"}})

            def do_POST(self) -> None:
                with outer._lock:
                    if outer.config.close_stale_once and not outer._stale_triggered:
                        outer._stale_triggered = True
                        self.close_connection = True
                        with contextlib.suppress(OSError):
                            self.connection.close()
                        return

                rec = self._record("POST")
                cfg = outer.config
                if cfg.custom_responder is not None:
                    custom = cfg.custom_responder(rec)
                    if custom is not None:
                        status, payload = custom
                        self._send_json(status, payload)
                        return

                # 1. SSE Streaming TTS endpoint
                if ":streamGenerateContent" in rec.path:
                    cfg.tts_received_event.set()
                    if cfg.tts_status_code != 200:
                        self._send_json(cfg.tts_status_code, cfg.tts_error_body)
                        return
                    if cfg.tts_first_chunk_delay_s > 0:
                        time.sleep(cfg.tts_first_chunk_delay_s)
                    try:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Transfer-Encoding", "chunked")
                        close_after = cfg.tts_close_after_first_request and self._req_seq == 1
                        self.send_header("Connection", "close" if close_after else "keep-alive")
                        self.end_headers()
                        chunks = list(cfg.tts_chunks)
                        for idx, pcm in enumerate(chunks):
                            if idx == len(chunks) - 1 and cfg.tts_hold_last_chunk is not None:
                                cfg.tts_hold_last_chunk.wait(timeout=5.0)
                            if idx > 0 and cfg.tts_inter_chunk_delay_s > 0:
                                time.sleep(cfg.tts_inter_chunk_delay_s)
                            b64 = base64.b64encode(pcm).decode("ascii")
                            ev = {
                                "candidates": [
                                    {
                                        "content": {
                                            "parts": [
                                                {
                                                    "inlineData": {
                                                        "mimeType": (
                                                            "audio/l16; rate=24000; channels=1"
                                                        ),
                                                        "data": b64,
                                                    }
                                                }
                                            ],
                                            "role": "model",
                                        },
                                        "index": 0,
                                    }
                                ]
                            }
                            frame = f"data: {json.dumps(ev, separators=(',', ':'))}\r\n\r\n"
                            self._write_chunk(frame.encode("utf-8"))
                        if not cfg.tts_omit_stop_reason:
                            stop_ev = {
                                "candidates": [
                                    {
                                        "content": {"role": "model", "parts": []},
                                        "finishReason": "STOP",
                                        "index": 0,
                                    }
                                ]
                            }
                            stop_frame = (
                                f"data: {json.dumps(stop_ev, separators=(',', ':'))}\r\n\r\n"
                            )
                            self._write_chunk(stop_frame.encode("utf-8"))
                        # Terminating zero-length HTTP/1.1 chunk
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                        if close_after:
                            self.close_connection = True
                    except (BrokenPipeError, ConnectionResetError, ssl.SSLError, OSError):
                        self.close_connection = True
                    return

                # 2. Unary TTS synthesize fallback endpoint (if used for precache or legacy tests)
                if "tts:generateContent" in rec.path:
                    cfg.tts_received_event.set()
                    if cfg.tts_status_code != 200:
                        self._send_json(cfg.tts_status_code, cfg.tts_error_body)
                        return
                    pcm_all = b"".join(cfg.tts_chunks)
                    b64 = base64.b64encode(pcm_all).decode("ascii")
                    self._send_json(
                        200,
                        {
                            "candidates": [
                                {
                                    "content": {
                                        "role": "model",
                                        "parts": [
                                            {
                                                "inlineData": {
                                                    "mimeType": "audio/l16; rate=24000; channels=1",
                                                    "data": b64,
                                                }
                                            }
                                        ],
                                    },
                                    "finishReason": "STOP",
                                }
                            ]
                        },
                    )
                    return

                # 3. Tier 2 Hydrogen Fast Grasp Gate
                if "hydrogen" in rec.path or (
                    ":generateContent" in rec.path and "argon" not in rec.path
                ):
                    cfg.hydrogen_started_event.set()
                    if cfg.hydrogen_barrier is not None:
                        cfg.hydrogen_barrier.wait()
                    if cfg.hydrogen_hold_event is not None:
                        cfg.hydrogen_hold_event.wait(timeout=5.0)
                    if cfg.hydrogen_status_code != 200:
                        self._send_json(
                            cfg.hydrogen_status_code,
                            {"error": {"code": cfg.hydrogen_status_code, "message": "gate error"}},
                        )
                        return
                    text = (
                        cfg.hydrogen_malformed_text
                        if cfg.hydrogen_malformed_text is not None
                        else json.dumps(cfg.hydrogen_verdict)
                    )
                    parts: list[dict[str, Any]] = []
                    if cfg.hydrogen_include_thought_signature:
                        parts.append(
                            {
                                "thought": True,
                                "text": "Checking flap depression and HV clearance.",
                                "thoughtSignature": "opaque-hydrogen-sig-001",
                            }
                        )
                    answer_part: dict[str, Any] = {"text": text}
                    if cfg.hydrogen_include_thought_signature:
                        answer_part["thoughtSignature"] = "opaque-hydrogen-sig-002"
                    parts.append(answer_part)
                    self._send_json(
                        200,
                        {
                            "candidates": [
                                {
                                    "content": {"role": "model", "parts": parts},
                                    "finishReason": "STOP",
                                }
                            ]
                        },
                    )
                    return

                # 4. Tier 3 Argon Silent Deep Audit
                if "argon" in rec.path:
                    cfg.argon_started_event.set()
                    if cfg.argon_hold_event is not None:
                        cfg.argon_hold_event.wait(timeout=5.0)
                    if cfg.argon_status_code != 200:
                        self._send_json(
                            cfg.argon_status_code,
                            {"error": {"code": cfg.argon_status_code, "message": "argon error"}},
                        )
                        return
                    text = (
                        cfg.argon_malformed_text
                        if cfg.argon_malformed_text is not None
                        else json.dumps(cfg.argon_report)
                    )
                    self._send_json(
                        200,
                        {
                            "candidates": [
                                {
                                    "content": {
                                        "role": "model",
                                        "parts": [
                                            {
                                                "text": text,
                                                "thoughtSignature": "opaque-argon-sig-001",
                                            }
                                        ],
                                    },
                                    "finishReason": "STOP",
                                }
                            ]
                        },
                    )
                    return

                self._send_json(404, {"error": {"code": 404, "message": f"unhandled {rec.path}"}})

        class _ReusableThreadingTLSServer(http.server.ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._httpd = _ReusableThreadingTLSServer(("127.0.0.1", 0), _Handler)
        self.port = int(self._httpd.server_address[1])
        self._httpd.socket = self._server_ctx.wrap_socket(self._httpd.socket, server_side=True)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="mock-gemini-tls",
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        return f"https://generativelanguage.googleapis.com:{self.port}/v1beta"

    @property
    def connection_count(self) -> int:
        with self._lock:
            return len(self.connection_ids)

    def requests_matching(self, substring: str) -> list[RecordedHttpRequest]:
        with self._lock:
            return [r for r in self.requests if substring in r.path]

    def reset(self) -> None:
        with self._lock:
            self.requests.clear()
            self.connection_ids.clear()
            self._stale_triggered = False
        self.config = MockGeminiConfig()

    def start(self) -> LoopbackGeminiTLSServer:
        self._thread.start()
        return self

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._httpd.shutdown()
            self._httpd.server_close()
        self._thread.join(timeout=3.0)


@pytest.fixture(scope="session")
def loopback_tls_cert_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Generate a 1-day self-signed SAN certificate for loopback TLS tests."""
    cert_dir = tmp_path_factory.mktemp("loopback_tls")
    cert_pem = cert_dir / "cert.pem"
    key_pem = cert_dir / "key.pem"
    subprocess.run(
        [
            "/usr/bin/openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key_pem),
            "-out",
            str(cert_pem),
            "-days",
            "1",
            "-subj",
            "/CN=generativelanguage.googleapis.com",
            "-addext",
            "subjectAltName=DNS:generativelanguage.googleapis.com,DNS:localhost,IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
        timeout=15,
    )
    return cert_pem, key_pem


@pytest.fixture
def mock_gemini_server(
    loopback_tls_cert_pair: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[LoopbackGeminiTLSServer]:
    """Start loopback IPv4 TLS Gemini server and route generativelanguage.googleapis.com to it."""
    cert_pem, key_pem = loopback_tls_cert_pair
    srv = LoopbackGeminiTLSServer(cert_pem, key_pem).start()

    real_create_default_context = ssl.create_default_context
    real_getaddrinfo = socket.getaddrinfo

    def trusted_default_context(
        purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH,
        *,
        cafile: str | None = None,
        capath: str | None = None,
        cadata: str | bytes | None = None,
    ) -> ssl.SSLContext:
        if cafile is None and capath is None and cadata is None:
            return real_create_default_context(purpose, cafile=str(cert_pem))
        return real_create_default_context(purpose, cafile=cafile, capath=capath, cadata=cadata)

    def routed_getaddrinfo(
        host: Any,
        port: Any,
        family: int = 0,
        type: int = 0,  # noqa: A002
        proto: int = 0,
        flags: int = 0,
    ) -> list[Any]:
        host_str = host.decode("ascii", "ignore") if isinstance(host, bytes) else str(host or "")
        if host_str.lower() == "generativelanguage.googleapis.com":
            target_port = srv.port if int(port or 443) == 443 else int(port)
            # Include a synthetic IPv6 entry when family is AF_UNSPEC (0) so tests verify
            # IPv4HTTPSConnection explicitly requests/filters socket.AF_INET!
            entries: list[Any] = [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("127.0.0.1", target_port),
                )
            ]
            if family == 0:
                entries.insert(
                    0,
                    (
                        socket.AF_INET6,
                        socket.SOCK_STREAM,
                        socket.IPPROTO_TCP,
                        "",
                        ("::1", target_port, 0, 0),
                    ),
                )
            elif family == socket.AF_INET6:
                return [
                    (
                        socket.AF_INET6,
                        socket.SOCK_STREAM,
                        socket.IPPROTO_TCP,
                        "",
                        ("::1", target_port, 0, 0),
                    )
                ]
            return entries
        return real_getaddrinfo(host, port, family, type, proto, flags)

    monkeypatch.setattr(ssl, "create_default_context", trusted_default_context)
    monkeypatch.setattr(socket, "getaddrinfo", routed_getaddrinfo)
    try:
        yield srv
    finally:
        srv.close()


# ---------------------------------------------------------------------------
# Plain HTTP Loopback OpenAI-Compatible Local Gemma Server (/v1/chat/completions)
# ---------------------------------------------------------------------------


@dataclass
class LocalGemmaServer:
    """Loopback HTTP/1.1 OpenAI-compatible server for ``LocalModelClient`` (`gemma4:e4b`)."""

    response_json: dict[str, Any] = field(
        default_factory=lambda: {
            "verdict": "approve",
            "say": "Local Gemma verified latch clearance.",
            "reason": "Flap depressed and bolt cleared safely.",
            "flap_depressed": True,
            "bolt_clearance_ok": True,
            "hv_clearance_ok": True,
            "torque_controlled": True,
            "confidence": 0.89,
        }
    )
    status_code: int = 200
    raw_content_override: str | None = None
    delay_s: float = 0.0
    requests: list[RecordedHttpRequest] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        outer = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length > 0 else b""
                parsed: dict[str, Any] | None = None
                with contextlib.suppress(ValueError, UnicodeDecodeError):
                    obj = json.loads(raw.decode("utf-8"))
                    if isinstance(obj, dict):
                        parsed = obj
                rec = RecordedHttpRequest(
                    conn_id=1,
                    request_seq_on_conn=1,
                    method="POST",
                    path=self.path,
                    headers={k.lower(): v for k, v in self.headers.items()},
                    raw_body=raw,
                    json_body=parsed,
                    sni_hostname=None,
                    peer_ip=str(self.client_address[0]),
                )
                with outer._lock:
                    outer.requests.append(rec)
                if outer.delay_s > 0:
                    time.sleep(outer.delay_s)
                if outer.status_code != 200:
                    err = json.dumps({"error": {"message": "local model error"}}).encode("utf-8")
                    self.send_response(outer.status_code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                    return
                content = (
                    outer.raw_content_override
                    if outer.raw_content_override is not None
                    else json.dumps(outer.response_json)
                )
                payload = json.dumps(
                    {
                        "id": "chatcmpl-local-001",
                        "object": "chat.completion",
                        "model": "gemma4:e4b",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        class _Server(http.server.ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._httpd = _Server(("127.0.0.1", 0), _Handler)
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="mock-local-gemma",
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self) -> LocalGemmaServer:
        self._thread.start()
        return self

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._httpd.shutdown()
            self._httpd.server_close()
        self._thread.join(timeout=3.0)


@pytest.fixture
def local_gemma_server() -> Iterator[LocalGemmaServer]:
    srv = LocalGemmaServer().start()
    try:
        yield srv
    finally:
        srv.close()


# ---------------------------------------------------------------------------
# Shadowed $PATH + shutil.which Fake Audio CLI Binaries
# ---------------------------------------------------------------------------

_FAKE_AUDIO_SCRIPT = """#!{python}
import json
import os
import sys
import time
from pathlib import Path

tool = {tool_name!r}
argv = sys.argv[1:]
log_path = Path(os.environ["FAKE_AUDIO_LOG"])
mode = os.environ.get(f"FAKE_{{tool.upper().replace('-', '_')}}_MODE", "ok")

t_start_ns = time.monotonic_ns()
t_first_byte_ns = None
stdin_total = 0
wav_file_bytes = None

if mode == "fail_early":
    rec = {{
        "tool": tool,
        "argv": argv,
        "mode": mode,
        "t_start_ns": t_start_ns,
        "t_first_byte_ns": None,
        "stdin_bytes": 0,
        "wav_file_bytes": None,
    }}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\\n")
    sys.exit(1)

reads_stdin = (
    (tool == "paplay" and "--raw" in argv)
    or (tool == "aplay" and ("-" in argv or "-t" in argv))
)

if reads_stdin:
    while True:
        chunk = sys.stdin.buffer.read(512)
        if not chunk:
            break
        if t_first_byte_ns is None:
            t_first_byte_ns = time.monotonic_ns()
        stdin_total += len(chunk)
        if mode == "broken_pipe_midstream" and stdin_total >= 1024:
            rec = {{
                "tool": tool,
                "argv": argv,
                "mode": mode,
                "t_start_ns": t_start_ns,
                "t_first_byte_ns": t_first_byte_ns,
                "stdin_bytes": stdin_total,
                "wav_file_bytes": None,
            }}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\\n")
            sys.exit(1)
else:
    for arg in reversed(argv):
        if not arg.startswith("-") and os.path.isfile(arg):
            data = Path(arg).read_bytes()
            wav_file_bytes = len(data)
            wav_riff = data[:4].decode("latin1", "ignore")
            break
    else:
        wav_riff = None

rec = {{
    "tool": tool,
    "argv": argv,
    "mode": mode,
    "t_start_ns": t_start_ns,
    "t_first_byte_ns": t_first_byte_ns,
    "stdin_bytes": stdin_total,
    "wav_file_bytes": wav_file_bytes,
}}
with log_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(rec) + "\\n")
sys.exit(0)
"""


@dataclass
class FakeAudioHarness:
    bin_dir: Path
    log_path: Path

    def calls(self, tool: str | None = None) -> list[dict[str, Any]]:
        if not self.log_path.is_file():
            return []
        rows = [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if tool is not None:
            return [r for r in rows if r.get("tool") == tool]
        return rows

    def remove_tool(self, name: str) -> None:
        target = self.bin_dir / name
        if target.exists():
            target.unlink()

    def clear_log(self) -> None:
        if self.log_path.exists():
            self.log_path.unlink()


@pytest.fixture
def fake_audio_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> FakeAudioHarness:
    """Shadow host audio players (`paplay`, `aplay`, `espeak-ng`, `espeak`, `spd-say`, `afplay`)."""
    bin_dir = tmp_path / "fake_audio_bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    log_path = tmp_path / "audio_calls.jsonl"

    tools = ("paplay", "aplay", "espeak-ng", "espeak", "spd-say", "afplay")
    for tool in tools:
        script_path = bin_dir / tool
        script_path.write_text(
            _FAKE_AUDIO_SCRIPT.format(python=sys.executable, tool_name=tool),
            encoding="utf-8",
        )
        script_path.chmod(0o755)

    audio_names = frozenset(tools)
    real_which = shutil.which

    def shadowed_which(cmd: str, mode: int = os.F_OK | os.X_OK, path: str | None = None) -> Any:
        base = os.path.basename(cmd)
        if base in audio_names:
            candidate = bin_dir / base
            return str(candidate) if candidate.is_file() and os.access(candidate, mode) else None
        return real_which(cmd, mode=mode, path=path)

    monkeypatch.setenv("FAKE_AUDIO_LOG", str(log_path))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setattr(shutil, "which", shadowed_which)
    return FakeAudioHarness(bin_dir=bin_dir, log_path=log_path)


# ---------------------------------------------------------------------------
# Disk-backed StubLeRobotDataset fallback (0 skips when lerobot is absent)
# ---------------------------------------------------------------------------


class _StubMeta:
    def __init__(
        self,
        root: Path,
        fps: int,
        features: dict[str, Any],
        robot_type: str,
        episodes: list[dict[str, int]],
    ) -> None:
        self.root = root
        self.fps = fps
        self.features = features
        self.robot_type = robot_type
        self.episodes = episodes

    @property
    def total_episodes(self) -> int:
        return len(self.episodes)


class _StubHfSlice:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __getitem__(self, key: int | slice | str) -> Any:
        if isinstance(key, slice):
            return _StubHfSlice(self._rows[key])
        if isinstance(key, int):
            return self._rows[key]
        return [r[key] for r in self._rows]


class StubLeRobotDataset:
    """Minimal disk-backed v3 LeRobotDataset stand-in when optional ``lerobot`` is not installed."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        *,
        _create: bool = False,
        _fps: int = 30,
        _features: dict[str, Any] | None = None,
        _robot_type: str = "unknown",
    ) -> None:
        self.repo_id = repo_id
        self.root = Path(root)
        self._buffer: list[dict[str, Any]] = []
        self._frames: list[dict[str, Any]] = []
        if _create:
            self.fps = int(_fps)
            self.meta = _StubMeta(
                self.root,
                self.fps,
                dict(_features or {}),
                str(_robot_type),
                [],
            )
            self._persist()
        else:
            info = json.loads((self.root / "meta" / "info.json").read_text(encoding="utf-8"))
            self.fps = int(info["fps"])
            self.meta = _StubMeta(
                self.root,
                self.fps,
                dict(info["features"]),
                str(info.get("robot_type", "unknown")),
                list(info.get("episodes", [])),
            )
            npz_path = self.root / "data" / "stub_frames.npz"
            if npz_path.is_file():
                with np.load(npz_path, allow_pickle=False) as archive:
                    tasks = json.loads(
                        (self.root / "meta" / "stub_tasks.json").read_text(encoding="utf-8")
                    )
                    states = archive["state"]
                    actions = archive["action"]
                    cam_keys = [k for k in archive.files if k.startswith("img__")]
                    for i in range(len(tasks)):
                        row: dict[str, Any] = {
                            "observation.state": states[i],
                            "action": actions[i],
                            "task": tasks[i],
                        }
                        for ck in cam_keys:
                            cam_name = ck[len("img__") :]
                            row[f"observation.images.{cam_name}"] = archive[ck][i]
                        self._frames.append(row)

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict[str, Any],
        *,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        **_kwargs: Any,
    ) -> StubLeRobotDataset:
        del use_videos
        assert root is not None
        return cls(
            repo_id,
            Path(root),
            _create=True,
            _fps=fps,
            _features=features,
            _robot_type=robot_type or "unknown",
        )

    @property
    def num_episodes(self) -> int:
        return self.meta.total_episodes

    @property
    def hf_dataset(self) -> _StubHfSlice:
        return _StubHfSlice(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raw = self._frames[idx]
        out: dict[str, Any] = {
            "observation.state": np.asarray(raw["observation.state"], dtype=np.float32),
            "action": np.asarray(raw["action"], dtype=np.float32),
            "task": str(raw.get("task", "")),
        }
        for k, v in raw.items():
            if k.startswith("observation.images."):
                arr = np.asarray(v, dtype=np.float32)
                if arr.ndim == 3 and arr.shape[-1] == 3:
                    arr = arr.transpose(2, 0, 1) / 255.0
                out[k] = arr
        return out

    def add_frame(self, frame: Mapping[str, Any]) -> None:
        copied: dict[str, Any] = {
            "observation.state": np.asarray(frame["observation.state"], dtype=np.float32).copy(),
            "action": np.asarray(frame["action"], dtype=np.float32).copy(),
            "task": str(frame.get("task", "")),
        }
        for k, v in frame.items():
            if k.startswith("observation.images."):
                copied[k] = np.asarray(v, dtype=np.uint8).copy()
        self._buffer.append(copied)

    def save_episode(self, *_args: Any, **_kwargs: Any) -> None:
        lo = len(self._frames)
        self._frames.extend(self._buffer)
        hi = len(self._frames)
        self._buffer.clear()
        self.meta.episodes.append({"dataset_from_index": lo, "dataset_to_index": hi})
        self._persist()

    def clear_episode_buffer(self, **_kwargs: Any) -> None:
        self._buffer.clear()

    def finalize(self) -> None:
        self._persist()

    def _persist(self) -> None:
        meta_dir = self.root / "meta"
        data_dir = self.root / "data"
        meta_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        info = {
            "repo_id": self.repo_id,
            "fps": self.fps,
            "robot_type": self.meta.robot_type,
            "total_episodes": self.meta.total_episodes,
            "features": self.meta.features,
            "episodes": self.meta.episodes,
        }
        (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        if self._frames:
            arrays: dict[str, Any] = {
                "state": np.stack([f["observation.state"] for f in self._frames]),
                "action": np.stack([f["action"] for f in self._frames]),
            }
            for k in self._frames[0]:
                if k.startswith("observation.images."):
                    cam_name = k[len("observation.images.") :]
                    arrays[f"img__{cam_name}"] = np.stack([f[k] for f in self._frames])
            np.savez(data_dir / "stub_frames.npz", **arrays)
            tasks = [str(f.get("task", "")) for f in self._frames]
            (meta_dir / "stub_tasks.json").write_text(json.dumps(tasks), encoding="utf-8")


@pytest.fixture(autouse=True)
def ensure_lerobot_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install StubLeRobotDataset ONLY when optional ``lerobot`` is absent in `.venv`."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    if importlib.util.find_spec("lerobot") is not None:
        return
    pkg = ModuleType("lerobot")
    datasets_pkg = ModuleType("lerobot.datasets")
    ds_mod = ModuleType("lerobot.datasets.lerobot_dataset")
    ds_mod.LeRobotDataset = StubLeRobotDataset  # type: ignore[attr-defined]
    pkg.datasets = datasets_pkg  # type: ignore[attr-defined]
    datasets_pkg.lerobot_dataset = ds_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot", pkg)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets_pkg)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", ds_mod)


# ---------------------------------------------------------------------------
# Millisecond FastSkillValidator & 4-Bay AtcCoordinator Rescue Helpers
# ---------------------------------------------------------------------------


def assert_mujoco_models_present() -> None:
    """Hard-assert Panda and Inspire MJCF models are symlinked and readable."""
    assert PANDA_XML.is_file(), f"Missing Panda MJCF at {PANDA_XML}; check third_party symlinks"
    right_mjcf = load_hand("inspire").mjcf["right"]
    assert right_mjcf.is_file(), f"Missing Inspire MJCF at {right_mjcf}; check third_party symlinks"


class RecordingComplianceSink:
    def __init__(self) -> None:
        self.events: list[ComplianceEvent] = []

    def emit(self, event: ComplianceEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass

    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]


class QuietReasoner:
    def submit(self, ctx: ReasoningContext) -> None:
        del ctx

    def latest_verdict(self, bay_id: int) -> ReasoningVerdict | None:
        del bay_id
        return None

    def close(self) -> None:
        pass


class FastSkillValidator:
    """Drop-in replacement for SkillValidator executing in <1 ms instead of ~7 s."""

    def __init__(
        self,
        *,
        passed: bool = True,
        pass_min: int = GATE_PASS_MIN,
        barrier: threading.Barrier | None = None,
        hold_event: threading.Event | None = None,
    ) -> None:
        self.passed = passed
        self.pass_min = pass_min
        self._barrier = barrier
        self._hold_event = hold_event
        self.calls: list[SkillPrimitive] = []
        self.started_event = threading.Event()

    def validate(self, skill: SkillPrimitive) -> ValidationReport:
        self.calls.append(skill)
        self.started_event.set()
        if self._barrier is not None:
            self._barrier.wait()
        if self._hold_event is not None:
            self._hold_event.wait(timeout=20.0)
        cond = RolloutConditions((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0, 35.0, 1.0)
        ok_res = RolloutResult((0.0, 0.0, 0.0), 0.06, 1.0, 0.1, True, True, True, cond)
        bad_res = RolloutResult((0.0, 0.0, 0.0), 0.03, 1.0, 0.1, True, False, False, cond)
        results = (ok_res,) * 18 + (bad_res,) * 2 if self.passed else (bad_res,) * 20
        return ValidationReport(results=results, elapsed_s=0.001, threshold=self.pass_min)


@pytest.fixture(scope="session")
def inspire_hand_spec() -> HandSpec:
    assert_mujoco_models_present()
    return load_hand("inspire")


@pytest.fixture
def unoq_client() -> Iterator[UnoQClient]:
    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="e2e-unoq").start()
    client = UnoQClient(
        "127.0.0.1",
        server.udp_port,
        server.tcp_port,
        local_server=server,
    ).connect()
    try:
        yield client
    finally:
        client.close()


@dataclass
class E2ECoordinatorRig:
    co: AtcCoordinator
    sink: RecordingComplianceSink
    reasoner: QuietReasoner
    unoq: UnoQClient
    validator: FastSkillValidator
    registered_skills: list[SkillPrimitive]

    def stall_bay(self, bay_id: int = 1, **fault: bool) -> None:
        if not fault:
            fault = {"flap_jammed": True}
        self.co.start_cycle(bay_id, **fault)
        for _ in range(1500):
            snaps = self.co.tick()
            if snaps[bay_id - 1].state in {
                BayState.RESCUE,
                BayState.OOD_ESCALATED,
                BayState.QUEUED,
            }:
                break
        else:
            raise AssertionError(f"Bay {bay_id} never stalled into RESCUE")
        # Crucial 40-tick wait so the 0.5s (30-tick) -X hover retract completes and bay.move is None
        for _ in range(40):
            self.co.tick()

    def operator_samples(self, bay_id: int = 1) -> Iterator[OperatorInput]:
        bay_rig = self.co.rig(bay_id)
        hover = bay_rig.wrist_pose()
        assert hover.position is not None
        r_world = quat_to_mat(bay_rig.world_from_latch().orientation)
        origin = np.asarray(hover.position)
        base = dict(ROUTINE.fingers)
        waypoints = list(RESCUE_WAYPOINTS)
        for i in range(1, len(waypoints)):
            dur, p1, f1 = waypoints[i]
            _, p0, f0 = waypoints[i - 1]
            n = max(int(round(dur / CONTROL_DT)), 1)
            for k in range(1, n + 1):
                a = k / n
                delta = np.asarray(p0) + a * (np.asarray(p1) - np.asarray(p0))
                f = f0 + a * (f1 - f0)
                q = dict(base)
                for name, v in RESCUE_PRESS.items():
                    q[name] = base[name] + f * v
                pos = origin + r_world @ delta
                yield OperatorInput(
                    Pose(orientation=hover.orientation, position=(pos[0], pos[1], pos[2])),
                    q,
                )

    def run_rescue(self, bay_id: int = 1, *, max_extra_ticks: int | None = None) -> None:
        if max_extra_ticks is None:
            max_extra_ticks = round(self.co.config.gate_deadline_s / CONTROL_DT) + 80
        console = ConsoleInputState(selected_bay=bay_id, clutch=True)
        for sample in self.operator_samples(bay_id):
            snaps = self.co.tick(console, sample)
            if snaps[bay_id - 1].x_bolt_m >= BOLT_CLEARED_M + 0.0015:
                break
        released = ConsoleInputState(selected_bay=bay_id, clutch=False)
        stop = BayState.VALIDATING if max_extra_ticks == 0 else BayState.NOMINAL
        for _ in range(max_extra_ticks or 30):
            self.co.tick(released, None)
            if self.co.state(bay_id) == stop:
                return

    def run_short_rescue(self, bay_id: int = 1, *, ticks: int = 18) -> float:
        console = ConsoleInputState(selected_bay=bay_id, clutch=True)
        for idx, sample in enumerate(self.operator_samples(bay_id)):
            self.co.tick(console, sample)
            if idx + 1 >= ticks:
                break
        bolt_before_release = float(self.co.rig(bay_id).bolt_x())
        released = ConsoleInputState(selected_bay=bay_id, clutch=False)
        for _ in range(5):
            self.co.tick(released, None)
        return bolt_before_release


def build_e2e_coordinator(
    unoq: UnoQClient,
    hand_spec: HandSpec,
    *,
    validator: FastSkillValidator | None = None,
    gate_deadline_s: float = 0.05,
    grasp_gate: Any = None,
    audit_sink: Any = None,
    observer: Any = None,
    require_grasp_gate_hook: bool = True,
) -> E2ECoordinatorRig:
    """Build a 4-bay AtcCoordinator with spied unoq.register_skill and flexible M3 kwargs."""
    fast_val = validator or FastSkillValidator()
    sink = RecordingComplianceSink()
    reasoner = QuietReasoner()
    cfg = CoordinatorConfig(gate_deadline_s=gate_deadline_s)

    registered: list[SkillPrimitive] = []
    real_register = unoq.register_skill

    def spy_register(skill: SkillPrimitive) -> Any:
        registered.append(skill)
        return real_register(skill)

    unoq.register_skill = spy_register  # type: ignore[method-assign]

    sig = inspect.signature(AtcCoordinator.build)
    init_sig = inspect.signature(AtcCoordinator.__init__)
    if require_grasp_gate_hook and grasp_gate is not None:
        has_hook = "grasp_gate" in sig.parameters or "grasp_gate" in init_sig.parameters
        assert has_hook, (
            "AtcCoordinator.build/__init__ does not accept required M3 `grasp_gate` parameter yet"
        )

    build_kwargs: dict[str, Any] = {
        "hand_spec": hand_spec,
        "validator": fast_val,  # type: ignore[arg-type]
        "config": cfg,
        "observer": observer,
    }
    if "grasp_gate" in sig.parameters:
        build_kwargs["grasp_gate"] = grasp_gate
    if "audit_sink" in sig.parameters and audit_sink is not None:
        build_kwargs["audit_sink"] = audit_sink

    co = AtcCoordinator.build(unoq, sink, reasoner, **build_kwargs)
    return E2ECoordinatorRig(
        co=co,
        sink=sink,
        reasoner=reasoner,
        unoq=unoq,
        validator=fast_val,
        registered_skills=registered,
    )


@pytest.fixture
def coordinator_factory(
    unoq_client: UnoQClient,
    inspire_hand_spec: HandSpec,
) -> Iterator[Callable[..., E2ECoordinatorRig]]:
    created: list[AtcCoordinator] = []

    def _make(**kwargs: Any) -> E2ECoordinatorRig:
        rig = build_e2e_coordinator(unoq_client, inspire_hand_spec, **kwargs)
        created.append(rig.co)
        return rig

    try:
        yield _make
    finally:
        while created:
            created.pop().close()


# Suppress unused dataclass warning helper if needed
_ = dataclasses.is_dataclass
