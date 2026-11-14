"""Laptop <-> UNO Q wire protocol (spec 5.1): newline-delimited JSON objects.

UDP carries the fast ``filter_step`` stream (one datagram per request and reply). TCP carries
request/response ops (``ping``, ``classify_stall``, ``fit_dmp``, ``register_skill``,
``console_state``), one JSON object per line. Every reply has ``"ok"``; failures carry
``"error"``. Non-finite floats (an OOD distance of ``inf``) go on the wire as ``null``.
"""

from __future__ import annotations

import json
import math
from typing import Any, cast

DEFAULT_UDP_PORT = 15051
DEFAULT_TCP_PORT = 15052
MAX_DATAGRAM = 8192  # stays under macOS's 9216-byte UDP datagram cap

OP_PING = "ping"
OP_FILTER_STEP = "filter_step"
OP_CLASSIFY_STALL = "classify_stall"
OP_FIT_DMP = "fit_dmp"
OP_REGISTER_SKILL = "register_skill"
OP_CONSOLE_STATE = "console_state"

UDP_OPS = frozenset({OP_FILTER_STEP, OP_PING})
TCP_OPS = frozenset({OP_PING, OP_CLASSIFY_STALL, OP_FIT_DMP, OP_REGISTER_SKILL, OP_CONSOLE_STATE})


class ProtocolError(ValueError):
    """A line that is not a JSON object with a string ``op`` (requests) or ``ok`` (replies)."""


def _finite(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in cast(dict[Any, Any], value).items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in cast(list[Any] | tuple[Any, ...], value)]
    return value


def encode(message: dict[str, Any]) -> bytes:
    """One message as a UTF-8 JSON line (trailing newline included)."""
    return (json.dumps(_finite(message), separators=(",", ":"), allow_nan=False) + "\n").encode()


def decode(line: bytes | str) -> dict[str, Any]:
    text = line.decode() if isinstance(line, bytes) else line
    try:
        obj: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"bad JSON: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError("message must be a JSON object")
    return {str(k): v for k, v in cast(dict[Any, Any], obj).items()}


def decode_request(line: bytes | str) -> dict[str, Any]:
    msg = decode(line)
    if not isinstance(msg.get("op"), str):
        raise ProtocolError("request needs a string 'op'")
    return msg


def ok(**fields: Any) -> dict[str, Any]:
    return {"ok": True, **fields}


def error(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}
