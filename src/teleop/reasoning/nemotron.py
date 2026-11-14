"""Verda Nemotron fleet reasoner (T-161) and deterministic offline placeholder (T-100).

Pipeline for every ``ReasoningContext``::

    submit(ctx)     60 Hz control thread: immediately seeds a deterministic ``triage(ctx)``
      |             provisional verdict (so the HUD is never blank or stuck on a stale prior-phase
      |             verdict), coalesces pending work per bay under a lock, and wakes the daemon
      |             worker pool. Zero network or disk I/O, bounded O(bays) memory, never raises.
      |
    worker pool     Up to ``max_workers`` daemon threads (at most 1 in-flight request per bay)
                    POST OpenAI-compatible chat completions to Verda (``VERDA_BASE_URL``,
                    ``VERDA_API_KEY``, ``NEMOTRON_MODEL``; default model ``kimi-k3-1-pro``
                    with ``reasoning_effort="none"``), ignore ``reasoning_content``, retry
                    without ``response_format`` if an endpoint rejects ``json_object`` mode with
                    HTTP 400/422, strip ``<think>`` / prompt-prefilled ``</think>`` / code-fence
                    wrappers, and upgrade ``ReasoningVerdict`` if no newer event superseded
                    that bay.
                    On missing ``VERDA_API_KEY``, timeout, HTTP error, or malformed model output,
                    retains deterministic ``triage(ctx)``.

``VerdaNemotronPlaceholder`` (from T-100) is kept unchanged for deterministic zero-thread tests;
``VerdaNemotronReasoner.from_env()`` selects ``"verda"`` mode when ``VERDA_API_KEY`` is configured
and ``"offline"`` mode otherwise.

CLI: ``uv run python -m teleop.reasoning.nemotron {status,demo}``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from http.client import HTTPMessage
from typing import IO, Any, Literal, Protocol, Self, cast, runtime_checkable
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit, urlunsplit

from dotenv import find_dotenv, load_dotenv

from teleop.reasoning.protocol import ReasoningContext, ReasoningVerdict

log = logging.getLogger(__name__)

OOD_THRESHOLD = 3.0  # gamma_ood, spec 4.2; quoted in rationales and system prompt
BOLT_CLEARED_CM = 5.2  # x_bolt threshold in cm (spec 2.1)

DEFAULT_VERDA_BASE_URL = "https://inference.verda.com/v1"
DEFAULT_NEMOTRON_MODEL = "kimi-k3-1-pro"  # Verda Inference API model ID (name kept for compat)
# Kimi reasons before answering (default effort "max"), which would burn the 4 s budget and
# max_tokens on hidden reasoning; "none" returns the JSON verdict directly.
DEFAULT_REASONING_EFFORT = "none"
REASONING_EFFORTS = ("none", "low", "high", "max")
DEFAULT_TIMEOUT_S = 4.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_MAX_WORKERS = 4
DEFAULT_CLOSE_GRACE_S = 2.0

MAX_HEADLINE_CHARS = 120
MAX_FIELD_CHARS = 280

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_FENCE_RE = re.compile(r"\A```(?:json)?\s*(.*?)\s*```\Z", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are the Fleet Reasoning Co-Pilot for a 4-bay EV-battery disassembly cell "
    "(7-DoF Franka Panda + compliant wrist + 6-actuator Inspire five-finger hand unlatching a "
    "3-box sprung slide-bolt latch: index hooks the bolt, thumb presses the stopper flap flat). "
    "Physics reference: bolt clears at x_bolt >= 5.2 cm; a sprung stopper flap stalls the bolt "
    "near x_bolt ~ 3.5 cm; a pinned-bolt fault locks near x_bolt = 0.0 cm; stall triggers when "
    "confidence C < 0.3 for 300 ms; Mahalanobis OOD threshold gamma_ood = 3.0. "
    "Respond ONLY with a single JSON object containing exactly three concise single-line string "
    'keys: "headline", "operator_guidance", and "compliance_rationale".'
)


# ── Deterministic offline triage (T-100 baseline + fallback) ─────────────────────────────────


def _cm(x_m: float) -> str:
    if not math.isfinite(x_m):
        return "n/a"
    return f"{x_m * 100.0:.1f} cm"


def _fmt_float(val: float, fmt: str) -> str:
    if not math.isfinite(val):
        return "inf" if val > 0 else ("-inf" if val < 0 else "n/a")
    return format(val, fmt)


def _distance(ctx: ReasoningContext) -> str:
    d = ctx.mahalanobis_distance
    if d is None:
        return "n/a"
    return _fmt_float(d, ".2f")


def triage(ctx: ReasoningContext) -> tuple[str, str, str]:
    """Deterministic (headline, operator_guidance, compliance_rationale) for a context."""
    bay = f"Bay {ctx.bay_id}"
    tau_s = _fmt_float(ctx.tau_ext_inf_nm, ".1f")
    conf_s = _fmt_float(ctx.confidence, ".2f")
    stall = f"x_bolt={_cm(ctx.x_bolt_m)}, |tau_ext|={tau_s} Nm, C={conf_s}"
    match ctx.event_kind:
        case "STALL_ALERT":
            if ctx.is_ood or ctx.matched_signature_id is None:
                return (
                    f"{bay}: unseen contact stall at {_cm(ctx.x_bolt_m)}",
                    "Select the bay, hold the clutch, depress the stopper flap with the thumb "
                    f"and slide the bolt past {BOLT_CLEARED_CM:.1f} cm with the index finger.",
                    f"No validated skill for {ctx.part_sku_id} matches this stall ({stall}); "
                    "the bay retracted to a safe hover and was queued for a human rescue.",
                )
            return (
                f"{bay}: known stall {ctx.matched_signature_id} at {_cm(ctx.x_bolt_m)}",
                "No action needed: the fleet skill is replaying inside the torque tube.",
                f"Stall matched {ctx.matched_signature_id} (d_M={_distance(ctx)} <= "
                f"{OOD_THRESHOLD:.1f}); autonomous replay authorised ({stall}).",
            )
        case "OOD_REFUSAL":
            return (
                f"{bay}: out-of-distribution stall at {_cm(ctx.x_bolt_m)} refused",
                "Inspect the latch for a pinned or damaged bolt before any rescue attempt.",
                f"Mahalanobis distance {_distance(ctx)} exceeds gamma_ood={OOD_THRESHOLD:.1f}; "
                f"refusing skill transfer and escalating to a human ({stall}).",
            )
        case "VALIDATION_GATE":
            passed = ctx.rollout_passed or 0
            total = ctx.rollout_total or 0
            need = total if ctx.rollout_threshold is None else ctx.rollout_threshold
            ok = total > 0 and passed >= need
            return (
                f"{bay}: skill validation {passed}/{total} {'PASS' if ok else 'FAIL'}",
                "Skill promoted to the fleet library."
                if ok
                else "Skill rejected; repeat the rescue with a cleaner motion.",
                f"Headless sim gate cleared the bolt in {passed} of {total} perturbed "
                f"randomized rollouts (pass at {need}) for {ctx.part_sku_id}; "
                + ("deployment approved." if ok else "deployment withheld."),
            )
        case "RESCUE_LOGGED":
            return (
                f"{bay}: operator rescue logged at {_cm(ctx.x_bolt_m)}",
                "Release the clutch; the demonstration is being fitted and validated.",
                f"Human rescue trajectory recorded for {ctx.part_sku_id} ({stall}).",
            )
        case "SKILL_DEPLOYED":
            return (
                f"{bay}: skill {ctx.matched_signature_id or 'new'} deployed to fleet",
                "No action needed.",
                f"Validated skill for {ctx.part_sku_id} broadcast to all bays.",
            )
        case _:
            return (
                f"{bay}: event {ctx.event_kind} at {_cm(ctx.x_bolt_m)}",
                "Monitor bay telemetry.",
                f"Recorded {ctx.event_kind} for {ctx.part_sku_id} ({stall}).",
            )


class VerdaNemotronPlaceholder:
    """Zero-network deterministic placeholder from T-100."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = f"placeholder:{model or 'offline'}"
        self._lock = threading.Lock()
        self._verdicts: dict[int, ReasoningVerdict] = {}

    @classmethod
    def from_env(cls) -> VerdaNemotronPlaceholder:
        return cls(
            base_url=os.environ.get("VERDA_BASE_URL") or None,
            api_key=os.environ.get("VERDA_API_KEY") or None,
            model=os.environ.get("NEMOTRON_MODEL") or None,
        )

    def submit(self, ctx: ReasoningContext) -> None:
        t0 = time.perf_counter()
        headline, guidance, rationale = triage(ctx)
        verdict = ReasoningVerdict(
            headline=headline,
            operator_guidance=guidance,
            compliance_rationale=rationale,
            model=self.model,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
        with self._lock:
            self._verdicts[ctx.bay_id] = verdict

    def latest_verdict(self, bay_id: int) -> ReasoningVerdict | None:
        with self._lock:
            return self._verdicts.get(bay_id)

    def close(self) -> None:
        pass


# ── Verda Nemotron HTTP transport & response parsing (T-161) ─────────────────────────────────


class ReasonerError(RuntimeError):
    """Any Verda endpoint, HTTP, timeout, or model output validation failure."""


class HttpStatusError(ReasonerError):
    """HTTP 4xx/5xx failure carrying the status code for capability-fallback retry."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


@runtime_checkable
class ChatTransport(Protocol):
    def complete(
        self,
        url: str,
        api_key: str,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]: ...


def resolve_completions_url(base_url: str) -> str:
    """Normalise ``VERDA_BASE_URL`` into a full ``.../chat/completions`` HTTP(S) endpoint."""
    raw = base_url.strip()
    if not raw or any(ch.isspace() for ch in raw):
        raise ValueError(f"invalid VERDA_BASE_URL: {base_url!r}")
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"VERDA_BASE_URL must be an http(s) URL with a hostname, got {base_url!r}")
    _ = parts.port  # raises ValueError on malformed port (e.g. "http://host:badport")
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        norm_path = path
    elif path.endswith("/v1"):
        norm_path = f"{path}/chat/completions"
    else:
        norm_path = f"{path}/v1/chat/completions"
    return urlunsplit((parts.scheme, parts.netloc, norm_path, parts.query, ""))


def _json_safe_float(val: float | None) -> float | str | None:
    if val is None:
        return None
    if math.isfinite(val):
        return float(val)
    return "inf" if val > 0 else ("-inf" if val < 0 else "nan")


def context_to_dict(ctx: ReasoningContext) -> dict[str, Any]:
    """JSON-compliant telemetry dict (converts ``inf``/``nan`` so strict JSON never fails)."""
    return {
        "event_kind": str(ctx.event_kind),
        "bay_id": int(ctx.bay_id),
        "part_sku_id": str(ctx.part_sku_id),
        "confidence": _json_safe_float(float(ctx.confidence)),
        "x_bolt_m": _json_safe_float(float(ctx.x_bolt_m)),
        "x_bolt_cm": _json_safe_float(round(float(ctx.x_bolt_m) * 100.0, 2))
        if math.isfinite(ctx.x_bolt_m)
        else _json_safe_float(float(ctx.x_bolt_m)),
        "tau_ext_inf_nm": _json_safe_float(float(ctx.tau_ext_inf_nm)),
        "matched_signature_id": ctx.matched_signature_id,
        "mahalanobis_distance": _json_safe_float(ctx.mahalanobis_distance),
        "ood_threshold": OOD_THRESHOLD,
        "is_ood": bool(ctx.is_ood),
        "rollout_passed": ctx.rollout_passed,
        "rollout_total": ctx.rollout_total,
        "rollout_threshold": ctx.rollout_threshold,
    }


def build_messages(ctx: ReasoningContext) -> list[dict[str, str]]:
    user_json = json.dumps(context_to_dict(ctx), sort_keys=True, allow_nan=False)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Telemetry event:\n{user_json}"},
    ]


def _http_error_detail(exc: HTTPError) -> str:
    try:
        raw = exc.read()
        doc: object = json.loads(raw)
        if isinstance(doc, dict):
            obj = cast(dict[str, object], doc)
            err = obj.get("error")
            if isinstance(err, dict):
                msg = cast(dict[str, object], err).get("message")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
            if isinstance(err, str) and err.strip():
                return err.strip()
            top_msg = obj.get("message")
            if isinstance(top_msg, str) and top_msg.strip():
                return top_msg.strip()
            detail = obj.get("detail")
            if isinstance(detail, str) and detail.strip():
                return detail.strip()
            if isinstance(detail, list) and detail:
                first_item = cast(list[object], detail)[0]
                if isinstance(first_item, dict):
                    item_msg = cast(dict[str, object], first_item).get("msg")
                    if isinstance(item_msg, str) and item_msg.strip():
                        return item_msg.strip()
    except (OSError, ValueError):
        pass
    return str(exc.reason)


class _SafePostRedirectHandler(urlrequest.HTTPRedirectHandler):
    """Restrict redirects to HTTP(S), block TLS downgrade, and preserve POST body on 307/308."""

    def redirect_request(
        self,
        req: urlrequest.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urlrequest.Request | None:
        target = urljoin(req.full_url, newurl.replace(" ", "%20"))
        target_parts = urlsplit(target)
        orig_parts = urlsplit(req.full_url)
        if target_parts.scheme not in ("http", "https") or not target_parts.hostname:
            raise HTTPError(
                req.full_url,
                code,
                f"unsafe redirect target: {newurl!r}",
                headers,
                fp,
            )
        if orig_parts.scheme == "https" and target_parts.scheme != "https":
            raise HTTPError(
                req.full_url,
                code,
                "refusing HTTPS to HTTP redirect downgrade",
                headers,
                fp,
            )
        method = req.get_method()
        same_origin = (target_parts.scheme, target_parts.netloc) == (
            orig_parts.scheme,
            orig_parts.netloc,
        )
        if code in (307, 308):
            forwarded_headers: dict[str, str] = {}
            for k, v in req.header_items():
                low = k.lower()
                if low == "content-length":
                    continue
                if low == "authorization" and not same_origin:
                    continue
                forwarded_headers[k] = v
            return urlrequest.Request(
                target,
                data=req.data,
                headers=forwarded_headers,
                origin_req_host=req.origin_req_host,
                unverifiable=True,
                method=method,
            )
        if method == "POST":
            raise HTTPError(
                req.full_url,
                code,
                f"HTTP {code} redirect drops POST body; point VERDA_BASE_URL to {target}",
                headers,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, target)


class UrllibChatTransport:
    """OpenAI-compatible ``/v1/chat/completions`` client using stdlib ``urllib.request``."""

    def __init__(self) -> None:
        self._opener = urlrequest.build_opener(
            urlrequest.HTTPHandler(),
            urlrequest.HTTPSHandler(),
            _SafePostRedirectHandler(),
        )

    def complete(
        self,
        url: str,
        api_key: str,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        req = urlrequest.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "teleop-atc-reasoner/0.1",
            },
            method="POST",
        )
        try:
            with self._opener.open(req, timeout=timeout_s) as resp:
                raw = cast(bytes, resp.read())
            parsed: object = json.loads(raw.decode("utf-8"))
        except HTTPError as exc:
            raise HttpStatusError(exc.code, _http_error_detail(exc)) from exc
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            raise ReasonerError(str(exc)) from exc
        if not isinstance(parsed, dict):
            raise ReasonerError(f"expected JSON object from Verda, got {type(parsed).__name__}")
        doc = cast(dict[str, Any], parsed)
        if doc.get("error") is not None:
            raise ReasonerError(f"API error: {doc['error']}")
        return doc


def _clamp_single_line(value: object, field_name: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise ReasonerError(f"field {field_name!r} must be a string")
    collapsed = " ".join(value.split())
    if not collapsed:
        raise ReasonerError(f"field {field_name!r} must not be empty")
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = _THINK_RE.sub("", text).strip()
    if "<think>" in cleaned.lower():
        raise ReasonerError("unterminated <think> block in Nemotron response")
    if "</think>" in cleaned.lower():
        # Chat templates that prefill "<think>\n" in the prompt emit only "... </think>{...}".
        cleaned = _THINK_CLOSE_RE.split(cleaned)[-1].strip()
    fence = _FENCE_RE.match(cleaned)
    if fence is not None:
        cleaned = fence.group(1).strip()
    try:
        direct: object = json.loads(cleaned)
        if isinstance(direct, dict):
            return cast(dict[str, Any], direct)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    idx = cleaned.find("{")
    while idx != -1:
        try:
            candidate, _ = decoder.raw_decode(cleaned, idx)
            if isinstance(candidate, dict):
                obj = cast(dict[str, Any], candidate)
                if "headline" in obj:
                    return obj
        except ValueError:
            pass
        idx = cleaned.find("{", idx + 1)
    raise ReasonerError("no valid JSON verdict object found in Nemotron response")


def _extract_message_text(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ReasonerError("response missing non-empty 'choices' list")
    first = cast(object, choices[0])
    if not isinstance(first, dict):
        raise ReasonerError("choices[0] is not an object")
    msg = cast(dict[str, Any], first).get("message")
    if not isinstance(msg, dict):
        raise ReasonerError("choices[0].message is not an object")
    content = cast(dict[str, Any], msg).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in cast(list[object], content):
            if isinstance(item, dict):
                txt = cast(dict[str, Any], item).get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        if parts:
            return "\n".join(parts)
    raise ReasonerError("choices[0].message.content is missing or empty")


def parse_verdict_response(
    response: Mapping[str, Any],
    *,
    default_model: str,
    latency_ms: float,
) -> ReasoningVerdict:
    """Parse an OpenAI-compatible chat completion response into a ``ReasoningVerdict``."""
    raw_text = _extract_message_text(response)
    fields = _extract_json_object(raw_text)
    headline = _clamp_single_line(fields.get("headline"), "headline", MAX_HEADLINE_CHARS)
    guidance = _clamp_single_line(
        fields.get("operator_guidance"), "operator_guidance", MAX_FIELD_CHARS
    )
    rationale = _clamp_single_line(
        fields.get("compliance_rationale"), "compliance_rationale", MAX_FIELD_CHARS
    )
    raw_model = response.get("model")
    model_name = (
        raw_model.strip() if isinstance(raw_model, str) and raw_model.strip() else default_model
    )
    return ReasoningVerdict(
        headline=headline,
        operator_guidance=guidance,
        compliance_rationale=rationale,
        model=model_name,
        latency_ms=max(0.0, float(latency_ms)),
    )


# ── Async non-blocking fleet reasoner (T-161) ────────────────────────────────────────────────


class VerdaNemotronReasoner:
    """Asynchronous Verda-hosted Nemotron fleet reasoner with deterministic offline fallback.

    * When ``api_key`` is absent/blank (or ``base_url`` is invalid), runs in ``"offline"`` mode
      with zero background threads and zero network calls, formatting deterministic ``triage(ctx)``
      verdicts directly in ``submit()``.
    * When ``api_key`` is set, ``submit()`` immediately records a deterministic provisional
      ``triage(ctx)`` verdict (preventing blank or stale prior-phase HUD guidance) and enqueues the
      latest context per bay in microseconds. A bounded daemon worker pool queries Verda Nemotron
      (at most 1 in-flight HTTP call per bay, concurrent across independent bays) and upgrades the
      verdict if no newer event superseded that bay.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        *,
        transport: ChatTransport | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.1,
        close_grace_s: float = DEFAULT_CLOSE_GRACE_S,
        max_workers: int = DEFAULT_MAX_WORKERS,
        reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    ) -> None:
        if timeout_s <= 0.0 or not math.isfinite(timeout_s):
            raise ValueError(f"timeout_s must be positive and finite, got {timeout_s}")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if temperature < 0.0 or not math.isfinite(temperature):
            raise ValueError(f"temperature must be non-negative and finite, got {temperature}")
        if close_grace_s < 0.0 or not math.isfinite(close_grace_s):
            raise ValueError(f"close_grace_s must be non-negative and finite, got {close_grace_s}")
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {REASONING_EFFORTS} or None, "
                f"got {reasoning_effort!r}"
            )

        clean_key = (api_key or "").strip() or None
        raw_model = (model or "").strip() or None
        raw_url = (base_url or "").strip() or None

        endpoint_url: str | None = None
        if clean_key is not None:
            candidate_url = raw_url or DEFAULT_VERDA_BASE_URL
            try:
                endpoint_url = resolve_completions_url(candidate_url)
            except ValueError as exc:
                log.warning("nemotron: %s; degrading to deterministic offline fallback", exc)
                clean_key = None

        self.base_url = raw_url
        self.endpoint_url = endpoint_url
        self.api_key = clean_key
        self.configured_model = raw_model or DEFAULT_NEMOTRON_MODEL
        self.fallback_model = f"placeholder:{raw_model or 'offline'}"
        self.model = self.configured_model if clean_key is not None else self.fallback_model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self._close_grace_s = close_grace_s
        self._max_workers = max_workers

        self._transport: ChatTransport | None = None
        if clean_key is not None and endpoint_url is not None:
            self._transport = UrllibChatTransport() if transport is None else transport

        self._cond = threading.Condition()
        self._verdicts: dict[int, ReasoningVerdict] = {}
        self._bay_seq: dict[int, int] = {}
        self._pending: OrderedDict[int, tuple[int, ReasoningContext]] = OrderedDict()
        self._in_flight_bays: set[int] = set()
        self._use_json_mode = True
        self._closed = False
        self._close_deadline: float | None = None
        self._counts: dict[str, int] = {
            "submitted": 0,
            "coalesced": 0,
            "remote_ok": 0,
            "fallbacks": 0,
            "errors": 0,
        }

        self._workers: list[threading.Thread] = []
        if self._transport is not None:
            for idx in range(self._max_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"verda-nemotron-{idx}",
                    daemon=True,
                )
                worker.start()
                self._workers.append(worker)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        transport: ChatTransport | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        close_grace_s: float = DEFAULT_CLOSE_GRACE_S,
        max_workers: int = DEFAULT_MAX_WORKERS,
        reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    ) -> Self:
        """Construct from ``VERDA_BASE_URL``, ``VERDA_API_KEY``, and ``NEMOTRON_MODEL``."""
        src = os.environ if env is None else env
        return cls(
            base_url=src.get("VERDA_BASE_URL") or None,
            api_key=src.get("VERDA_API_KEY") or None,
            model=src.get("NEMOTRON_MODEL") or None,
            transport=transport,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            close_grace_s=close_grace_s,
            max_workers=max_workers,
            reasoning_effort=reasoning_effort,
        )

    @property
    def mode(self) -> Literal["verda", "offline"]:
        return "verda" if self._transport is not None else "offline"

    def counts(self) -> dict[str, int]:
        with self._cond:
            return dict(self._counts)

    def submit(self, ctx: ReasoningContext) -> None:
        """Queue ``ctx`` for triage without blocking the 60 Hz control loop; never raises."""
        try:
            t0 = time.perf_counter()
            headline, guidance, rationale = triage(ctx)
            provisional = ReasoningVerdict(
                headline=headline,
                operator_guidance=guidance,
                compliance_rationale=rationale,
                model=self.fallback_model,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
            if self._transport is None:
                with self._cond:
                    if self._closed:
                        return
                    self._counts["submitted"] += 1
                    self._counts["fallbacks"] += 1
                    self._bay_seq[ctx.bay_id] = self._bay_seq.get(ctx.bay_id, 0) + 1
                    self._verdicts[ctx.bay_id] = provisional
                return

            with self._cond:
                if self._closed:
                    return
                self._counts["submitted"] += 1
                seq = self._bay_seq.get(ctx.bay_id, 0) + 1
                self._bay_seq[ctx.bay_id] = seq
                self._verdicts[ctx.bay_id] = provisional
                if ctx.bay_id in self._pending:
                    self._counts["coalesced"] += 1
                self._pending[ctx.bay_id] = (seq, ctx)
                self._cond.notify_all()
        except Exception:
            log.exception("nemotron: submit failed")

    def latest_verdict(self, bay_id: int) -> ReasoningVerdict | None:
        with self._cond:
            return self._verdicts.get(bay_id)

    def flush(self, timeout_s: float = 5.0) -> bool:
        """Wait up to ``timeout_s`` for queued and in-flight requests to finish."""
        safe_timeout = 0.0 if not math.isfinite(timeout_s) else max(0.0, timeout_s)
        deadline = time.monotonic() + safe_timeout
        with self._cond:
            while self._pending or self._in_flight_bays:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._cond.wait(timeout=remaining)
            return True

    def close(self) -> None:
        """Drain queued contexts up to ``close_grace_s`` and stop the worker pool."""
        deadline = time.monotonic() + self._close_grace_s
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._close_deadline = deadline
            self._cond.notify_all()
        for worker in self._workers:
            remaining = max(0.0, deadline - time.monotonic())
            worker.join(timeout=remaining)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── worker internals ──

    def _fallback_verdict(self, ctx: ReasoningContext, latency_ms: float) -> ReasoningVerdict:
        headline, guidance, rationale = triage(ctx)
        return ReasoningVerdict(
            headline=headline,
            operator_guidance=guidance,
            compliance_rationale=rationale,
            model=self.fallback_model,
            latency_ms=max(0.0, latency_ms),
        )

    def _pop_ready_locked(self) -> tuple[int, int, ReasoningContext] | None:
        """Pop the oldest pending bay that does not already have an in-flight request."""
        for bay_id in list(self._pending.keys()):
            if bay_id not in self._in_flight_bays:
                seq, ctx = self._pending.pop(bay_id)
                self._in_flight_bays.add(bay_id)
                return bay_id, seq, ctx
        return None

    def _evaluate_one(self, ctx: ReasoningContext) -> tuple[ReasoningVerdict, bool]:
        """Return ``(verdict, from_remote)`` without holding the lock."""
        assert self._transport is not None and self.endpoint_url is not None
        assert self.api_key is not None
        payload: dict[str, Any] = {
            "model": self.configured_model,
            "messages": build_messages(ctx),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self._use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        try:
            try:
                response = self._transport.complete(
                    self.endpoint_url,
                    self.api_key,
                    payload,
                    self.timeout_s,
                )
            except HttpStatusError as exc:
                if exc.status_code in (400, 422) and "response_format" in payload:
                    self._use_json_mode = False
                    retry_payload = {k: v for k, v in payload.items() if k != "response_format"}
                    remaining_s = max(0.05, self.timeout_s - (time.perf_counter() - t0))
                    log.info(
                        "nemotron: endpoint rejected response_format (%s); retrying plain",
                        exc,
                    )
                    response = self._transport.complete(
                        self.endpoint_url,
                        self.api_key,
                        retry_payload,
                        remaining_s,
                    )
                else:
                    raise
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            verdict = parse_verdict_response(
                response,
                default_model=self.configured_model,
                latency_ms=elapsed_ms,
            )
            return verdict, True
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            log.warning(
                "nemotron: remote triage failed for bay %d (%s); using offline fallback",
                ctx.bay_id,
                exc,
            )
            return self._fallback_verdict(ctx, elapsed_ms), False

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while True:
                    if (
                        self._closed
                        and self._close_deadline is not None
                        and time.monotonic() >= self._close_deadline
                    ):
                        self._pending.clear()
                    ready = self._pop_ready_locked()
                    if ready is not None:
                        bay_id, seq, ctx = ready
                        break
                    if self._closed and not self._pending:
                        self._cond.notify_all()
                        return
                    if self._closed and self._close_deadline is not None:
                        remaining = self._close_deadline - time.monotonic()
                        if remaining <= 0.0:
                            self._pending.clear()
                            self._cond.notify_all()
                            return
                        self._cond.wait(timeout=remaining)
                    else:
                        self._cond.wait()

            try:
                verdict, remote_ok = self._evaluate_one(ctx)
            except Exception:  # defensive: _evaluate_one already catches, never die silently
                log.exception("nemotron: unexpected worker failure")
                verdict, remote_ok = self._fallback_verdict(ctx, 0.0), False

            with self._cond:
                self._in_flight_bays.discard(bay_id)
                if remote_ok:
                    self._counts["remote_ok"] += 1
                else:
                    self._counts["errors"] += 1
                    self._counts["fallbacks"] += 1
                if self._bay_seq.get(bay_id) == seq:
                    self._verdicts[bay_id] = verdict
                self._cond.notify_all()


# ── Module CLI (uv run python -m teleop.reasoning.nemotron {status,demo}) ────────────────────


def _demo_contexts() -> list[ReasoningContext]:
    return [
        ReasoningContext(
            event_kind="STALL_ALERT",
            bay_id=1,
            part_sku_id="ev_latch_v1",
            confidence=0.21,
            x_bolt_m=0.035,
            tau_ext_inf_nm=7.8,
            matched_signature_id=None,
            mahalanobis_distance=float("inf"),
            is_ood=True,
        ),
        ReasoningContext(
            event_kind="VALIDATION_GATE",
            bay_id=1,
            part_sku_id="ev_latch_v1",
            confidence=0.92,
            x_bolt_m=0.056,
            tau_ext_inf_nm=2.9,
            matched_signature_id="Jam_Sig_1",
            mahalanobis_distance=0.42,
            is_ood=False,
            rollout_passed=3,
            rollout_total=3,
        ),
        ReasoningContext(
            event_kind="OOD_REFUSAL",
            bay_id=2,
            part_sku_id="ev_latch_v1",
            confidence=0.18,
            x_bolt_m=0.0,
            tau_ext_inf_nm=8.4,
            matched_signature_id=None,
            mahalanobis_distance=7.4,
            is_ood=True,
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    load_dotenv(find_dotenv(usecwd=True))
    parser = argparse.ArgumentParser(
        prog="python -m teleop.reasoning.nemotron",
        description="Verda Nemotron fleet reasoning co-pilot status and smoke test",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="Show configured mode, endpoint URL and model")
    sub.add_parser("demo", help="Run sample STALL_ALERT, VALIDATION_GATE and OOD_REFUSAL events")
    args = parser.parse_args(argv)

    with VerdaNemotronReasoner.from_env() as reasoner:
        print(f"mode:     {reasoner.mode}")
        print(f"endpoint: {reasoner.endpoint_url or '(offline)'}")
        print(f"model:    {reasoner.model}")
        if args.cmd == "status":
            return 0
        for ctx in _demo_contexts():
            reasoner.submit(ctx)
            reasoner.flush(timeout_s=reasoner.timeout_s + 1.0)
            verdict = reasoner.latest_verdict(ctx.bay_id)
            assert verdict is not None
            header = (
                f"[{ctx.event_kind} bay={ctx.bay_id}] "
                f"({verdict.model}, {verdict.latency_ms:.1f} ms)"
            )
            print(f"\n{header}")
            print(f"  headline:  {verdict.headline}")
            print(f"  guidance:  {verdict.operator_guidance}")
            print(f"  rationale: {verdict.compliance_rationale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
