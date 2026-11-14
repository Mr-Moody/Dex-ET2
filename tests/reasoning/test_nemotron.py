"""T-161: Verda Nemotron fleet reasoning co-pilot — mocked HTTP, loopback HTTP & edge cases."""

from __future__ import annotations

import io
import json
import threading
import time
from collections.abc import Callable
from email.message import Message as EmailMessage
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

from teleop.reasoning import FleetReasoner, ReasoningContext, ReasoningVerdict
from teleop.reasoning.nemotron import (
    DEFAULT_NEMOTRON_MODEL,
    DEFAULT_VERDA_BASE_URL,
    MAX_FIELD_CHARS,
    MAX_HEADLINE_CHARS,
    SYSTEM_PROMPT,
    ChatTransport,
    HttpStatusError,
    ReasonerError,
    UrllibChatTransport,
    VerdaNemotronPlaceholder,
    VerdaNemotronReasoner,
    _http_error_detail,
    build_messages,
    context_to_dict,
    main,
    parse_verdict_response,
    resolve_completions_url,
    triage,
)


def make_ctx(**overrides: object) -> ReasoningContext:
    base: dict[str, object] = {
        "event_kind": "STALL_ALERT",
        "bay_id": 1,
        "part_sku_id": "ev_latch_v1",
        "confidence": 0.21,
        "x_bolt_m": 0.035,
        "tau_ext_inf_nm": 7.9,
        "matched_signature_id": None,
        "mahalanobis_distance": float("inf"),
        "is_ood": True,
        "rollout_passed": None,
        "rollout_total": None,
    }
    base.update(overrides)
    return ReasoningContext(**base)  # type: ignore[arg-type]


def chat_envelope(
    content: str | list[dict[str, Any]],
    *,
    model: str = "nvidia/nemotron-verda-live",
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test-1",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


class FakeVerdaTransport:
    """Thread-safe injectable ``ChatTransport`` recording every call."""

    def __init__(
        self,
        responder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        *,
        delay_s: float = 0.0,
    ) -> None:
        self.delay_s = delay_s
        self._responder = responder
        self.calls: list[tuple[str, str, dict[str, Any], float]] = []
        self._lock = threading.Lock()

    def complete(
        self,
        url: str,
        api_key: str,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        with self._lock:
            self.calls.append((url, api_key, payload, timeout_s))
        if self.delay_s > 0.0:
            if self.delay_s > timeout_s:
                time.sleep(timeout_s)
                raise ReasonerError(f"request timed out after {timeout_s:.3f}s")
            time.sleep(self.delay_s)
        if self._responder is not None:
            return self._responder(payload)
        user_msg = str(payload["messages"][-1]["content"])
        telemetry = json.loads(user_msg.split("\n", 1)[1])
        kind = telemetry["event_kind"]
        bay = telemetry["bay_id"]
        verdict_json = json.dumps(
            {
                "headline": f"Nemotron Bay {bay} {kind}",
                "operator_guidance": f"Action for Bay {bay} ({kind}).",
                "compliance_rationale": f"Audit rationale for {telemetry['part_sku_id']} ({kind}).",
            }
        )
        return chat_envelope(verdict_json)


# ── Protocol conformance, URL normalisation & env configuration ──────────────────────────────


def test_reasoners_satisfy_fleet_reasoner_protocol() -> None:
    placeholder = VerdaNemotronPlaceholder()
    reasoner = VerdaNemotronReasoner()
    try:
        assert isinstance(placeholder, FleetReasoner)
        assert isinstance(reasoner, FleetReasoner)
        assert isinstance(UrllibChatTransport(), ChatTransport)
    finally:
        placeholder.close()
        reasoner.close()


def test_resolve_completions_url_normalises_variants_and_rejects_bad_schemes() -> None:
    assert DEFAULT_VERDA_BASE_URL == "https://inference.verda.com/v1"
    assert "detailed thinking" not in SYSTEM_PROMPT  # Nemotron-only toggle, removed for Kimi
    assert (
        resolve_completions_url("https://containers.verda.com/my-nemotron")
        == "https://containers.verda.com/my-nemotron/v1/chat/completions"
    )
    assert (
        resolve_completions_url("https://containers.verda.com/my-nemotron/")
        == "https://containers.verda.com/my-nemotron/v1/chat/completions"
    )
    assert (
        resolve_completions_url("https://inference.verda.com/v1")
        == "https://inference.verda.com/v1/chat/completions"
    )
    assert (
        resolve_completions_url("https://inference.verda.com/v1/")
        == "https://inference.verda.com/v1/chat/completions"
    )
    assert (
        resolve_completions_url("https://inference.verda.com/v1?routing=fast")
        == "https://inference.verda.com/v1/chat/completions?routing=fast"
    )
    assert (
        resolve_completions_url("http://127.0.0.1:8000/v1/chat/completions?foo=1")
        == "http://127.0.0.1:8000/v1/chat/completions?foo=1"
    )
    for bad in (
        "",
        "   ",
        "not-a-url",
        "file:///etc/passwd",
        "ftp://verda.com",
        "https://a b.com",
        "http://:8000",
        "https://@",
        "http://verda.com:notaport",
    ):
        with pytest.raises(ValueError):
            resolve_completions_url(bad)


def test_from_env_selects_mode_and_never_calls_network_at_init() -> None:
    fake = FakeVerdaTransport()

    # Unset / blank API key -> offline mode, zero worker threads, zero network calls.
    for env in ({}, {"VERDA_API_KEY": "   ", "VERDA_BASE_URL": "https://api.verda.com/v1"}):
        with VerdaNemotronReasoner.from_env(env, transport=fake) as r:
            assert r.mode == "offline"
            assert r.endpoint_url is None
            assert r.model == "placeholder:offline"

    # API key without base URL -> defaults to DEFAULT_VERDA_BASE_URL.
    with VerdaNemotronReasoner.from_env({"VERDA_API_KEY": "sk-test"}, transport=fake) as r:
        assert r.mode == "verda"
        assert r.endpoint_url == f"{DEFAULT_VERDA_BASE_URL}/chat/completions"
        assert r.model == DEFAULT_NEMOTRON_MODEL == "kimi-k3-1-pro"
        assert r.reasoning_effort == "none"

    # Malformed URL with API key degrades cleanly to offline mode instead of raising.
    with VerdaNemotronReasoner.from_env(
        {"VERDA_API_KEY": "sk-test", "VERDA_BASE_URL": "file:///tmp/bad", "NEMOTRON_MODEL": "m1"},
        transport=fake,
    ) as r:
        assert r.mode == "offline"
        assert r.model == "placeholder:m1"

    assert fake.calls == []

    for bad_kwargs in (
        {"timeout_s": 0.0},
        {"timeout_s": float("nan")},
        {"max_tokens": 0},
        {"temperature": -0.1},
        {"temperature": float("nan")},
        {"close_grace_s": -1.0},
        {"close_grace_s": float("nan")},
        {"max_workers": 0},
        {"reasoning_effort": "extreme"},
    ):
        with pytest.raises(ValueError):
            VerdaNemotronReasoner(**bad_kwargs)  # type: ignore[arg-type]


# ── Offline deterministic fallback across all event kinds & non-finite floats ────────────────


def test_offline_fallback_handles_all_event_kinds_and_non_finite_floats() -> None:
    with VerdaNemotronReasoner(model="nemotron-mini") as r:
        assert r.mode == "offline"
        assert r.latest_verdict(1) is None

        cases = [
            make_ctx(event_kind="STALL_ALERT", bay_id=1, is_ood=True, mahalanobis_distance=None),
            make_ctx(
                event_kind="STALL_ALERT",
                bay_id=2,
                is_ood=False,
                matched_signature_id="Jam_Sig_1",
                mahalanobis_distance=0.85,
            ),
            make_ctx(event_kind="OOD_REFUSAL", bay_id=3, x_bolt_m=0.0, mahalanobis_distance=7.4),
            make_ctx(
                event_kind="VALIDATION_GATE",
                bay_id=4,
                is_ood=False,
                rollout_passed=2,
                rollout_total=3,
            ),
            make_ctx(event_kind="RESCUE_LOGGED", bay_id=1, x_bolt_m=0.055),
            make_ctx(event_kind="SKILL_DEPLOYED", bay_id=2, matched_signature_id="Jam_Sig_1"),
        ]
        for ctx in cases:
            r.submit(ctx)
            v = r.latest_verdict(ctx.bay_id)
            assert isinstance(v, ReasoningVerdict)
            assert v.model == "placeholder:nemotron-mini"
            assert v.headline and v.operator_guidance and v.compliance_rationale

        v4 = r.latest_verdict(4)
        assert (
            v4 is not None and "2/3 FAIL" in v4.headline and "withheld" in v4.compliance_rationale
        )

        # Non-finite floats in context never crash triage or JSON prompt construction.
        weird = make_ctx(
            bay_id=1,
            confidence=float("nan"),
            x_bolt_m=float("nan"),
            tau_ext_inf_nm=float("inf"),
            mahalanobis_distance=float("-inf"),
        )
        r.submit(weird)
        vw = r.latest_verdict(1)
        assert vw is not None and "n/a" in vw.headline
        serialized = json.dumps(context_to_dict(weird), allow_nan=False)
        assert '"mahalanobis_distance": "-inf"' in serialized
        assert len(build_messages(weird)) == 2


def test_defensive_unknown_event_kind_does_not_crash() -> None:
    ctx = make_ctx(event_kind="CUSTOM_EVENT")  # type: ignore[arg-type]
    headline, guidance, rationale = triage(ctx)
    assert "CUSTOM_EVENT" in headline and guidance and rationale


# ── Async worker with mocked Verda Nemotron responses ────────────────────────────────────────


def test_async_worker_triages_stall_validation_gate_and_ood_refusal() -> None:
    fake = FakeVerdaTransport()
    with VerdaNemotronReasoner(
        base_url="https://verda.example/v1",
        api_key="sk-verda-123",
        model="nvidia/Llama-3.3-Nemotron-Super-49B-v1",
        transport=fake,
    ) as r:
        assert r.mode == "verda"
        r.submit(make_ctx(event_kind="STALL_ALERT", bay_id=1, mahalanobis_distance=float("inf")))
        r.submit(
            make_ctx(
                event_kind="VALIDATION_GATE",
                bay_id=2,
                is_ood=False,
                matched_signature_id="Jam_Sig_1",
                rollout_passed=3,
                rollout_total=3,
            )
        )
        r.submit(
            make_ctx(event_kind="OOD_REFUSAL", bay_id=3, x_bolt_m=0.0, mahalanobis_distance=8.1)
        )
        assert r.flush(timeout_s=2.0)

        v1 = r.latest_verdict(1)
        v2 = r.latest_verdict(2)
        v3 = r.latest_verdict(3)
        assert v1 is not None and v1.headline == "Nemotron Bay 1 STALL_ALERT"
        assert v2 is not None and v2.headline == "Nemotron Bay 2 VALIDATION_GATE"
        assert v3 is not None and v3.headline == "Nemotron Bay 3 OOD_REFUSAL"
        assert v1.model == "nvidia/nemotron-verda-live"
        assert r.latest_verdict(4) is None

        assert len(fake.calls) == 3
        url, key, payload, _ = fake.calls[0]
        assert url == "https://verda.example/v1/chat/completions"
        assert key == "sk-verda-123"
        assert payload["model"] == "nvidia/Llama-3.3-Nemotron-Super-49B-v1"
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["reasoning_effort"] == "none"
        assert r.counts()["remote_ok"] == 3
        assert r.counts()["errors"] == 0


def test_submit_never_blocks_60hz_control_loop_under_slow_network() -> None:
    fake = FakeVerdaTransport(delay_s=0.12)
    with VerdaNemotronReasoner(
        api_key="sk-verda",
        transport=fake,
        timeout_s=1.0,
        close_grace_s=1.0,
    ) as r:
        t0 = time.perf_counter()
        for i in range(40):
            r.submit(make_ctx(bay_id=(i % 4) + 1, x_bolt_m=0.01 * (i % 5)))
        per_submit_ms = (time.perf_counter() - t0) * 1000.0 / 40
        assert per_submit_ms < 0.5  # strictly non-blocking on the 60 Hz (16.6 ms) control thread
        assert r.flush(timeout_s=2.0)
        for bay in (1, 2, 3, 4):
            assert r.latest_verdict(bay) is not None
        assert r.counts()["coalesced"] > 0


def test_per_bay_coalescing_and_stale_response_guard() -> None:
    started_first = threading.Event()
    release_first = threading.Event()

    def gated_responder(payload: dict[str, Any]) -> dict[str, Any]:
        telemetry = json.loads(str(payload["messages"][-1]["content"]).split("\n", 1)[1])
        kind = telemetry["event_kind"]
        if kind == "STALL_ALERT":
            started_first.set()
            release_first.wait(timeout=2.0)
        body = json.dumps(
            {
                "headline": f"Verdict for {kind}",
                "operator_guidance": f"Guidance {kind}",
                "compliance_rationale": f"Rationale {kind}",
            }
        )
        return chat_envelope(body)

    fake = FakeVerdaTransport(responder=gated_responder)
    with VerdaNemotronReasoner(api_key="sk-verda", transport=fake) as r:
        # 1. Start an in-flight STALL_ALERT for Bay 1.
        r.submit(make_ctx(bay_id=1, event_kind="STALL_ALERT"))
        assert started_first.wait(timeout=1.0)

        # 2. Submit two newer events for Bay 1 while STALL_ALERT is still in flight.
        # RESCUE_LOGGED should be coalesced into VALIDATION_GATE.
        r.submit(make_ctx(bay_id=1, event_kind="RESCUE_LOGGED"))
        r.submit(
            make_ctx(
                bay_id=1,
                event_kind="VALIDATION_GATE",
                is_ood=False,
                rollout_passed=3,
                rollout_total=3,
            )
        )
        release_first.set()
        assert r.flush(timeout_s=2.0)

        verdict = r.latest_verdict(1)
        assert verdict is not None
        assert verdict.headline == "Verdict for VALIDATION_GATE"
        assert (
            len(fake.calls) == 2
        )  # RESCUE_LOGGED was coalesced; only STALL_ALERT + VALIDATION_GATE ran
        assert r.counts()["coalesced"] == 1


# ── Nemotron output parsing edge cases (<think>, code fences, clamping, bad JSON) ────────────


def test_parser_handles_think_blocks_markdown_fences_content_arrays_and_clamping() -> None:
    # 1. <think>...</think> reasoning block + ```json fence (common Nemotron output).
    raw_think = (
        "<think>\nBay 2 stalled at x_bolt=0.0 cm; d_M=7.4 > 3.0 -> OOD pinned bolt.\n</think>\n"
        "```json\n"
        + json.dumps(
            {
                "headline": "Bay 2:\n  pinned bolt\trefused",
                "operator_guidance": "A" * 400,
                "compliance_rationale": "Rejected unsafe replay.",
            }
        )
        + "\n```"
    )
    v = parse_verdict_response(
        chat_envelope(raw_think, model="  nvidia/nemotron-super  "),
        default_model="fallback",
        latency_ms=42.5,
    )
    assert v.headline == "Bay 2: pinned bolt refused"
    assert len(v.operator_guidance) == MAX_FIELD_CHARS
    assert v.operator_guidance.endswith("...")
    assert v.model == "nvidia/nemotron-super"
    assert v.latency_ms == pytest.approx(42.5)

    # 2. Structured content parts list + prose wrapper + headline length clamping.
    parts_content = [
        {"type": "text", "text": "Here is the triage result:\n"},
        {
            "type": "text",
            "text": json.dumps(
                {
                    "headline": "H" * 200,
                    "operator_guidance": "Hold clutch and depress flap.",
                    "compliance_rationale": "Unseen jam queued for operator.",
                }
            ),
        },
    ]
    v2 = parse_verdict_response(
        chat_envelope(parts_content, model=""),
        default_model="nvidia/default",
        latency_ms=10.0,
    )
    assert len(v2.headline) == MAX_HEADLINE_CHARS and v2.headline.endswith("...")
    assert v2.model == "nvidia/default"

    # 3. Prompt-prefilled assistant "<think>\n" (only closing </think> in content, even with a
    # draft JSON object inside the thinking trace) strips everything through </think>.
    prefilled_think = 'Draft check: {"headline": "draft-only"}.\n</think>\n' + json.dumps(
        {
            "headline": "Bay 2: pinned bolt refused",
            "operator_guidance": "Inspect bolt before rescue.",
            "compliance_rationale": "d_M=7.40 > 3.0.",
        }
    )
    v3 = parse_verdict_response(
        chat_envelope(prefilled_think, model="nvidia/nemotron-prefill"),
        default_model="fallback",
        latency_ms=15.0,
    )
    assert v3.headline == "Bay 2: pinned bolt refused"
    assert v3.operator_guidance == "Inspect bolt before rescue."


def test_provisional_verdict_prevents_stale_hud_across_phase_transitions() -> None:
    started_gate = threading.Event()
    release_gate = threading.Event()

    def responder(payload: dict[str, Any]) -> dict[str, Any]:
        user_msg = str(payload["messages"][-1]["content"])
        if "STALL_ALERT" in user_msg:
            return chat_envelope(
                json.dumps(
                    {
                        "headline": "Live STALL_ALERT verdict",
                        "operator_guidance": "Hold clutch.",
                        "compliance_rationale": "Stall detected.",
                    }
                )
            )
        started_gate.set()
        release_gate.wait(timeout=2.0)
        return chat_envelope(
            json.dumps(
                {
                    "headline": "Live VALIDATION_GATE verdict",
                    "operator_guidance": "Skill promoted.",
                    "compliance_rationale": "3/3 rollouts passed.",
                }
            )
        )

    fake = FakeVerdaTransport(responder=responder)
    with VerdaNemotronReasoner(api_key="sk-verda", model="nemotron-live", transport=fake) as r:
        r.submit(make_ctx(bay_id=1, event_kind="STALL_ALERT"))
        assert r.flush(timeout_s=1.0)
        v_stall = r.latest_verdict(1)
        assert v_stall is not None and v_stall.headline == "Live STALL_ALERT verdict"

        # Transition Bay 1 to VALIDATION_GATE while the remote call is still in flight:
        # HUD immediately sees fresh VALIDATION_GATE provisional guidance (not stale STALL_ALERT),
        # then upgrades seamlessly once the remote call completes.
        r.submit(
            make_ctx(
                bay_id=1,
                event_kind="VALIDATION_GATE",
                is_ood=False,
                rollout_passed=3,
                rollout_total=3,
            )
        )
        assert started_gate.wait(timeout=1.0)
        v_mid = r.latest_verdict(1)
        assert v_mid is not None
        assert "3/3 PASS" in v_mid.headline
        assert v_mid.model == "placeholder:nemotron-live"

        release_gate.set()
        assert r.flush(timeout_s=1.0)
        v_done = r.latest_verdict(1)
        assert v_done is not None
        assert v_done.headline == "Live VALIDATION_GATE verdict"
        assert v_done.model == "nvidia/nemotron-verda-live"


def test_http_400_on_response_format_retries_without_json_mode_and_caches_capability() -> None:
    seen_formats: list[bool] = []

    def strict_vllm_responder(payload: dict[str, Any]) -> dict[str, Any]:
        has_rf = "response_format" in payload
        seen_formats.append(has_rf)
        if has_rf:
            raise HttpStatusError(400, "guided decoding response_format unsupported with thinking")
        return chat_envelope(
            "<think>checking telemetry</think>"
            + json.dumps(
                {
                    "headline": "Recovered without response_format",
                    "operator_guidance": "Hold clutch and depress flap.",
                    "compliance_rationale": "Fallback to prompt-enforced JSON succeeded.",
                }
            )
        )

    fake = FakeVerdaTransport(responder=strict_vllm_responder)
    with VerdaNemotronReasoner(api_key="sk-verda", transport=fake) as r:
        r.submit(make_ctx(bay_id=1))
        assert r.flush(timeout_s=1.0)
        v1 = r.latest_verdict(1)
        assert v1 is not None and v1.headline == "Recovered without response_format"
        # First call tried response_format=True, retried False.
        assert seen_formats == [True, False]

        # Subsequent call skips response_format directly (1 HTTP call, no wasted 400).
        r.submit(make_ctx(bay_id=2))
        assert r.flush(timeout_s=1.0)
        assert seen_formats == [True, False, False]
        assert r.counts()["remote_ok"] == 2
        assert r.counts()["errors"] == 0


def test_multi_bay_worker_pool_prevents_cross_bay_head_of_line_blocking() -> None:
    bay1_entered = threading.Event()
    release_bay1 = threading.Event()

    def responder(payload: dict[str, Any]) -> dict[str, Any]:
        telemetry = json.loads(str(payload["messages"][-1]["content"]).split("\n", 1)[1])
        bay = int(telemetry["bay_id"])
        if bay == 1:
            bay1_entered.set()
            release_bay1.wait(timeout=2.0)
        return chat_envelope(
            json.dumps(
                {
                    "headline": f"Live Bay {bay}",
                    "operator_guidance": "ok",
                    "compliance_rationale": "ok",
                }
            )
        )

    fake = FakeVerdaTransport(responder=responder)
    with VerdaNemotronReasoner(api_key="sk-verda", transport=fake, max_workers=4) as r:
        r.submit(make_ctx(bay_id=1))
        assert bay1_entered.wait(timeout=1.0)

        # Bay 2 submits while Bay 1 is completely blocked on the remote server:
        # Bay 2 must complete independently without waiting for Bay 1!
        t0 = time.perf_counter()
        r.submit(make_ctx(bay_id=2))
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            v2 = r.latest_verdict(2)
            if v2 is not None and v2.headline == "Live Bay 2":
                break
            time.sleep(0.005)
        b2_ms = (time.perf_counter() - t0) * 1000.0
        v2 = r.latest_verdict(2)
        assert v2 is not None and v2.headline == "Live Bay 2"
        assert b2_ms < 250.0

        release_bay1.set()
        assert r.flush(timeout_s=1.0)
        v1 = r.latest_verdict(1)
        assert v1 is not None and v1.headline == "Live Bay 1"


def test_http_error_detail_extracts_openai_vllm_and_fastapi_messages() -> None:
    hdrs = EmailMessage()
    e_vllm = HTTPError(
        "http://x",
        400,
        "Bad Request",
        hdrs,
        io.BytesIO(b'{"object":"error","message":"model does not exist"}'),
    )
    assert _http_error_detail(e_vllm) == "model does not exist"

    e_fastapi_str = HTTPError(
        "http://x", 422, "Unprocessable", hdrs, io.BytesIO(b'{"detail":"invalid max_tokens"}')
    )
    assert _http_error_detail(e_fastapi_str) == "invalid max_tokens"

    e_fastapi_list = HTTPError(
        "http://x",
        422,
        "Unprocessable",
        hdrs,
        io.BytesIO(b'{"detail":[{"msg":"Extra inputs are not permitted"}]}'),
    )
    assert _http_error_detail(e_fastapi_list) == "Extra inputs are not permitted"


@pytest.mark.parametrize(
    "bad_envelope",
    [
        {},
        {"choices": []},
        {"choices": ["not-a-dict"]},
        {"choices": [{"message": "not-a-dict"}]},
        {"choices": [{"message": {"content": None}}]},
        chat_envelope("<think>truncated mid-thought without closing tag"),
        chat_envelope("plain text with no braces at all"),
        chat_envelope(
            json.dumps({"headline": "", "operator_guidance": "ok", "compliance_rationale": "ok"})
        ),
        chat_envelope(
            json.dumps({"headline": 123, "operator_guidance": "ok", "compliance_rationale": "ok"})
        ),
        chat_envelope(json.dumps({"headline": "only one key"})),
    ],
)
def test_malformed_model_responses_fall_back_to_deterministic_triage(
    bad_envelope: dict[str, Any],
) -> None:
    fake = FakeVerdaTransport(responder=lambda _p: bad_envelope)
    with VerdaNemotronReasoner(api_key="sk-verda", model="nemotron-x", transport=fake) as r:
        ctx = make_ctx(event_kind="OOD_REFUSAL", bay_id=2, x_bolt_m=0.0, mahalanobis_distance=7.4)
        r.submit(ctx)
        assert r.flush(timeout_s=1.0)
        verdict = r.latest_verdict(2)
        assert verdict is not None
        assert verdict.model == "placeholder:nemotron-x"
        assert "0.0 cm" in verdict.headline and "7.40" in verdict.compliance_rationale
        assert r.counts()["errors"] == 1
        assert r.counts()["fallbacks"] == 1


# ── Real loopback HTTP server end-to-end tests (auth, 503 error, socket timeout) ─────────────


def test_urllib_transport_end_to_end_success_http_error_and_timeout() -> None:
    state: dict[str, Any] = {"mode": "ok", "auth": None, "path": None}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            state["auth"] = self.headers.get("Authorization")
            state["path"] = self.path
            mode = state["mode"]
            if mode == "timeout":
                time.sleep(0.25)
                return
            if mode == "error503":
                err = json.dumps({"error": {"message": "Verda replica scaling up"}}).encode("utf-8")
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
                return
            reply = json.dumps(
                chat_envelope(
                    json.dumps(
                        {
                            "headline": "Live Loopback Nemotron Verdict",
                            "operator_guidance": "Hold clutch and slide bolt past 5.2 cm.",
                            "compliance_rationale": "Contact stall confirmed over HTTP.",
                        }
                    )
                )
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, format: str, *args: object) -> None:  # silence stderr in pytest
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{port}"
        with VerdaNemotronReasoner(
            base_url=base_url,
            api_key="sk-live-check",
            model="nemotron-loopback",
            timeout_s=0.08,
        ) as r:
            # 1. 200 OK over real HTTP socket.
            r.submit(make_ctx(bay_id=1))
            assert r.flush(timeout_s=2.0)
            v_ok = r.latest_verdict(1)
            assert v_ok is not None and v_ok.headline == "Live Loopback Nemotron Verdict"
            assert state["auth"] == "Bearer sk-live-check"
            assert state["path"] == "/v1/chat/completions"

            # 2. HTTP 503 JSON error -> graceful deterministic fallback.
            state["mode"] = "error503"
            r.submit(
                make_ctx(bay_id=2, event_kind="OOD_REFUSAL", x_bolt_m=0.0, mahalanobis_distance=7.2)
            )
            assert r.flush(timeout_s=2.0)
            v_503 = r.latest_verdict(2)
            assert v_503 is not None and v_503.model == "placeholder:nemotron-loopback"
            assert "refus" in v_503.compliance_rationale.lower()

            # 3. Socket timeout (> 80 ms) -> graceful deterministic fallback.
            state["mode"] = "timeout"
            t0 = time.perf_counter()
            r.submit(make_ctx(bay_id=3, event_kind="STALL_ALERT", x_bolt_m=0.035))
            submit_ms = (time.perf_counter() - t0) * 1000.0
            assert submit_ms < 5.0  # submit() returned immediately despite slow server
            assert r.flush(timeout_s=2.0)
            v_timeout = r.latest_verdict(3)
            assert v_timeout is not None and v_timeout.model == "placeholder:nemotron-loopback"
            assert "3.5 cm" in v_timeout.headline
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def test_urllib_transport_handles_307_post_redirect_and_blocks_unsafe_redirects() -> None:
    seen_target: dict[str, Any] = {"body": b"", "auth": None}

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            if self.path == "/v1/chat/completions":
                self.send_response(307)
                self.send_header("Location", "/v2/chat/completions")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path == "/v2/chat/completions":
                seen_target["body"] = body
                seen_target["auth"] = self.headers.get("Authorization")
                reply = json.dumps(
                    chat_envelope(
                        json.dumps(
                            {
                                "headline": "Redirected 307 OK",
                                "operator_guidance": "Proceed.",
                                "compliance_rationale": "Preserved POST body across 307.",
                            }
                        )
                    )
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)
                return
            if self.path == "/ssrf/v1/chat/completions":
                self.send_response(302)
                self.send_header("Location", "file:///etc/passwd")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        transport = UrllibChatTransport()
        res = transport.complete(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            "sk-redir",
            {"model": "m", "messages": []},
            timeout_s=1.0,
        )
        assert res["choices"][0]["message"]["content"]
        assert seen_target["auth"] == "Bearer sk-redir"
        assert json.loads(seen_target["body"].decode("utf-8"))["model"] == "m"

        with pytest.raises(HttpStatusError, match=r"unsafe redirect target|not allowed"):
            transport.complete(
                f"http://127.0.0.1:{port}/ssrf/v1/chat/completions",
                "sk-redir",
                {"model": "m", "messages": []},
                timeout_s=1.0,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


# ── Lifecycle & CLI smoke tests ──────────────────────────────────────────────────────────────


def test_close_is_idempotent_and_ignores_post_close_submits() -> None:
    fake = FakeVerdaTransport()
    r = VerdaNemotronReasoner(api_key="sk-verda", transport=fake)
    r.close()
    r.close()  # idempotent
    r.submit(make_ctx(bay_id=1))  # dropped cleanly after close, never raises
    assert r.latest_verdict(1) is None
    assert fake.calls == []


def test_module_cli_status_and_demo_run_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)  # isolate load_dotenv(find_dotenv(usecwd=True)) from repo .env
    monkeypatch.delenv("VERDA_API_KEY", raising=False)
    monkeypatch.delenv("VERDA_BASE_URL", raising=False)
    monkeypatch.setenv("NEMOTRON_MODEL", "nvidia/nemotron-cli-test")

    assert main(["status"]) == 0
    out_status = capsys.readouterr().out
    assert "mode:     offline" in out_status
    assert "placeholder:nvidia/nemotron-cli-test" in out_status

    assert main(["demo"]) == 0
    out_demo = capsys.readouterr().out
    assert "[STALL_ALERT bay=1]" in out_demo
    assert "[VALIDATION_GATE bay=1]" in out_demo
    assert "[OOD_REFUSAL bay=2]" in out_demo


# ── Kimi K3.1 Pro (Verda Inference API) specifics ────────────────────────────────────────────


def kimi_envelope(content: str, *, finish_reason: str = "stop") -> dict[str, Any]:
    env = chat_envelope(content, model="kimi-k3-1-pro")
    env["choices"][0]["finish_reason"] = finish_reason
    env["choices"][0]["message"]["reasoning_content"] = "Bolt stalled at the flap; {not json"
    return env


def test_kimi_ignores_reasoning_content_and_parses_content() -> None:
    verdict = {
        "headline": "Bay 1 flap stall",
        "operator_guidance": "Lift the flap, then pull the bolt past 5.2 cm.",
        "compliance_rationale": "Contact stall logged for ev_latch_v1.",
    }
    fake = FakeVerdaTransport(lambda _p: kimi_envelope(json.dumps(verdict)))
    with VerdaNemotronReasoner(api_key="sk-verda", transport=fake) as r:
        r.submit(make_ctx(bay_id=1))
        assert r.flush(timeout_s=2.0)
        v = r.latest_verdict(1)
        assert v is not None and v.headline == "Bay 1 flap stall"
        assert v.model == "kimi-k3-1-pro"
        payload = fake.calls[0][2]
        assert payload["model"] == "kimi-k3-1-pro"
        assert payload["reasoning_effort"] == "none"


def test_kimi_empty_content_after_length_cutoff_falls_back_to_triage() -> None:
    fake = FakeVerdaTransport(lambda _p: kimi_envelope("", finish_reason="length"))
    ctx = make_ctx(bay_id=2)
    with VerdaNemotronReasoner(api_key="sk-verda", transport=fake) as r:
        r.submit(ctx)
        assert r.flush(timeout_s=2.0)
        v = r.latest_verdict(2)
        assert v is not None and v.headline == triage(ctx)[0]
        assert r.counts()["remote_ok"] == 0


def test_reasoning_effort_can_be_omitted_or_overridden() -> None:
    fake = FakeVerdaTransport()
    with VerdaNemotronReasoner(api_key="k", transport=fake, reasoning_effort=None) as r:
        r.submit(make_ctx(bay_id=1))
        assert r.flush(timeout_s=2.0)
    with VerdaNemotronReasoner.from_env(
        {"VERDA_API_KEY": "k"}, transport=fake, reasoning_effort="low"
    ) as r:
        r.submit(make_ctx(bay_id=1))
        assert r.flush(timeout_s=2.0)
    assert "reasoning_effort" not in fake.calls[0][2]
    assert fake.calls[1][2]["reasoning_effort"] == "low"
