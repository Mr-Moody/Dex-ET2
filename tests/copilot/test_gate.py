"""Tier 2 grasp gate: verdict parsing, fallbacks and the spoken verdict, on fake clients."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from teleop.copilot.config import CopilotConfig
from teleop.copilot.gate import (
    GATE_SCHEMA,
    GraspGate,
    GraspTelemetrySummary,
    KeyframeJpeg,
    PoseSnapshot,
    gate_parts,
    parse_verdict,
)
from teleop.copilot.gemini import GeminiClient, ImagePart, Part, TextPart
from teleop.copilot.local import LocalModelClient

SUMMARY = GraspTelemetrySummary(
    bay_id=1,
    sku="EV-PACK",
    skill_id="skill_1",
    signature_id="sig_1",
    duration_s=2.5,
    bolt_start_m=0.014,
    bolt_final_m=0.056,
    peak_tau_hand_nm=0.44,
    mean_tau_hand_nm=0.28,
    peak_arm_ratio=0.31,
    min_wrist_x_latch_m=0.012,
    min_wrist_z_latch_m=0.045,
)


def _jpeg() -> bytes:
    from teleop.copilot.gate import encode_jpeg

    return encode_jpeg(np.zeros((64, 96, 3), dtype=np.uint8))


KEYFRAMES = [
    KeyframeJpeg(label, t, x, 0.3, _jpeg())
    for label, t, x in (("stall", 1.0, 0.014), ("mid-hold", 2.0, 0.035), ("bolt-clear", 3.0, 0.056))
]


class FakeTransport:
    def __init__(self, answer: str, status: int = 200) -> None:
        self.answer = answer
        self.status = status
        self.bodies: list[bytes] = []

    def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
    ) -> tuple[int, bytes]:
        del method, url, headers, timeout_s
        self.bodies.append(body or b"")
        import json

        doc = {"candidates": [{"content": {"parts": [{"text": self.answer}]}}]}
        return self.status, json.dumps(doc).encode()


class FakeSpeaker:
    def __init__(self) -> None:
        self.said: list[tuple[str, str, int | None]] = []

    def say(self, text: str, *, severity: Any = "info", bay: int | None = None) -> int:
        self.said.append((text, severity, bay))
        return len(self.said)


def _gemini(answer: str, status: int = 200) -> tuple[GeminiClient, FakeTransport]:
    t = FakeTransport(answer, status)
    return GeminiClient("k", transport=t), t


def test_approve_is_spoken_and_request_uses_low_thinking_and_media() -> None:
    client, transport = _gemini(
        '{"verdict": "approve", "say": "Bay 1 clean hook, skill approved.", "reason": "ok",'
        ' "confidence": 0.9, "hv_clearance_ok": true}'
    )
    speaker = FakeSpeaker()
    gate = GraspGate(client, cfg=CopilotConfig(gemini_api_key="k"), speaker=speaker)
    try:
        verdict = gate.evaluate(KEYFRAMES, SUMMARY)
    finally:
        gate.close()
    assert verdict.approved and verdict.source == "gemini"
    assert verdict.hv_clearance_ok is True
    assert speaker.said == [("Bay 1 clean hook, skill approved.", "info", 1)]
    body = transport.bodies[0].decode()
    assert '"thinkingLevel":"LOW"' in body
    assert '"mediaResolution":"MEDIA_RESOLUTION_LOW"' in body
    assert body.count("inlineData") == 3
    assert "additionalProperties" not in str(GATE_SCHEMA)


def test_reject_is_spoken_as_alarm() -> None:
    client, _ = _gemini('{"verdict": "reject", "say": "Bay 1 thumb near HV.", "reason": "x"}')
    speaker = FakeSpeaker()
    gate = GraspGate(client, cfg=CopilotConfig(gemini_api_key="k"), speaker=speaker)
    try:
        assert not gate.evaluate(KEYFRAMES, SUMMARY).approved
    finally:
        gate.close()
    assert speaker.said[0][1] == "alarm"


def test_unknown_verdict_rejects_and_confidence_is_clamped() -> None:
    v = parse_verdict(
        {"verdict": "maybe", "say": "x", "reason": "y", "confidence": 1.7},
        latency_ms=1.0,
        model="m",
        source="gemini",
    )
    assert v.verdict == "reject" and v.confidence == 1.0


def test_no_client_is_physics_only_and_never_renders() -> None:
    gate = GraspGate(None, cfg=CopilotConfig())
    snap = PoseSnapshot(
        "stall", 0.0, 0.0, 0.0, *(np.zeros(1) for _ in range(5))
    )  # model=None: nothing to render
    try:
        verdict = gate.submit(SUMMARY, [snap]).result(timeout=5)
    finally:
        gate.close()
    assert verdict.approved and verdict.source == "physics_only"


def test_model_error_falls_back_to_physics_and_stays_silent() -> None:
    client, _ = _gemini("{}", status=500)
    speaker = FakeSpeaker()
    gate = GraspGate(client, cfg=CopilotConfig(gemini_api_key="k"), speaker=speaker)
    try:
        verdict = gate.evaluate(KEYFRAMES, SUMMARY)
    finally:
        gate.close()
    assert verdict.approved and verdict.source == "error"
    assert speaker.said == []


def test_local_client_reports_local_source() -> None:
    class LocalTransport:
        def request(
            self,
            method: str,
            url: str,
            headers: Mapping[str, str],
            body: bytes | None,
            timeout_s: float,
        ) -> tuple[int, bytes]:
            del method, url, headers, body, timeout_s
            return 200, b'{"choices": [{"message": {"content": "{\\"verdict\\": \\"approve\\"}"}}]}'

    client = LocalModelClient("http://127.0.0.1:1", transport=LocalTransport())
    gate = GraspGate(client, cfg=CopilotConfig(local_url="http://127.0.0.1:1"))
    try:
        verdict = gate.evaluate(KEYFRAMES, SUMMARY)
    finally:
        gate.close()
    assert verdict.source == "local" and verdict.model == "gemma4:e4b"


def test_parts_carry_labels_telemetry_and_images() -> None:
    parts: Sequence[Part] = gate_parts(KEYFRAMES, SUMMARY)
    text = " ".join(p.text for p in parts if isinstance(p, TextPart))
    for label in ("stall", "mid-hold", "bolt-clear"):
        assert label in text
    assert "0.056" in text and "0.44" in text
    assert sum(isinstance(p, ImagePart) for p in parts) == 3


def test_closed_gate_refuses_submissions() -> None:
    gate = GraspGate(None)
    gate.close()
    with pytest.raises(RuntimeError):
        gate.submit(SUMMARY, [])
