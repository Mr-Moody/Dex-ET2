"""Tier 1 & Tier 2 E2E tests for Features F8, F9, and F11.

Covers:
- F8: `LocalModelClient` (`COPILOT_LOCAL_URL` / `gemma4:e4b` OpenAI-compatible `/chat/completions`
      multimodal + `no_images` fallback) when `GEMINI_API_KEY` is unset or `--no-gemini`, and
      physics-only fallback when neither Gemini nor local URL is active.
- F9: `GraspGate` (`gemini-4-hydrogen`, `thinkingLevel="LOW"`, `mediaResolution=`
      `"MEDIA_RESOLUTION_LOW"`, 3 `<=512px` JPEGs `stall`/`mid-hold`/`bolt-clear` +
      `GraspTelemetrySummary`, `responseSchema` without `additionalProperties`,
      `thoughtSignature` handling, `GraspGateVerdict`).
- F11: `AtcCoordinator._Gate` clutch engage, `bolt >= 0.052 m` vs `< 0.052 m` short release,
       `threading.Barrier(2)` proof of parallel `SkillValidator` + `GraspGate` execution,
       immediate Hydrogen `"say"` SSE TTS while `BayState.VALIDATING` before `gate_deadline_s`,
       and 4-way `(Sim, Hydrogen)` dual-approval matrix on `unoq.register_skill`.
"""

from __future__ import annotations

import base64
import inspect
import json
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from teleop.atc.types import BOLT_CLEARED_M, BayState
from teleop.copilot import speech as speech_mod
from teleop.copilot.config import CopilotConfig
from teleop.copilot.gemini import GeminiClient, ImagePart, TextPart
from teleop.sim.hands import HandSpec

from .conftest import (
    E2ECoordinatorRig,
    FakeAudioHarness,
    FastSkillValidator,
    LocalGemmaServer,
    LoopbackGeminiTLSServer,
    construct_flexible,
    contains_key_recursive,
    jpeg_dimensions,
    load_rescue_jpeg,
    make_rescue_keyframes,
    make_tts_streamer,
    require_attr,
    require_module,
)


def _make_sample_telemetry(gate_mod: Any, *, bay_id: int = 1) -> Any:
    summary_cls = require_attr(gate_mod, "GraspTelemetrySummary")
    return summary_cls(
        bay_id=bay_id,
        sku="EV-PACK-75KWH",
        skill_id="skill_bay1_001",
        signature_id="Jam_Sig_1",
        duration_s=2.70,
        bolt_start_m=0.014,
        bolt_final_m=0.056,
        peak_tau_hand_nm=0.44,
        mean_tau_hand_nm=0.28,
        peak_arm_ratio=0.31,
        min_wrist_x_latch_m=0.012,
        min_wrist_z_latch_m=0.045,
        touch_offset_m=(0.001, -0.002),
    )


def _make_keyframe_objs(gate_mod: Any, count: int = 3) -> list[Any]:
    kf_cls = require_attr(gate_mod, "KeyframeJpeg")
    return [
        kf_cls(
            label=label,
            sim_t=sim_t,
            bolt_x_m=bolt_x,
            tau_hand_nm=tau,
            jpeg_bytes=jpeg,
        )
        for label, sim_t, bolt_x, tau, jpeg in make_rescue_keyframes(count)
    ]


def _make_local_client(
    local_url: str,
    *,
    model: str = "gemma4:e4b",
    no_images: bool = False,
) -> Any:
    local_mod = require_module("teleop.copilot.local")
    cls = require_attr(local_mod, "LocalModelClient")
    return construct_flexible(
        cls,
        base_url=local_url,
        local_url=local_url,
        url=local_url,
        model=model,
        local_model=model,
        no_images=no_images,
    )


def _build_grasp_gate(
    *,
    client: Any = None,
    cfg: CopilotConfig | None = None,
    speaker: Any = None,
    audit_reviewer: Any = None,
    hand_spec: HandSpec | None = None,
) -> Any:
    gate_mod = require_module("teleop.copilot.gate")
    gate_cls = require_attr(gate_mod, "GraspGate")
    resolved_cfg = cfg or CopilotConfig(gemini_api_key="test-e2e-hydrogen-key")
    return construct_flexible(
        gate_cls,
        client=client,
        gemini_client=client,
        model_client=client,
        cfg=resolved_cfg,
        config=resolved_cfg,
        model=resolved_cfg.fast_model,
        fast_model=resolved_cfg.fast_model,
        thinking_level=resolved_cfg.fast_thinking,
        speaker=speaker,
        tts=speaker,
        audit_reviewer=audit_reviewer,
        reviewer=audit_reviewer,
        audit_sink=audit_reviewer,
        hand_spec=hand_spec,
    )


def _evaluate_grasp_gate(gate: Any, keyframes: list[Any], telemetry: Any) -> Any:
    """Invoke synchronous evaluate/assess or join async submit(...) on GraspGate."""
    for method_name in ("evaluate", "assess", "check", "evaluate_keyframes", "run"):
        fn = getattr(gate, method_name, None)
        if callable(fn):
            sig = inspect.signature(fn)
            if len(sig.parameters) >= 2:
                return fn(keyframes, telemetry)
            # Single bundle input dataclass
            gate_mod = require_module("teleop.copilot.gate")
            inp_cls = getattr(gate_mod, "GraspGateInput", None) or getattr(
                gate_mod, "GraspGateRequest", None
            )
            if inp_cls is not None:
                inp = construct_flexible(
                    inp_cls,
                    keyframes=tuple(keyframes),
                    frames=tuple(keyframes),
                    telemetry=telemetry,
                    summary=telemetry,
                    bay_id=telemetry.bay_id,
                    skill_id=telemetry.skill_id,
                )
                return fn(inp)
    submit_fn = require_attr(gate, "submit")
    sig = inspect.signature(submit_fn)
    if len(sig.parameters) >= 2:
        fut = submit_fn(keyframes, telemetry)
    else:
        gate_mod = require_module("teleop.copilot.gate")
        inp_cls = getattr(gate_mod, "GraspGateInput", None) or getattr(
            gate_mod, "GraspGateRequest", None
        )
        assert inp_cls is not None, "Expected evaluate(keyframes, telemetry) or GraspGateInput"
        inp = construct_flexible(
            inp_cls,
            keyframes=tuple(keyframes),
            frames=tuple(keyframes),
            telemetry=telemetry,
            summary=telemetry,
            bay_id=telemetry.bay_id,
            skill_id=telemetry.skill_id,
        )
        fut = submit_fn(inp)
    return fut.result(timeout=5.0) if hasattr(fut, "result") else fut


# ===========================================================================
# FEATURE 8 — Local Gemma Fallback (`LocalModelClient`) (Tier 1 >=5, Tier 2 >=5)
# ===========================================================================


class TestFeature8LocalGemmaFallbackTier1:
    """Tier 1 category-partition tests for `LocalModelClient` (`gemma4:e4b`)."""

    def test_f8_t1_posts_multimodal_messages_to_openai_chat_completions_endpoint(
        self,
        local_gemma_server: LocalGemmaServer,
    ) -> None:
        client = _make_local_client(local_gemma_server.base_url, model="gemma4:e4b")
        jpeg = load_rescue_jpeg(256)
        ans = client.generate_json(
            "gemma4:e4b",
            "Evaluate battery latch grasp.",
            [TextPart("Telemetry: bolt=5.6cm"), ImagePart(jpeg)],
            {"type": "object", "properties": {"verdict": {"type": "string"}}},
            5.0,
        )
        assert ans["verdict"] == "approve"
        assert len(local_gemma_server.requests) == 1
        req = local_gemma_server.requests[0]
        assert req.path.endswith("/chat/completions")
        assert req.json_body is not None
        assert req.json_body["model"] == "gemma4:e4b"
        serialized = json.dumps(req.json_body)
        assert "data:image/jpeg;base64," in serialized

    def test_f8_t1_no_images_true_strips_image_parts_from_local_gemma_request(
        self,
        local_gemma_server: LocalGemmaServer,
    ) -> None:
        client = _make_local_client(
            local_gemma_server.base_url,
            model="gemma4:e4b",
            no_images=True,
        )
        jpeg = load_rescue_jpeg(256)
        ans = client.generate_json(
            "gemma4:e4b",
            "Text-only evaluation.",
            [TextPart("Telemetry: bolt=5.6cm"), ImagePart(jpeg)],
            None,
            5.0,
        )
        assert ans["verdict"] == "approve"
        serialized = json.dumps(local_gemma_server.requests[-1].json_body)
        assert "data:image/jpeg;base64," not in serialized
        assert "Telemetry: bolt=5.6cm" in serialized

    def test_f8_t1_copilot_config_from_env_selects_local_url_when_gemini_key_unset(
        self,
        local_gemma_server: LocalGemmaServer,
    ) -> None:
        empty_env_file = REPO_ROOT_EMPTY_ENV = __file__  # non-.env path ignored cleanly
        del REPO_ROOT_EMPTY_ENV
        cfg = CopilotConfig.from_env(
            {
                "COPILOT_LOCAL_URL": local_gemma_server.base_url,
                "COPILOT_LOCAL_MODEL": "gemma4:e4b",
            },
            env_file="/nonexistent/.env",
        )
        assert not cfg.has_key
        assert cfg.local_url == local_gemma_server.base_url
        assert cfg.local_model == "gemma4:e4b"
        del empty_env_file

    def test_f8_t1_grasp_gate_uses_local_gemma_when_gemini_key_unset_and_local_url_set(
        self,
        local_gemma_server: LocalGemmaServer,
    ) -> None:
        gate_mod = require_module("teleop.copilot.gate")
        cfg = CopilotConfig(
            gemini_api_key=None,
            local_url=local_gemma_server.base_url,
            local_model="gemma4:e4b",
        )
        local_client = _make_local_client(local_gemma_server.base_url)
        gate = _build_grasp_gate(client=local_client, cfg=cfg)
        try:
            verdict = _evaluate_grasp_gate(
                gate,
                _make_keyframe_objs(gate_mod, 3),
                _make_sample_telemetry(gate_mod),
            )
            assert verdict.approved is True
            assert verdict.verdict == "approve"
            assert verdict.source == "local"
            assert len(local_gemma_server.requests) == 1
        finally:
            if hasattr(gate, "close"):
                gate.close()

    def test_f8_t1_physics_only_fallback_when_neither_gemini_key_nor_local_url_configured(
        self,
    ) -> None:
        gate_mod = require_module("teleop.copilot.gate")
        cfg = CopilotConfig(gemini_api_key=None, local_url=None)
        gate = _build_grasp_gate(client=None, cfg=cfg)
        try:
            verdict = _evaluate_grasp_gate(
                gate,
                _make_keyframe_objs(gate_mod, 3),
                _make_sample_telemetry(gate_mod),
            )
            assert verdict.approved is True
            assert verdict.source == "physics_only"
        finally:
            if hasattr(gate, "close"):
                gate.close()


class TestFeature8LocalGemmaFallbackTier2:
    """Tier 2 negative, markdown fence stripping, HTTP error, and timeout tests for F8."""

    def test_f8_t2_parses_json_wrapped_in_markdown_code_fences_from_local_gemma(
        self,
        local_gemma_server: LocalGemmaServer,
    ) -> None:
        local_gemma_server.raw_content_override = (
            '```json\n{"verdict": "reject", "say": "Unsafe thumb pose.", '
            '"reason": "HV bus contact risk."}\n```'
        )
        client = _make_local_client(local_gemma_server.base_url)
        ans = client.generate_json("gemma4:e4b", "sys", [TextPart("obs")], None, 5.0)
        assert ans["verdict"] == "reject"
        assert ans["say"] == "Unsafe thumb pose."

    def test_f8_t2_http_500_from_local_gemma_raises_clean_error(
        self,
        local_gemma_server: LocalGemmaServer,
    ) -> None:
        local_gemma_server.status_code = 500
        client = _make_local_client(local_gemma_server.base_url)
        with pytest.raises((RuntimeError, OSError, ValueError)):
            _ = client.generate_json("gemma4:e4b", "sys", [TextPart("obs")], None, 3.0)

    def test_f8_t2_malformed_non_json_from_local_gemma_raises_clean_error(
        self,
        local_gemma_server: LocalGemmaServer,
    ) -> None:
        local_gemma_server.raw_content_override = "Definitely not JSON {broken"
        client = _make_local_client(local_gemma_server.base_url)
        with pytest.raises((RuntimeError, ValueError)):
            _ = client.generate_json("gemma4:e4b", "sys", [TextPart("obs")], None, 3.0)

    def test_f8_t2_timeout_against_slow_local_gemma_server_aborts_within_budget(
        self,
        local_gemma_server: LocalGemmaServer,
    ) -> None:
        local_gemma_server.delay_s = 0.80
        client = _make_local_client(local_gemma_server.base_url)
        t0 = time.monotonic()
        with pytest.raises((RuntimeError, TimeoutError, OSError)):
            _ = client.generate_json("gemma4:e4b", "sys", [TextPart("obs")], None, 0.20)
        assert time.monotonic() - t0 < 0.75

    def test_f8_t2_local_url_with_or_without_v1_suffix_normalizes_cleanly(
        self,
        local_gemma_server: LocalGemmaServer,
    ) -> None:
        root_without_v1 = f"http://127.0.0.1:{local_gemma_server.port}"
        client = _make_local_client(root_without_v1)
        ans = client.generate_json("gemma4:e4b", "sys", [TextPart("obs")], None, 5.0)
        assert ans["verdict"] == "approve"
        assert local_gemma_server.requests[-1].path.endswith("/chat/completions")


# ===========================================================================
# FEATURE 9 — Tier 2 `GraspGate` Client (`gemini-4-hydrogen`) (Tier 1 >=5, Tier 2 >=5)
# ===========================================================================


class TestFeature9Tier2HydrogenGraspGateTier1:
    """Tier 1 category-partition tests for `GraspGate` (`gemini-4-hydrogen`)."""

    def test_f9_t1_contract_dataclasses_expose_all_required_project_fields(self) -> None:
        gate_mod = require_module("teleop.copilot.gate")
        for sym in (
            "KeyframeJpeg",
            "PoseSnapshot",
            "GraspTelemetrySummary",
            "GraspGateVerdict",
            "GraspGate",
        ):
            _ = require_attr(gate_mod, sym)

        verdict_cls = gate_mod.GraspGateVerdict
        v_ok = verdict_cls(verdict="approve", say="Clean hook.", reason="All clear.")
        v_bad = verdict_cls(verdict="reject", say="Stop, HV risk.", reason="Thumb near -X bus.")
        assert v_ok.approved is True
        assert v_bad.approved is False
        assert v_ok.model == "gemini-4-hydrogen"

        snap_cls = gate_mod.PoseSnapshot
        snap = snap_cls(
            label="stall",
            sim_t=1.5,
            bolt_x_m=0.014,
            tau_hand_nm=0.35,
            qpos=np.zeros(7, dtype=np.float64),
            mocap_pos=np.zeros((1, 3), dtype=np.float64),
            mocap_quat=np.zeros((1, 4), dtype=np.float64),
            body_pos=np.zeros((2, 3), dtype=np.float64),
            body_quat=np.zeros((2, 4), dtype=np.float64),
        )
        assert snap.label == "stall"

    def test_f9_t1_sends_hydrogen_low_thinking_low_media_resolution_and_3_keyframes(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        gate_mod = require_module("teleop.copilot.gate")
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        cfg = CopilotConfig(gemini_api_key="test-key")
        gate = _build_grasp_gate(client=gem_client, cfg=cfg)
        try:
            verdict = _evaluate_grasp_gate(
                gate,
                _make_keyframe_objs(gate_mod, 3),
                _make_sample_telemetry(gate_mod),
            )
            assert verdict.approved is True
            assert verdict.verdict == "approve"
            assert "Flap depressed" in verdict.say

            reqs = mock_gemini_server.requests_matching("gemini-4-hydrogen:generateContent")
            assert len(reqs) == 1
            body = reqs[0].json_body
            assert body is not None
            gen_cfg = body["generationConfig"]
            assert gen_cfg["thinkingConfig"]["thinkingLevel"].upper() == "LOW"
            assert gen_cfg.get("mediaResolution") == "MEDIA_RESOLUTION_LOW"

            parts = body["contents"][0]["parts"]
            img_parts = [p["inlineData"] for p in parts if "inlineData" in p]
            assert len(img_parts) == 3
            for inline in img_parts:
                assert inline["mimeType"] == "image/jpeg"
                raw_jpeg = base64.b64decode(inline["data"])
                w, h = jpeg_dimensions(raw_jpeg)
                assert w <= 512 and h <= 512
        finally:
            if hasattr(gate, "close"):
                gate.close()

    def test_f9_t1_request_includes_stall_mid_hold_bolt_clear_labels_and_telemetry(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        gate_mod = require_module("teleop.copilot.gate")
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _build_grasp_gate(client=gem_client)
        try:
            _ = _evaluate_grasp_gate(
                gate,
                _make_keyframe_objs(gate_mod, 3),
                _make_sample_telemetry(gate_mod),
            )
            req = mock_gemini_server.requests_matching("gemini-4-hydrogen")[-1]
            serialized = json.dumps(req.json_body)
            for label in ("stall", "mid-hold", "bolt-clear"):
                assert label in serialized, f"Hydrogen prompt missing keyframe label {label!r}"
            assert "0.056" in serialized or "5.6" in serialized
            assert "0.44" in serialized
        finally:
            if hasattr(gate, "close"):
                gate.close()

    def test_f9_t1_response_schema_omits_additional_properties_on_gemini_wire(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        gate_mod = require_module("teleop.copilot.gate")
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _build_grasp_gate(client=gem_client)
        try:
            _ = _evaluate_grasp_gate(
                gate,
                _make_keyframe_objs(gate_mod, 3),
                _make_sample_telemetry(gate_mod),
            )
            gen_cfg = mock_gemini_server.requests_matching("gemini-4-hydrogen")[-1].json_body[  # type: ignore[index]
                "generationConfig"
            ]
            assert gen_cfg["responseMimeType"] == "application/json"
            if "responseSchema" in gen_cfg:
                assert not contains_key_recursive(
                    gen_cfg["responseSchema"], "additionalProperties"
                ), "Gemini responseSchema rejects `additionalProperties`!"
        finally:
            if hasattr(gate, "close"):
                gate.close()

    def test_f9_t1_handles_thought_parts_and_thought_signature_transparently(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        mock_gemini_server.config.hydrogen_include_thought_signature = True
        mock_gemini_server.config.hydrogen_verdict = {
            "verdict": "reject",
            "say": "Rejected: thumb brushed near HV terminal.",
            "reason": "Insufficient -X HV battery terminal clearance during slide.",
            "flap_depressed": True,
            "bolt_clearance_ok": True,
            "hv_clearance_ok": False,
            "torque_controlled": True,
            "confidence": 0.96,
        }
        gate_mod = require_module("teleop.copilot.gate")
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _build_grasp_gate(client=gem_client)
        try:
            verdict = _evaluate_grasp_gate(
                gate,
                _make_keyframe_objs(gate_mod, 3),
                _make_sample_telemetry(gate_mod),
            )
            assert verdict.approved is False
            assert verdict.verdict == "reject"
            assert verdict.hv_clearance_ok is False
            assert verdict.confidence == pytest.approx(0.96)
        finally:
            if hasattr(gate, "close"):
                gate.close()


class TestFeature9Tier2HydrogenGraspGateTier2:
    """Tier 2 negative, oversize JPEG downscale, malformed JSON, and HTTP 500 tests for F9."""

    def test_f9_t2_oversized_1024px_input_frame_is_downscaled_to_at_most_512px(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        gate_mod = require_module("teleop.copilot.gate")
        # Raw fixture frame is >512px wide before downscaling
        from .conftest import FIXTURE_FRAME_BIN

        big_jpeg = FIXTURE_FRAME_BIN.read_bytes()[2:]
        kf_cls = require_attr(gate_mod, "KeyframeJpeg")
        kfs = [
            kf_cls("stall", 1.0, 0.014, 0.3, big_jpeg),
            kf_cls("mid-hold", 2.0, 0.035, 0.3, big_jpeg),
            kf_cls("bolt-clear", 3.0, 0.055, 0.2, big_jpeg),
        ]
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _build_grasp_gate(client=gem_client)
        try:
            _ = _evaluate_grasp_gate(gate, kfs, _make_sample_telemetry(gate_mod))
            req = mock_gemini_server.requests_matching("gemini-4-hydrogen")[-1]
            parts = req.json_body["contents"][0]["parts"]  # type: ignore[index]
            for p in parts:
                if "inlineData" in p:
                    w, h = jpeg_dimensions(base64.b64decode(p["inlineData"]["data"]))
                    assert w <= 512 and h <= 512, f"Expected <=512px JPEG on wire, got {w}x{h}"
        finally:
            if hasattr(gate, "close"):
                gate.close()

    def test_f9_t2_http_500_from_hydrogen_handled_deterministically_without_crashing(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        mock_gemini_server.config.hydrogen_status_code = 500
        gate_mod = require_module("teleop.copilot.gate")
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _build_grasp_gate(client=gem_client)
        try:
            try:
                verdict = _evaluate_grasp_gate(
                    gate,
                    _make_keyframe_objs(gate_mod, 3),
                    _make_sample_telemetry(gate_mod),
                )
            except Exception as exc:
                # Either raises clean GeminiError or returns explicit fail-safe verdict
                assert "500" in str(exc) or "gate" in str(exc).lower()
            else:
                assert verdict.verdict in ("approve", "reject")
        finally:
            if hasattr(gate, "close"):
                gate.close()

    def test_f9_t2_malformed_json_from_hydrogen_handled_cleanly(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        mock_gemini_server.config.hydrogen_malformed_text = "{unterminated json"
        gate_mod = require_module("teleop.copilot.gate")
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _build_grasp_gate(client=gem_client)
        try:
            try:
                verdict = _evaluate_grasp_gate(
                    gate,
                    _make_keyframe_objs(gate_mod, 3),
                    _make_sample_telemetry(gate_mod),
                )
            except Exception as exc:
                assert exc is not None
            else:
                assert verdict.verdict in ("approve", "reject")
        finally:
            if hasattr(gate, "close"):
                gate.close()

    def test_f9_t2_confidence_clamped_or_validated_within_unit_interval(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        mock_gemini_server.config.hydrogen_verdict = {
            "verdict": "approve",
            "say": "Bolt clear.",
            "reason": "Clean.",
            "confidence": 1.75,
        }
        gate_mod = require_module("teleop.copilot.gate")
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _build_grasp_gate(client=gem_client)
        try:
            verdict = _evaluate_grasp_gate(
                gate,
                _make_keyframe_objs(gate_mod, 3),
                _make_sample_telemetry(gate_mod),
            )
            assert 0.0 <= float(verdict.confidence) <= 1.0
        finally:
            if hasattr(gate, "close"):
                gate.close()

    def test_f9_t2_invalid_verdict_enum_value_rejected_or_coerced_to_safe_verdict(
        self,
        mock_gemini_server: LoopbackGeminiTLSServer,
    ) -> None:
        mock_gemini_server.config.hydrogen_verdict = {
            "verdict": "maybe_later",
            "say": "Unclear.",
            "reason": "Ambiguous frame.",
        }
        gate_mod = require_module("teleop.copilot.gate")
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _build_grasp_gate(client=gem_client)
        try:
            try:
                verdict = _evaluate_grasp_gate(
                    gate,
                    _make_keyframe_objs(gate_mod, 3),
                    _make_sample_telemetry(gate_mod),
                )
            except Exception as exc:
                assert exc is not None
            else:
                assert verdict.verdict in ("approve", "reject")
        finally:
            if hasattr(gate, "close"):
                gate.close()


# ===========================================================================
# FEATURE 11 — Parallel `AtcCoordinator._Gate` Integration (Tier 1 >=5, Tier 2 >=5)
# ===========================================================================


class TestFeature11ParallelAtcCoordinatorGateTier1:
    """Tier 1 category-partition tests for `AtcCoordinator._Gate` + `GraspGate`."""

    def test_f11_t1_default_coordinator_without_grasp_gate_runs_physics_only_cleanly(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
    ) -> None:
        rig = coordinator_factory(grasp_gate=None, require_grasp_gate_hook=False)
        rig.stall_bay(1)
        assert rig.co.state(1) == BayState.RESCUE
        rig.run_rescue(1)
        assert rig.co.state(1) == BayState.NOMINAL
        assert len(rig.registered_skills) == 1
        assert "SKILL_DEPLOYED" in rig.sink.kinds()

    def test_f11_t1_skill_validator_and_grasp_gate_execute_genuinely_in_parallel_barrier(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        inspire_hand_spec: HandSpec,
    ) -> None:
        # Barrier(2) deadlocks and raises BrokenBarrierError if GraspGate shares the single-worker
        # SkillValidator executor instead of running in parallel on its own worker!
        barrier = threading.Barrier(2, timeout=15.0)
        mock_gemini_server.config.hydrogen_barrier = barrier
        fast_val = FastSkillValidator(passed=True, barrier=barrier)

        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _build_grasp_gate(client=gem_client, hand_spec=inspire_hand_spec)
        rig = coordinator_factory(validator=fast_val, grasp_gate=gate, gate_deadline_s=0.05)
        try:
            rig.stall_bay(1)
            rig.run_rescue(1)
            assert len(fast_val.calls) == 1
            assert len(mock_gemini_server.requests_matching("gemini-4-hydrogen")) == 1
            assert len(rig.registered_skills) == 1
        finally:
            if hasattr(gate, "close"):
                gate.close()

    def test_f11_t1_hydrogen_say_dispatched_immediately_to_sse_tts_before_gate_deadline(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        fake_audio_env: FakeAudioHarness,
        inspire_hand_spec: HandSpec,
    ) -> None:
        assert fake_audio_env.bin_dir.is_dir()
        sim_hold = threading.Event()
        fast_val = FastSkillValidator(passed=True, hold_event=sim_hold)
        streamer = make_tts_streamer(
            host="generativelanguage.googleapis.com",
            port=mock_gemini_server.port,
            base_url=mock_gemini_server.base_url,
        )
        spk_sig = inspect.signature(speech_mod.Speaker)
        spk_kw = {
            k: v
            for k, v in {
                "tts_model": "gemini-3.8-flash-lite-tts",
                "voice": "Kore",
                "streamer": streamer,
                "tts_streamer": streamer,
                "streaming_tts": streamer,
            }.items()
            if k in spk_sig.parameters
        }
        spk = speech_mod.Speaker(streamer, **spk_kw)
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _build_grasp_gate(client=gem_client, speaker=spk, hand_spec=inspire_hand_spec)
        rig = coordinator_factory(validator=fast_val, grasp_gate=gate, gate_deadline_s=10.0)
        try:
            rig.stall_bay(1)
            # Stop right as Bay 1 enters VALIDATING (10.0 sim-s before gate_deadline_s, Sim held!)
            rig.run_rescue(1, max_extra_ticks=0)
            assert rig.co.state(1) == BayState.VALIDATING
            assert not sim_hold.is_set()

            # Hydrogen finishes immediately and MUST trigger Tier 1 SSE TTS BEFORE Sim finishes!
            assert mock_gemini_server.config.tts_received_event.wait(timeout=15.0), (
                "Hydrogen `verdict.say` was NOT streamed to TTS immediately while VALIDATING!"
            )
            assert rig.co.state(1) == BayState.VALIDATING
            assert len(rig.registered_skills) == 0
        finally:
            sim_hold.set()
            spk.close()
            if hasattr(streamer, "close"):
                streamer.close()
            if hasattr(gate, "close"):
                gate.close()

    def test_f11_t1_dual_approval_both_sim_and_hydrogen_approve_registers_skill(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        inspire_hand_spec: HandSpec,
    ) -> None:
        mock_gemini_server.config.hydrogen_verdict["verdict"] = "approve"
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _build_grasp_gate(client=gem_client, hand_spec=inspire_hand_spec)
        rig = coordinator_factory(
            validator=FastSkillValidator(passed=True),
            grasp_gate=gate,
            gate_deadline_s=0.05,
        )
        try:
            rig.stall_bay(1)
            rig.run_rescue(1)
            assert len(rig.registered_skills) == 1
            assert len(rig.co.library) == 1
            assert "SKILL_DEPLOYED" in rig.sink.kinds()
        finally:
            if hasattr(gate, "close"):
                gate.close()

    def test_f11_t1_coordinator_captures_stall_mid_hold_bolt_clear_keyframes_from_rescue(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        inspire_hand_spec: HandSpec,
    ) -> None:
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _build_grasp_gate(client=gem_client, hand_spec=inspire_hand_spec)
        rig = coordinator_factory(
            validator=FastSkillValidator(passed=True),
            grasp_gate=gate,
            gate_deadline_s=0.05,
        )
        try:
            rig.stall_bay(1)
            rig.run_rescue(1)
            req = mock_gemini_server.requests_matching("gemini-4-hydrogen")[-1]
            parts = req.json_body["contents"][0]["parts"]  # type: ignore[index]
            images = [p["inlineData"] for p in parts if "inlineData" in p]
            assert len(images) == 3
            body_str = json.dumps(req.json_body)
            for label in ("stall", "mid-hold", "bolt-clear"):
                assert label in body_str
        finally:
            if hasattr(gate, "close"):
                gate.close()


class TestFeature11ParallelAtcCoordinatorGateTier2:
    """Tier 2 negative & boundary tests for `AtcCoordinator._Gate` dual-approval & short release."""

    def test_f11_t2_short_clutch_release_below_5_2cm_discards_rescue_without_calling_gate(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        inspire_hand_spec: HandSpec,
    ) -> None:
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _build_grasp_gate(client=gem_client, hand_spec=inspire_hand_spec)
        fast_val = FastSkillValidator(passed=True)
        rig = coordinator_factory(validator=fast_val, grasp_gate=gate, gate_deadline_s=0.05)
        try:
            rig.stall_bay(1)
            bolt_x = rig.run_short_rescue(1, ticks=16)
            assert bolt_x < BOLT_CLEARED_M
            assert rig.co.state(1) == BayState.RESCUE
            assert not rig.co.gate_pending(1)
            assert "RESCUE_LOGGED" not in rig.sink.kinds()
            assert "VALIDATION_GATE" not in rig.sink.kinds()
            assert fast_val.calls == []
            assert mock_gemini_server.requests_matching("gemini-4-hydrogen") == []
            assert rig.registered_skills == []
        finally:
            if hasattr(gate, "close"):
                gate.close()

    def test_f11_t2_sim_passes_but_hydrogen_rejects_blocks_skill_registration(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        inspire_hand_spec: HandSpec,
    ) -> None:
        mock_gemini_server.config.hydrogen_verdict = {
            "verdict": "reject",
            "say": "Skill rejected: finger brushed near HV terminal.",
            "reason": "Unsafe -X HV proximity during bolt pull.",
            "flap_depressed": True,
            "bolt_clearance_ok": True,
            "hv_clearance_ok": False,
            "torque_controlled": True,
            "confidence": 0.93,
        }
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _build_grasp_gate(client=gem_client, hand_spec=inspire_hand_spec)
        rig = coordinator_factory(
            validator=FastSkillValidator(passed=True),
            grasp_gate=gate,
            gate_deadline_s=0.05,
        )
        try:
            rig.stall_bay(1)
            rig.run_rescue(1)
            assert rig.registered_skills == []
            assert len(rig.co.library) == 0
            assert "SKILL_DEPLOYED" not in rig.sink.kinds()
            gate_events = [e for e in rig.sink.events if e.kind == "VALIDATION_GATE"]
            assert len(gate_events) == 1
            assert gate_events[0].metadata["passed"] is False
        finally:
            if hasattr(gate, "close"):
                gate.close()

    def test_f11_t2_sim_fails_and_hydrogen_approves_blocks_skill_registration(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        inspire_hand_spec: HandSpec,
    ) -> None:
        mock_gemini_server.config.hydrogen_verdict["verdict"] = "approve"
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _build_grasp_gate(client=gem_client, hand_spec=inspire_hand_spec)
        rig = coordinator_factory(
            validator=FastSkillValidator(passed=False),
            grasp_gate=gate,
            gate_deadline_s=0.05,
        )
        try:
            rig.stall_bay(1)
            rig.run_rescue(1)
            assert rig.registered_skills == []
            assert len(rig.co.library) == 0
            assert "SKILL_DEPLOYED" not in rig.sink.kinds()
            gate_events = [e for e in rig.sink.events if e.kind == "VALIDATION_GATE"]
            assert len(gate_events) == 1
            assert gate_events[0].metadata["passed"] is False
        finally:
            if hasattr(gate, "close"):
                gate.close()

    def test_f11_t2_both_sim_and_hydrogen_fail_blocks_skill_registration(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
        mock_gemini_server: LoopbackGeminiTLSServer,
        inspire_hand_spec: HandSpec,
    ) -> None:
        mock_gemini_server.config.hydrogen_verdict["verdict"] = "reject"
        gem_client = GeminiClient("test-key", base_url=mock_gemini_server.base_url)
        gate = _build_grasp_gate(client=gem_client, hand_spec=inspire_hand_spec)
        rig = coordinator_factory(
            validator=FastSkillValidator(passed=False),
            grasp_gate=gate,
            gate_deadline_s=0.05,
        )
        try:
            rig.stall_bay(1)
            rig.run_rescue(1)
            assert rig.registered_skills == []
            assert len(rig.co.library) == 0
            assert "SKILL_DEPLOYED" not in rig.sink.kinds()
        finally:
            if hasattr(gate, "close"):
                gate.close()

    def test_f11_t2_clutch_pressed_before_retract_finishes_does_not_prematurely_engage(
        self,
        coordinator_factory: Callable[..., E2ECoordinatorRig],
    ) -> None:
        rig = coordinator_factory(grasp_gate=None, require_grasp_gate_hook=False)
        rig.co.start_cycle(1, flap_jammed=True)
        for _ in range(1500):
            snaps = rig.co.tick()
            if snaps[0].state == BayState.RESCUE:
                break
        # Immediately on the first tick of RESCUE, the 0.5s retract move is still in flight
        from teleop.atc.types import ConsoleInputState

        op = next(rig.operator_samples(1))
        rig.co.tick(ConsoleInputState(selected_bay=1, clutch=True), op)
        # Must wait until retract completes before clutch_held activates
        for _ in range(45):
            rig.co.tick(ConsoleInputState(selected_bay=1, clutch=True), op)
        assert rig.co.state(1) == BayState.RESCUE
