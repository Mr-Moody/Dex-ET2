"""Tier 2 inline grasp gate: Gemini 4 Hydrogen checks a rescue before its skill is deployed.

The coordinator captures cheap :class:`PoseSnapshot` copies during a clutch-held rescue. On a
release that clears the bolt it hands three of them (``stall``, ``mid-hold``, ``bolt-clear``) and
a :class:`GraspTelemetrySummary` to :meth:`GraspGate.submit`, next to the randomized
``SkillValidator`` rollouts. The gate renders the snapshots on its own worker thread (a private
``MjData`` and ``mujoco.Renderer``, never the 60 Hz loop's), asks the fast model for a structured
verdict, speaks the verdict's ``say`` line at once and returns a :class:`GraspGateVerdict`. The
coordinator deploys the skill only when the rollouts and the verdict both approve.

Fallbacks: a :class:`~teleop.copilot.local.LocalModelClient` answers with ``source="local"``; no
client at all gives a physics-only approval (``source="physics_only"``) without any network call;
a model error (HTTP, timeout, malformed JSON) also falls back to physics-only with
``source="error"``, so an unreachable model never blocks a skill the rollouts approved.

The gate never sends console keys or commands a robot: its only outputs are the verdict, a spoken
line and (optionally) a fire-and-forget Tier 3 audit submission.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from teleop.copilot.config import CopilotConfig
from teleop.copilot.frames import MAX_SIDE_PX, downscale_jpeg
from teleop.copilot.gemini import GeminiClient, GeminiError, ImagePart, Part, TextPart
from teleop.copilot.local import LocalModelClient
from teleop.copilot.prompts import GATE_SCHEMA, GATE_SYSTEM, SAY_MAX_WORDS

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
GateSource = Literal["gemini", "local", "physics_only", "error"]

GATE_TIMEOUT_S = 8.0
MEDIA_RESOLUTION = "MEDIA_RESOLUTION_LOW"
KEYFRAME_LABELS = ("stall", "mid-hold", "bolt-clear")
RENDER_HW = (384, 512)  # height, width: inside the 512 px budget
JPEG_QUALITY = 85


# --------------------------------------------------------------------------- contracts


@dataclass(frozen=True)
class PoseSnapshot:
    """The sim state at one rescue instant: a few numpy copies, cheap on the 60 Hz thread."""

    label: str
    sim_t: float
    bolt_x_m: float
    tau_hand_nm: float
    qpos: FloatArray
    mocap_pos: FloatArray
    mocap_quat: FloatArray
    body_pos: FloatArray
    body_quat: FloatArray


@dataclass(frozen=True)
class KeyframeJpeg:
    """One rendered rescue keyframe (at most 512 px on its long side once sent)."""

    label: str
    sim_t: float
    bolt_x_m: float
    tau_hand_nm: float
    jpeg_bytes: bytes


@dataclass(frozen=True)
class GraspTelemetrySummary:
    """Contact torque, bolt travel and latch clearance of one rescue, for Tier 2 and Tier 3."""

    bay_id: int
    sku: str
    skill_id: str
    signature_id: str
    duration_s: float
    bolt_start_m: float
    bolt_final_m: float
    peak_tau_hand_nm: float
    mean_tau_hand_nm: float
    peak_arm_ratio: float
    min_wrist_x_latch_m: float  # clearance towards the -X HV terminals
    min_wrist_z_latch_m: float  # flap depression geometry
    touch_offset_m: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bay_id": self.bay_id,
            "sku": self.sku,
            "skill_id": self.skill_id,
            "signature_id": self.signature_id,
            "duration_s": round(self.duration_s, 3),
            "bolt_start_m": round(self.bolt_start_m, 4),
            "bolt_final_m": round(self.bolt_final_m, 4),
            "peak_tau_hand_nm": round(self.peak_tau_hand_nm, 3),
            "mean_tau_hand_nm": round(self.mean_tau_hand_nm, 3),
            "peak_arm_ratio": round(self.peak_arm_ratio, 3),
            "min_wrist_x_latch_m": round(self.min_wrist_x_latch_m, 4),
            "min_wrist_z_latch_m": round(self.min_wrist_z_latch_m, 4),
            "touch_offset_m": None
            if self.touch_offset_m is None
            else [round(v, 4) for v in self.touch_offset_m],
        }


@dataclass(frozen=True)
class GraspGateVerdict:
    """The fast model's physical verdict on one rescue."""

    verdict: Literal["approve", "reject"]
    say: str
    reason: str
    flap_depressed: bool | None = None
    bolt_clearance_ok: bool | None = None
    hv_clearance_ok: bool | None = None
    torque_controlled: bool | None = None
    confidence: float | None = None
    latency_ms: float = 0.0
    model: str = "gemini-4-hydrogen"
    source: GateSource = "gemini"

    @property
    def approved(self) -> bool:
        return self.verdict == "approve"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "say": self.say,
            "reason": self.reason,
            "flap_depressed": self.flap_depressed,
            "bolt_clearance_ok": self.bolt_clearance_ok,
            "hv_clearance_ok": self.hv_clearance_ok,
            "torque_controlled": self.torque_controlled,
            "confidence": self.confidence,
            "latency_ms": round(self.latency_ms, 1),
            "model": self.model,
            "source": self.source,
        }


ModelClient = GeminiClient | LocalModelClient


class SpeakerLike(Protocol):
    def say(self, text: str, *, severity: Any = ..., bay: int | None = ...) -> int: ...


class AuditSubmitter(Protocol):
    def submit(
        self,
        *,
        bay: int,
        skill_id: str,
        episode_uid: str | None,
        frames: Sequence[KeyframeJpeg],
        timeline: Mapping[str, Any],
    ) -> str: ...


# --------------------------------------------------------------------------- rendering


class SnapshotRenderer:
    """Renders :class:`PoseSnapshot` s of one model into JPEGs. Use from one thread only."""

    def __init__(self, model: Any, hw: tuple[int, int] = RENDER_HW) -> None:
        import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]

        self._mj: Any = _mujoco
        self.model = model
        self._data: Any = self._mj.MjData(model)
        self._hw = hw
        self._renderer: Any = None

    def render(self, snap: PoseSnapshot, bay_id: int) -> bytes:
        from teleop.policy.schema import camera_name

        mj = self._mj
        if self._renderer is None:
            self._renderer = mj.Renderer(self.model, self._hw[0], self._hw[1])
        d = self._data
        d.qpos[:] = snap.qpos
        if snap.mocap_pos.size:
            d.mocap_pos[:] = snap.mocap_pos
            d.mocap_quat[:] = snap.mocap_quat
        mj.mj_forward(self.model, d)
        self._renderer.update_scene(d, camera_name(bay_id, "top"))
        rgb = np.asarray(self._renderer.render(), dtype=np.uint8)
        return encode_jpeg(rgb)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def encode_jpeg(rgb: NDArray[np.uint8], quality: int = JPEG_QUALITY) -> bytes:
    from PIL import Image

    img = Image.fromarray(rgb, mode="RGB")
    img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# --------------------------------------------------------------------------- the gate


class GraspGate:
    """Tier 2 grasp gate on its own worker thread (never the coordinator's rollout worker)."""

    def __init__(
        self,
        client: ModelClient | None = None,
        *,
        cfg: CopilotConfig | None = None,
        speaker: SpeakerLike | None = None,
        audit_reviewer: AuditSubmitter | None = None,
        timeout_s: float = GATE_TIMEOUT_S,
    ) -> None:
        self.cfg = cfg or CopilotConfig()
        self.client = client
        self.speaker = speaker
        self.audit_reviewer = audit_reviewer
        self.timeout_s = timeout_s
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="grasp-gate"
        )
        self._renderer: SnapshotRenderer | None = None
        self._lock = threading.Lock()

    @property
    def source(self) -> GateSource:
        if self.client is None:
            return "physics_only"
        return "local" if isinstance(self.client, LocalModelClient) else "gemini"

    @property
    def model(self) -> str:
        if isinstance(self.client, LocalModelClient):
            return self.cfg.local_model
        return self.cfg.fast_model if self.client is not None else "physics_only"

    # -- async (coordinator) ----------------------------------------------------------------

    def submit(
        self,
        summary: GraspTelemetrySummary,
        snapshots: Sequence[PoseSnapshot],
        *,
        model: Any = None,
        audit_snapshots: Sequence[PoseSnapshot] = (),
        episode_uid: str | None = None,
    ) -> Future[GraspGateVerdict]:
        """Render ``snapshots`` with ``model`` and evaluate them on the gate's worker."""
        with self._lock:
            executor = self._executor
            if executor is None:
                raise RuntimeError("grasp gate is closed")
            return executor.submit(
                self._run, summary, tuple(snapshots), model, tuple(audit_snapshots), episode_uid
            )

    def _run(
        self,
        summary: GraspTelemetrySummary,
        snapshots: tuple[PoseSnapshot, ...],
        model: Any,
        audit_snapshots: tuple[PoseSnapshot, ...],
        episode_uid: str | None,
    ) -> GraspGateVerdict:
        keyframes = self._render(snapshots, summary.bay_id, model)
        audit_frames = self._render(audit_snapshots, summary.bay_id, model)
        verdict = self.evaluate(keyframes, summary)
        if self.audit_reviewer is not None:
            try:
                self.audit_reviewer.submit(
                    bay=summary.bay_id,
                    skill_id=summary.skill_id,
                    episode_uid=episode_uid,
                    frames=audit_frames or keyframes,
                    timeline={**summary.to_dict(), "gate": verdict.to_dict()},
                )
            except Exception:
                logger.exception("grasp gate: audit submission failed")
        return verdict

    def _render(
        self, snapshots: tuple[PoseSnapshot, ...], bay_id: int, model: Any
    ) -> list[KeyframeJpeg]:
        if model is None or not snapshots or self.cfg.no_images or self.client is None:
            return []
        try:
            if self._renderer is None or self._renderer.model is not model:
                if self._renderer is not None:
                    self._renderer.close()
                self._renderer = SnapshotRenderer(model)
            return [
                KeyframeJpeg(
                    s.label, s.sim_t, s.bolt_x_m, s.tau_hand_nm, self._renderer.render(s, bay_id)
                )
                for s in snapshots
            ]
        except Exception:
            logger.exception("grasp gate: keyframe rendering failed, sending telemetry only")
            return []

    # -- sync -------------------------------------------------------------------------------

    def evaluate(
        self, keyframes: Sequence[KeyframeJpeg], telemetry: GraspTelemetrySummary
    ) -> GraspGateVerdict:
        """Ask the model for a verdict on ``keyframes`` + ``telemetry`` and speak it."""
        if self.client is None:
            return GraspGateVerdict(
                "approve", "", "no model configured: physics gate only",
                model="physics_only", source="physics_only",
            )  # fmt: skip
        t0 = time.monotonic()
        parts = gate_parts(keyframes, telemetry)
        try:
            if isinstance(self.client, GeminiClient):
                answer = self.client.generate_json(
                    self.model, GATE_SYSTEM, parts, GATE_SCHEMA, self.timeout_s,
                    thinking_level=self.cfg.fast_thinking, media_resolution=MEDIA_RESOLUTION,
                )  # fmt: skip
            else:
                answer = self.client.generate_json(
                    self.model, GATE_SYSTEM, parts, GATE_SCHEMA, self.timeout_s,
                    thinking_level=self.cfg.fast_thinking,
                )  # fmt: skip
        except (GeminiError, OSError, TimeoutError, ValueError) as exc:
            logger.warning("grasp gate: model unavailable (%s), physics gate only", exc)
            return GraspGateVerdict(
                "approve", "", f"model unavailable: {exc}"[:200],
                latency_ms=(time.monotonic() - t0) * 1000.0, model=self.model, source="error",
            )  # fmt: skip
        verdict = parse_verdict(
            answer, latency_ms=(time.monotonic() - t0) * 1000.0, model=self.model,
            source=self.source,
        )  # fmt: skip
        self._speak(verdict, telemetry.bay_id)
        return verdict

    def _speak(self, verdict: GraspGateVerdict, bay_id: int) -> None:
        if self.speaker is None or not verdict.say:
            return
        try:
            self.speaker.say(
                verdict.say, severity="info" if verdict.approved else "alarm", bay=bay_id
            )
        except Exception:
            logger.exception("grasp gate: speaking the verdict failed")

    def close(self) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.submit(self._close_renderer)
            executor.shutdown(wait=True, cancel_futures=False)

    def _close_renderer(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# --------------------------------------------------------------------------- prompt + parsing


def gate_parts(keyframes: Sequence[KeyframeJpeg], telemetry: GraspTelemetrySummary) -> list[Part]:
    """Telemetry text, then each keyframe as a label line and a <=512 px JPEG."""
    t = telemetry
    lines = [
        f"Bay {t.bay_id}, SKU {t.sku}, skill {t.skill_id}, stall signature {t.signature_id}.",
        f"Rescue duration {t.duration_s:.2f} s.",
        f"Bolt travel {t.bolt_start_m:.3f} m -> {t.bolt_final_m:.3f} m "
        f"({t.bolt_final_m * 100:.1f} cm; clear at 5.2 cm).",
        f"Hand contact torque peak {t.peak_tau_hand_nm:.2f} N m, "
        f"mean {t.mean_tau_hand_nm:.2f} N m.",
        f"Peak arm torque ratio {t.peak_arm_ratio:.2f}.",
        f"Closest wrist approach in the latch frame: x {t.min_wrist_x_latch_m:.3f} m "
        f"(-X is the HV terminals), z {t.min_wrist_z_latch_m:.3f} m (flap depth).",
        f"Telemetry JSON: {t.to_dict()}",
    ]
    parts: list[Part] = [TextPart("\n".join(lines))]
    for kf in keyframes:
        parts.append(
            TextPart(
                f"Keyframe '{kf.label}' at t={kf.sim_t:.2f} s, bolt {kf.bolt_x_m:.3f} m, "
                f"hand torque {kf.tau_hand_nm:.2f} N m:"
            )
        )
        parts.append(ImagePart(downscale_jpeg(kf.jpeg_bytes)))
    return parts


def _bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _clip_words(text: str, limit: int) -> str:
    words = text.split()
    return " ".join(words[:limit])


def parse_verdict(
    answer: Mapping[str, Any], *, latency_ms: float, model: str, source: GateSource
) -> GraspGateVerdict:
    """A :class:`GraspGateVerdict` from the model's JSON. Anything but ``approve`` rejects."""
    raw = answer.get("verdict")
    verdict: Literal["approve", "reject"] = (
        "approve" if isinstance(raw, str) and raw.strip().lower() == "approve" else "reject"
    )
    conf_raw = answer.get("confidence")
    confidence = (
        min(max(float(conf_raw), 0.0), 1.0)
        if isinstance(conf_raw, int | float) and not isinstance(conf_raw, bool)
        else None
    )
    say = answer.get("say")
    reason = answer.get("reason")
    return GraspGateVerdict(
        verdict=verdict,
        say=_clip_words(say, SAY_MAX_WORDS) if isinstance(say, str) else "",
        reason=reason.strip() if isinstance(reason, str) else "",
        flap_depressed=_bool(answer.get("flap_depressed")),
        bolt_clearance_ok=_bool(answer.get("bolt_clearance_ok")),
        hv_clearance_ok=_bool(answer.get("hv_clearance_ok")),
        torque_controlled=_bool(answer.get("torque_controlled")),
        confidence=confidence,
        latency_ms=latency_ms,
        model=model,
        source=source,
    )


def build_grasp_gate(
    cfg: CopilotConfig,
    *,
    use_gemini: bool = True,
    speaker: SpeakerLike | None = None,
    audit_reviewer: AuditSubmitter | None = None,
) -> GraspGate:
    """Gemini when a key is set (and allowed), else local Gemma when configured, else physics."""
    client: ModelClient | None = None
    if use_gemini and cfg.has_key and cfg.gemini_api_key:
        client = GeminiClient(cfg.gemini_api_key, no_images=cfg.no_images)
    else:
        client = LocalModelClient.from_config(cfg)
    return GraspGate(client, cfg=cfg, speaker=speaker, audit_reviewer=audit_reviewer)
