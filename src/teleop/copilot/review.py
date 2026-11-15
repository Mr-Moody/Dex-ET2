"""Tier 3 silent deep audit: Gemini 4 Argon reviews a finished rescue for the ACT dataset.

After the Tier 2 grasp gate has judged a rescue, it hands :meth:`AuditReviewer.submit` the rescue
keyframes and a telemetry timeline. ``submit`` returns an ``audit_id`` at once and reports a
``pending`` :class:`AuditReport`; one background worker then asks the review model (default
``gemini-4-argon``, ``thinkingLevel="LOW"``) for a structured verdict on up to six ``<= 512 px``
frames and reports ``complete``, ``skipped`` (no model, or the hourly budget
``COPILOT_MAX_REVIEWS_PER_HOUR`` is spent) or ``error``.

The audit is strictly silent and never blocks anything: it makes no TTS call (a ``speaker`` passed
in is ignored), never waits on the gate, the coordinator or ``register_skill``, and its only
outputs are the ``on_update`` callback (the console's ``#audit-panel`` card) and, on ``complete``,
``episode_sink.tag_episode(...)`` so ``teleop process`` drops ``"flag"`` episodes from the ACT
training set (``"copilot flagged"``).
"""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Protocol

from teleop.copilot.config import CopilotConfig
from teleop.copilot.frames import MAX_SIDE_PX, downscale_jpeg
from teleop.copilot.gemini import ImagePart, Part, TextPart, ThinkingLevel
from teleop.copilot.prompts import AUDIT_SCHEMA, AUDIT_SYSTEM

logger = logging.getLogger(__name__)

AuditStatus = Literal["pending", "complete", "skipped", "error"]
AuditVerdict = Literal["approve", "flag"]
MAX_AUDIT_FRAMES = 6
REVIEW_TIMEOUT_S = 60.0
HOUR_S = 3600.0
_VERDICTS: tuple[AuditVerdict, ...] = ("approve", "flag")


class ReviewClient(Protocol):
    """What the reviewer needs from a model client (``GeminiClient`` or ``LocalModelClient``)."""

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


class EpisodeTagger(Protocol):
    """``teleop.atc.episodes.EpisodeSink``, duck-typed: the copilot never imports ``teleop.atc``."""

    def tag_episode(
        self, uid: str, *, dataset_tag: str, audit: Mapping[str, Any] | None = None
    ) -> bool: ...


@dataclass(frozen=True)
class AuditReport:
    """One audit's state, as shown on the console card. ``to_dict`` is JSON-safe."""

    audit_id: str
    status: AuditStatus
    bay: int
    skill_id: str
    episode_uid: str | None
    verdict: AuditVerdict | None = None
    dataset_tag: AuditVerdict | None = None
    confidence: float | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple[str, ...])
    operator_note: str = ""
    model: str = ""
    frame_count: int = 0
    latency_ms: float | None = None
    wall: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        return d


@dataclass(frozen=True)
class _Job:
    report: AuditReport
    jpegs: tuple[bytes, ...]
    labels: tuple[str, ...]
    timeline: Mapping[str, Any]


def subsample(items: Sequence[Any], limit: int = MAX_AUDIT_FRAMES) -> list[Any]:
    """At most ``limit`` items spread evenly over ``items``, keeping the first and the last."""
    n = len(items)
    if n <= limit:
        return list(items)
    if limit == 1:
        return [items[-1]]
    picks = sorted({round(i * (n - 1) / (limit - 1)) for i in range(limit)})
    return [items[i] for i in picks]


def _frame_bytes(frame: Any) -> bytes:
    data: Any = getattr(frame, "jpeg_bytes", frame)
    if not isinstance(data, bytes | bytearray):
        raise TypeError(f"audit frame must be JPEG bytes or have .jpeg_bytes, got {type(frame)}")
    return bytes(data)


def _frame_label(frame: Any, i: int) -> str:
    label = getattr(frame, "label", None)
    sim_t = getattr(frame, "sim_t", None)
    bolt = getattr(frame, "bolt_x_m", None)
    text = f"frame {i + 1}" + (f" ({label})" if label else "")
    if isinstance(sim_t, int | float):
        text += f", t={sim_t:.2f}s"
    if isinstance(bolt, int | float):
        text += f", bolt={bolt * 100:.1f}cm"
    return text


def _now_wall() -> str:
    return time.strftime("%H:%M:%S")


class AuditReviewer:
    """Fire-and-forget Tier 3 audits on one worker thread. ``client=None`` skips every audit."""

    def __init__(
        self,
        client: ReviewClient | None = None,
        *,
        cfg: CopilotConfig | None = None,
        on_update: Callable[[AuditReport], None] | None = None,
        episode_sink: EpisodeTagger | None = None,
        speaker: object = None,
        timeout_s: float = REVIEW_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        del speaker  # Tier 3 is silent by contract: never speak, even if handed a speaker
        self._client = client
        self._cfg = cfg if cfg is not None else CopilotConfig()
        self._on_update = on_update
        self._episode_sink = episode_sink
        self._timeout_s = timeout_s
        self._clock = clock
        self._lock = threading.Lock()
        self._seq = itertools.count(1)
        self._recent: deque[float] = deque()
        self._closed = False
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="copilot-audit")

    @property
    def model(self) -> str:
        return self._cfg.review_model

    def submit(
        self,
        *,
        bay: int,
        skill_id: str,
        episode_uid: str | None,
        frames: Sequence[Any],
        timeline: Mapping[str, Any] | None = None,
        telemetry: Mapping[str, Any] | None = None,
    ) -> str:
        """Queue an audit of one rescue and return its ``audit_id`` without waiting.

        ``frames`` are ``KeyframeJpeg`` (``teleop.copilot.gate``) or raw JPEG bytes, oldest first.
        ``telemetry`` is an alias of ``timeline`` (the bay's rescue telemetry summary).
        """
        facts = dict(timeline if timeline is not None else (telemetry or {}))
        audit_id = f"bay{bay}-{next(self._seq):03d}"
        picked = subsample(list(frames))
        base = AuditReport(
            audit_id=audit_id,
            status="pending",
            bay=int(bay),
            skill_id=str(skill_id),
            episode_uid=episode_uid,
            model=self.model,
            frame_count=len(picked),
            wall=_now_wall(),
        )
        if self._client is None:
            self._emit(replace_report(base, status="skipped", error="no review model configured"))
            return audit_id
        if not self._take_budget():
            self._emit(
                replace_report(
                    base,
                    status="skipped",
                    error=f"hourly review budget ({self._cfg.max_reviews_per_hour}) spent",
                )
            )
            return audit_id
        try:
            jpegs = tuple(_frame_bytes(f) for f in picked)
        except TypeError as exc:
            self._emit(replace_report(base, status="error", error=str(exc)))
            return audit_id
        labels = tuple(_frame_label(f, i) for i, f in enumerate(picked))
        self._emit(base)
        job = _Job(base, jpegs, labels, facts)
        with self._lock:
            if self._closed:
                self._emit(replace_report(base, status="skipped", error="reviewer closed"))
                return audit_id
            self._pool.submit(self._run, job)
        return audit_id

    def close(self) -> None:
        """Stop taking audits, drop queued ones and wait for the one in flight."""
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=True)

    # -- worker -------------------------------------------------------------------------------

    def _take_budget(self) -> bool:
        now = self._clock()
        with self._lock:
            while self._recent and now - self._recent[0] >= HOUR_S:
                self._recent.popleft()
            if len(self._recent) >= self._cfg.max_reviews_per_hour:
                return False
            self._recent.append(now)
            return True

    def _run(self, job: _Job) -> None:
        client = self._client
        assert client is not None
        t0 = time.perf_counter()
        try:
            parts: list[Part] = [TextPart(_timeline_text(job.report, job.timeline))]
            for label, jpeg in zip(job.labels, job.jpegs, strict=True):
                parts.append(TextPart(label))
                parts.append(ImagePart(downscale_jpeg(jpeg, MAX_SIDE_PX)))
            answer = client.generate_json(
                self.model,
                AUDIT_SYSTEM,
                parts,
                AUDIT_SCHEMA,
                self._timeout_s,
                thinking_level=self._cfg.review_thinking,
            )
            report = parse_audit(answer, job.report)
        except Exception as exc:  # noqa: BLE001 - an audit failure must never escape the worker
            logger.warning("audit %s failed: %s", job.report.audit_id, exc)
            report = replace_report(job.report, status="error", error=str(exc)[:300])
        report = replace_report(
            report, latency_ms=round((time.perf_counter() - t0) * 1000.0, 1), wall=_now_wall()
        )
        self._emit(report)
        if report.status == "complete":
            self._tag(report)

    def _tag(self, report: AuditReport) -> None:
        sink, uid = self._episode_sink, report.episode_uid
        if sink is None or not uid or report.dataset_tag is None:
            return
        try:
            if not sink.tag_episode(uid, dataset_tag=report.dataset_tag, audit=report.to_dict()):
                logger.warning("audit %s: episode %s not found to tag", report.audit_id, uid)
        except Exception:  # noqa: BLE001
            logger.exception("audit %s: tagging episode %s failed", report.audit_id, uid)

    def _emit(self, report: AuditReport) -> None:
        if self._on_update is None:
            return
        try:
            self._on_update(report)
        except Exception:  # noqa: BLE001 - a broken console must not break the audit
            logger.exception("audit %s: on_update raised", report.audit_id)


def replace_report(report: AuditReport, **changes: Any) -> AuditReport:
    """``dataclasses.replace`` for :class:`AuditReport` (kept typed for callers)."""
    return replace(report, **changes)


def _timeline_text(report: AuditReport, timeline: Mapping[str, Any]) -> str:
    facts = json.dumps(dict(timeline), default=str, sort_keys=True)
    return (
        f"Bay {report.bay} rescue, skill {report.skill_id}. "
        f"{report.frame_count} chronological frames follow. Telemetry timeline: {facts}"
    )


def _verdict(value: object) -> AuditVerdict | None:
    if isinstance(value, str) and value.strip().lower() in _VERDICTS:
        return "approve" if value.strip().lower() == "approve" else "flag"
    return None


def parse_audit(answer: Mapping[str, Any], pending: AuditReport) -> AuditReport:
    """A ``complete`` report from the model's JSON, or ``error`` when the verdict is unusable."""
    verdict = _verdict(answer.get("verdict"))
    if verdict is None:
        return replace_report(
            pending, status="error", error=f"unusable verdict {answer.get('verdict')!r}"
        )
    tag = _verdict(answer.get("dataset_tag")) or verdict
    raw_conf = answer.get("confidence")
    confidence = (
        min(max(float(raw_conf), 0.0), 1.0)
        if isinstance(raw_conf, int | float) and not isinstance(raw_conf, bool)
        else None
    )
    raw_reasons = answer.get("reasons")
    reasons: tuple[str, ...] = ()
    if isinstance(raw_reasons, list):
        reasons = tuple(str(r).strip() for r in raw_reasons if str(r).strip())  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    elif isinstance(raw_reasons, str) and raw_reasons.strip():
        reasons = (raw_reasons.strip(),)
    note = answer.get("operator_note")
    return replace_report(
        pending,
        status="complete",
        verdict=verdict,
        dataset_tag=tag,
        confidence=confidence,
        reasons=reasons or ("no reason given",),
        operator_note=str(note).strip() if isinstance(note, str) else "",
    )
