"""T-213: the Tier 3 silent Argon audit (``teleop.copilot.review``), with a fake model client."""

from __future__ import annotations

import io
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from PIL import Image

from teleop.copilot.config import CopilotConfig
from teleop.copilot.gemini import GeminiHttpError, ImagePart, Part
from teleop.copilot.review import AuditReport, AuditReviewer, parse_audit, subsample

APPROVE = {
    "verdict": "approve",
    "dataset_tag": "approve",
    "confidence": 0.9,
    "reasons": ["Flap depressed before the slide"],
    "operator_note": "Clean demo.",
}


def _jpeg(w: int = 1024, h: int = 768) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 120, 150)).save(buf, "JPEG")
    return buf.getvalue()


class FakeClient:
    def __init__(self, answer: Mapping[str, Any] | Exception = APPROVE) -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def generate_json(
        self,
        model: str,
        system: str,
        parts: Sequence[Part],
        schema: Mapping[str, Any] | None,
        timeout_s: float,
        *,
        thinking_level: Any = None,
    ) -> dict[str, Any]:
        self.calls.append({"model": model, "parts": list(parts), "thinking": thinking_level})
        if isinstance(self.answer, Exception):
            raise self.answer
        return dict(self.answer)


class FakeSink:
    def __init__(self) -> None:
        self.tags: list[tuple[str, str, Mapping[str, Any] | None]] = []

    def tag_episode(
        self, uid: str, *, dataset_tag: str, audit: Mapping[str, Any] | None = None
    ) -> bool:
        self.tags.append((uid, dataset_tag, audit))
        return True


def _run(reviewer: AuditReviewer, n_frames: int = 3, **kw: Any) -> list[AuditReport]:
    updates: list[AuditReport] = []
    done = threading.Event()

    def on_update(r: AuditReport) -> None:
        updates.append(r)
        if r.status != "pending":
            done.set()

    reviewer._on_update = on_update  # pyright: ignore[reportPrivateUsage]
    reviewer.submit(
        bay=1,
        skill_id="skill_1",
        episode_uid=kw.get("episode_uid", "ep-1"),
        frames=[_jpeg()] * n_frames,
        timeline={"bolt_final_m": 0.055},
    )
    assert done.wait(5.0)
    reviewer.close()
    return updates


def test_complete_report_tags_episode_and_downscales_frames() -> None:
    client, sink = FakeClient(), FakeSink()
    reviewer = AuditReviewer(client, cfg=CopilotConfig(), episode_sink=sink)
    updates = _run(reviewer, n_frames=9)
    assert [u.status for u in updates] == ["pending", "complete"]
    final = updates[-1]
    assert (final.verdict, final.dataset_tag, final.frame_count) == ("approve", "approve", 6)
    assert final.latency_ms is not None
    (call,) = client.calls
    assert call["model"] == "gemini-4-argon" and call["thinking"] == "low"
    images = [p for p in call["parts"] if isinstance(p, ImagePart)]
    assert len(images) == 6
    for img in images:
        assert max(Image.open(io.BytesIO(img.jpeg_bytes)).size) <= 512
    assert sink.tags == [("ep-1", "approve", final.to_dict())]


def test_model_error_reports_error_and_never_tags() -> None:
    sink = FakeSink()
    reviewer = AuditReviewer(
        FakeClient(GeminiHttpError(500, "boom")), cfg=CopilotConfig(), episode_sink=sink
    )
    final = _run(reviewer)[-1]
    assert final.status == "error" and final.dataset_tag is None
    assert sink.tags == []


def test_no_client_and_spent_budget_are_skipped_without_calls() -> None:
    offline = AuditReviewer(None, cfg=CopilotConfig())
    assert _run(offline)[-1].status == "skipped"

    client = FakeClient()
    reviewer = AuditReviewer(client, cfg=CopilotConfig(max_reviews_per_hour=1))
    assert _run(reviewer)[-1].status == "complete"
    reviewer2 = AuditReviewer(client, cfg=CopilotConfig(max_reviews_per_hour=1))
    reviewer2._recent.append(reviewer2._clock())  # pyright: ignore[reportPrivateUsage]
    assert _run(reviewer2)[-1].status == "skipped"
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("answer", "status", "tag"),
    [
        ({"verdict": "FLAG", "reasons": "HV too close"}, "complete", "flag"),
        ({"verdict": "approve", "dataset_tag": "flag", "confidence": 3}, "complete", "flag"),
        ({"verdict": "maybe"}, "error", None),
    ],
)
def test_parse_audit(answer: dict[str, Any], status: str, tag: str | None) -> None:
    pending = AuditReport("a", "pending", 1, "s", None)
    rep = parse_audit(answer, pending)
    assert (rep.status, rep.dataset_tag) == (status, tag)
    if status == "complete":
        assert rep.reasons
        assert rep.confidence is None or 0.0 <= rep.confidence <= 1.0


def test_subsample_keeps_ends_and_order() -> None:
    assert subsample(list(range(10))) == [0, 2, 4, 5, 7, 9]
    assert subsample([1, 2]) == [1, 2]
