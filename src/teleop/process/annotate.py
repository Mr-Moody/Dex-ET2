"""Episode annotators: ``RuleAnnotator`` labels from score/success; a VLM one plugs in later."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

# LiftCube score = max cube height x 100, so 3.0 is a 3 cm lift.
SUCCESS_SCORE = 3.0


@dataclass(frozen=True)
class Annotation:
    label: str
    success: bool | None
    notes: str


class Annotator(Protocol):
    def annotate(
        self, episode_frames: Sequence[np.ndarray[Any, Any]], meta: Mapping[str, Any]
    ) -> Annotation: ...


class RuleAnnotator:
    """Label by the recorded ``success`` flag, falling back to ``score`` >= ``SUCCESS_SCORE``."""

    def annotate(
        self, episode_frames: Sequence[np.ndarray[Any, Any]], meta: Mapping[str, Any]
    ) -> Annotation:
        score = meta.get("score")
        success = meta.get("success")
        if success is None and isinstance(score, int | float):
            success = float(score) >= SUCCESS_SCORE
        label = "unknown" if success is None else ("success" if success else "failure")
        notes = "" if score is None else f"score={score:g}"
        return Annotation(label, None if success is None else bool(success), notes)
