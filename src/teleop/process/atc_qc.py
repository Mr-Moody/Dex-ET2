"""QC, idle trim and labels for ATC recovery datasets (spec section 6).

An ATC dataset is one whose sidecar (``meta/teleop_episodes.jsonl``) lines carry ``source``
(:class:`teleop.policy.schema.EpisodeMeta`). Episodes are dropped for one of the exact reason
strings in :data:`DROP_REASONS`; kept episodes are trimmed to the frames between the first and
the last commanded motion. ``write_clean`` then carries every sidecar label into the clean set.

The timestamp-gap rule reads ``max_gap_s`` from the sidecar (top level, or ``source_status`` as
in legacy QC): LeRobot writes ``timestamp = frame_index / fps``, so the dataset itself never
shows a gap. Episodes without it are kept and carry the ``no_max_gap_s`` warning.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, get_args

import numpy as np
from numpy.typing import NDArray

from teleop.atc.validator import ARM_TORQUE_MARGIN, TAU_HARD_MAX_NM
from teleop.policy.schema import ACTION_NAMES, EpisodeSource
from teleop.process.qc import QCResult, episode_actions, num_episodes, open_dataset, read_sidecar

SOURCES: tuple[str, ...] = get_args(EpisodeSource)
DEFAULT_SOURCES: frozenset[str] = frozenset({"glove", "scripted", "dmp"})
DROP_REASONS: tuple[str, ...] = (
    "not successful",
    "source excluded",
    "too short",
    "timestamp gap",
    "torque breach",
    "action out of range",
)
RANGE_TOL = 1e-6
_HAND_PREFIX = "hand/"


@dataclass
class AtcQCThresholds:
    min_len_s: float = 0.5
    max_gap_s: float = 0.2
    idle_eps: float = 1e-3  # action units per frame (max abs delta from the previous frame)


def is_atc_dataset(root: Path) -> bool:
    """True when the dataset has sidecar lines and every one carries ``source``."""
    rows = read_sidecar(Path(root))
    return bool(rows) and all("source" in row for row in rows.values())


def atc_action_ranges(hand: str = "inspire") -> dict[str, tuple[float, float]] | None:
    """``hand/<actuator>`` -> ctrlrange of the fleet's right hand; None (warned) on failure."""
    try:
        from teleop.sim.hands import actuator_ranges, load_hand

        ranges = actuator_ranges(load_hand(hand), "right")
    except Exception as exc:  # noqa: BLE001 - QC must still run without ranges
        print(f"teleop process: warning: cannot load hand {hand!r} ranges: {exc}", file=sys.stderr)
        return None
    return {f"{_HAND_PREFIX}{name}": rng for name, rng in ranges.items()}


def atc_trim(actions: NDArray[Any], eps: float) -> tuple[int, int]:
    """``[start, end)`` from the first to the last frame whose action moved more than ``eps``.

    A frame's motion is the max abs change from the previous frame. No motion at all gives an
    empty range, which the length rule then drops.
    """
    a = np.asarray(actions, dtype=np.float64)
    if len(a) < 2:
        return 0, 0
    moving = np.flatnonzero(np.abs(np.diff(a, axis=0)).max(axis=1) > eps) + 1
    if moving.size == 0:
        return 0, 0
    return int(moving[0]), int(moving[-1]) + 1


def _max_gap_s(meta: Mapping[str, Any]) -> float | None:
    gap: Any = meta.get("max_gap_s")
    status: Any = meta.get("source_status")
    if gap is None and isinstance(status, Mapping):
        gap = cast("Mapping[str, Any]", status).get("max_gap_s")
    return float(gap) if isinstance(gap, int | float) else None


def _out_of_range(actions: NDArray[Any], ranges: Mapping[str, tuple[float, float]] | None) -> bool:
    if not ranges or not len(actions):
        return False
    for i, name in enumerate(ACTION_NAMES[: actions.shape[1]]):
        if name.startswith(_HAND_PREFIX) and name in ranges:
            lo, hi = ranges[name]
            col = actions[:, i]
            if bool(np.any(col < lo - RANGE_TOL) or np.any(col > hi + RANGE_TOL)):
                return True
    return False


def _reasons(
    actions: NDArray[Any],
    trim: tuple[int, int],
    fps: int,
    meta: Mapping[str, Any],
    th: AtcQCThresholds,
    sources: frozenset[str],
    ranges: Mapping[str, tuple[float, float]] | None,
) -> list[str]:
    reasons: list[str] = []
    if meta.get("success") is not True:
        reasons.append("not successful")
    if meta.get("source") not in sources:
        reasons.append("source excluded")
    if (trim[1] - trim[0]) / fps < th.min_len_s:
        reasons.append("too short")
    gap = _max_gap_s(meta)
    if gap is not None and gap > th.max_gap_s:
        reasons.append("timestamp gap")
    tau: Any = meta.get("max_tau_hand_inf")
    ratio: Any = meta.get("max_arm_ratio")
    if (isinstance(tau, int | float) and tau >= TAU_HARD_MAX_NM) or (
        isinstance(ratio, int | float) and ratio >= ARM_TORQUE_MARGIN
    ):
        reasons.append("torque breach")
    if _out_of_range(actions, ranges):
        reasons.append("action out of range")
    return reasons


def run_atc_qc(
    root: Path,
    thresholds: AtcQCThresholds,
    sources: frozenset[str],
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> list[QCResult]:
    """QC every episode of one ATC dataset. ``ranges`` defaults to the Inspire hand's."""
    root = Path(root)
    if num_episodes(root) == 0:
        return []
    if ranges is None:
        ranges = atc_action_ranges()
    ds = open_dataset(root)
    sidecar = read_sidecar(root)
    fps = int(ds.fps)
    results: list[QCResult] = []
    for ep in range(int(ds.meta.total_episodes)):
        actions = episode_actions(ds, ep)
        meta = sidecar.get(ep, {})
        trim = atc_trim(actions, thresholds.idle_eps)
        reasons = _reasons(actions, trim, fps, meta, thresholds, sources, ranges)
        warnings = [] if _max_gap_s(meta) is not None else ["no_max_gap_s"]
        results.append(QCResult(ep, not reasons, reasons, trim, None, root, warnings))
    return results


def _cell(counts: Mapping[tuple[str, str], tuple[int, int]], fault: str, source: str) -> str:
    kept, total = counts.get((fault, source), (0, 0))
    return f"{kept}/{total}" if total else "-"


def summarize(
    results: Sequence[QCResult],
    sidecars: Mapping[Path, Mapping[int, Mapping[str, Any]]],
) -> str:
    """A fault x source table of kept/total episodes, with row and column totals."""
    counts: dict[tuple[str, str], tuple[int, int]] = {}
    empty: Mapping[str, Any] = {}
    for r in results:
        rows = sidecars.get(r.source_root) if r.source_root else None
        row = (rows.get(r.episode_index) if rows else None) or empty
        fault, source = str(row.get("fault") or "none"), str(row.get("source") or "unknown")
        for key in ((fault, source), (fault, "total"), ("total", source), ("total", "total")):
            kept, total = counts.get(key, (0, 0))
            counts[key] = (kept + int(r.keep), total + 1)
    faults = sorted({f for f, _ in counts if f != "total"}) + ["total"]
    present = {s for _, s in counts if s != "total"}
    order = [s for s in SOURCES if s in present] + sorted(present - set(SOURCES))
    cols = [*order, "total"]
    table = [["fault", *cols]] + [[f, *(_cell(counts, f, s) for s in cols)] for f in faults]
    widths = [max(len(row[i]) for row in table) for i in range(len(cols) + 1)]
    return "\n".join(
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip()
        for row in table
    )
