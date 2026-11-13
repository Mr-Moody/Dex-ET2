"""Episode QC, idle trimming and clean-dataset writing for LeRobot v3 datasets."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from teleop.process.annotate import Annotator
from teleop.record.recorder import EpisodeRecorder

SIDECAR = Path("meta") / "teleop_episodes.jsonl"
_AUTO_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
_IMG_PREFIX = "observation.images."


@dataclass
class QCThresholds:
    min_len_s: float = 2.0
    max_gap_s: float = 0.5
    out_of_range_frac: float = 0.05
    min_progress_m: float = 0.03
    idle_vel_eps: float = 1e-3  # action units per frame (max abs delta between frames)


@dataclass
class QCResult:
    episode_index: int
    keep: bool
    reasons: list[str]
    trim: tuple[int, int]  # [start, end) frame range within the episode
    score: float | None = None
    source_root: Path | None = field(default=None, compare=False)


def _open_dataset(root: Path) -> Any:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    mod: Any = importlib.import_module("lerobot.datasets.lerobot_dataset")
    return mod.LeRobotDataset(f"local/{root.name}", root=root)


def _num_episodes(root: Path) -> int:
    info = root / "meta" / "info.json"
    if not info.is_file():
        return 0
    return int(json.loads(info.read_text(encoding="utf-8")).get("total_episodes", 0))


def _read_sidecar(root: Path) -> dict[int, dict[str, Any]]:
    path = root / SIDECAR
    if not path.is_file():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[int(row["episode_index"])] = row
    return out


def _score_frames(meta: Mapping[str, Any], n: int) -> list[int] | None:
    frames: Any = meta.get("score_increase_frames")
    if isinstance(frames, list) and frames:
        return [int(f) for f in cast("list[Any]", frames)]
    curve: Any = meta.get("score_curve")
    if isinstance(curve, list) and len(curve) > 1:  # pyright: ignore[reportUnknownArgumentType]
        d = np.diff(np.asarray(curve, dtype=float))
        idx = [int(i) + 1 for i in np.flatnonzero(d > 0) if i + 1 < n]
        return idx or None
    return None


def compute_trim(
    actions: NDArray[Any], fps: int, eps: float, score_frames: Sequence[int] | None
) -> tuple[int, int]:
    """Drop idle head/tail (action velocity < eps), keeping >= 1 s and all score-increase frames."""
    n = len(actions)
    if n < 2:
        return 0, n
    vel = np.abs(np.diff(np.asarray(actions, dtype=np.float64), axis=0)).max(axis=1)
    moving = np.flatnonzero(vel >= eps)
    if moving.size == 0:
        start, end = 0, n
    else:
        start, end = int(moving[0]), int(moving[-1]) + 2
    if score_frames:
        start = min(start, max(0, min(score_frames)))
        end = max(end, min(n, max(score_frames) + 1))
    min_keep = min(n, max(1, fps))
    deficit = min_keep - (end - start)
    if deficit > 0:
        grow_end = min(deficit, n - end)
        end += grow_end
        start -= min(deficit - grow_end, start)
    return start, end


def _height(meta: Mapping[str, Any]) -> float | None:
    z = meta.get("max_cube_z")
    if isinstance(z, int | float):
        return float(z)
    score = meta.get("score")
    if isinstance(score, int | float):
        return float(score) / 100.0
    return None


def _reasons(
    actions: NDArray[Any],
    names: Sequence[str],
    fps: int,
    meta: Mapping[str, Any],
    th: QCThresholds,
    ranges: Mapping[str, tuple[float, float]] | None,
) -> list[str]:
    reasons: list[str] = []
    n = len(actions)
    if n / fps < th.min_len_s:
        reasons.append("too_short")
    status: Any = meta.get("source_status")
    if isinstance(status, Mapping):
        status = cast("Mapping[str, Any]", status)
        gap: Any = status.get("max_gap_s")
        if status.get("connected") is False or (
            isinstance(gap, int | float) and gap > th.max_gap_s
        ):
            reasons.append("input_dropout")
    if meta.get("hotspot_events"):
        reasons.append("hotspot_event")
    if ranges and n:
        bad = np.zeros(n, dtype=bool)
        for i, name in enumerate(names):
            if name in ranges:
                lo, hi = ranges[name]
                bad |= (actions[:, i] < lo) | (actions[:, i] > hi)
        if bad.mean() > th.out_of_range_frac:
            reasons.append("out_of_range")
    height = _height(meta)
    if height is not None and height < th.min_progress_m:
        reasons.append("no_progress")
    return reasons


def run_qc(
    ds_root: Path,
    thresholds: QCThresholds,
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> list[QCResult]:
    root = Path(ds_root)
    if _num_episodes(root) == 0:
        return []
    ds = _open_dataset(root)
    names = list(ds.meta.features["action"].get("names") or [])
    sidecar = _read_sidecar(root)
    fps = int(ds.fps)
    results: list[QCResult] = []
    for ep in range(int(ds.meta.total_episodes)):
        row = ds.meta.episodes[ep]
        lo, hi = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        col = ds.hf_dataset[lo:hi]["action"]
        actions = np.stack([np.asarray(a, dtype=np.float32) for a in col])
        meta = sidecar.get(ep, {})
        reasons = _reasons(actions, names, fps, meta, thresholds, ranges)
        trim = compute_trim(
            actions, fps, thresholds.idle_vel_eps, _score_frames(meta, len(actions))
        )
        score = meta.get("score")
        results.append(
            QCResult(
                ep,
                not reasons,
                reasons,
                trim,
                float(score) if isinstance(score, int | float) else None,
                root,
            )
        )
    return results


def _frame_arrays(ds: Any, lo: int, hi: int) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for i in range(lo, hi):
        item = ds[i]
        images = {
            k[len(_IMG_PREFIX) :]: np.clip(
                np.rint(np.asarray(v).transpose(1, 2, 0) * 255.0), 0, 255
            ).astype(np.uint8)
            for k, v in item.items()
            if k.startswith(_IMG_PREFIX)
        }
        frames.append(
            {
                "state": np.asarray(item["observation.state"], dtype=np.float32),
                "action": np.asarray(item["action"], dtype=np.float32),
                "images": images,
                "task": str(item.get("task", "")),
            }
        )
    return frames


def _clean_features(ds: Any) -> dict[str, Any]:
    return {
        k: {kk: vv for kk, vv in v.items() if kk in ("dtype", "shape", "names")}
        for k, v in ds.meta.features.items()
        if k not in _AUTO_FEATURES
    }


def _write_reports(
    out_root: Path,
    results: Sequence[QCResult],
    inputs: Sequence[Path],
    frames_written: int,
    empty_reason: str | None,
) -> None:
    kept = sum(r.keep for r in results)
    totals = {
        "episodes": len(results),
        "kept": kept,
        "rejected": len(results) - kept,
        "frames_written": frames_written,
    }
    report = {
        "inputs": [str(p) for p in inputs],
        "totals": totals,
        "empty_reason": empty_reason,
        "episodes": [
            {
                "source_root": str(r.source_root) if r.source_root else None,
                "episode_index": r.episode_index,
                "keep": r.keep,
                "reasons": r.reasons,
                "trim": list(r.trim),
                "score": r.score,
            }
            for r in results
        ],
    }
    (out_root / "qc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# QC report", ""]
    if empty_reason:
        lines += [f"**{empty_reason}**", ""]
    lines += ["| episode | keep | reasons | trim | score |", "|---|---|---|---|---|"]
    for r in results:
        src = f"{r.source_root.name}:" if r.source_root else ""
        score = "" if r.score is None else f"{r.score:g}"
        lines.append(
            f"| {src}{r.episode_index} | {'yes' if r.keep else 'no'} | "
            f"{', '.join(r.reasons)} | {r.trim[0]}-{r.trim[1]} | {score} |"
        )
    lines += [
        "",
        f"Totals: {totals['episodes']} episodes, {totals['kept']} kept, "
        f"{totals['rejected']} rejected, {totals['frames_written']} frames written.",
        "",
    ]
    (out_root / "qc_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_clean(
    ds_root: Path,
    results: Sequence[QCResult],
    out_root: Path,
    annotator: Annotator | None = None,
) -> None:
    """Write kept, trimmed episodes to a fresh v3 dataset at ``out_root`` plus the QC reports.

    ``results`` may mix sessions; each result's ``source_root`` wins over ``ds_root``.
    """
    out_root = Path(out_root)
    kept = [r for r in results if r.keep]
    inputs: list[Path] = []
    for r in results:
        src = r.source_root or Path(ds_root)
        if src not in inputs:
            inputs.append(src)
    if not inputs:
        inputs.append(Path(ds_root))
    frames_written = 0
    empty_reason: str | None = None
    if not results:
        empty_reason = "no episodes in the input datasets"
    elif not kept:
        empty_reason = f"all {len(results)} episodes were rejected by QC"

    if kept:
        datasets: dict[Path, Any] = {}
        sidecars: dict[Path, dict[int, dict[str, Any]]] = {}
        rec: EpisodeRecorder | None = None
        try:
            for r in kept:
                src = r.source_root or Path(ds_root)
                if src not in datasets:
                    datasets[src] = _open_dataset(src)
                    sidecars[src] = _read_sidecar(src)
                ds = datasets[src]
                if rec is None:
                    rec = EpisodeRecorder(
                        f"local/{out_root.name}",
                        out_root,
                        int(ds.fps),
                        _clean_features(ds),
                        str(ds.meta.robot_type or "unknown"),
                    )
                elif int(ds.fps) != rec.fps or _clean_features(ds) != rec.features:
                    raise ValueError(f"{src} has a different fps or schema than the first input")
                start = int(ds.meta.episodes[r.episode_index]["dataset_from_index"])
                frames = _frame_arrays(ds, start + r.trim[0], start + r.trim[1])
                meta = dict(sidecars[src].get(r.episode_index, {}))
                meta.pop("episode_index", None)
                meta["source_root"] = str(src)
                meta["source_episode_index"] = r.episode_index
                if annotator is not None:
                    cam_frames = [next(iter(f["images"].values())) for f in frames if f["images"]]
                    ann = annotator.annotate(cam_frames, meta)
                    meta["annotation"] = {
                        "label": ann.label,
                        "success": ann.success,
                        "notes": ann.notes,
                    }
                rec.start_episode(frames[0]["task"] if frames else "")
                for f in frames:
                    rec.add_frame(f["state"], f["action"], f["images"])
                rec.end_episode(meta)
                frames_written += len(frames)
        finally:
            if rec is not None:
                rec.close()
    out_root.mkdir(parents=True, exist_ok=True)
    _write_reports(out_root, results, inputs, frames_written, empty_reason)
