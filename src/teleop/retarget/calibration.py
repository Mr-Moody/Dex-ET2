"""Per-actuator (open, closed) flexion calibration."""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from teleop.retarget.flexion import flex_value
from teleop.sim.hands import FlexMap, HandSpec
from teleop.types import SIDES, HandFrame, Side

DEFAULT_CLOSED_PER_JOINT = 1.4


def _default_range(fm: FlexMap) -> tuple[float, float]:
    if fm.range is not None:
        return (float(fm.range[0]), float(fm.range[1]))
    scale = sum(fm.weights) if fm.weights is not None else float(len(fm.joints))
    return (0.0, DEFAULT_CLOSED_PER_JOINT * scale)


@dataclass
class Calibration:
    ranges: dict[Side, dict[str, tuple[float, float]]]

    @classmethod
    def default(cls, spec: HandSpec) -> Calibration:
        return cls({side: {fm.actuator: _default_range(fm) for fm in spec.flex} for side in SIDES})

    def to_json(self, path: Path | str) -> None:
        data = {s: {a: list(r) for a, r in per.items()} for s, per in self.ranges.items()}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path | str) -> Calibration:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        ranges: dict[Side, dict[str, tuple[float, float]]] = {}
        for side in SIDES:
            if side in raw:
                ranges[side] = {a: (float(r[0]), float(r[1])) for a, r in raw[side].items()}
        return cls(ranges)


def _median_flex(frames: Sequence[HandFrame], fm: FlexMap) -> float | None:
    values = [v for f in frames if (v := flex_value(f, fm)) is not None]
    return statistics.median(values) if values else None


def calibrate(
    spec: HandSpec, open_frames: Sequence[HandFrame], closed_frames: Sequence[HandFrame]
) -> Calibration:
    """Median flex per actuator in the open and closed poses, per side.

    Sides without frames in either pose, actuators with no data, and actuators with a fixed
    ``range`` use defaults.
    """
    calib = Calibration.default(spec)
    for side in SIDES:
        of = [f for f in open_frames if f.side == side]
        cf = [f for f in closed_frames if f.side == side]
        for fm in spec.flex:
            if fm.range is not None:
                continue
            o = _median_flex(of, fm)
            c = _median_flex(cf, fm)
            if o is not None and c is not None:
                calib.ranges[side][fm.actuator] = (o, c)
    return calib
