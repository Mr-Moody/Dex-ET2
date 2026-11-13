"""Direct per-actuator retargeting from calibrated glove flexion."""

from __future__ import annotations

from collections.abc import Mapping

from teleop.retarget.calibration import Calibration
from teleop.retarget.flexion import flex_value
from teleop.sim.hands import FlexMap, HandSpec
from teleop.types import SIDES, HandFrame, JointTargets, Side

MIN_SPAN = 0.05  # rad


class DirectRetargeter:
    def __init__(
        self,
        spec: HandSpec,
        ctrlrange: Mapping[Side, Mapping[str, tuple[float, float]]],
        calib: Calibration,
    ) -> None:
        self.spec = spec
        self.degenerate: set[tuple[Side, str]] = set()
        default = Calibration.default(spec)
        self._maps: dict[Side, list[tuple[FlexMap, float, float, float, float]]] = {}
        self._last: dict[Side, dict[str, float]] = {}
        for side in SIDES:
            entries: list[tuple[FlexMap, float, float, float, float]] = []
            self._last[side] = {}
            for fm in spec.flex:
                rng = ctrlrange.get(side, {}).get(fm.actuator)
                if rng is None:
                    continue
                lo, hi = rng
                open_, closed = calib.ranges.get(side, {}).get(
                    fm.actuator, default.ranges[side][fm.actuator]
                )
                if abs(closed - open_) < MIN_SPAN:
                    self.degenerate.add((side, fm.actuator))
                    open_, closed = default.ranges[side][fm.actuator]
                entries.append((fm, open_, closed, lo, hi))
                self._last[side][fm.actuator] = lo
            self._maps[side] = entries

    def retarget(self, frame: HandFrame) -> JointTargets:
        last = self._last[frame.side]
        for fm, open_, closed, lo, hi in self._maps[frame.side]:
            v = flex_value(frame, fm)
            if v is None:
                continue  # hold previous target
            n = min(1.0, max(0.0, (v - open_) / (closed - open_)))
            last[fm.actuator] = lo + n * (hi - lo)
        return JointTargets(side=frame.side, host_time_ns=frame.host_time_ns, positions=dict(last))
