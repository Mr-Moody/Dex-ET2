"""Fixed-rate loop scheduled against absolute deadlines (pure Python, injectable clock)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class LoopStats:
    steps: int = 0
    overruns: int = 0


class FixedRateLoop:
    """Call ``step(i)`` at ``hz``. Late loops skip ahead instead of bursting."""

    def __init__(
        self,
        hz: int,
        clock: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if hz <= 0:
            raise ValueError("hz must be positive")
        self.hz = hz
        self._period_ns = 1_000_000_000 // hz
        self._clock = clock
        self._sleep = sleep

    def run(self, step: Callable[[int], bool], max_steps: int | None = None) -> LoopStats:
        """Run until ``step`` returns False or ``max_steps`` steps have been made."""
        stats = LoopStats()
        period = self._period_ns
        deadline = self._clock()
        while max_steps is None or stats.steps < max_steps:
            keep_going = step(stats.steps)
            stats.steps += 1
            if not keep_going:
                break
            deadline += period
            now = self._clock()
            if now - deadline > period:
                stats.overruns += 1
                missed = (now - deadline + period - 1) // period
                deadline += missed * period
            if deadline > now:
                self._sleep((deadline - now) / 1e9)
        return stats
