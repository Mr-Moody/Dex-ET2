"""Rolling latency statistics."""

from __future__ import annotations

from collections import deque

import numpy as np

_WINDOW = 2048


class LatencyStats:
    """Ring buffer of the last 2048 latency samples (stored in ms)."""

    def __init__(self) -> None:
        self._ms: deque[float] = deque(maxlen=_WINDOW)

    def add(self, ns: int) -> None:
        self._ms.append(ns / 1e6)

    def summary(self) -> dict[str, float]:
        """Return p50_ms, p95_ms, max_ms, n. On an empty buffer all values are 0.0."""
        if not self._ms:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "n": 0.0}
        arr = np.asarray(self._ms, dtype=np.float64)
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max()),
            "n": float(arr.size),
        }
