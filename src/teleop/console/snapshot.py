"""Hand-off from the 60 Hz physics thread to the console render thread.

The physics thread captures a ``SimSnapshot`` (copied arrays plus the HUD view) after every
tick and puts it in a ``LatestSlot``; the render thread takes whatever is newest. Neither side
ever waits on the other.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from teleop.atc.hud import HudView


@dataclass(frozen=True)
class SimSnapshot:
    qpos: NDArray[np.float64]
    mocap_pos: NDArray[np.float64]
    mocap_quat: NDArray[np.float64]
    view: HudView
    sim_t: float

    @classmethod
    def capture(cls, data: Any, view: HudView, sim_t: float) -> SimSnapshot:
        return cls(
            qpos=np.array(data.qpos, dtype=np.float64, copy=True),
            mocap_pos=np.array(data.mocap_pos, dtype=np.float64, copy=True),
            mocap_quat=np.array(data.mocap_quat, dtype=np.float64, copy=True),
            view=view,
            sim_t=float(sim_t),
        )


class LatestSlot[T]:
    """Thread-safe single-value mailbox: ``put`` overwrites, ``get`` returns the newest."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: T | None = None
        self._version = 0

    def put(self, item: T) -> None:
        with self._lock:
            self._item = item
            self._version += 1

    def get(self) -> T | None:
        with self._lock:
            return self._item

    @property
    def version(self) -> int:
        with self._lock:
            return self._version
