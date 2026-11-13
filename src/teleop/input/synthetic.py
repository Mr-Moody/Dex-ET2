"""Synthetic source: both hands open and close sinusoidally. No hardware needed."""

from __future__ import annotations

import math
import threading
import time

from teleop.config import WristConfig
from teleop.input.buffer import ThreadedSource
from teleop.types import OPENXR_JOINTS, HandFrame, Pose, Quat, Side, Vec3

_IDENTITY: Quat = (0.0, 0.0, 0.0, 1.0)
_FLEXION_JOINTS = frozenset(
    j
    for j in OPENXR_JOINTS
    if j.endswith(("_proximal", "_intermediate", "_distal")) or j == "thumb_metacarpal"
)
_MAX_ANGLE = 0.7  # angle = 0.7 * (1 - cos) spans 0..1.4 rad


def _rot_x(angle: float) -> Quat:
    half = angle / 2.0
    return (math.sin(half), 0.0, 0.0, math.cos(half))


class SyntheticSource(ThreadedSource):
    def __init__(self, rate_hz: float = 60.0, period_s: float = 2.0) -> None:
        super().__init__("synthetic")
        self._rate_hz = rate_hz
        self._period_s = period_s
        wrist = WristConfig()
        self._home: dict[Side, Vec3] = {"left": wrist.home_left, "right": wrist.home_right}

    def _make_frame(self, side: Side, t_s: float, now_ns: int, seq: int) -> HandFrame:
        angle = _MAX_ANGLE * (1.0 - math.cos(2.0 * math.pi * t_s / self._period_s))
        flex = _rot_x(angle)
        joints = {j: (flex if j in _FLEXION_JOINTS else _IDENTITY) for j in OPENXR_JOINTS}
        return HandFrame(
            side=side,
            host_time_ns=now_ns,
            seq=seq,
            source=self.name,
            joints=joints,
            device_time_us=now_ns // 1000,
            wrist=Pose(orientation=_IDENTITY, position=self._home[side]),
        )

    def _run(self, stop: threading.Event) -> None:
        period = 1.0 / self._rate_hz
        t0 = time.monotonic_ns()
        seq = 0
        while not stop.is_set():
            now = time.monotonic_ns()
            t_s = (now - t0) / 1e9
            for side in ("left", "right"):
                self._buffer.put(self._make_frame(side, t_s, now, seq))
            seq += 1
            stop.wait(period)
