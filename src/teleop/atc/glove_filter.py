"""1-Euro smoothing of the rescue glove: wrist pose and finger targets, once per glove frame.

The control loop polls the glove at 60 Hz while the glove streams at its own rate, so the same
frame is often read twice. Each frame is filtered once, timed by the device clock when the
source provides one, and repeats return the cached result. A gap in the stream (or the clock
running backwards, as when a source restarts) restarts the filters so the hand does not glide
from where it was before the gap.
"""

from __future__ import annotations

from collections.abc import Mapping

from teleop.types import HandFrame, Pose
from teleop.unoq.filter import JointsOneEuroFilter, PoseOneEuroFilter

GAP_RESET_S = 0.25


def frame_time_s(frame: HandFrame) -> float:
    """Sample time in seconds: the device clock if present, else the host receive time."""
    if frame.device_time_us is not None:
        return frame.device_time_us * 1e-6
    return frame.host_time_ns * 1e-9


class GloveInputFilter:
    """Smooths one hand's wrist ``Pose`` and actuator targets with 1-Euro filters."""

    def __init__(
        self,
        wrist: PoseOneEuroFilter | None = None,
        joints: JointsOneEuroFilter | None = None,
        gap_reset_s: float = GAP_RESET_S,
    ) -> None:
        if gap_reset_s <= 0.0:
            raise ValueError("gap_reset_s must be positive")
        self.wrist = wrist if wrist is not None else PoseOneEuroFilter()
        self.joints = joints if joints is not None else JointsOneEuroFilter()
        self.gap_reset_s = gap_reset_s
        self._key: tuple[int, int] | None = None
        self._t: float | None = None
        self._out: tuple[Pose, dict[str, float]] | None = None

    def reset(self) -> None:
        self.wrist.reset()
        self.joints.reset()
        self._key = None
        self._t = None
        self._out = None

    def step(
        self, frame: HandFrame, wrist: Pose, targets: Mapping[str, float]
    ) -> tuple[Pose, dict[str, float]]:
        """Filtered ``(wrist, targets)`` for ``frame``; a repeated frame returns the last result."""
        key = (frame.seq, frame.host_time_ns)
        if key == self._key and self._out is not None:
            return self._out[0], dict(self._out[1])
        t = frame_time_s(frame)
        if self._t is not None and (t < self._t or t - self._t > self.gap_reset_s):
            self.reset()
        out = (self.wrist.filter(wrist, t), self.joints.filter(dict(targets), t))
        self._key = key
        self._t = t
        self._out = out
        return out[0], dict(out[1])
