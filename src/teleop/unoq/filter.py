"""1-Euro filters (Casiez et al., CHI 2012) for the 83.3 Hz glove wrist and joint streams.

The cutoff adapts to speed: ``f_c = min_cutoff + beta * |dx/dt|``, so a still hand is heavily
smoothed while fast motion passes with little lag. Defaults are tuned for wrist positions in
metres sampled at 83.3 Hz.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from teleop.types import Pose

STREAM_HZ = 83.3

FloatArray = NDArray[np.float64]


def _alpha(cutoff_hz: FloatArray | float, dt: float) -> FloatArray:
    tau = 1.0 / (2.0 * math.pi * np.asarray(cutoff_hz, dtype=np.float64))
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """1-Euro filter over a scalar or a fixed-shape vector (each element adapts on its own)."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 10.0,
        d_cutoff: float = 1.0,
        freq_hz: float = STREAM_HZ,
    ) -> None:
        if min_cutoff <= 0.0 or d_cutoff <= 0.0 or freq_hz <= 0.0:
            raise ValueError("min_cutoff, d_cutoff and freq_hz must be positive")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.freq_hz = freq_hz
        self._x: FloatArray | None = None
        self._dx: FloatArray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = None
        self._dx = None
        self._t = None

    def _dt(self, t_s: float | None) -> float:
        nominal = 1.0 / self.freq_hz
        if t_s is None or self._t is None:
            return nominal
        dt = t_s - self._t
        return dt if dt > 0.0 else nominal

    def filter(self, x: ArrayLike, t_s: float | None = None) -> FloatArray:
        """Filter one sample. ``t_s`` (seconds) overrides the nominal sample period."""
        xa = np.array(x, dtype=np.float64)
        dt = self._dt(t_s)
        if t_s is not None:
            self._t = t_s
        if self._x is None or self._dx is None:
            self._x = xa
            self._dx = np.zeros_like(xa)
            return xa.copy()
        if xa.shape != self._x.shape:
            raise ValueError(f"sample shape {xa.shape} != filter shape {self._x.shape}")
        dx = (xa - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * xa + (1.0 - a) * self._x
        self._x = x_hat
        self._dx = dx_hat
        return x_hat.copy()

    def __call__(self, x: ArrayLike, t_s: float | None = None) -> FloatArray:
        return self.filter(x, t_s)


class PoseOneEuroFilter:
    """1-Euro smoothing of a wrist ``Pose``: 3D position plus an xyzw quaternion.

    Quaternions are flipped onto the previous sample's hemisphere before filtering (``q`` and
    ``-q`` are the same rotation) and re-normalised afterwards.
    """

    def __init__(
        self,
        pos_min_cutoff: float = 1.0,
        pos_beta: float = 10.0,
        rot_min_cutoff: float = 1.0,
        rot_beta: float = 0.3,
        d_cutoff: float = 1.0,
        freq_hz: float = STREAM_HZ,
    ) -> None:
        self.position = OneEuroFilter(pos_min_cutoff, pos_beta, d_cutoff, freq_hz)
        self.rotation = OneEuroFilter(rot_min_cutoff, rot_beta, d_cutoff, freq_hz)
        self._q_prev: FloatArray | None = None

    def reset(self) -> None:
        self.position.reset()
        self.rotation.reset()
        self._q_prev = None

    def filter(self, pose: Pose, t_s: float | None = None) -> Pose:
        q: FloatArray = np.asarray(pose.orientation, dtype=np.float64)
        n = float(np.linalg.norm(q))
        if n == 0.0:
            raise ValueError("zero quaternion")
        q = q / n
        if self._q_prev is not None and float(np.dot(q, self._q_prev)) < 0.0:
            q = -q
        q_hat = self.rotation.filter(q, t_s)
        q_hat = q_hat / float(np.linalg.norm(q_hat))
        self._q_prev = q_hat
        orientation = (float(q_hat[0]), float(q_hat[1]), float(q_hat[2]), float(q_hat[3]))
        if pose.position is None:
            return Pose(orientation=orientation)
        p = self.position.filter(pose.position, t_s)
        return Pose(orientation=orientation, position=(float(p[0]), float(p[1]), float(p[2])))

    def __call__(self, pose: Pose, t_s: float | None = None) -> Pose:
        return self.filter(pose, t_s)


class JointsOneEuroFilter:
    """Independent 1-Euro filters per named joint (radians); new names start a fresh filter."""

    def __init__(
        self, min_cutoff: float = 1.5, beta: float = 0.5, freq_hz: float = STREAM_HZ
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.freq_hz = freq_hz
        self._filters: dict[str, OneEuroFilter] = {}

    def reset(self) -> None:
        self._filters.clear()

    def filter(self, joints: dict[str, float], t_s: float | None = None) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, value in joints.items():
            f = self._filters.get(name)
            if f is None:
                f = self._filters[name] = OneEuroFilter(
                    self.min_cutoff, self.beta, freq_hz=self.freq_hz
                )
            out[name] = float(f.filter(value, t_s))
        return out
