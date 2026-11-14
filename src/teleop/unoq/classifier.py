"""Mahalanobis stall-signature gate (spec 4.2).

Each ``(part_sku_id, signature_id)`` prototype keeps every stall vector registered for it:

    mu = mean(z),  Sigma = cov(z) + diag(sigma_floor^2)

The floor makes a prototype usable after a single rescue (one 1x4 sample, zero sample
covariance) and the sample covariance takes over as more rescues arrive. ``classify`` returns
the nearest prototype of that SKU, or an OOD result when the nearest distance exceeds
``ood_threshold`` or the SKU has no prototypes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from teleop.atc.types import StallMatch

N_FEATURES = 4  # x_bolt, mean |tau|_inf, max |tau|_inf, |d tau_wrist|_2
DEFAULT_SIGMA_FLOOR = (0.005, 0.4, 0.4, 0.4)

FloatArray = NDArray[np.float64]


def _as_samples(samples: ArrayLike) -> FloatArray:
    a = np.array(samples, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != N_FEATURES or a.shape[0] == 0:
        raise ValueError(f"stall samples must be n x {N_FEATURES}, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("stall samples must be finite")
    return a


class _Prototype:
    def __init__(self, samples: FloatArray, floor_var: FloatArray) -> None:
        self.samples = samples
        self._floor_var = floor_var
        self.mean: FloatArray = np.zeros(N_FEATURES)
        self.cov: FloatArray = np.eye(N_FEATURES)
        self._precision: FloatArray = np.eye(N_FEATURES)
        self._refit()

    def add(self, samples: FloatArray) -> None:
        self.samples = np.vstack([self.samples, samples])
        self._refit()

    def _refit(self) -> None:
        self.mean = self.samples.mean(axis=0)
        n = self.samples.shape[0]
        sample_cov = np.cov(self.samples, rowvar=False) if n > 1 else np.zeros((4, 4))
        self.cov = np.asarray(sample_cov, dtype=np.float64) + np.diag(self._floor_var)
        self._precision = np.asarray(np.linalg.inv(self.cov), dtype=np.float64)

    def distance(self, z: FloatArray) -> float:
        d = z - self.mean
        return math.sqrt(max(float(d @ self._precision @ d), 0.0))


class MahalanobisStallClassifier:
    def __init__(
        self,
        ood_threshold: float = 3.0,
        sigma_floor: Sequence[float] = DEFAULT_SIGMA_FLOOR,
    ) -> None:
        floor = np.asarray(sigma_floor, dtype=np.float64)
        if floor.shape != (N_FEATURES,) or np.any(floor <= 0.0):
            raise ValueError(f"sigma_floor needs {N_FEATURES} positive values")
        if ood_threshold <= 0.0:
            raise ValueError("ood_threshold must be positive")
        self.ood_threshold = ood_threshold
        self._floor_var: FloatArray = floor**2
        self._protos: dict[str, dict[str, _Prototype]] = {}

    def register_prototype(self, part_sku_id: str, signature_id: str, samples: ArrayLike) -> None:
        """Add one (1x4 or length-4) or more (n x 4) stall vectors to a prototype."""
        a = _as_samples(samples)
        by_sig = self._protos.setdefault(part_sku_id, {})
        proto = by_sig.get(signature_id)
        if proto is None:
            by_sig[signature_id] = _Prototype(a, self._floor_var)
        else:
            proto.add(a)

    def signatures(self, part_sku_id: str) -> tuple[str, ...]:
        return tuple(self._protos.get(part_sku_id, {}))

    def sample_count(self, part_sku_id: str, signature_id: str) -> int:
        return int(self._protos[part_sku_id][signature_id].samples.shape[0])

    def classify(self, part_sku_id: str, z: ArrayLike) -> StallMatch:
        zv = _as_samples(z)
        if zv.shape[0] != 1:
            raise ValueError("classify takes a single stall vector")
        best_id: str | None = None
        best = math.inf
        for sig, proto in self._protos.get(part_sku_id, {}).items():
            d = proto.distance(zv[0])
            if d < best:
                best_id, best = sig, d
        if best_id is None or best > self.ood_threshold:
            return StallMatch(signature_id=None, distance=best, is_ood=True)
        return StallMatch(signature_id=best_id, distance=best, is_ood=False)
