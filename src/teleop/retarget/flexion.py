"""Glove joint rotations to scalar flexion values (swing-twist decomposition)."""

from __future__ import annotations

import math
from typing import Literal

from teleop.sim.hands import FlexMap
from teleop.types import HandFrame, Quat

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_EPS = 1e-9


def _wrap(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    a = math.remainder(angle, 2.0 * math.pi)
    return math.pi if a <= -math.pi else a


def twist_angle(q: Quat, axis: Literal["x", "y", "z"]) -> float:
    """Signed angle in (-pi, pi] of the twist of ``q`` (xyzw) about ``axis``."""
    proj = q[_AXIS_INDEX[axis]]  # vector part projected on a unit basis axis
    w = q[3]
    if math.hypot(proj, w) < _EPS:  # 180 degree pure swing: twist undefined
        return 0.0
    return _wrap(2.0 * math.atan2(proj, w))


def flex_value(frame: HandFrame, fm: FlexMap) -> float | None:
    """Weighted sum of twist angles over present joints, or None if none are present."""
    weights = fm.weights if fm.weights is not None else [1.0] * len(fm.joints)
    total = 0.0
    found = False
    for joint, weight in zip(fm.joints, weights, strict=True):
        q = frame.joints.get(joint)
        if q is None:
            continue
        total += weight * twist_angle(q, fm.axis)
        found = True
    if not found:
        return None
    return -total if fm.invert else total
