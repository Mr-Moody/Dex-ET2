"""Minimal structural types for the MuJoCo model/data attributes this package touches.

MuJoCo's bindings are untyped for pyright; these protocols list only what sink/task code
uses, so a typo in an attribute name on ``model``/``data`` is caught statically.
"""

from __future__ import annotations

from typing import Any, Protocol


class MjOptionLike(Protocol):
    timestep: float


class MjModelLike(Protocol):
    nu: int
    njnt: int
    opt: MjOptionLike
    actuator_ctrllimited: Any
    actuator_ctrlrange: Any
    body_mocapid: Any
    jnt_type: Any
    jnt_qposadr: Any

    def actuator(self, i: int | str) -> Any: ...
    def joint(self, i: int | str) -> Any: ...


class MjDataLike(Protocol):
    time: float
    qpos: Any
    qvel: Any
    ctrl: Any
    mocap_pos: Any
    mocap_quat: Any
