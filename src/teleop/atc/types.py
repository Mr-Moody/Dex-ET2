"""Shared ATC value types (T-100). Pure stdlib so ``teleop.unoq`` can import them on the UNO Q.

Positions in metres, angles in radians, quaternions xyzw. Bays are numbered 1-4.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from teleop.types import Pose, Quat, Vec3

N_BAYS = 4
BOLT_CLEARED_M = 0.052  # x_bolt at which a latch counts as unlatched (spec 2.1)


def _check_bay(bay_id: int) -> None:
    if not 1 <= bay_id <= N_BAYS:
        raise ValueError(f"bay_id must be in 1..{N_BAYS}, got {bay_id}")


class BayState(StrEnum):
    NOMINAL = "NOMINAL"
    STALLED = "STALLED"
    QUEUED = "QUEUED"
    RESCUE = "RESCUE"
    VALIDATING = "VALIDATING"
    AUTONOMOUS_SKILL = "AUTONOMOUS_SKILL"
    OOD_ESCALATED = "OOD_ESCALATED"


@dataclass(frozen=True, slots=True)
class LatchPose:
    """Latch base-track frame in the world: position plus yaw about world +Z."""

    position: Vec3
    yaw_rad: float = 0.0

    def as_pose(self) -> Pose:
        half = 0.5 * self.yaw_rad
        return Pose(orientation=(0.0, 0.0, math.sin(half), math.cos(half)), position=self.position)

    def offset(self, dx: float = 0.0, dy: float = 0.0, dyaw_rad: float = 0.0) -> LatchPose:
        """World-frame shift plus extra yaw (used for Bay 2 and the validation perturbations)."""
        x, y, z = self.position
        return LatchPose(position=(x + dx, y + dy, z), yaw_rad=self.yaw_rad + dyaw_rad)


@dataclass(frozen=True, slots=True)
class StallSignature:
    """Feature vector z over the 300 ms stall window (spec 4.2)."""

    x_bolt_m: float
    tau_ext_inf_mean_nm: float
    tau_ext_inf_max_nm: float
    wrist_tau_delta_nm: float

    def as_vector(self) -> tuple[float, float, float, float]:
        return (
            self.x_bolt_m,
            self.tau_ext_inf_mean_nm,
            self.tau_ext_inf_max_nm,
            self.wrist_tau_delta_nm,
        )

    @classmethod
    def from_vector(cls, z: Sequence[float]) -> StallSignature:
        if len(z) != 4:
            raise ValueError(f"stall signature needs 4 features, got {len(z)}")
        return cls(float(z[0]), float(z[1]), float(z[2]), float(z[3]))


@dataclass(frozen=True, slots=True)
class StallMatch:
    """Mahalanobis classifier result. OOD results carry no signature."""

    signature_id: str | None
    distance: float
    is_ood: bool

    def __post_init__(self) -> None:
        if not self.is_ood and self.signature_id is None:
            raise ValueError("an in-distribution StallMatch needs a signature_id")


def _floats(values: Sequence[Any]) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


@dataclass(frozen=True, slots=True)
class SkillPrimitive:
    """Object-centric RBF-DMP fitted in the latch frame (spec 4.3).

    Each of ``y0``, ``y_goal`` and the rows of ``weights`` has ``6 + len(actuator_names)``
    entries: wrist position in the latch frame (3), rotation log relative to
    ``goal_orientation`` (3), then hand actuator targets in ``actuator_names`` order.
    ``tau_demo`` is the bin-averaged nominal ``|tau_ext|_inf`` profile, one value per kernel.
    """

    skill_id: str
    part_sku_id: str
    signature_id: str
    duration_s: float
    alpha_s: float
    actuator_names: tuple[str, ...]
    goal_orientation: Quat  # xyzw, wrist orientation in the latch frame at the goal
    y0: tuple[float, ...]
    y_goal: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]  # n_kernels x n_dims
    tau_demo: tuple[float, ...]  # n_kernels
    prototype: StallSignature
    fit_ms: float = 0.0
    # Index-tip contact offset along latch +X, calibrated from this skill's own demonstration:
    # (tip - origin)·x - (HOOK_BACK_X_M + x_bolt) at the first bolt-encoder movement. None: no
    # replay touch re-localization for this skill.
    touch_offset_m: float | None = None

    def __post_init__(self) -> None:
        d = self.n_dims
        if len(self.y0) != d or len(self.y_goal) != d:
            raise ValueError(f"y0 and y_goal need {d} entries")
        if not self.weights or any(len(row) != d for row in self.weights):
            raise ValueError(f"weights must be n_kernels x {d}")
        if len(self.tau_demo) != self.n_kernels:
            raise ValueError(f"tau_demo needs {self.n_kernels} entries")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")

    @property
    def n_kernels(self) -> int:
        return len(self.weights)

    @property
    def n_dims(self) -> int:
        return 6 + len(self.actuator_names)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict for the UNO Q wire protocol (spec 5.1)."""
        return {
            "skill_id": self.skill_id,
            "part_sku_id": self.part_sku_id,
            "signature_id": self.signature_id,
            "duration_s": self.duration_s,
            "alpha_s": self.alpha_s,
            "actuator_names": list(self.actuator_names),
            "goal_orientation": list(self.goal_orientation),
            "y0": list(self.y0),
            "y_goal": list(self.y_goal),
            "weights": [list(row) for row in self.weights],
            "tau_demo": list(self.tau_demo),
            "prototype": list(self.prototype.as_vector()),
            "fit_ms": self.fit_ms,
            "touch_offset_m": self.touch_offset_m,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SkillPrimitive:
        q = _floats(raw["goal_orientation"])
        if len(q) != 4:
            raise ValueError("goal_orientation must be an xyzw quaternion")
        return cls(
            skill_id=str(raw["skill_id"]),
            part_sku_id=str(raw["part_sku_id"]),
            signature_id=str(raw["signature_id"]),
            duration_s=float(raw["duration_s"]),
            alpha_s=float(raw["alpha_s"]),
            actuator_names=tuple(str(n) for n in raw["actuator_names"]),
            goal_orientation=(q[0], q[1], q[2], q[3]),
            y0=_floats(raw["y0"]),
            y_goal=_floats(raw["y_goal"]),
            weights=tuple(_floats(row) for row in raw["weights"]),
            tau_demo=_floats(raw["tau_demo"]),
            prototype=StallSignature.from_vector(_floats(raw["prototype"])),
            fit_ms=float(raw.get("fit_ms", 0.0)),
            touch_offset_m=(
                None if raw.get("touch_offset_m") is None else float(raw["touch_offset_m"])
            ),
        )


@dataclass(frozen=True, slots=True)
class BaySnapshot:
    """Per-bay status published by the coordinator to the HUD, LED matrix and hooks."""

    bay_id: int
    state: BayState
    confidence: float  # C(t) in [0, 1]
    x_bolt_m: float
    tau_ext_inf_nm: float
    part_sku_id: str
    queue_position: int | None = None  # 0 = next in the FIFO rescue queue
    active_skill_id: str | None = None
    last_match: StallMatch | None = None

    def __post_init__(self) -> None:
        _check_bay(self.bay_id)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True, slots=True)
class ConsoleInputState:
    """Operator console (UNO Q MCU buttons or keyboard mirror, spec 5.2)."""

    selected_bay: int | None = None
    clutch: bool = False
    estop: bool = False
    fault_rig_latched: bool = False
    timestamp_ns: int = 0  # time.monotonic_ns() of the last change

    def __post_init__(self) -> None:
        if self.selected_bay is not None:
            _check_bay(self.selected_bay)
