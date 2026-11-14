"""Recovery backends: how a bay recovers by itself after a matched stall (ACT recovery spec 1).

``recovery_backend`` picks between today's DMP replay (``dmp``, the default and the fallback),
a gated ACT policy with the DMP behind it (``hybrid``), or ACT only (``act``). The coordinator
asks :func:`choose_recovery` at each stall and drives the chosen :class:`RecoveryController`
once per 60 Hz tick: ``command`` in the control phase, ``check`` after physics. Bolt clearance,
the arm torque guard and escalation stay in the coordinator, the same for every backend.

:class:`DmpRecovery` is the coordinator's original autonomous path moved here unchanged: a 1 cm
back-off along latch -X, a blend into the DMP, the replay, the additive torque tube and the
skill timeout.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

from teleop.atc.types import SkillPrimitive, StallSignature
from teleop.atc.validator import (
    CONTROL_DT,
    BayRig,
    ReplayCommand,
    SkillReplay,
    blend_pose,
    local_axis_world,
    torque_tube_ok,
    translate_pose,
)
from teleop.types import Pose

RecoveryKind = Literal["act", "dmp"]


class RecoveryBackend(StrEnum):
    DMP = "dmp"
    HYBRID = "hybrid"
    ACT = "act"


_CYCLE = (RecoveryBackend.DMP, RecoveryBackend.HYBRID, RecoveryBackend.ACT)


def next_backend(backend: RecoveryBackend) -> RecoveryBackend:
    """The ``M`` key order: dmp -> hybrid -> act -> dmp."""
    return _CYCLE[(_CYCLE.index(backend) + 1) % len(_CYCLE)]


def choose_recovery(
    backend: RecoveryBackend,
    *,
    act_covers: bool,
    act_healthy: bool,
    dmp_available: bool,
    act_failed: bool = False,
) -> RecoveryKind | None:
    """Which recovery runs for a stall, or ``None`` for the operator queue."""
    act_ok = act_covers and act_healthy and not act_failed
    if backend is RecoveryBackend.DMP:
        return "dmp" if dmp_available else None
    if act_ok:
        return "act"
    if backend is RecoveryBackend.HYBRID and dmp_available:
        return "dmp"
    return None


class SmoothMove:
    """Smoothstep wrist move between two world poses, one ``CONTROL_DT`` per :meth:`step`."""

    def __init__(self, start: Pose, end: Pose, duration_s: float) -> None:
        self.start, self.end, self.duration_s = start, end, max(duration_s, CONTROL_DT)
        self.t = 0.0

    def step(self) -> Pose:
        self.t = min(self.t + CONTROL_DT, self.duration_s)
        u = self.t / self.duration_s
        return blend_pose(self.start, self.end, u * u * (3.0 - 2.0 * u))

    @property
    def done(self) -> bool:
        return self.t >= self.duration_s


@dataclass(frozen=True)
class BaySignals:
    """What a recovery controller may read about its bay each tick, besides the rig."""

    bay_id: int
    t_s: float  # seconds since this recovery started
    tau_hand_inf: float
    confidence: float
    fingers: Mapping[str, float]  # current hand targets by name


@dataclass(frozen=True)
class RecoveryCommand:
    wrist: Pose  # world frame
    fingers: Mapping[str, float] | None = None  # None keeps the current hand targets
    v_cmd: float | None = None  # None: the coordinator derives it from the wrist motion


class RecoveryController(Protocol):
    name: str

    def command(self, rig: BayRig, s: BaySignals) -> RecoveryCommand | None:
        """This tick's command, or ``None`` to hold the current pose."""
        ...

    def check(self, rig: BayRig, s: BaySignals) -> str | None:
        """Abort reason after physics, or ``None`` to continue."""
        ...

    def close(self) -> None: ...


class RecoveryFactory(Protocol):
    name: str
    unhealthy_reason: str

    def covers(self, z: StallSignature) -> bool: ...
    def healthy(self) -> bool: ...
    def create(self, rig: BayRig, z: StallSignature) -> RecoveryController: ...
    def close(self) -> None: ...


EpisodeStartSource = Literal["operator", "dmp", "act"]


class CoordinatorObserver(Protocol):
    """Episode hook (recording): one episode per rescue or autonomous recovery attempt."""

    def on_episode_start(
        self,
        bay_id: int,
        source: EpisodeStartSource,
        signature_id: str | None,
        signature: StallSignature | None,
    ) -> None: ...

    def on_tick(
        self, bay_id: int, rig: BayRig, wrist_world: Pose, fingers: Mapping[str, float]
    ) -> None: ...

    def on_episode_end(self, bay_id: int, success: bool, aborted_reason: str | None) -> None: ...


class DmpRecovery:
    """Back off, blend into the DMP and replay it inside the additive torque tube."""

    name = "dmp"

    def __init__(
        self,
        skill: SkillPrimitive,
        rig: BayRig,
        *,
        backoff_m: float,
        backoff_s: float,
        blend_s: float,
        hold_s: float,
        tau_margin: float,
        tau_hard_max: float,
        timeout_s: float,
        on_replay_start: Callable[[], None] | None = None,
    ) -> None:
        self.skill = skill
        self._on_replay_start = on_replay_start
        self.blend_s, self.hold_s = blend_s, hold_s
        self.tau_margin, self.tau_hard_max, self.timeout_s = tau_margin, tau_hard_max, timeout_s
        wrist = rig.wrist_pose()
        ax = local_axis_world(rig.world_from_latch(), 0)
        self.backoff: SmoothMove | None = SmoothMove(
            wrist, translate_pose(wrist, -backoff_m * ax), backoff_s
        )
        self.replay: SkillReplay | None = None
        self.t = 0.0
        self.last: ReplayCommand | None = None

    def command(self, rig: BayRig, s: BaySignals) -> RecoveryCommand | None:
        if self.backoff is not None:
            pose = self.backoff.step()
            if self.backoff.done:
                self.replay = SkillReplay(
                    self.skill,
                    rig.world_from_latch(),
                    start_wrist=pose,
                    start_fingers=s.fingers,
                    blend_s=self.blend_s,
                    hold_s=self.hold_s,
                )
                self.backoff = None
                self.t = 0.0
                if self._on_replay_start is not None:
                    self._on_replay_start()
            return RecoveryCommand(pose)
        assert self.replay is not None
        cmd = self.replay.command(self.t)
        self.last = cmd
        self.t += CONTROL_DT
        return RecoveryCommand(cmd.wrist, cmd.fingers)

    @property
    def in_replay(self) -> bool:
        """True once the back-off is done and the replay has issued its first command."""
        return self.replay is not None and self.last is not None

    @property
    def touch_offset_m(self) -> float | None:
        """The skill's demo-calibrated contact offset (``None``: no replay touch)."""
        return self.skill.touch_offset_m

    def rebase(self, world_from_latch: Pose) -> None:
        """Re-anchor the running replay on a touch-corrected latch estimate."""
        if self.replay is not None:
            self.replay.rebase(world_from_latch, self.t)

    def check(self, rig: BayRig, s: BaySignals) -> str | None:
        if self.replay is None or self.last is None:
            return None
        in_dmp = self.t > self.replay.blend_s + CONTROL_DT
        timed_out = self.t > self.replay.total_s + self.timeout_s
        if in_dmp and not torque_tube_ok(
            s.tau_hand_inf, self.last.tau_demo, self.tau_margin, self.tau_hard_max
        ):
            return "torque tube violated"
        if timed_out:
            return "skill timeout"
        return None

    def close(self) -> None:
        pass
