"""Headless scripted rescue episode generator (``teleop atc --headless --generate N``).

Runs the fleet through jam cycles with no human: each cycle arms one rig fault on a bay, nudges
the latch to a random pose while the bay is between cycles, holds the clutch on the queued bay
and feeds it the scripted rescue (with time-scaling and wrist noise so no two demonstrations are
identical). The ``EpisodeSink`` records every attempt; DMP recoveries of an already learned fault
are recorded too (``source: "dmp"``). Generation stops after ``n`` successful episodes or ``3n``
attempts.

The latch only moves while its bay is between cycles (``set_world_latch`` resets the estimate to
the exact pose; the coordinator samples a fresh noisy estimate when the next cycle starts), and
the generator never calls ``set_estimate``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from teleop.atc.coordinator import OperatorInput
from teleop.atc.stage import BAY_KEYS, ScriptedGlove
from teleop.atc.types import BOLT_CLEARED_M, BayState, LatchPose
from teleop.types import Pose

if TYPE_CHECKING:
    from teleop.atc.episodes import EpisodeSink
    from teleop.atc.stage import StageController

logger = logging.getLogger(__name__)

HOLD_AFTER_CLEAR_TICKS = 15  # keep the clutch a moment past the cleared bolt, as an operator would
MAX_ATTEMPT_FACTOR = 3
FAULT_ARMS: dict[str, tuple[int, bool, bool]] = {  # name -> (bay, flap_jammed, bolt_pinned)
    "bay1_flap": (1, True, False),
    "bay2_flap": (2, True, False),
    "bay2_pin": (2, False, True),
}


@dataclass(frozen=True)
class GeneratorConfig:
    perturb_xy_m: float = 0.10
    perturb_yaw_deg: float = 40.0
    timing_jitter: float = 0.2
    pose_noise_m: float = 0.002
    # A pinned bolt is refused to the human and no rescue can clear it, so "bay2_pin" is accepted
    # but not a default: its episodes could only fail and every one would burn the time budget.
    faults: tuple[str, ...] = ("bay1_flap", "bay2_flap")
    max_sim_s_per_episode: float = 30.0


def latch_perturbation(
    rng: np.random.Generator, config: GeneratorConfig
) -> tuple[float, float, float]:
    """One latch nudge: ``(dx m, dy m, dyaw deg)``, uniform within the configured bounds."""
    dx, dy = rng.uniform(-config.perturb_xy_m, config.perturb_xy_m, size=2)
    dyaw = rng.uniform(-config.perturb_yaw_deg, config.perturb_yaw_deg)
    return float(dx), float(dy), float(dyaw)


class _JitteredGlove(ScriptedGlove):
    """``ScriptedGlove`` with the rescue played slightly faster or slower and a noisy wrist."""

    def __init__(
        self, controller: StageController, rng: np.random.Generator, config: GeneratorConfig
    ) -> None:
        super().__init__(controller)
        self._rng = rng
        self._config = config
        self._scale = 1.0
        self._acc = 0.0
        self._held: OperatorInput | None = None

    def latest(self) -> OperatorInput | None:
        if self.controller.rescue_bay() is None or not self.controller.console.clutch:
            self._held = None
            self._acc = 0.0
            return super().latest()  # resets the base class; returns None
        if self._held is None:
            self._scale = 1.0 + float(
                self._rng.uniform(-self._config.timing_jitter, self._config.timing_jitter)
            )
        self._acc += 1.0 / self._scale
        steps = int(self._acc)
        self._acc -= steps
        sample = self._held
        for _ in range(max(steps, 1 if self._held is None else 0)):
            fresh = super().latest()
            if fresh is not None:
                sample = fresh
        if sample is None:
            return None
        self._held = sample
        return self._noisy(sample)

    def _noisy(self, sample: OperatorInput) -> OperatorInput:
        pos = sample.glove_wrist.position
        if pos is None or self._config.pose_noise_m <= 0.0:
            return sample
        noise = self._rng.normal(0.0, self._config.pose_noise_m, size=3)
        wrist = Pose(
            orientation=sample.glove_wrist.orientation,
            position=(pos[0] + noise[0], pos[1] + noise[1], pos[2] + noise[2]),
        )
        return OperatorInput(wrist, sample.hand_targets)


class _Counter:
    """Observer shim: forwards to the sink and counts successful episodes it saved."""

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.successes = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    def on_episode_end(self, bay_id: int, success: bool, aborted_reason: str | None) -> None:
        if success:
            self.successes += 1
        self.inner.on_episode_end(bay_id, success, aborted_reason)  # type: ignore[attr-defined]


def _between_cycles(controller: StageController, bay: int) -> bool:
    co = controller.coordinator
    return co.state(bay) == BayState.NOMINAL and bool(co.outcome(bay)) and not co.in_routine(bay)


def run_generation(
    controller: StageController,
    sink: EpisodeSink | None,
    n: int,
    seed: int,
    config: GeneratorConfig = GeneratorConfig(),  # noqa: B008
) -> int:
    """Generate scripted rescue episodes into ``sink``; returns the successful ones recorded."""
    if sink is None:
        raise ValueError("run_generation needs an episode sink")
    rng = np.random.default_rng(seed)
    co = controller.coordinator
    counter = _Counter(sink)
    co.set_observer(counter)  # type: ignore[arg-type]
    nominal = {b: co.rig(b).true_world_latch() for b in controller.active_bays}
    glove = _JitteredGlove(controller, rng, config)
    driver = _Driver(controller, glove, config, nominal)
    faults = [f for f in config.faults if FAULT_ARMS[f][0] in controller.active_bays]
    if not faults:
        raise ValueError("no configured fault targets an active bay")
    attempts = 0
    while counter.successes < n and attempts < MAX_ATTEMPT_FACTOR * n:
        attempts += 1
        fault = faults[int(rng.integers(len(faults)))]
        bay, jam, pin = FAULT_ARMS[fault]
        while not driver.run(lambda bay=bay: _between_cycles(controller, bay)):  # noqa: B023
            pass
        _perturb_latch(controller, bay, nominal[bay], rng, config)
        controller.arm(bay, flap_jammed=jam, bolt_pinned=pin)
        logger.info("generate: attempt %d fault %s (%d ok)", attempts, fault, counter.successes)
        _attempt(driver, bay)
    return counter.successes


class _Driver:
    """Ticks the stage, rescuing whichever bay asks for the operator, until a goal holds."""

    def __init__(
        self,
        controller: StageController,
        glove: _JitteredGlove,
        config: GeneratorConfig,
        nominal: dict[int, LatchPose],
    ) -> None:
        self.controller = controller
        self.glove = glove
        self.config = config
        self.nominal = nominal
        self._engaged = False
        self._clear_ticks = 0

    def run(self, goal: Callable[[], bool]) -> bool:
        """Tick until ``goal()``; on a stuck stage reset it and return ``False``."""
        ctl = self.controller
        co = ctl.coordinator
        budget = self.config.max_sim_s_per_episode
        t_end = co.time_s + budget
        while True:
            self._rescue_step()
            ctl.tick(self.glove.latest() if self._engaged else None)
            if goal():
                return True
            if co.time_s > t_end:
                logger.warning("generate: no progress in %.0f s; resetting the stage", budget)
                self._engaged = False
                self._clear_ticks = 0
                for bay, pose in self.nominal.items():  # a bad nudge must not stall the bay forever
                    co.rig(bay).set_world_latch(pose)
                ctl.reset()  # restarts every bay and releases the clutch
                return False

    def _rescue_step(self) -> None:
        ctl = self.controller
        bay = ctl.rescue_bay()
        if self._engaged and bay is None:
            self._release()  # the rescue left the operator's hands (validation, abort)
            return
        if bay is None:
            return
        if not self._engaged:
            ctl.on_key(BAY_KEYS[bay])
            ctl.mcu.set_clutch(True)  # absolute, never a toggle that can drift
            self._engaged = True
            return
        if ctl.coordinator.rig(bay).bolt_x() >= BOLT_CLEARED_M + ScriptedGlove.CLEAR_MARGIN_M:
            self._clear_ticks += 1
            if self._clear_ticks >= HOLD_AFTER_CLEAR_TICKS:
                self._release()

    def _release(self) -> None:
        self.controller.mcu.set_clutch(False)
        self._engaged = False
        self._clear_ticks = 0


def _perturb_latch(
    controller: StageController,
    bay: int,
    base: LatchPose,
    rng: np.random.Generator,
    config: GeneratorConfig,
) -> None:
    dx, dy, dyaw = latch_perturbation(rng, config)
    pose = base.offset(dx, dy, math.radians(dyaw))
    controller.coordinator.rig(bay).set_world_latch(pose)


def _attempt(driver: _Driver, bay: int) -> None:
    """One armed cycle: run until the bay has left its idle state and come back to it."""
    controller = driver.controller
    started = [False]

    def finished() -> bool:
        if not _between_cycles(controller, bay):
            started[0] = True
        return started[0] and _between_cycles(controller, bay)

    driver.run(finished)
    for _ in range(3):  # let a released clutch register before the next attempt
        controller.tick(None)
