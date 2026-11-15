"""Hard-coded "human operator" rescue for ``teleop atc --demo-rescue``.

One take of the pitch: Bay 1 runs a clean cycle, the fault rig jams its flap, the arm stalls
and asks for help, then an operator selects Bay 1, holds the clutch and rescues it. The thumb
presses the raised flap flat while the wrist pushes the bolt clear. The glove is scripted
(:data:`DEMO_WAYPOINTS`) but moves like a person: a reaction delay before the operator reaches
in, a hesitation at the hover, a sidestep-then-drop reach, minimum-jerk easing between
waypoints, and hand shake on top (physiological tremor at 8-11 Hz plus a slow wandering drift).

The shake is zero-mean and is scaled down while the thumb is on the flap, so the press still
lands. If the scripted push ends short of the cleared line, the operator keeps leaning in, so
the take always ends with the bolt cleared.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Generator, Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from teleop.atc.coordinator import ROUTINE, AtcCoordinator, OperatorInput
from teleop.atc.stage import BAY_KEYS, CONTROL_DT, KEY_SPACE, RESCUE_PRESS, StageController
from teleop.atc.types import BOLT_CLEARED_M, BayState
from teleop.types import Pose
from teleop.unoq.dmp import quat_to_mat

logger = logging.getLogger(__name__)

DEMO_BAY = 1
CLEAR_MARGIN_M = 0.0015
EXTRA_PUSH_M = 0.015  # extra lean past the scripted push if the bolt is not clear yet
EXTRA_PUSH_S = 1.5

# (duration s, wrist delta from the safe hover in the latch frame in metres, thumb press).
# With the 45 deg hand mount the thumb sits ~3 cm to the side of the index, so the wrist steps
# -y until the thumb is over the raised flap, drops, presses it flat, then pushes the bolt clear
# with the index on the hook. Swept on Bay 1 (seeds 0, 3, 7, background bays on): y -22..-25 mm
# and z -9..-15 mm all clear, so the centre below leaves ~1.5-3 mm for the hand shake.
DEMO_Y = -0.0235
DEMO_Z = -0.012
DEMO_PUSH_X = 0.046
DEMO_WAYPOINTS: tuple[tuple[float, tuple[float, float, float], float], ...] = (
    (0.0, (0.0, 0.0, 0.0), 0.0),
    (0.8, (0.0, DEMO_Y, -0.004), 0.0),  # sidestep over the flap
    (0.6, (0.0, DEMO_Y, DEMO_Z), 0.0),  # drop: index behind the hook, thumb above the flap
    (0.8, (0.0, DEMO_Y, DEMO_Z), 1.0),  # thumb presses the flap flat
    (0.3, (0.0, DEMO_Y, DEMO_Z), 1.0),  # hold it down
    (1.6, (DEMO_PUSH_X, DEMO_Y, DEMO_Z), 1.0),  # push the bolt clear
)


@dataclass(frozen=True)
class Shake:
    """Hand shake, in metres (latch frame) and finger fraction."""

    tremor_m: float = 0.0005  # peak of the 8-11 Hz tremor per axis
    drift_m: float = 0.0008  # std of the slow drift per axis
    drift_tau_s: float = 0.45  # drift correlation time
    finger: float = 0.02  # thumb jitter (fraction of the press)
    contact_scale: float = 0.5  # shake scale while the thumb is on the flap

    def scaled(self, factor: float) -> Shake:
        return Shake(
            self.tremor_m * factor,
            self.drift_m * factor,
            self.drift_tau_s,
            self.finger * factor,
            self.contact_scale,
        )


class HandNoise:
    """Zero-mean tremor + Ornstein-Uhlenbeck drift, one sample per control tick."""

    def __init__(self, shake: Shake, rng: np.random.Generator) -> None:
        self.shake = shake
        self.rng = rng
        self.t = 0.0
        self.freqs = rng.uniform(8.0, 11.0, size=(2, 3))  # two tremor tones per axis
        self.phases = rng.uniform(0.0, 2.0 * math.pi, size=(2, 3))
        self.drift = np.zeros(3)
        self.finger_drift = 0.0

    def step(self, scale: float = 1.0) -> tuple[npt.NDArray[np.float64], float]:
        s = self.shake
        self.t += CONTROL_DT
        tremor = (
            0.5 * s.tremor_m * np.sin(2.0 * math.pi * self.freqs * self.t + self.phases).sum(axis=0)
        )
        a = math.exp(-CONTROL_DT / s.drift_tau_s)
        self.drift = a * self.drift + math.sqrt(1.0 - a * a) * s.drift_m * self.rng.standard_normal(
            3
        )
        self.finger_drift = 0.8 * self.finger_drift + 0.6 * s.finger * self.rng.standard_normal()
        return scale * (tremor + self.drift), scale * self.finger_drift


def _min_jerk(a: float) -> float:
    return a * a * a * (10.0 + a * (-15.0 + 6.0 * a))


def human_rescue(
    coordinator: AtcCoordinator,
    bay_id: int,
    shake: Shake,
    rng: np.random.Generator,
    hesitate_s: float = 0.6,
) -> Iterator[OperatorInput]:
    """Glove samples of the flap-hold rescue with human timing and shake, from the hover.

    Hesitates at the hover, then follows :data:`DEMO_WAYPOINTS` with minimum-jerk easing,
    then leans a further :data:`EXTRA_PUSH_M` along the push. The caller stops pulling once the
    bolt is clear.
    """
    rig = coordinator.rig(bay_id)
    hover = rig.wrist_pose()
    if hover.position is None:
        raise ValueError("wrist pose has no position")
    r_world = quat_to_mat(rig.world_from_latch().orientation)
    origin = np.asarray(hover.position, dtype=np.float64)
    base = dict(ROUTINE.fingers)
    noise = HandNoise(shake, rng)

    def sample(delta: npt.NDArray[np.float64], press: float, scale: float) -> OperatorInput:
        d_pos, d_finger = noise.step(scale)
        pos = origin + r_world @ (delta + d_pos)
        fingers = dict(base)
        for name, v in RESCUE_PRESS.items():
            fingers[name] = base.get(name, 0.0) + min(max(press + d_finger, 0.0), 1.0) * v
        wrist = Pose(orientation=hover.orientation, position=(pos[0], pos[1], pos[2]))
        return OperatorInput(wrist, fingers)

    for _ in range(max(round(hesitate_s / CONTROL_DT), 1)):
        yield sample(np.zeros(3), 0.0, 1.0)

    for (_, p0, f0), (dur, p1, f1) in zip(DEMO_WAYPOINTS, DEMO_WAYPOINTS[1:], strict=False):
        n = max(round(dur / CONTROL_DT), 1)
        for k in range(1, n + 1):
            a = _min_jerk(k / n)
            delta = np.asarray(p0) + a * (np.asarray(p1) - np.asarray(p0))
            press = f0 + a * (f1 - f0)
            yield sample(delta, press, shake.contact_scale if press > 0.5 else 1.0)

    _, last, last_f = DEMO_WAYPOINTS[-1]
    push_dir = np.array([1.0, 0.0, 0.0])
    n = max(round(EXTRA_PUSH_S / CONTROL_DT), 1)
    for k in range(1, n + 1):
        delta = np.asarray(last) + _min_jerk(k / n) * EXTRA_PUSH_M * push_dir
        yield sample(delta, last_f, shake.contact_scale)
    while True:  # hold the final pose
        yield sample(np.asarray(last) + EXTRA_PUSH_M * push_dir, last_f, shake.contact_scale)


Script = Generator[OperatorInput | None, None, None]


class DemoRescueDirector:
    """Jam Bay 1, then rescue it by "hand". Same ``step()``/``done`` surface as the stage director.

    Call :meth:`step` once per tick and pass the result to ``controller.tick``. ``rescued`` is
    set once the bolt is cleared; ``deployed`` once the bay is back to nominal (skill gate run).
    """

    def __init__(
        self,
        controller: StageController,
        *,
        shake: Shake | None = None,
        seed: int = 0,
        lead_in_s: float = 2.0,
        reaction_s: float = 1.2,
        timeout_s: float = 60.0,
    ) -> None:
        self.controller = controller
        self.co = controller.coordinator
        self.shake = shake or Shake()
        self.rng = np.random.default_rng(seed)
        self.lead_in_s = lead_in_s
        self.reaction_s = reaction_s
        self.timeout_s = timeout_s
        self.rescued = False
        self.deployed = False
        self.done = False
        self.rescue_s: float | None = None
        self._script = self._run()

    @property
    def ok(self) -> bool:
        return self.rescued

    def step(self) -> OperatorInput | None:
        if self.done:
            return None
        try:
            return next(self._script)
        except StopIteration:
            self.done = True
            self.controller.beat = "demo rescue complete" if self.ok else "demo rescue FAILED"
            return None

    def _press(self, key: int | str) -> None:
        self.controller.on_key(ord(key) if isinstance(key, str) else key)

    def _idle(self, seconds: float) -> Generator[None]:
        for _ in range(max(round(seconds / CONTROL_DT), 1)):
            yield None

    def _wait(self, pred: object, budget_s: float) -> Generator[None]:
        assert callable(pred)
        t_end = self.co.time_s + budget_s
        while not pred():
            if self.co.time_s > t_end:
                raise TimeoutError
            yield None

    def _cleared(self) -> bool:
        return self.co.rig(DEMO_BAY).bolt_x() >= BOLT_CLEARED_M + CLEAR_MARGIN_M

    def _run(self) -> Script:
        co, ctl = self.co, self.controller
        try:
            ctl.beat = "demo: routine cycle"
            yield from self._wait(lambda: ctl.cleared_cycles.get(DEMO_BAY, 0) >= 1, self.timeout_s)
            yield from self._idle(self.lead_in_s)

            ctl.beat = "demo: flap jam"
            self._press("F")
            yield from self._wait(lambda: co.state(DEMO_BAY) == BayState.RESCUE, self.timeout_s)
            yield from self._idle(0.7 + self.reaction_s)  # retract settles, operator notices

            ctl.beat = "demo: operator rescue"
            self._press(BAY_KEYS[DEMO_BAY])
            yield from self._idle(0.5)
            self._press(KEY_SPACE)
            t0 = co.time_s
            for sample in human_rescue(co, DEMO_BAY, self.shake, self.rng):
                yield sample
                if self._cleared():
                    break
                if co.time_s - t0 > self.timeout_s:
                    raise TimeoutError
            self.rescued = True
            self.rescue_s = co.time_s - t0
            logger.info("demo rescue: bolt cleared in %.1f s", self.rescue_s)
            yield from self._idle(0.3)  # hold a beat before letting go
            self._press(KEY_SPACE)

            ctl.beat = "demo: skill gate"
            budget = self.timeout_s + co.config.gate_deadline_s
            yield from self._wait(lambda: co.state(DEMO_BAY) == BayState.NOMINAL, budget)
            self.deployed = True
        except TimeoutError:
            logger.warning("demo rescue: timed out at %s", ctl.beat)


__all__ = ["DEMO_WAYPOINTS", "DemoRescueDirector", "HandNoise", "Shake", "human_rescue"]
