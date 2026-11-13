"""Lift-cube task: pick a 4 cm cube off the table and hold it above 12 cm for 0.5 s."""

from __future__ import annotations

from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np

from teleop.sim.mjtypes import MjDataLike, MjModelLike
from teleop.types import TaskScore

mj: Any = _mujoco

CUBE_HALF = 0.02
CUBE_START = (0.0, 0.1, CUBE_HALF)
LIFT_Z = 0.12
HOLD_S = 0.5
RANDOM_XY = 0.03


class LiftCube:
    name = "lift_cube"
    instruction = "pick up the cube"

    def __init__(self) -> None:
        self._reset_state(0.0)

    def _reset_state(self, t0: float) -> None:
        self._t0 = t0
        self._max_z = 0.0
        self._above_since: float | None = None
        self._success = False

    def add_to_spec(self, spec: Any) -> None:
        body = spec.worldbody.add_body(name="cube", pos=list(CUBE_START))
        body.add_joint(name="cube_free", type=mj.mjtJoint.mjJNT_FREE)
        body.add_geom(
            name="cube_geom",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[CUBE_HALF] * 3,
            mass=0.05,
            rgba=[0.9, 0.2, 0.2, 1],
            friction=[1.0, 0.005, 0.0001],
        )

    def reset(self, model: MjModelLike, data: MjDataLike, seed: int) -> None:
        rng = np.random.default_rng(seed)
        dx, dy = rng.uniform(-RANDOM_XY, RANDOM_XY, size=2)
        adr = int(model.joint("cube_free").qposadr[0])
        data.qpos[adr : adr + 7] = [
            CUBE_START[0] + dx,
            CUBE_START[1] + dy,
            CUBE_START[2],
            1,
            0,
            0,
            0,
        ]
        vadr = int(model.joint("cube_free").dofadr[0])
        data.qvel[vadr : vadr + 6] = 0.0
        mj.mj_forward(model, data)
        self._reset_state(float(data.time))

    def score(self, model: MjModelLike, data: MjDataLike) -> TaskScore:
        adr = int(model.joint("cube_free").qposadr[0])
        z = float(data.qpos[adr + 2])
        t = float(data.time)
        self._max_z = max(self._max_z, z)
        if z > LIFT_Z:
            if self._above_since is None:
                self._above_since = t
            if t - self._above_since >= HOLD_S:
                self._success = True
        else:
            self._above_since = None
        return TaskScore(success=self._success, score=self._max_z * 100.0, elapsed_s=t - self._t0)
