"""External joint-torque residual for one bay's arm + hand via ``mj_inverse`` (spec 4.1).

    tau_ext = tau_act - tau_id = data.qfrc_actuator[dofs] - inv.qfrc_inverse[dofs]

``tau_id`` is the rigid-body inverse dynamics (``M qacc + bias - passive``, gravity
compensation included) of the live state, computed in a scratch ``MjData`` with constraints
disabled, so contact and limit forces show up in ``tau_ext`` instead of being explained away.
The live ``data`` is only read. Not thread-safe: the model's disable flags are toggled for the
duration of the ``mj_inverse`` call.
"""

from __future__ import annotations

from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from numpy.typing import NDArray

from teleop.sim.bays import WRIST_COMPLIANCE_AXES
from teleop.sim.ik import ARM_JOINTS

mj: Any = _mujoco

FloatArray = NDArray[np.float64]

LATCH_JOINTS = ("bolt_slide", "flap_hinge")


def _norm_prefix(prefix: str) -> str:
    return prefix if prefix.endswith("/") else f"{prefix}/"


class ExternalTorqueEstimator:
    """``tau_ext`` over a bay's 7 arm + hand finger DoFs (arm first, then hand in model order)."""

    def __init__(self, model: Any, bay_prefix: str) -> None:
        p = _norm_prefix(bay_prefix)
        self._model = model
        self._scratch: Any = mj.MjData(model)
        arm = [f"{p}{n}" for n in ARM_JOINTS]
        # Passive joints that are not hand joints: the latch's and the compliant wrist's (its
        # spring load depends on which Panda posture reached the latch, the fingers' does not).
        latch = {f"{p}{n}" for n in (*LATCH_JOINTS, *WRIST_COMPLIANCE_AXES)}
        hand = [
            str(model.joint(j).name)
            for j in range(int(model.njnt))
            if str(model.joint(j).name).startswith(p)
            and str(model.joint(j).name) not in latch
            and str(model.joint(j).name) not in arm
        ]
        self.joint_names: tuple[str, ...] = (*arm, *hand)
        self.dofs = np.array([int(model.joint(n).dofadr[0]) for n in self.joint_names])

    def estimate(self, data: Any) -> FloatArray:
        m, inv = self._model, self._scratch
        inv.qpos[:] = data.qpos
        inv.qvel[:] = data.qvel
        inv.qacc[:] = data.qacc
        inv.act[:] = data.act
        inv.qfrc_applied[:] = 0.0
        inv.xfrc_applied[:] = 0.0
        saved = int(m.opt.disableflags)
        m.opt.disableflags = saved | int(mj.mjtDisableBit.mjDSBL_CONSTRAINT)
        try:
            mj.mj_inverse(m, inv)
        finally:
            m.opt.disableflags = saved
        tau_act = np.asarray(data.qfrc_actuator, dtype=float)[self.dofs]
        tau_id = np.asarray(inv.qfrc_inverse, dtype=float)[self.dofs]
        return tau_act - tau_id

    def estimate_inf(self, data: Any) -> float:
        """``|tau_ext|_inf`` in Nm, the scalar fed to C(t) and the stall signature."""
        return float(np.max(np.abs(self.estimate(data))))
