"""MuJoCo robot sink: name-mapped joint targets plus mocap wrist poses."""
# pyright: reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np

from teleop.types import SIDES, JointTargets, Pose, RobotState, Side, WristTarget


class SimSink:
    def __init__(self, model: Any, data: Any, physics_hz: int, control_hz: int) -> None:
        self.model = model
        self.data = data
        self.substeps = max(1, physics_hz // control_hz)
        model.opt.timestep = 1.0 / physics_hz
        self.unknown_names = 0
        self._act_ids: dict[str, int] = {}
        for i in range(int(model.nu)):
            self._act_ids[str(model.actuator(i).name)] = i
        self._mocap: dict[Side, int] = {}
        for side in SIDES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}/mocap")
            if bid >= 0:
                self._mocap[side] = int(model.body_mocapid[bid])
        self._joints: list[tuple[str, int]] = []
        scalar = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        for j in range(int(model.njnt)):
            name = str(model.joint(j).name)
            if name.startswith(("left/", "right/")) and int(model.jnt_type[j]) in scalar:
                self._joints.append((name, int(model.jnt_qposadr[j])))

    def apply(self, targets: Sequence[JointTargets], wrists: Sequence[WristTarget]) -> None:
        for t in targets:
            for name, value in t.positions.items():
                aid = self._act_ids.get(f"{t.side}/{name}")
                if aid is None:
                    self.unknown_names += 1
                    continue
                lo, hi = self.model.actuator_ctrlrange[aid]
                self.data.ctrl[aid] = float(np.clip(value, lo, hi))
        for w in wrists:
            mid = self._mocap.get(w.side)
            if mid is None or w.pose.position is None:
                continue
            x, y, z, ww = w.pose.orientation
            self.data.mocap_pos[mid] = w.pose.position
            self.data.mocap_quat[mid] = (ww, x, y, z)  # MuJoCo is wxyz
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

    def state(self) -> RobotState:
        qpos = {name: float(self.data.qpos[adr]) for name, adr in self._joints}
        wrist: dict[Side, Pose] = {}
        for side, mid in self._mocap.items():
            w, x, y, z = (float(v) for v in self.data.mocap_quat[mid])
            px, py, pz = (float(v) for v in self.data.mocap_pos[mid])
            wrist[side] = Pose(orientation=(x, y, z, w), position=(px, py, pz))
        return RobotState(qpos=qpos, wrist=wrist)
