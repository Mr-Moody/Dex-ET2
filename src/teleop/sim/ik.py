"""Damped-least-squares differential IK for one bay's 7-DoF Franka Panda (spec 2.2).

Each ``solve`` call is one DLS step from the current ``data`` state towards a world wrist pose
on ``bay{k}/wrist_site``:

    dq = J^T (J J^T + lambda^2 I)^-1 e + (I - J^+ J) k_null (q_home - q)

clamped to ``|dq|_inf <= qdot_max * dt`` and clipped to the arm actuators' ``ctrlrange``. The
result is 7 position targets for ``bay{k}/actuator1..7``.

The step is taken from the last commanded arm configuration, not the measured one: the Panda
position servos lag by ~0.1 s, and re-linearising at the lagging measurement every 60 Hz tick
would cap the arm at a fraction of ``qdot_max``. Forward kinematics for the commanded
configuration run in a scratch ``MjData``, so the live ``data`` is only read. When the
measurement drifts more than ``resync_tol`` rad from the command (contact, a reset, a clutch
bleed-off) the solver restarts from the measurement, which bounds how far a blocked arm winds
up. ``reset()`` forces that restart.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from numpy.typing import NDArray

from teleop.types import Pose, Quat

mj: Any = _mujoco

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
# Menagerie "home" keyframe: flange pointing straight down in front of the base.
PANDA_HOME = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853)
QDOT_MAX = 2.0  # rad/s, below the Panda's 2.175-2.61 rad/s joint limits

FloatArray = NDArray[np.float64]


def _norm_prefix(prefix: str) -> str:
    return prefix if prefix.endswith("/") else f"{prefix}/"


def quat_xyzw_to_mat(q: Quat) -> FloatArray:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        raise ValueError("zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def so3_log(r: FloatArray) -> FloatArray:
    """Rotation vector (axis * angle) of a rotation matrix, stable near 0 and pi."""
    cos = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cos)
    if angle < 1e-6:
        return 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if math.pi - angle < 1e-4:
        # axis from the dominant column of (R + I) / 2 = a a^T
        b = 0.5 * (r + np.eye(3))
        k = int(np.argmax(np.diag(b)))
        axis = b[:, k] / math.sqrt(max(float(b[k, k]), 1e-12))
        return angle * axis / float(np.linalg.norm(axis))
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return angle / (2.0 * math.sin(angle)) * v


def pose_error(
    pos_cur: FloatArray, mat_cur: FloatArray, pos_des: FloatArray, mat_des: FloatArray
) -> FloatArray:
    """Stacked world-frame error ``[p_des - p_cur; R_cur Log(R_cur^T R_des)]``."""
    e_rot = mat_cur @ so3_log(mat_cur.T @ mat_des)
    return np.concatenate([pos_des - pos_cur, e_rot])


class PandaDiffIK:
    """One DLS differential-IK step per ``solve`` for the Panda of ``bay_prefix``."""

    def __init__(
        self,
        model: Any,
        bay_prefix: str,
        damping: float = 0.05,
        nullspace_gain: float = 0.5,
        qdot_max: float = QDOT_MAX,
        q_home: tuple[float, ...] = PANDA_HOME,
        resync_tol: float = 0.1,
    ) -> None:
        p = _norm_prefix(bay_prefix)
        self._model = model
        self._scratch: Any = mj.MjData(model)
        self._q_cmd: FloatArray | None = None
        self.damping = damping
        self.nullspace_gain = nullspace_gain
        self.qdot_max = qdot_max
        self.resync_tol = resync_tol
        self.q_home: FloatArray = np.asarray(q_home, dtype=float)
        self.site_id = int(model.site(f"{p}wrist_site").id)
        joints = [model.joint(f"{p}{n}") for n in ARM_JOINTS]
        self.qpos_adr = np.array([int(j.qposadr[0]) for j in joints])
        self.dof_adr = np.array([int(j.dofadr[0]) for j in joints])
        acts = [model.actuator(f"{p}{n}") for n in ARM_ACTUATORS]
        self.actuator_ids = np.array([int(a.id) for a in acts])
        ranges = np.asarray(model.actuator_ctrlrange)[self.actuator_ids]
        self.q_min: FloatArray = ranges[:, 0].astype(float)
        self.q_max: FloatArray = ranges[:, 1].astype(float)
        nv = int(model.nv)
        self._jacp: FloatArray = np.zeros((3, nv))
        self._jacr: FloatArray = np.zeros((3, nv))

    def reset(self) -> None:
        """Forget the commanded configuration; the next ``solve`` starts from the measurement."""
        self._q_cmd = None

    def site_pose(self, data: Any) -> tuple[FloatArray, FloatArray]:
        """World position and rotation matrix of the wrist site in ``data``."""
        pos = np.array(data.site_xpos[self.site_id], dtype=float)
        mat = np.array(data.site_xmat[self.site_id], dtype=float).reshape(3, 3)
        return pos, mat

    def error(self, data: Any, target_pose: Pose) -> FloatArray:
        """6-vector pose error of the wrist site in ``data`` (measured, not commanded)."""
        if target_pose.position is None:
            raise ValueError("target_pose needs a position")
        pos, mat = self.site_pose(data)
        des_pos = np.asarray(target_pose.position, dtype=float)
        return pose_error(pos, mat, des_pos, quat_xyzw_to_mat(target_pose.orientation))

    def _base_config(self, data: Any) -> FloatArray:
        q_meas = np.asarray(data.qpos, dtype=float)[self.qpos_adr]
        q_cmd = self._q_cmd
        if q_cmd is None or float(np.max(np.abs(q_cmd - q_meas))) > self.resync_tol:
            return q_meas
        return q_cmd

    def solve(self, data: Any, target_pose: Pose, dt: float) -> FloatArray:
        """Return 7 arm joint position targets one DLS step closer to ``target_pose``."""
        m, s = self._model, self._scratch
        q = self._base_config(data)
        s.qpos[:] = data.qpos
        s.qpos[self.qpos_adr] = q
        mj.mj_kinematics(m, s)
        mj.mj_comPos(m, s)

        e = self.error(s, target_pose)
        mj.mj_jacSite(m, s, self._jacp, self._jacr, self.site_id)
        jac = np.vstack([self._jacp[:, self.dof_adr], self._jacr[:, self.dof_adr]])

        jjt = jac @ jac.T + (self.damping**2) * np.eye(6)
        dq_task = jac.T @ np.linalg.solve(jjt, e)
        null = np.eye(7) - np.linalg.pinv(jac) @ jac
        dq = dq_task + null @ (self.nullspace_gain * (self.q_home - q))

        step = self.qdot_max * dt
        q_next = np.clip(q + np.clip(dq, -step, step), self.q_min, self.q_max)
        self._q_cmd = q_next
        return q_next

    def apply(self, data: Any, q_target: FloatArray) -> None:
        """Write arm position targets into ``data.ctrl`` by actuator name."""
        data.ctrl[self.actuator_ids] = q_target
