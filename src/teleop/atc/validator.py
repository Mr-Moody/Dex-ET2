"""Bay physics rig, DMP replay and the 3-rollout headless validation gate (spec 4.4).

``BayRig`` wraps one bay of a compiled scene (Panda DLS IK, ``tau_ext`` estimator, latch
controller, LEAP actuators by name). ``SkillReplay`` turns a ``SkillPrimitive`` into per-tick
wrist and finger commands in any latch frame. ``SkillValidator`` owns one reusable headless
single-bay ``MjModel``/``MjData``: the three pose perturbations only move the latch body in the
model (no XML recompile), so a gate run costs a few hundred physics steps per rollout.

Torque convention: every ``tau_ext_inf`` used for stall signatures, the torque tube and the
hard-max check is the inf-norm over the 16 LEAP hand joints only. Hand torques depend on the
contact in the latch frame, not on which Panda reached it, so they transfer between bays.

Nothing here reads ``flap_hinge``: clearance is judged from ``x_bolt``, ``tau_ext`` and the
Panda joint limits only. The latch starts each rollout in its jammed stall state through
``LatchController.reset(flap_jammed=True)``, the same rig fault the live bays use.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from numpy.typing import NDArray

from teleop.atc.types import BOLT_CLEARED_M, LatchPose, SkillPrimitive
from teleop.sim.bays import (
    CANONICAL_LATCH,
    bay_prefix,
    build_single_bay_scene,
    latch_world_pose,
    reset_robot_home,
)
from teleop.sim.hands import HandSpec
from teleop.sim.ik import PandaDiffIK
from teleop.sim.latch import LatchController, yaw_quat_wxyz
from teleop.sim.torques import ExternalTorqueEstimator
from teleop.types import Pose
from teleop.unoq.dmp import (
    LEAP_ACTUATORS,
    compose_pose,
    evaluate_rbf_dmp,
    mat_to_quat,
    quat_to_mat,
    so3_exp,
    so3_log,
)

mj: Any = _mujoco

FloatArray = NDArray[np.float64]

N_ARM_DOF = 7

PHYSICS_HZ = 600
SUBSTEPS = 10  # 60 Hz control on 600 Hz physics
CONTROL_DT = SUBSTEPS / PHYSICS_HZ

TAU_MARGIN_NM = 3.5
TAU_HARD_MAX_NM = 12.0
JOINT_LIMIT_TOL_RAD = 0.02

# (dx m, dy m, dyaw deg) in the bay-local latch frame, spec 4.4.
PERTURBATIONS_DEG: tuple[tuple[float, float, float], ...] = (
    (-0.08, -0.06, -30.0),
    (0.0, 0.0, 0.0),
    (0.08, 0.06, 30.0),
)
VALIDATION_BUDGET_S = 0.4


# --------------------------------------------------------------------------- small helpers


def pose_position(pose: Pose) -> FloatArray:
    if pose.position is None:
        raise ValueError("pose needs a position")
    return np.asarray(pose.position, dtype=np.float64)


def blend_pose(a: Pose, b: Pose, alpha: float) -> Pose:
    """Linear position and geodesic rotation interpolation, ``alpha`` in [0, 1]."""
    pa, pb = pose_position(a), pose_position(b)
    ra, rb = quat_to_mat(a.orientation), quat_to_mat(b.orientation)
    p = pa + alpha * (pb - pa)
    r = ra @ so3_exp(alpha * so3_log(ra.T @ rb))
    return Pose(orientation=mat_to_quat(r), position=(float(p[0]), float(p[1]), float(p[2])))


def local_axis_world(world_from_latch: Pose, axis: int = 0) -> FloatArray:
    """Unit vector of latch-frame axis ``axis`` (0 = X) expressed in the world."""
    return np.asarray(quat_to_mat(world_from_latch.orientation)[:, axis], dtype=np.float64)


def translate_pose(pose: Pose, delta: Sequence[float] | FloatArray) -> Pose:
    p = pose_position(pose) + np.asarray(delta, dtype=np.float64)
    return Pose(orientation=pose.orientation, position=(float(p[0]), float(p[1]), float(p[2])))


def torque_tube_ok(
    tau_ext_inf: float,
    tau_demo: float,
    tau_margin: float = TAU_MARGIN_NM,
    tau_hard_max: float = TAU_HARD_MAX_NM,
) -> bool:
    """Additive Residual Torque Tube (spec 4.4).

    ``|tau_ext - tau_demo|_inf <= tau_margin`` and ``|tau_ext|_inf < tau_hard_max``. The
    residual is additive, so a zero demo torque in free space never trips on a percentage of
    zero.
    """
    return abs(tau_ext_inf - tau_demo) <= tau_margin and tau_ext_inf < tau_hard_max


# --------------------------------------------------------------------------- bay rig


class BayRig:
    """One bay's IK, torque estimator, latch fault controller and LEAP actuators by name."""

    def __init__(self, model: Any, data: Any, bay_id: int) -> None:
        p = bay_prefix(bay_id)
        self.model = model
        self.data = data
        self.bay_id = bay_id
        self.prefix = p
        self.ik = PandaDiffIK(model, p)
        self.torque = ExternalTorqueEstimator(model, p)
        self.latch = LatchController(model, data, p)
        self._leap_act = {n: int(model.actuator(f"{p}{n}").id) for n in LEAP_ACTUATORS}
        self._leap_qadr = {
            n: int(model.jnt_qposadr[int(model.actuator_trnid[a][0])])
            for n, a in self._leap_act.items()
        }
        self._latch_body = int(model.body(f"{p}latch_base").id)
        self._site = int(model.site(f"{p}wrist_site").id)
        self._bolt_qadr = int(model.joint(f"{p}bolt_slide").qposadr[0])
        arm = [model.joint(n) for n in self.torque.joint_names[:7]]
        self._arm_range = np.array([model.jnt_range[int(j.id)] for j in arm], dtype=np.float64)
        self._arm_qadr = np.array([int(j.qposadr[0]) for j in arm])

    # -- frames

    def world_latch(self) -> LatchPose:
        pos = self.model.body_pos[self._latch_body]
        q = self.model.body_quat[self._latch_body]  # wxyz, a pure yaw
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return LatchPose(position=(float(pos[0]), float(pos[1]), float(pos[2])), yaw_rad=yaw)

    def world_from_latch(self) -> Pose:
        return self.world_latch().as_pose()

    def set_world_latch(self, pose: LatchPose) -> None:
        """Move the latch body in the compiled model (no recompile). Call before ``reset``."""
        self.model.body_pos[self._latch_body] = np.asarray(pose.position, dtype=np.float64)
        self.model.body_quat[self._latch_body] = np.asarray(
            yaw_quat_wxyz(pose.yaw_rad), dtype=np.float64
        )

    # -- state

    def wrist_pose(self) -> Pose:
        pos = np.asarray(self.data.site_xpos[self._site], dtype=np.float64)
        mat = np.asarray(self.data.site_xmat[self._site], dtype=np.float64).reshape(3, 3)
        return Pose(
            orientation=mat_to_quat(mat), position=(float(pos[0]), float(pos[1]), float(pos[2]))
        )

    def bolt_x(self) -> float:
        return self.latch.bolt_x()

    def set_bolt_x(self, x: float) -> None:
        self.data.qpos[self._bolt_qadr] = x
        mj.mj_forward(self.model, self.data)

    def tau_ext(self) -> FloatArray:
        return self.torque.estimate(self.data)

    def tau_hand_inf(self, tau: FloatArray | None = None) -> float:
        """``|tau_ext|_inf`` over the 16 LEAP joints only (arm joints come first in ``tau``)."""
        t = self.tau_ext() if tau is None else tau
        return float(np.max(np.abs(t[N_ARM_DOF:])))

    def tau_arm_inf(self, tau: FloatArray | None = None) -> float:
        t = self.tau_ext() if tau is None else tau
        return float(np.max(np.abs(t[:N_ARM_DOF])))

    def arm_in_limits(self) -> bool:
        q = np.asarray(self.data.qpos, dtype=np.float64)[self._arm_qadr]
        lo, hi = self._arm_range[:, 0], self._arm_range[:, 1]
        return bool(np.all(q >= lo - JOINT_LIMIT_TOL_RAD) and np.all(q <= hi + JOINT_LIMIT_TOL_RAD))

    def leap_q(self) -> FloatArray:
        qpos = np.asarray(self.data.qpos, dtype=np.float64)
        return np.array([qpos[self._leap_qadr[n]] for n in LEAP_ACTUATORS], dtype=np.float64)

    # -- control

    def reset(self, flap_jammed: bool = False, bolt_pinned: bool = False) -> None:
        reset_robot_home(self.model, self.data, self.bay_id)
        self.ik.reset()
        self.latch.reset(flap_jammed=flap_jammed, bolt_pinned=bolt_pinned)

    def command(self, wrist: Pose, fingers: Mapping[str, float] | None = None) -> None:
        """One 60 Hz control tick: a DLS step towards ``wrist`` plus LEAP targets by name."""
        self.ik.apply(self.data, self.ik.solve(self.data, wrist, CONTROL_DT))
        self.set_fingers(fingers)

    def set_fingers(self, fingers: Mapping[str, float] | None) -> None:
        for name, value in (fingers or {}).items():
            self.data.ctrl[self._leap_act[name]] = value

    def teleport_to(self, wrist: Pose, fingers: Mapping[str, float], iterations: int = 120) -> None:
        """Kinematically place the arm at ``wrist`` and the hand at ``fingers`` (validation)."""
        m, d = self.model, self.data
        self.ik.reset()
        for _ in range(iterations):
            d.qpos[self.ik.qpos_adr] = self.ik.solve(d, wrist, 0.2)
            mj.mj_kinematics(m, d)
            err = self.ik.error(d, wrist)
            if float(np.linalg.norm(err[:3])) < 5e-4 and float(np.linalg.norm(err[3:])) < 5e-3:
                break
        d.ctrl[self.ik.actuator_ids] = d.qpos[self.ik.qpos_adr]
        for name, value in fingers.items():
            d.qpos[self._leap_qadr[name]] = value
            d.ctrl[self._leap_act[name]] = value
        d.qvel[:] = 0.0
        self.ik.reset()
        mj.mj_forward(m, d)


def step_physics(model: Any, data: Any, n: int = SUBSTEPS) -> None:
    for _ in range(n):
        mj.mj_step(model, data)


# --------------------------------------------------------------------------- DMP replay


@dataclass(frozen=True, slots=True)
class ReplayCommand:
    wrist: Pose  # world frame
    fingers: dict[str, float]
    tau_demo: float  # bin-averaged nominal |tau_ext|_inf at this phase, Nm
    done: bool


class SkillReplay:
    """Time-indexed replay of a ``SkillPrimitive`` in the latch frame ``world_from_latch``.

    Optional blend-in: a smoothstep from the current wrist and finger state to the DMP start
    ``y0`` (``blend_s``). The DMP then runs for the demo duration ``T`` with phase
    ``s = exp(-alpha_s t / T)`` and holds its goal for ``hold_s``.
    """

    def __init__(
        self,
        skill: SkillPrimitive,
        world_from_latch: Pose,
        start_wrist: Pose | None = None,
        start_fingers: Mapping[str, float] | None = None,
        blend_s: float = 0.0,
        hold_s: float = 0.3,
    ) -> None:
        self.skill = skill
        self.world_from_latch = world_from_latch
        self.blend_s = blend_s if start_wrist is not None else 0.0
        self.hold_s = hold_s
        self._start_wrist = start_wrist
        self._start_fingers = dict(start_fingers or {})
        self.total_s = self.blend_s + skill.duration_s + hold_s

    def command(self, t_s: float) -> ReplayCommand:
        sk = self.skill
        if t_s < self.blend_s and self._start_wrist is not None:
            wrist0, fingers0, _ = evaluate_rbf_dmp(sk, 1.0, self.world_from_latch)
            u = min(max(t_s / self.blend_s, 0.0), 1.0)
            a = u * u * (3.0 - 2.0 * u)
            fingers = {
                n: self._start_fingers.get(n, v) + a * (v - self._start_fingers.get(n, v))
                for n, v in fingers0.items()
            }
            return ReplayCommand(blend_pose(self._start_wrist, wrist0, a), fingers, 0.0, False)
        t = min(max(t_s - self.blend_s, 0.0), sk.duration_s)
        s = math.exp(-sk.alpha_s * t / sk.duration_s)
        wrist, fingers, tau_demo = evaluate_rbf_dmp(sk, s, self.world_from_latch)
        return ReplayCommand(wrist, fingers, tau_demo, t_s >= self.total_s)


# --------------------------------------------------------------------------- validation gate


@dataclass(frozen=True, slots=True)
class RolloutResult:
    perturbation: tuple[float, float, float]  # dx m, dy m, dyaw deg
    bolt_x_m: float
    max_tau_ext_inf_nm: float
    joint_limits_ok: bool
    cleared: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class ValidationReport:
    results: tuple[RolloutResult, ...]
    elapsed_s: float

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def n_passed(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def passed(self) -> bool:
        return self.n_passed == self.total


class SkillValidator:
    """Pre-deployment gate: 3 headless single-bay rollouts across latch pose perturbations.

    The bolt starts at the skill's recorded stall position (``prototype.x_bolt_m``) with the
    stopper flap jammed, the arm teleported to the DMP start for the perturbed latch frame. A
    rollout passes when ``x_bolt >= 0.052``, ``max |tau_ext|_inf < tau_hard_max`` and the Panda
    stays inside its joint limits.
    """

    def __init__(
        self,
        hand_spec: HandSpec,
        tau_hard_max: float = TAU_HARD_MAX_NM,
        base_latch: LatchPose = CANONICAL_LATCH,
        perturbations_deg: Sequence[tuple[float, float, float]] = PERTURBATIONS_DEG,
    ) -> None:
        self.hand_spec = hand_spec
        self.tau_hard_max = tau_hard_max
        self.base_latch = base_latch
        self.perturbations_deg = tuple(perturbations_deg)
        self.model: Any = build_single_bay_scene(base_latch)
        self.data: Any = mj.MjData(self.model)
        self.rig = BayRig(self.model, self.data, 1)

    def _rollout(
        self, skill: SkillPrimitive, perturbation: tuple[float, float, float]
    ) -> RolloutResult:
        dx, dy, dyaw_deg = perturbation
        rig, m, d = self.rig, self.model, self.data
        local = self.base_latch.offset(dx, dy, math.radians(dyaw_deg))
        rig.set_world_latch(latch_world_pose(1, local))
        world_from_latch = rig.world_from_latch()

        rig.reset(flap_jammed=True)
        start = SkillReplay(skill, world_from_latch, hold_s=0.0).command(0.0)
        rig.teleport_to(start.wrist, start.fingers)
        rig.set_bolt_x(skill.prototype.x_bolt_m)

        replay = SkillReplay(skill, world_from_latch, hold_s=0.2)
        max_tau = 0.0
        limits_ok = True
        cleared = False
        ticks = int(math.ceil(replay.total_s / CONTROL_DT))
        for k in range(ticks):
            cmd = replay.command(k * CONTROL_DT)
            rig.command(cmd.wrist, cmd.fingers)
            step_physics(m, d)
            max_tau = max(max_tau, rig.tau_hand_inf())
            limits_ok = limits_ok and rig.arm_in_limits()
            if rig.bolt_x() >= BOLT_CLEARED_M:
                cleared = True
                break
        bolt_x = rig.bolt_x()
        passed = cleared and max_tau < self.tau_hard_max and limits_ok
        return RolloutResult(perturbation, bolt_x, max_tau, limits_ok, cleared, passed)

    def validate(self, skill: SkillPrimitive) -> ValidationReport:
        t0 = time.perf_counter()
        results = tuple(self._rollout(skill, p) for p in self.perturbations_deg)
        return ValidationReport(results, time.perf_counter() - t0)


__all__ = [
    "CONTROL_DT",
    "PERTURBATIONS_DEG",
    "TAU_HARD_MAX_NM",
    "TAU_MARGIN_NM",
    "BayRig",
    "ReplayCommand",
    "RolloutResult",
    "SkillReplay",
    "SkillValidator",
    "ValidationReport",
    "blend_pose",
    "compose_pose",
    "local_axis_world",
    "pose_position",
    "step_physics",
    "torque_tube_ok",
    "translate_pose",
]
