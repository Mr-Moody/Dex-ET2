"""Bay physics rig, DMP replay and the 3-rollout headless validation gate (spec 4.4).

``BayRig`` wraps one bay of a compiled scene (Panda DLS IK, ``tau_ext`` estimator, latch
controller, hand actuators by name). ``SkillReplay`` turns a ``SkillPrimitive`` into per-tick
wrist and finger commands in any latch frame. ``SkillValidator`` owns one reusable headless
single-bay ``MjModel``/``MjData``: the three pose perturbations only move the latch body in the
model (no XML recompile), so a gate run costs a few hundred physics steps per rollout.

Torque convention: every ``tau_ext_inf`` used for stall signatures, the torque tube and the
hard-max check is the inf-norm over the hand finger joints only. Hand torques depend on the
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
    wrist_pose_over_latch,
)
from teleop.sim.hands import HandSpec
from teleop.sim.ik import PandaDiffIK
from teleop.sim.latch import LatchController, yaw_quat_wxyz
from teleop.sim.torques import ExternalTorqueEstimator
from teleop.types import Pose
from teleop.unoq.dmp import (
    HAND_ACTUATORS,
    compose_pose,
    evaluate_latch_frame,
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
# Fraction of each Panda joint's actuator force limit that the arm may load before a breach.
ARM_TORQUE_MARGIN = 0.8
APPROACH_SPEED_MPS = 0.4
COARSE_FACTOR = 2  # approach-only timestep multiplier (free space, no contacts)
SETTLE_MAX_TICKS = 300  # 1.5 s to converge on an approach target
APPROACH_ROT_SPEED = 1.5  # rad/s
LIFT_HEIGHT_M = 0.18  # wrist height above the latch origin for free-space transits (clears
# the curled Inspire index over the bolt hook)
PRE_APPROACH_M = 0.04  # DMP pre-approach pose: this far above the DMP start, latch +Z

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


class ApproachPlan:
    """Rate-limited free-space path through world wrist waypoints (smoothstep per segment).

    Segment durations come from ``speed`` (m/s) and ``rot_speed`` (rad/s), so the arm always
    moves under IK and the position servos, never by kinematic placement. ``index`` is the
    segment being executed (``len(waypoints)`` once finished).
    """

    def __init__(
        self,
        start: Pose,
        waypoints: Sequence[Pose],
        speed: float = APPROACH_SPEED_MPS,
        rot_speed: float = APPROACH_ROT_SPEED,
    ) -> None:
        self._segments: list[tuple[Pose, Pose, float]] = []
        prev = start
        for wp in waypoints:
            dist = float(np.linalg.norm(pose_position(wp) - pose_position(prev)))
            ra, rb = quat_to_mat(prev.orientation), quat_to_mat(wp.orientation)
            ang = float(np.linalg.norm(so3_log(ra.T @ rb)))
            dur = max(dist / speed, ang / rot_speed, CONTROL_DT)
            self._segments.append((prev, wp, dur))
            prev = wp
        self.index = 0
        self._t = 0.0
        self.last = start

    @property
    def done(self) -> bool:
        return self.index >= len(self._segments)

    def step(self) -> Pose:
        if self.done:
            return self.last
        a, b, dur = self._segments[self.index]
        self._t = min(self._t + CONTROL_DT, dur)
        u = self._t / dur
        self.last = blend_pose(a, b, u * u * (3.0 - 2.0 * u))
        if self._t >= dur:
            self.index += 1
            self._t = 0.0
        return self.last


def lifted(pose: Pose, z_world: float) -> Pose:
    """``pose`` raised to at least ``z_world``."""
    p = pose_position(pose)
    if p[2] >= z_world:
        return pose
    return Pose(orientation=pose.orientation, position=(float(p[0]), float(p[1]), z_world))


class BayRig:
    """One bay's IK, torque estimator, latch fault controller and hand actuators by name.

    The arm is only ever moved by ``command`` (IK + position servos). The single kinematic
    placement is ``home_keyframe`` once, when a scene or model is first built.
    """

    def __init__(self, model: Any, data: Any, bay_id: int) -> None:
        p = bay_prefix(bay_id)
        self.model = model
        self.data = data
        self.bay_id = bay_id
        self.prefix = p
        self.ik = PandaDiffIK(model, p)
        self.torque = ExternalTorqueEstimator(model, p)
        self.latch = LatchController(model, data, p)
        self._hand_act = {n: int(model.actuator(f"{p}{n}").id) for n in HAND_ACTUATORS}
        self._hand_qadr = {
            n: int(model.jnt_qposadr[int(model.actuator_trnid[a][0])])
            for n, a in self._hand_act.items()
        }
        self._latch_body = int(model.body(f"{p}latch_base").id)
        self._site = int(model.site(f"{p}wrist_site").id)
        self._bolt_qadr = int(model.joint(f"{p}bolt_slide").qposadr[0])
        arm = [model.joint(n) for n in self.torque.joint_names[:7]]
        self._arm_range = np.array([model.jnt_range[int(j.id)] for j in arm], dtype=np.float64)
        self._arm_qadr = np.array([int(j.qposadr[0]) for j in arm])
        # Per-joint Panda torque limits by actuator name: the menagerie panda_nohand.xml
        # forcerange (87 Nm joints 1-4, 12 Nm joints 5-7, the Franka rated torques).
        self.arm_force_limit: FloatArray = np.array(
            [
                float(model.actuator_forcerange[int(model.actuator(f"{p}actuator{i}").id)][1])
                for i in range(1, N_ARM_DOF + 1)
            ],
            dtype=np.float64,
        )

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
        """``|tau_ext|_inf`` over the hand finger joints only (arm joints come first in ``tau``)."""
        t = self.tau_ext() if tau is None else tau
        return float(np.max(np.abs(t[N_ARM_DOF:])))

    def tau_arm_inf(self, tau: FloatArray | None = None) -> float:
        t = self.tau_ext() if tau is None else tau
        return float(np.max(np.abs(t[:N_ARM_DOF])))

    def reached(self, target: Pose, pos_tol: float = 0.004, rot_tol: float = 0.04) -> bool:
        """True once the measured wrist is within tolerance of ``target`` (world frame)."""
        err = self.ik.error(self.data, target)
        return bool(np.linalg.norm(err[:3]) < pos_tol and np.linalg.norm(err[3:]) < rot_tol)

    def arm_torque_ratio(self, tau: FloatArray | None = None) -> float:
        """Largest per-joint ``|tau_ext| / force_limit`` over the 7 Panda joints."""
        t = self.tau_ext() if tau is None else tau
        return float(np.max(np.abs(t[:N_ARM_DOF]) / self.arm_force_limit))

    def arm_in_limits(self) -> bool:
        q = np.asarray(self.data.qpos, dtype=np.float64)[self._arm_qadr]
        lo, hi = self._arm_range[:, 0], self._arm_range[:, 1]
        return bool(np.all(q >= lo - JOINT_LIMIT_TOL_RAD) and np.all(q <= hi + JOINT_LIMIT_TOL_RAD))

    def hand_q(self) -> FloatArray:
        qpos = np.asarray(self.data.qpos, dtype=np.float64)
        return np.array([qpos[self._hand_qadr[n]] for n in HAND_ACTUATORS], dtype=np.float64)

    # -- control

    def home_keyframe(self) -> None:
        """One-time kinematic home keyframe (arm at ``PANDA_HOME``, open hand) at scene build."""
        reset_robot_home(self.model, self.data, self.bay_id)
        self.ik.reset()

    def reset_latch(self, flap_jammed: bool = False, bolt_pinned: bool = False) -> None:
        """New pallet cycle on the latch only (the rig fault toggles); the arm is not moved."""
        self.latch.reset(flap_jammed=flap_jammed, bolt_pinned=bolt_pinned)

    def command(self, wrist: Pose, fingers: Mapping[str, float] | None = None) -> None:
        """One 60 Hz control tick: a DLS step towards ``wrist`` plus hand targets by name."""
        self.ik.apply(self.data, self.ik.solve(self.data, wrist, CONTROL_DT))
        self.set_fingers(fingers)

    def set_fingers(self, fingers: Mapping[str, float] | None) -> None:
        for name, value in (fingers or {}).items():
            self.data.ctrl[self._hand_act[name]] = value


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
        latch_cache: dict[int, tuple[Pose, dict[str, float], float]] | None = None,
    ) -> None:
        self._latch_cache = latch_cache  # latch-frame DMP samples, shared across rollouts
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
        key = round(t * PHYSICS_HZ)
        hit = self._latch_cache.get(key) if self._latch_cache is not None else None
        if hit is None:
            hit = evaluate_latch_frame(sk, s)
            if self._latch_cache is not None:
                self._latch_cache[key] = hit
        local, fingers, tau_demo = hit
        wrist = compose_pose(self.world_from_latch, local)
        return ReplayCommand(wrist, fingers, tau_demo, t_s >= self.total_s)


# --------------------------------------------------------------------------- validation gate


@dataclass(frozen=True, slots=True)
class RolloutResult:
    perturbation: tuple[float, float, float]  # dx m, dy m, dyaw deg
    bolt_x_m: float
    max_tau_ext_inf_nm: float  # hand-only
    max_arm_torque_ratio: float
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

    Every rollout is physical: the arm lifts clear of the previous latch pose, the latch is
    moved to the perturbed frame and reset to its jammed stall state (bolt at the skill's
    recorded stall x, flap released), then the arm approaches a pre-approach pose 4 cm above
    the DMP start under IK and the position servos, blends into the DMP and replays it. A
    rollout passes when ``x_bolt >= 0.052``, hand ``max |tau_ext|_inf < tau_hard_max``, every
    Panda joint stays below ``arm_margin`` of its force limit and inside its joint limits.

    The only kinematic placement is the home keyframe applied once when the model is built.
    """

    def __init__(
        self,
        hand_spec: HandSpec,
        tau_hard_max: float = TAU_HARD_MAX_NM,
        base_latch: LatchPose = CANONICAL_LATCH,
        perturbations_deg: Sequence[tuple[float, float, float]] = PERTURBATIONS_DEG,
        arm_margin: float = ARM_TORQUE_MARGIN,
    ) -> None:
        self.hand_spec = hand_spec
        self.tau_hard_max = tau_hard_max
        self.arm_margin = arm_margin
        self.base_latch = base_latch
        self.perturbations_deg = tuple(perturbations_deg)
        self.model: Any = build_single_bay_scene(base_latch)
        self.data: Any = mj.MjData(self.model)
        self.rig = BayRig(self.model, self.data, 1)
        self.rig.home_keyframe()
        self._move_to_ready_pose()

    def _move_to_ready_pose(self) -> None:
        """Physically bring the arm from the home keyframe to a ready pose above the latch.

        Done once at construction (not part of any timed rollout), so the first rollout starts
        from a pose comparable to where every later one starts.
        """
        rig = self.rig
        ready = wrist_pose_over_latch(rig.world_latch(), LIFT_HEIGHT_M)
        plan = ApproachPlan(rig.wrist_pose(), [ready])
        rig.ik.reset()
        for _ in range(600):
            rig.command(plan.step())
            step_physics(self.model, self.data)
            if plan.done and rig.reached(ready):
                break

    def _rollout(
        self,
        skill: SkillPrimitive,
        perturbation: tuple[float, float, float],
        cache: dict[int, tuple[Pose, dict[str, float], float]],
    ) -> RolloutResult:
        dx, dy, dyaw_deg = perturbation
        rig, m, d = self.rig, self.model, self.data
        new_latch = latch_world_pose(1, self.base_latch.offset(dx, dy, math.radians(dyaw_deg)))
        new_wfl = new_latch.as_pose()
        y0 = SkillReplay(skill, new_wfl, hold_s=0.0).command(0.0)
        up = local_axis_world(new_wfl, 2)
        pre = translate_pose(y0.wrist, PRE_APPROACH_M * up)
        lift_z = new_latch.position[2] + LIFT_HEIGHT_M
        plan = ApproachPlan(
            rig.wrist_pose(), [lifted(rig.wrist_pose(), lift_z), lifted(pre, lift_z), pre]
        )

        max_tau = 0.0
        max_ratio = 0.0
        limits_ok = True

        def monitor() -> None:
            nonlocal max_tau, max_ratio, limits_ok
            tau = rig.tau_ext()
            max_tau = max(max_tau, rig.tau_hand_inf(tau))
            max_ratio = max(max_ratio, rig.arm_torque_ratio(tau))
            limits_ok = limits_ok and rig.arm_in_limits()

        latch_ready = False
        rig.ik.reset()
        settle_ticks = 0
        m.opt.timestep = COARSE_FACTOR / PHYSICS_HZ
        while (
            not plan.done
            or not latch_ready
            or (not rig.reached(pre) and settle_ticks < SETTLE_MAX_TICKS)
        ):
            settle_ticks += 1 if plan.done else 0
            if plan.index >= 1 and not latch_ready:  # lifted clear: move + reset the latch
                rig.set_world_latch(new_latch)
                rig.reset_latch(flap_jammed=True)
                rig.set_bolt_x(skill.prototype.x_bolt_m)
                latch_ready = True
            rig.command(plan.step(), y0.fingers)
            step_physics(m, d, SUBSTEPS // COARSE_FACTOR)  # coarse physics, free space only
            monitor()

        m.opt.timestep = 1.0 / PHYSICS_HZ  # full-rate physics for the contact replay
        replay = SkillReplay(
            skill,
            new_wfl,
            start_wrist=pre,
            start_fingers=y0.fingers,
            blend_s=0.2,
            hold_s=0.2,
            latch_cache=cache,
        )
        cleared = False
        for k in range(int(math.ceil(replay.total_s / CONTROL_DT))):
            cmd = replay.command(k * CONTROL_DT)
            rig.command(cmd.wrist, cmd.fingers)
            step_physics(m, d)
            monitor()
            if rig.bolt_x() >= BOLT_CLEARED_M:
                cleared = True
                break
        passed = (
            cleared and max_tau < self.tau_hard_max and max_ratio < self.arm_margin and limits_ok
        )
        return RolloutResult(
            perturbation, rig.bolt_x(), max_tau, max_ratio, limits_ok, cleared, passed
        )

    def validate(self, skill: SkillPrimitive) -> ValidationReport:
        t0 = time.perf_counter()
        cache: dict[int, tuple[Pose, dict[str, float], float]] = {}
        results = tuple(self._rollout(skill, p, cache) for p in self.perturbations_deg)
        return ValidationReport(results, time.perf_counter() - t0)


__all__ = [
    "CONTROL_DT",
    "PERTURBATIONS_DEG",
    "TAU_HARD_MAX_NM",
    "TAU_MARGIN_NM",
    "ARM_TORQUE_MARGIN",
    "ApproachPlan",
    "BayRig",
    "ReplayCommand",
    "RolloutResult",
    "SkillReplay",
    "SkillValidator",
    "ValidationReport",
    "blend_pose",
    "lifted",
    "compose_pose",
    "local_axis_world",
    "pose_position",
    "step_physics",
    "torque_tube_ok",
    "translate_pose",
]
