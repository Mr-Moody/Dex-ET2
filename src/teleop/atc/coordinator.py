"""ATC fleet coordinator: confidence C(t), contact-only stall trigger, FIFO rescue queue (spec 4).

Active Bays 1 and 2 run a routine one-finger unlatching push. A stall is declared from contact
alone: ``C(t) < 0.3`` for 300 ms (18 ticks at 60 Hz) or a 3.0 s step watchdog, using only
``x_bolt``, ``tau_ext`` and the commanded versus measured wrist speed. The coordinator never
reads ``flap_hinge``; faults are injected through ``LatchController`` rig toggles and are never
queried.

On a stall the 4-feature signature goes to the UNO Q classifier:

* known signature with a deployed skill: back off 1 cm along latch -X, blend into the
  RBF-DMP warped to this bay's latch frame and run it inside the additive torque tube
  ``|tau_ext - tau_demo|_inf <= tau_margin`` and ``|tau_ext|_inf < tau_hard_max``;
* unseen or out-of-distribution signature: retract 3 cm to a zero-velocity hover and join the
  single-active-rescue FIFO queue. The queue head is handed to the operator (``RESCUE`` or,
  for a refused finite-distance OOD, ``OOD_ESCALATED``). At most one bay holds that slot.

After a clutch-held rescue clears the bolt the recording is fitted by the UNO Q, gated by
``SkillValidator`` (3 headless rollouts) and, if it passes, registered in the Fleet Library.
Compliance events and reasoning contexts are emitted without blocking the control loop.

Torque convention: ``tau_ext_inf`` for the stall signature, torque tube, rescue recordings and
snapshots is the inf-norm over the 16 LEAP joints (arm torques depend on which Panda reached
the contact, hand torques do not). C(t) keeps the spec's all-DoF (arm + hand) ``tau_ext``, which
already gives C > 0.85 nominal and C < 0.3 on a stall.

Arm torque safety: in every phase the per-joint ratio ``|tau_ext| / actuator forcerange`` of the
7 Panda joints (87 Nm joints 1-4, 12 Nm joints 5-7 from the menagerie model, read by actuator
name) must stay below ``arm_torque_margin`` (0.8). A breach in the routine is a fast-path stall
trigger (no 300 ms wait); in autonomous replay it aborts and escalates; in an operator rescue it
bleeds off and discards the recording until the clutch is released; elsewhere it bleeds off to a
zero-velocity hold. The validator fails a rollout on the same ratio.

Motion: the arm only moves through IK + the position servos. ``start_cycle`` lifts the hand
clear, resets the latch faults, then follows a rate-limited path to the routine start. The one
kinematic placement is the home keyframe applied once when the coordinator is built.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from teleop.atc.types import (
    BOLT_CLEARED_M,
    BaySnapshot,
    BayState,
    ConsoleInputState,
    LatchPose,
    SkillPrimitive,
    StallMatch,
    StallSignature,
)
from teleop.atc.validator import (
    ARM_TORQUE_MARGIN,
    CONTROL_DT,
    LIFT_HEIGHT_M,
    SETTLE_MAX_TICKS,
    TAU_HARD_MAX_NM,
    TAU_MARGIN_NM,
    ApproachPlan,
    BayRig,
    ReplayCommand,
    SkillReplay,
    SkillValidator,
    ValidationReport,
    blend_pose,
    lifted,
    local_axis_world,
    pose_position,
    step_physics,
    torque_tube_ok,
    translate_pose,
)
from teleop.compliance.protocol import ComplianceKind, ComplianceSink, make_event
from teleop.reasoning.protocol import FleetReasoner, ReasoningContext
from teleop.retarget.clutch import (
    RelativeClutchIndexer,
    RescueTrajectory,
    RescueTrajectoryBuffer,
    bleed_off_targets,
)
from teleop.sim.bays import (
    ACTIVE_BAYS,
    BACKGROUND_BAYS,
    BAY2_LATCH,
    CANONICAL_LATCH,
    build_four_bay_scene,
    reset_robot_home,
)
from teleop.sim.hands import HandSpec, load_hand
from teleop.types import Pose
from teleop.unoq.client import UnoQClient
from teleop.unoq.dmp import LEAP_ACTUATORS, compose_pose, mat_to_quat

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

DEFAULT_SKU = "ev_latch_v1"
CONTROL_HZ = 60.0


# --------------------------------------------------------------------------- confidence C(t)


def compute_confidence(
    tau_ext_inf: float,
    v_cmd: float,
    v_act: float,
    alpha: float = 0.8,
    beta: float = 0.9,
    tau_limit: float = 6.0,
    v_nom: float = 0.05,
) -> float:
    """Spec 4.1: ``C = exp(-alpha |tau_ext|_inf / tau_limit - beta max(0, v_cmd - v_act) / v_nom)``.

    ``tau_ext_inf`` in Nm, speeds in m/s. Always in (0, 1].
    """
    tau_term = alpha * abs(tau_ext_inf) / tau_limit
    lag_term = beta * max(0.0, v_cmd - v_act) / v_nom
    return math.exp(-(tau_term + lag_term))


# --------------------------------------------------------------------------- config


@dataclass(frozen=True, slots=True)
class CoordinatorConfig:
    part_sku_id: str = DEFAULT_SKU
    alpha: float = 0.8
    beta: float = 0.9
    tau_limit: float = 6.0
    v_nom: float = 0.05
    stall_confidence: float = 0.3
    stall_ticks: int = 18  # 300 ms at 60 Hz
    watchdog_s: float = 3.0
    tau_margin: float = TAU_MARGIN_NM
    tau_hard_max: float = TAU_HARD_MAX_NM
    arm_torque_margin: float = ARM_TORQUE_MARGIN  # fraction of each Panda joint's force limit
    retract_m: float = 0.03
    backoff_m: float = 0.01
    retract_s: float = 0.5
    backoff_s: float = 0.3
    blend_s: float = 0.6
    skill_hold_s: float = 0.3
    skill_timeout_s: float = 2.0
    max_classify_s: float = 0.3
    push_lead_limit_m: float = 0.012


@dataclass(frozen=True, slots=True)
class RoutineSpec:
    """One-finger unlatching push in the latch frame, shared by every active bay.

    The wrist orientation is ``Rz(yaw) Ry(pitch) Rx(roll)`` in the latch frame (palm down, hand
    yawed so the index rests behind the bolt hook and the middle finger hangs over the stopper
    flap). The wrist starts ``start_x`` along the latch +X axis and advances at ``speed``.
    """

    yaw: float
    pitch: float
    roll: float
    start_x: float
    y: float
    z: float
    fingers: Mapping[str, float]
    speed: float = 0.05

    def rotation(self) -> FloatArray:
        return _rz(self.yaw) @ _ry(self.pitch) @ _rx(self.roll)

    def pose_at(self, t_s: float) -> Pose:
        p = (self.start_x + self.speed * t_s, self.y, self.z)
        return Pose(orientation=mat_to_quat(self.rotation()), position=p)


def _rx(a: float) -> FloatArray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a: float) -> FloatArray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a: float) -> FloatArray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


ROUTINE = RoutineSpec(
    yaw=1.097,
    pitch=0.033,
    roll=0.135,
    start_x=-0.0777,
    y=-0.097,
    z=0.116,
    fingers={
        "if_mcp_act": 0.389,
        "if_pip_act": -0.068,
        "if_dip_act": 0.068,
        "mf_mcp_act": 0.202,
        "mf_pip_act": 0.093,
        "mf_dip_act": 0.789,
    },
)


# --------------------------------------------------------------------------- stall detector


@dataclass(frozen=True, slots=True)
class StallTrigger:
    reason: str  # "confidence" or "watchdog"
    signature: StallSignature


class StallDetector:
    """Contact-only stall trigger over a sliding 300 ms window (spec 4.1)."""

    def __init__(self, threshold: float = 0.3, ticks: int = 18, watchdog_s: float = 3.0) -> None:
        self.threshold = threshold
        self.ticks = ticks
        self.watchdog_s = watchdog_s
        self._window: deque[tuple[float, float, FloatArray]] = deque(maxlen=ticks)
        self._low = 0

    def reset(self) -> None:
        self._window.clear()
        self._low = 0

    def update(
        self,
        confidence: float,
        x_bolt: float,
        tau_ext_inf: float,
        tau_wrist: FloatArray,
        step_elapsed_s: float,
    ) -> StallTrigger | None:
        self._window.append((x_bolt, tau_ext_inf, np.asarray(tau_wrist, dtype=np.float64)))
        self._low = self._low + 1 if confidence < self.threshold else 0
        if self._low >= self.ticks:
            return StallTrigger("confidence", self.signature())
        if step_elapsed_s > self.watchdog_s and x_bolt < BOLT_CLEARED_M:
            return StallTrigger("watchdog", self.signature())
        return None

    def signature(self) -> StallSignature:
        """``z = [x_bolt, mean |tau|_inf, max |tau|_inf, |d tau_wrist|_2]`` over the window."""
        xs = [w[0] for w in self._window]
        taus = [w[1] for w in self._window]
        dtau = float(np.linalg.norm(self._window[-1][2] - self._window[0][2]))
        return StallSignature(
            x_bolt_m=xs[-1],
            tau_ext_inf_mean_nm=float(np.mean(taus)),
            tau_ext_inf_max_nm=float(np.max(taus)),
            wrist_tau_delta_nm=dtau,
        )


# --------------------------------------------------------------------------- operator input


@dataclass(frozen=True, slots=True)
class OperatorInput:
    """One glove sample: wrist pose in the glove/sim axes and LEAP actuator targets by name."""

    glove_wrist: Pose
    leap_targets: Mapping[str, float] = field(default_factory=dict[str, float])


# --------------------------------------------------------------------------- per-bay state


def _fingers_array(fingers: Mapping[str, float]) -> FloatArray:
    return np.array([fingers.get(n, 0.0) for n in LEAP_ACTUATORS], dtype=np.float64)


class _Move:
    """Smoothstep wrist move between two world poses."""

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


class _Auto:
    """Autonomous skill run: back off, blend into the DMP, replay under the torque tube."""

    def __init__(self, skill: SkillPrimitive, backoff: _Move) -> None:
        self.skill = skill
        self.backoff: _Move | None = backoff
        self.replay: SkillReplay | None = None
        self.t = 0.0
        self.last: ReplayCommand | None = None


@dataclass
class _Bay:
    bay_id: int
    rig: BayRig
    state: BayState = BayState.NOMINAL
    routine_active: bool = False
    routine_t: float = 0.0
    hold_pose: Pose | None = None
    fingers: dict[str, float] = field(default_factory=dict[str, float])
    move: _Move | None = None
    detector: StallDetector = field(default_factory=StallDetector)
    indexer: RelativeClutchIndexer = field(default_factory=RelativeClutchIndexer)
    buffer: RescueTrajectoryBuffer | None = None
    clutch_held: bool = False
    auto: _Auto | None = None
    refused: bool = False
    signature: StallSignature | None = None
    pending: tuple[RescueTrajectory, StallSignature] | None = None
    match: StallMatch | None = None
    skill_id: str | None = None
    confidence: float = 1.0
    tau_inf: float = 0.0
    tau_arm: float = 0.0
    arm_ratio: float = 0.0
    approach: ApproachPlan | None = None
    pending_latch: tuple[bool, bool] | None = None
    settle_ticks: int = 0
    breach_lock: bool = False
    trigger_reason: str = ""
    peak_ratio: dict[str, float] = field(default_factory=dict[str, float])
    v_cmd: float = 0.0
    v_act: float = 0.0
    prev_cmd_pos: FloatArray | None = None
    prev_wrist_pos: FloatArray | None = None
    outcome: str = ""


class AtcCoordinator:
    """Fleet supervisor for the active bays of a four-bay scene (``teleop.sim.bays``)."""

    def __init__(
        self,
        model: Any,
        data: Any,
        unoq: UnoQClient,
        sink: ComplianceSink,
        reasoner: FleetReasoner,
        hand_spec: HandSpec,
        config: CoordinatorConfig | None = None,
        validator: SkillValidator | None = None,
        routine: RoutineSpec | None = None,
        bays: tuple[int, ...] = ACTIVE_BAYS,
    ) -> None:
        self.model = model
        self.data = data
        self.unoq = unoq
        self.sink = sink
        self.reasoner = reasoner
        self.hand_spec = hand_spec
        self.config = config or CoordinatorConfig()
        self.validator = validator or SkillValidator(hand_spec, self.config.tau_hard_max)
        self.routine = routine or ROUTINE
        self.library: dict[str, SkillPrimitive] = {}
        self.time_s = 0.0
        self.last_classify_ms = 0.0
        self.last_report: ValidationReport | None = None
        self._queue: list[int] = []
        self._bays = {b: _Bay(b, BayRig(model, data, b)) for b in bays}
        for b in BACKGROUND_BAYS:
            with contextlib.suppress(KeyError):  # scene built without the background bays
                reset_robot_home(model, data, b)
        for bay in self._bays.values():
            bay.rig.home_keyframe()  # the only kinematic arm placement: once, at scene build
        for bay in self._bays.values():
            self.start_cycle(bay.bay_id)

    @classmethod
    def build(
        cls,
        unoq: UnoQClient,
        sink: ComplianceSink,
        reasoner: FleetReasoner,
        hand_spec: HandSpec | None = None,
        bay1_pose: LatchPose = CANONICAL_LATCH,
        bay2_pose: LatchPose = BAY2_LATCH,
        include_background_bays: bool = False,
        config: CoordinatorConfig | None = None,
        validator: SkillValidator | None = None,
    ) -> AtcCoordinator:
        import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]

        mj: Any = _mujoco
        model = build_four_bay_scene(bay1_pose, bay2_pose, include_background_bays)
        data = mj.MjData(model)
        return cls(
            model,
            data,
            unoq,
            sink,
            reasoner,
            hand_spec or load_hand("leap"),
            config=config,
            validator=validator,
        )

    # ------------------------------------------------------------------ public queries

    def rig(self, bay_id: int) -> BayRig:
        return self._bays[bay_id].rig

    def state(self, bay_id: int) -> BayState:
        return self._bays[bay_id].state

    @property
    def queue(self) -> tuple[int, ...]:
        return tuple(self._queue)

    def rescue_slot_holders(self) -> tuple[int, ...]:
        return tuple(
            b.bay_id
            for b in self._bays.values()
            if b.state in (BayState.RESCUE, BayState.OOD_ESCALATED, BayState.VALIDATING)
        )

    def snapshot(self, bay_id: int) -> BaySnapshot:
        bay = self._bays[bay_id]
        queue_position = self._queue.index(bay_id) if bay_id in self._queue else None
        return BaySnapshot(
            bay_id=bay_id,
            state=bay.state,
            confidence=min(max(bay.confidence, 0.0), 1.0),
            x_bolt_m=bay.rig.bolt_x(),
            tau_ext_inf_nm=bay.tau_inf,
            part_sku_id=self.config.part_sku_id,
            queue_position=queue_position,
            active_skill_id=bay.skill_id if bay.state == BayState.AUTONOMOUS_SKILL else None,
            last_match=bay.match,
        )

    def snapshots(self) -> tuple[BaySnapshot, ...]:
        return tuple(self.snapshot(b) for b in sorted(self._bays))

    def last_signature(self, bay_id: int) -> StallSignature | None:
        return self._bays[bay_id].signature

    def outcome(self, bay_id: int) -> str:
        return self._bays[bay_id].outcome

    # ------------------------------------------------------------------ cycle control

    def start_cycle(
        self, bay_id: int, flap_jammed: bool = False, bolt_pinned: bool = False
    ) -> None:
        """New pallet cycle: lift clear, reset the latch, move the hand to the routine start, run.

        The arm travels under IK and the position servos (rate-limited), never kinematically.
        ``flap_jammed`` and ``bolt_pinned`` are rig faults (the physical fault rig of spec 2.1);
        they are applied once the hand is lifted clear of the latch and are never read back.
        """
        bay = self._bays[bay_id]
        rig = bay.rig
        wfl = rig.world_from_latch()
        start = compose_pose(wfl, self.routine.pose_at(0.0))
        here = rig.wrist_pose()
        lift_z = rig.world_latch().position[2] + LIFT_HEIGHT_M
        # Every LEAP actuator, not just the routine's: clutch release pins all 16 targets to the
        # rescue posture, and a stale one (e.g. mf_rot_act) would rest a finger on the bolt.
        fingers = {name: 0.0 for name in LEAP_ACTUATORS} | dict(self.routine.fingers)
        rig.ik.reset()
        bay.approach = ApproachPlan(here, [lifted(here, lift_z), lifted(start, lift_z), start])
        bay.pending_latch = (flap_jammed, bolt_pinned)
        bay.state = BayState.NOMINAL
        bay.routine_active = False
        bay.routine_t = 0.0
        bay.hold_pose = here
        bay.fingers = fingers
        bay.move = None
        bay.auto = None
        bay.buffer = None
        bay.pending = None
        bay.clutch_held = False
        bay.refused = False
        bay.match = None
        bay.skill_id = None
        bay.outcome = ""
        bay.breach_lock = False
        bay.trigger_reason = ""
        bay.confidence = 1.0
        bay.tau_inf = 0.0
        bay.prev_cmd_pos = None
        bay.prev_wrist_pos = None
        bay.detector.reset()
        bay.indexer.reset()
        if bay_id in self._queue:
            self._queue.remove(bay_id)

    def arm_fault(self, bay_id: int) -> bool:
        """Rig toggle: release the flap catch (one-way until the next cycle)."""
        return self._bays[bay_id].rig.latch.arm_flap_fault()

    # ------------------------------------------------------------------ tick

    def tick(
        self, console: ConsoleInputState | None = None, operator: OperatorInput | None = None
    ) -> tuple[BaySnapshot, ...]:
        """One 60 Hz control tick: command every active bay, step physics, sense, transition."""
        console = console or ConsoleInputState()
        self._run_pending()
        for bay in self._bays.values():
            self._control(bay, console, operator)
        step_physics(self.model, self.data)
        self.time_s += CONTROL_DT
        for bay in self._bays.values():
            self._sense(bay)
        self._promote()
        return self.snapshots()

    # ------------------------------------------------------------------ control phase

    def _command(
        self,
        bay: _Bay,
        wrist: Pose,
        fingers: Mapping[str, float] | None = None,
        v_cmd: float | None = None,
    ) -> None:
        """Drive the bay one tick. ``v_cmd`` overrides the commanded speed used by C(t)."""
        if fingers is not None:
            bay.fingers.update(fingers)
        bay.rig.command(wrist, bay.fingers)
        pos = pose_position(wrist)
        if v_cmd is not None:
            bay.v_cmd = v_cmd
        else:
            bay.v_cmd = (
                0.0
                if bay.prev_cmd_pos is None
                else float(np.linalg.norm(pos - bay.prev_cmd_pos)) / CONTROL_DT
            )
        bay.prev_cmd_pos = pos

    def _hold(self, bay: _Bay) -> None:
        if bay.move is not None:
            bay.hold_pose = bay.move.step()
            if bay.move.done:
                bay.move = None
        if bay.hold_pose is None:
            bay.hold_pose = bay.rig.wrist_pose()
        self._command(bay, bay.hold_pose)

    def _control(
        self, bay: _Bay, console: ConsoleInputState, operator: OperatorInput | None
    ) -> None:
        if bay.state == BayState.NOMINAL:
            if bay.approach is not None:
                self._control_approach(bay)
            elif bay.routine_active:
                bay.routine_t += CONTROL_DT
                wfl = bay.rig.world_from_latch()
                target = compose_pose(wfl, self.routine.pose_at(bay.routine_t))
                bay.hold_pose = target
                # Bounded lead: the IK target never runs more than lead_limit_m ahead of the
                # measured wrist along the push axis, so a blocked finger loads the arm to a
                # bounded force instead of winding up until the Panda saturates.
                u = local_axis_world(wfl, 0)
                lead = float(np.dot(pose_position(target) - pose_position(bay.rig.wrist_pose()), u))
                if lead > self.config.push_lead_limit_m:
                    target = translate_pose(target, -(lead - self.config.push_lead_limit_m) * u)
                self._command(bay, target, v_cmd=self.routine.speed)
            else:
                self._hold(bay)
        elif bay.state in (BayState.RESCUE, BayState.OOD_ESCALATED):
            self._control_rescue(bay, console, operator)
        elif bay.state == BayState.AUTONOMOUS_SKILL:
            self._control_auto(bay)
        else:  # QUEUED, STALLED, VALIDATING: zero-velocity safe hover
            self._hold(bay)

    def _control_approach(self, bay: _Bay) -> None:
        plan = bay.approach
        assert plan is not None
        if plan.index >= 1 and bay.pending_latch is not None:  # lifted clear: reset the latch
            jam, pin = bay.pending_latch
            bay.pending_latch = None
            bay.rig.reset_latch(flap_jammed=jam, bolt_pinned=pin)
        pose = plan.step()
        bay.hold_pose = pose
        self._command(bay, pose, self.routine.fingers, v_cmd=0.0)
        bay.settle_ticks += 1 if plan.done else 0
        if plan.done and (bay.rig.reached(pose) or bay.settle_ticks >= SETTLE_MAX_TICKS):
            bay.approach = None
            bay.settle_ticks = 0
            bay.routine_active = True
            bay.routine_t = 0.0
            bay.detector.reset()
            bay.prev_cmd_pos = None

    def _control_rescue(
        self, bay: _Bay, console: ConsoleInputState, operator: OperatorInput | None
    ) -> None:
        if not console.clutch:
            bay.breach_lock = False
        engaged = (
            not bay.breach_lock
            and console.clutch
            and console.selected_bay == bay.bay_id
            and operator is not None
            and bay.move is None
        )
        if engaged and operator is not None:
            cmd = bay.indexer.update(True, operator.glove_wrist, bay.rig.wrist_pose())
            if cmd is not None:
                self._command(bay, cmd, operator.leap_targets)
                bay.hold_pose = cmd
                if bay.buffer is not None:
                    bay.buffer.record(
                        True, self.time_s, cmd, _fingers_array(bay.fingers), bay.tau_inf
                    )
            bay.clutch_held = True
            return
        if bay.clutch_held:
            self._release_clutch(bay)
            return
        self._hold(bay)

    def _release_clutch(self, bay: _Bay) -> None:
        bay.clutch_held = False
        bay.indexer.reset()
        bleed_off_targets(self.model, self.data, bay.rig.prefix)
        bay.rig.ik.reset()
        bay.prev_cmd_pos = None
        bay.hold_pose = bay.rig.wrist_pose()
        traj = (
            bay.buffer.finish(bay.rig.bolt_x() >= BOLT_CLEARED_M)
            if bay.buffer is not None
            else None
        )
        if traj is not None and bay.signature is not None:
            bay.pending = (traj, bay.signature)
            bay.state = BayState.VALIDATING
        else:
            logger.info("bay %d: rescue discarded (bolt not cleared), staying in hover", bay.bay_id)

    def _control_auto(self, bay: _Bay) -> None:
        auto = bay.auto
        assert auto is not None
        if auto.backoff is not None:
            pose = auto.backoff.step()
            self._command(bay, pose)
            if auto.backoff.done:
                wfl = bay.rig.world_from_latch()
                auto.replay = SkillReplay(
                    auto.skill,
                    wfl,
                    start_wrist=pose,
                    start_fingers=bay.fingers,
                    blend_s=self.config.blend_s,
                    hold_s=self.config.skill_hold_s,
                )
                auto.backoff = None
                auto.t = 0.0
            return
        assert auto.replay is not None
        cmd = auto.replay.command(auto.t)
        auto.last = cmd
        self._command(bay, cmd.wrist, cmd.fingers)
        auto.t += CONTROL_DT

    # ------------------------------------------------------------------ sense phase

    def _sense(self, bay: _Bay) -> None:
        rig = bay.rig
        cfg = self.config
        tau = rig.tau_ext()
        bay.tau_inf = rig.tau_hand_inf(tau)  # hand-only: signature, tube, records, snapshots
        bay.tau_arm = rig.tau_arm_inf(tau)
        tau_full = max(bay.tau_inf, bay.tau_arm)
        pos = pose_position(rig.wrist_pose())
        speed = (
            0.0
            if bay.prev_wrist_pos is None
            else float(np.linalg.norm(pos - bay.prev_wrist_pos)) / CONTROL_DT
        )
        bay.prev_wrist_pos = pos
        bay.v_act = 0.5 * bay.v_act + 0.5 * speed
        bay.confidence = compute_confidence(
            tau_full, bay.v_cmd, bay.v_act, cfg.alpha, cfg.beta, cfg.tau_limit, cfg.v_nom
        )
        x_bolt = rig.bolt_x()

        bay.arm_ratio = rig.arm_torque_ratio(tau)
        phase = self._phase(bay)
        bay.peak_ratio[phase] = max(bay.peak_ratio.get(phase, 0.0), bay.arm_ratio)
        breach = bay.arm_ratio >= cfg.arm_torque_margin
        if breach and phase != "routine" and self._arm_breach(bay, phase):
            return

        if bay.state == BayState.NOMINAL and bay.routine_active:
            if x_bolt >= BOLT_CLEARED_M:
                bay.routine_active = False
                bay.outcome = "cleared"
                bay.detector.reset()
                return
            trigger = bay.detector.update(
                bay.confidence, x_bolt, bay.tau_inf, tau[4:7], bay.routine_t
            )
            if trigger is None and breach:  # fast path: no 300 ms wait
                self._arm_breach(bay, phase)
                return
            if trigger is not None:
                self._on_stall(bay, trigger)
        elif bay.state == BayState.AUTONOMOUS_SKILL:
            self._sense_auto(bay, x_bolt)

    def _sense_auto(self, bay: _Bay, x_bolt: float) -> None:
        auto = bay.auto
        assert auto is not None
        if x_bolt >= BOLT_CLEARED_M:
            self._complete_autonomous(bay)
            return
        if auto.replay is None or auto.last is None:
            return
        in_dmp = auto.t > auto.replay.blend_s + CONTROL_DT
        timed_out = auto.t > auto.replay.total_s + self.config.skill_timeout_s
        if in_dmp and not torque_tube_ok(
            bay.tau_inf, auto.last.tau_demo, self.config.tau_margin, self.config.tau_hard_max
        ):
            self._abort_autonomous(bay, "torque tube violated")
        elif timed_out:
            self._abort_autonomous(bay, "skill timeout")

    # ------------------------------------------------------------------ arm torque safety

    def _phase(self, bay: _Bay) -> str:
        if bay.state == BayState.NOMINAL:
            if bay.approach is not None:
                return "approach"
            return "routine" if bay.routine_active else "idle"
        if bay.state == BayState.AUTONOMOUS_SKILL:
            return "autonomous"
        if bay.state in (BayState.RESCUE, BayState.OOD_ESCALATED) and bay.clutch_held:
            return "rescue"
        return "hover"  # STALLED, QUEUED, VALIDATING and an unclutched rescue hold

    def _arm_breach(self, bay: _Bay, phase: str) -> bool:
        """A Panda joint reached its torque limit fraction. Returns True if the tick is consumed.

        Routine: fast-path stall trigger (no 300 ms wait). Autonomous: abort and escalate.
        Rescue: bleed off and discard the recording until the clutch is released. Everything
        else (approach, hover): bleed off to a zero-velocity hold.
        """
        logger.warning(
            "bay %d: arm torque ratio %.2f >= %.2f in %s",
            bay.bay_id,
            bay.arm_ratio,
            self.config.arm_torque_margin,
            phase,
        )
        if phase == "routine":  # the detector window already holds this tick's sample
            trigger = StallTrigger("arm_torque", bay.detector.signature())
            self._on_stall(bay, trigger)
            return True
        if phase == "autonomous":
            self._abort_autonomous(bay, "arm torque limit")
            return True
        self._bleed_off(bay)
        if phase == "rescue":
            if bay.buffer is not None:
                bay.buffer.finish(False)
            bay.clutch_held = False
            bay.breach_lock = True
            bay.indexer.reset()
        elif phase == "approach":
            bay.approach = None
            bay.pending_latch = None
            bay.outcome = "approach aborted: arm torque limit"
        return True

    def _bleed_off(self, bay: _Bay) -> None:
        bleed_off_targets(self.model, self.data, bay.rig.prefix)
        bay.rig.ik.reset()
        bay.move = None
        bay.prev_cmd_pos = None
        bay.hold_pose = bay.rig.wrist_pose()

    def peak_arm_ratio(self, bay_id: int, phase: str | None = None) -> float:
        """Largest per-joint arm torque ratio seen (per phase, or over all phases)."""
        peaks = self._bays[bay_id].peak_ratio
        return max(peaks.values(), default=0.0) if phase is None else peaks.get(phase, 0.0)

    def arm_phases(self, bay_id: int) -> dict[str, float]:
        return dict(self._bays[bay_id].peak_ratio)

    def in_routine(self, bay_id: int) -> bool:
        return self._bays[bay_id].routine_active

    def last_trigger_reason(self, bay_id: int) -> str:
        return self._bays[bay_id].trigger_reason

    # ------------------------------------------------------------------ stall handling

    def _on_stall(self, bay: _Bay, trigger: StallTrigger) -> None:
        cfg = self.config
        bay.state = BayState.STALLED
        bay.routine_active = False
        bay.signature = trigger.signature
        bay.trigger_reason = trigger.reason
        t0 = time.monotonic_ns()
        match, skill_id = self.unoq.classify_stall(cfg.part_sku_id, trigger.signature)
        self.last_classify_ms = (time.monotonic_ns() - t0) * 1e-6
        bay.match = match
        skill = self.library.get(skill_id) if skill_id is not None else None
        if not match.is_ood and skill is not None:
            self._start_autonomous(bay, skill)
            return
        refused = match.is_ood and math.isfinite(match.distance)
        self._escalate(bay, refused=refused, reason=trigger.reason)

    def _start_autonomous(self, bay: _Bay, skill: SkillPrimitive) -> None:
        wrist = bay.rig.wrist_pose()
        ax = local_axis_world(bay.rig.world_from_latch(), 0)
        backoff = _Move(
            wrist, translate_pose(wrist, -self.config.backoff_m * ax), self.config.backoff_s
        )
        bay.auto = _Auto(skill, backoff)
        bay.skill_id = skill.skill_id
        bay.state = BayState.AUTONOMOUS_SKILL
        bay.prev_cmd_pos = None

    def _retract_to_hover(self, bay: _Bay) -> None:
        wrist = bay.rig.wrist_pose()
        ax = local_axis_world(bay.rig.world_from_latch(), 0)
        bay.move = _Move(
            wrist, translate_pose(wrist, -self.config.retract_m * ax), self.config.retract_s
        )
        bay.hold_pose = wrist
        bay.prev_cmd_pos = None
        bay.rig.ik.reset()

    def _escalate(self, bay: _Bay, refused: bool, reason: str) -> None:
        """Retract 3 cm to zero-velocity hover, join the FIFO queue, alert compliance + reasoner."""
        self._retract_to_hover(bay)
        bay.refused = refused
        bay.state = BayState.QUEUED
        self._queue.append(bay.bay_id)
        kind: ComplianceKind = "OOD_REFUSAL" if refused else "STALL_ALERT"
        match = bay.match
        sig = bay.signature
        summary = f"bay {bay.bay_id} stall ({reason}): " + (
            "OOD signature refused, escalated to operator" if refused else "unseen signature"
        )
        self._emit(
            kind,
            bay,
            summary,
            {
                "reason": reason,
                "z": list(sig.as_vector()) if sig else [],
                "distance": None
                if match is None or not math.isfinite(match.distance)
                else match.distance,
                "is_ood": True if match is None else match.is_ood,
                "queue": list(self._queue),
            },
        )
        self._submit_reasoning(kind, bay)

    def _abort_autonomous(self, bay: _Bay, reason: str) -> None:
        logger.warning("bay %d: autonomous skill aborted (%s)", bay.bay_id, reason)
        bay.auto = None
        bay.skill_id = None
        bay.outcome = f"aborted: {reason}"
        self._retract_to_hover(bay)
        bay.refused = False
        bay.state = BayState.QUEUED
        self._queue.append(bay.bay_id)
        self._emit(
            "STALL_ALERT",
            bay,
            f"bay {bay.bay_id} autonomous skill aborted ({reason}), escalated to operator",
            {"reason": reason, "queue": list(self._queue)},
        )
        self._submit_reasoning("STALL_ALERT", bay)

    def _complete_autonomous(self, bay: _Bay) -> None:
        skill = bay.auto.skill if bay.auto is not None else None
        bay.auto = None
        bay.outcome = "cleared autonomously"
        bay.state = BayState.NOMINAL
        bay.routine_active = False
        bay.hold_pose = bay.rig.wrist_pose()
        bay.prev_cmd_pos = None
        self._emit(
            "RESCUE_LOGGED",
            bay,
            f"bay {bay.bay_id} cleared autonomously by {skill.skill_id if skill else '?'}",
            {"autonomous": True, "skill_id": skill.skill_id if skill else None},
        )

    # ------------------------------------------------------------------ queue

    def _promote(self) -> None:
        """Hand the rescue slot to the FIFO head when no bay holds it."""
        if not self._queue or self.rescue_slot_holders():
            return
        head = self._bays[self._queue.pop(0)]
        head.state = BayState.OOD_ESCALATED if head.refused else BayState.RESCUE
        head.buffer = RescueTrajectoryBuffer(head.rig.world_latch())
        head.indexer.reset()
        head.clutch_held = False

    # ------------------------------------------------------------------ validation + library

    def _run_pending(self) -> None:
        for bay in self._bays.values():
            if bay.state == BayState.VALIDATING and bay.pending is not None:
                traj, sig = bay.pending
                bay.pending = None
                self._finalize_rescue(bay, traj, sig)

    def _finalize_rescue(self, bay: _Bay, traj: RescueTrajectory, sig: StallSignature) -> None:
        sku = self.config.part_sku_id
        skill = self.unoq.fit_dmp(
            sku, sig, traj.timestamps_s, traj.wrist_latch_se3, traj.fingers_q, traj.tau_ext_inf
        )
        self._emit(
            "RESCUE_LOGGED",
            bay,
            f"bay {bay.bay_id} operator rescue logged, DMP {skill.skill_id} fitted",
            {
                "autonomous": False,
                "skill_id": skill.skill_id,
                "signature_id": skill.signature_id,
                "samples": len(traj),
                "duration_s": traj.duration_s,
                "fit_ms": skill.fit_ms,
            },
        )
        self._submit_reasoning("RESCUE_LOGGED", bay)
        report = self.validator.validate(skill)
        self.last_report = report
        self._emit(
            "VALIDATION_GATE",
            bay,
            f"bay {bay.bay_id} gate {report.n_passed}/{report.total} for {skill.skill_id}",
            {
                "skill_id": skill.skill_id,
                "passed": report.passed,
                "rollouts_passed": report.n_passed,
                "rollouts_total": report.total,
                "elapsed_s": report.elapsed_s,
                "max_tau_ext_inf_nm": max(r.max_tau_ext_inf_nm for r in report.results),
            },
        )
        self._submit_reasoning(
            "VALIDATION_GATE", bay, rollout_passed=report.n_passed, rollout_total=report.total
        )
        if report.passed:
            self.unoq.register_skill(skill)
            self.library[skill.skill_id] = skill
            self._emit(
                "SKILL_DEPLOYED",
                bay,
                f"skill {skill.skill_id} promoted to the Fleet Library",
                {"skill_id": skill.skill_id, "signature_id": skill.signature_id},
            )
            bay.outcome = f"rescued, skill {skill.skill_id} deployed"
        else:
            bay.outcome = "rescued, skill rejected by validation gate"
        bay.state = BayState.NOMINAL
        bay.routine_active = False
        bay.hold_pose = bay.rig.wrist_pose()
        bay.prev_cmd_pos = None

    # ------------------------------------------------------------------ hooks

    def _emit(
        self, kind: ComplianceKind, bay: _Bay, summary: str, metadata: Mapping[str, Any]
    ) -> None:
        try:
            self.sink.emit(
                make_event(
                    kind,
                    bay.bay_id,
                    self.config.part_sku_id,
                    summary,
                    metadata,
                    timestamp_ns=time.monotonic_ns(),
                )
            )
        except Exception:
            logger.exception("compliance sink failed")

    def _submit_reasoning(
        self,
        kind: ComplianceKind,
        bay: _Bay,
        rollout_passed: int | None = None,
        rollout_total: int | None = None,
    ) -> None:
        match = bay.match
        try:
            self.reasoner.submit(
                ReasoningContext(
                    event_kind=kind,
                    bay_id=bay.bay_id,
                    part_sku_id=self.config.part_sku_id,
                    confidence=min(max(bay.confidence, 0.0), 1.0),
                    x_bolt_m=bay.rig.bolt_x(),
                    tau_ext_inf_nm=bay.tau_inf,
                    matched_signature_id=None if match is None else match.signature_id,
                    mahalanobis_distance=(
                        None
                        if match is None or not math.isfinite(match.distance)
                        else match.distance
                    ),
                    is_ood=True if match is None else match.is_ood,
                    rollout_passed=rollout_passed,
                    rollout_total=rollout_total,
                )
            )
        except Exception:
            logger.exception("fleet reasoner failed")


__all__ = [
    "ROUTINE",
    "AtcCoordinator",
    "CoordinatorConfig",
    "OperatorInput",
    "RoutineSpec",
    "StallDetector",
    "StallTrigger",
    "compute_confidence",
]
