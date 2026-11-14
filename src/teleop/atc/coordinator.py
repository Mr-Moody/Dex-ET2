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
snapshots is the inf-norm over the hand finger joints (arm torques depend on which Panda reached
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
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from teleop.atc.estimate import (
    POSE_SIGMA_XY_M,
    POSE_SIGMA_YAW_RAD,
    LatchEstimate,
    LatchEstimator,
    TouchDetector,
    contact_offset,
    touch_correct,
)
from teleop.atc.hud import ConveyorMode, ConveyorStatus
from teleop.atc.recovery import (
    BaySignals,
    CoordinatorObserver,
    DmpRecovery,
    EpisodeStartSource,
    RecoveryBackend,
    RecoveryController,
    RecoveryFactory,
    RecoveryKind,
    SmoothMove,
    choose_recovery,
)
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
    GATE_PASS_MIN,
    LIFT_HEIGHT_M,
    SETTLE_MAX_TICKS,
    TAU_HARD_MAX_NM,
    TAU_MARGIN_NM,
    ApproachPlan,
    BayRig,
    SkillValidator,
    ValidationReport,
    lifted,
    local_axis_world,
    pose_position,
    step_physics,
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
from teleop.sim.cells import CONVEYOR_SPEED_M_S, PACK_PITCH_M
from teleop.sim.hands import HandSpec, load_hand
from teleop.types import Pose
from teleop.unoq.client import UnoQClient
from teleop.unoq.dmp import HAND_ACTUATORS, compose_pose, mat_to_quat

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


# Default realism scale. Scans of the scripted show (seeds 0-7, Windows and Linux) pass every run
# at 0.25, 13 of 16 at 0.5 and almost none at 1.0 (gate about 14/20). A fresh rescue's gate
# scores 19-20 at 0.25 but hovers at 17-19 around the 18 bar at 0.5, so 0.25 (pose sigma 0.75 mm
# and 0.375 deg, quarter-range gate randomization) is the default that reliably deploys skills.
DEFAULT_REALISM = 0.25


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
    # The Inspire fingers have no abduction compliance, so a blocked push loads the arm almost
    # directly: a 3 mm lead keeps a flap stall under the arm torque guard.
    seed: int = 0
    pose_sigma_xy_m: float = POSE_SIGMA_XY_M
    pose_sigma_yaw_rad: float = POSE_SIGMA_YAW_RAD
    # Scales the latch-pose estimate error and the gate's randomization: 1 is the full T-179
    # realism, 0 is an exact placement and a gate on the spec-4.4 perturbations only.
    realism: float = DEFAULT_REALISM
    gate_pass_min: int = GATE_PASS_MIN  # randomized gate: deploy at >= this many of 20
    # Sim seconds from submission to applying the skill-gate result. The gate runs in a worker
    # (6-9 s wall); applying at a fixed sim time keeps seeded runs tick-for-tick reproducible.
    gate_deadline_s: float = 10.0
    push_lead_limit_m: float = 0.006  # free slides track ~5 mm behind; a stall stays gentle


@dataclass(frozen=True, slots=True)
class RoutineSpec:
    """One-finger unlatching push in the latch frame, shared by every active bay.

    The wrist orientation is ``Rz(yaw) Ry(pitch) Rx(roll)`` in the latch frame (palm down and
    rolled, hand yawed so the index rests behind the bolt hook, pushing partly in its flexion
    plane, and the tucked thumb rides clear above the stopper flap). The wrist starts
    ``start_x`` along the latch +X axis and advances at ``speed``.
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
    yaw=-2.2,
    pitch=0.0,
    roll=-0.4,
    start_x=-0.0021,
    y=0.023,
    z=0.1163,
    fingers={
        "index": 0.5,
        "middle": 0.5,
        "ring": 0.3,
        "pinky": 0.3,
        "thumb_proximal": 0.1,
        "thumb_yaw": 0.0,
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


# --------------------------------------------------------------------------- touch detector

TOUCH_LOG_LEN = 64


@dataclass(frozen=True, slots=True)
class TouchEvent:
    """A HUD / log-feed entry for an accepted touch correction (not a compliance event)."""

    bay_id: int
    correction_mm: float
    time_s: float
    kind: str = "TOUCH_CORRECTED"


# --------------------------------------------------------------------------- operator input


@dataclass(frozen=True, slots=True)
class OperatorInput:
    """One glove sample: wrist pose in the glove/sim axes and hand actuator targets by name."""

    glove_wrist: Pose
    hand_targets: Mapping[str, float] = field(default_factory=dict[str, float])


# --------------------------------------------------------------------------- per-bay state


def _fingers_array(fingers: Mapping[str, float]) -> FloatArray:
    return np.array([fingers.get(n, 0.0) for n in HAND_ACTUATORS], dtype=np.float64)


@dataclass
class _Recovery:
    """One autonomous recovery run in a bay: its controller, which backend, its DMP skill."""

    controller: RecoveryController
    kind: RecoveryKind
    skill: SkillPrimitive | None = None
    t: float = 0.0
    error: str | None = None  # the controller raised: abort in the sense phase


@dataclass(slots=True)
class _Gate:
    """A skill gate running in the worker, applied at the fixed sim-time ``deadline``."""

    skill: SkillPrimitive
    future: Future[ValidationReport]
    deadline: float


@dataclass
class _Bay:
    bay_id: int
    rig: BayRig
    state: BayState = BayState.NOMINAL
    routine_active: bool = False
    routine_t: float = 0.0
    hold_pose: Pose | None = None
    fingers: dict[str, float] = field(default_factory=dict[str, float])
    move: SmoothMove | None = None
    detector: StallDetector = field(default_factory=StallDetector)
    touch: TouchDetector = field(default_factory=TouchDetector)
    last_touch: LatchEstimate | None = None
    # Rescue-demo contact calibration: encoder-only touch relative to the hold's start.
    demo_touch: TouchDetector = field(default_factory=TouchDetector)
    demo_offset_m: float | None = None
    indexer: RelativeClutchIndexer = field(default_factory=RelativeClutchIndexer)
    buffer: RescueTrajectoryBuffer | None = None
    clutch_held: bool = False
    auto: _Recovery | None = None
    fallback_skill: SkillPrimitive | None = None  # deployed DMP for this stall, if any
    recording: bool = False  # an observer episode is open
    refused: bool = False
    signature: StallSignature | None = None
    gate: _Gate | None = None
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
    belt_mode: ConveyorMode = "paused"
    belt_transfer: bool = False
    belt_dist_m: float = 0.0
    pack_dy_m: float = 0.0
    powered: bool = True  # False while the console E-stop has this robot's power cut
    brake_pose: Pose | None = None  # where the arm was when its power was cut


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
        recovery_backend: RecoveryBackend = RecoveryBackend.DMP,
        act: RecoveryFactory | None = None,
        observer: CoordinatorObserver | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.unoq = unoq
        self.sink = sink
        self.reasoner = reasoner
        self.hand_spec = hand_spec
        self.config = config or CoordinatorConfig()
        self.validator = validator or SkillValidator(
            hand_spec,
            self.config.tau_hard_max,
            pass_min=self.config.gate_pass_min,
            seed=self.config.seed,
            realism=self.config.realism,
        )
        self.routine = routine or ROUTINE
        self.library: dict[str, SkillPrimitive] = {}
        self.time_s = 0.0
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="skill-gate"
        )
        self.last_classify_ms = 0.0
        self.last_report: ValidationReport | None = None
        self._queue: list[int] = []
        self._estimator = LatchEstimator(
            self.config.seed,
            self.config.realism * self.config.pose_sigma_xy_m,
            self.config.realism * self.config.pose_sigma_yaw_rad,
        )
        self._cycles: dict[int, int] = {}
        self._touch_log: deque[TouchEvent] = deque(maxlen=TOUCH_LOG_LEN)
        self._backend = RecoveryBackend(recovery_backend)
        self._act = act
        self._observer = observer
        self.backend_note = ""
        self._bays = {b: _Bay(b, BayRig(model, data, b)) for b in bays}
        for b in BACKGROUND_BAYS:
            with contextlib.suppress(KeyError):  # scene built without the background bays
                reset_robot_home(model, data, b)
        for bay in self._bays.values():
            bay.rig.home_keyframe()  # the only kinematic arm placement: once, at scene build
        for bay in self._bays.values():
            self._cycles[bay.bay_id] = 0
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
        recovery_backend: RecoveryBackend = RecoveryBackend.DMP,
        act: RecoveryFactory | None = None,
        observer: CoordinatorObserver | None = None,
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
            hand_spec or load_hand("inspire"),
            config=config,
            validator=validator,
            recovery_backend=recovery_backend,
            act=act,
            observer=observer,
        )

    # ------------------------------------------------------------------ public queries

    def rig(self, bay_id: int) -> BayRig:
        return self._bays[bay_id].rig

    # ------------------------------------------------------------------ recovery backend

    @property
    def recovery_backend(self) -> RecoveryBackend:
        return self._backend

    @property
    def act(self) -> RecoveryFactory | None:
        return self._act

    def set_recovery_backend(self, backend: RecoveryBackend) -> None:
        """Switch backend from the next stall; a running recovery finishes on its own."""
        backend = RecoveryBackend(backend)
        if backend is self._backend:
            return
        self._backend = backend
        self.backend_note = f"recovery backend {backend.value} (from the next stall)"
        logger.info("atc: %s", self.backend_note)

    def set_act(self, factory: RecoveryFactory | None) -> None:
        self._act = factory

    def set_observer(self, observer: CoordinatorObserver | None) -> None:
        self._observer = observer

    def recovery_name(self, bay_id: int) -> str | None:
        """``"dmp"`` or ``"act"`` while the bay recovers autonomously, else ``None``."""
        bay = self._bays[bay_id]
        if bay.state != BayState.AUTONOMOUS_SKILL or bay.auto is None:
            return None
        return bay.auto.kind

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
            pose_sigma_mm=bay.rig.estimate.sigma_x_m * 1e3,
            pose_sigma_yaw_deg=math.degrees(bay.rig.estimate.sigma_yaw_rad),
            touch_correction_mm=self._touch_mm(bay),
            powered=bay.powered,
        )

    @staticmethod
    def _touch_mm(bay: _Bay) -> float | None:
        """Size of this cycle's accepted touch correction (needs no ground truth)."""
        touch = bay.last_touch
        if touch is None or touch.rejected_correction_m is not None:
            return None
        if touch.last_correction_m is None:
            return None
        return abs(touch.last_correction_m) * 1e3

    def snapshots(self) -> tuple[BaySnapshot, ...]:
        return tuple(self.snapshot(b) for b in sorted(self._bays))

    def last_signature(self, bay_id: int) -> StallSignature | None:
        return self._bays[bay_id].signature

    def outcome(self, bay_id: int) -> str:
        return self._bays[bay_id].outcome

    def conveyor_status(self, bay_id: int) -> ConveyorStatus:
        bay = self._bays[bay_id]
        return ConveyorStatus(
            bay_id=bay_id,
            mode=bay.belt_mode,
            belt_dist_m=bay.belt_dist_m,
            pack_dy_m=bay.pack_dy_m,
        )

    def conveyors(self) -> tuple[ConveyorStatus, ...]:
        return tuple(self.conveyor_status(b) for b in sorted(self._bays))

    def estimate(self, bay_id: int) -> LatchEstimate:
        return self._bays[bay_id].rig.estimate

    def cycle_count(self, bay_id: int) -> int:
        return self._cycles[bay_id]

    def last_touch(self, bay_id: int) -> LatchEstimate | None:
        """The estimate produced by this cycle's last touch (accepted or rejected), if any."""
        return self._bays[bay_id].last_touch

    def touch_events(self) -> tuple[TouchEvent, ...]:
        """Recent ``TOUCH_CORRECTED`` entries for the HUD and log feed, oldest first."""
        return tuple(self._touch_log)

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
        if bay.auto is not None:
            bay.auto.controller.close()
        self._end_episode(bay, False, "cycle restarted")
        rig = bay.rig
        # A new pallet is a new pose estimate (cycle 0 at build). Sampled here rather than at
        # the lift-clear reset so the approach waypoints below are built on the same estimate.
        rig.set_estimate(
            self._estimator.sample(rig.true_world_latch(), bay_id, self._cycles[bay_id])
        )
        self._cycles[bay_id] += 1
        wfl = rig.world_from_latch()
        start = compose_pose(wfl, self.routine.pose_at(0.0))
        here = rig.wrist_pose()
        lift_z = rig.world_latch().position[2] + LIFT_HEIGHT_M
        # Every hand actuator, not just the routine's: clutch release pins all targets to the
        # rescue posture, and a stale one (e.g. thumb_yaw) would rest a finger on the latch.
        fingers = {name: 0.0 for name in HAND_ACTUATORS} | dict(self.routine.fingers)
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
        bay.fallback_skill = None
        bay.buffer = None
        # A gate still running in the worker is simply never applied. It is deliberately not
        # cancelled: whether cancel() succeeds depends on wall-clock timing (queued vs running),
        # and the validator's state carries over between gates, so cancelling would make the
        # next gate's result nondeterministic.
        bay.gate = None
        bay.clutch_held = False
        bay.refused = False
        bay.match = None
        bay.skill_id = None
        bay.outcome = ""
        if not bay.belt_transfer:
            bay.pack_dy_m = 0.0
            bay.belt_mode = "paused"
        bay.breach_lock = False
        bay.trigger_reason = ""
        bay.confidence = 1.0
        bay.tau_inf = 0.0
        bay.prev_cmd_pos = None
        bay.prev_wrist_pos = None
        bay.detector.reset()
        bay.touch.disarm()  # armed when the routine starts, after the latch reset
        bay.last_touch = None
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
        self._apply_gates()
        for bay in self._bays.values():
            self._set_power(bay, bay.bay_id not in console.powered_off)
            if bay.powered:
                self._control(bay, console, operator)
            else:
                self._brake(bay)
        step_physics(self.model, self.data)
        self.time_s += CONTROL_DT
        for bay in self._bays.values():
            if bay.powered:
                self._sense(bay)
        self._promote()
        self._update_conveyors()
        return self.snapshots()

    def _set_power(self, bay: _Bay, powered: bool) -> None:
        """Console E-stop: a cut robot brakes where it is; restoring it resumes where it was.

        Timers (routine, approach, replay) only advance in ``_control``, so they pause with it.
        The clutch indexer re-anchors on the next engage, so the glove never jumps the arm.
        """
        if powered == bay.powered:
            return
        bay.powered = powered
        bay.indexer.reset()
        bay.rig.ik.reset()
        bay.prev_cmd_pos = None
        bay.prev_wrist_pos = None
        if powered:
            bay.brake_pose = None
            bay.detector.reset()
            logger.warning("bay %d: robot power restored", bay.bay_id)
        else:
            bay.brake_pose = bay.rig.wrist_pose()
            logger.warning("bay %d: robot power cut by the console E-stop", bay.bay_id)

    def _brake(self, bay: _Bay) -> None:
        assert bay.brake_pose is not None
        self._command(bay, bay.brake_pose, v_cmd=0.0)

    def _update_conveyors(self) -> None:
        step = CONVEYOR_SPEED_M_S * CONTROL_DT
        for bay in self._bays.values():
            if not bay.powered or bay.state in (
                BayState.STALLED,
                BayState.QUEUED,
                BayState.RESCUE,
                BayState.OOD_ESCALATED,
                BayState.VALIDATING,
            ):
                bay.belt_mode = "estop"
                continue
            if bay.belt_transfer:
                bay.belt_mode = "feeding"
                bay.pack_dy_m += step
                bay.belt_dist_m += step
                if bay.pack_dy_m >= PACK_PITCH_M:
                    # Seamless modular wrap: incoming pack lands at 0.0 right as latch resets.
                    bay.pack_dy_m = 0.0
                    bay.belt_transfer = False
                    bay.belt_mode = "paused"
                    if bay.pending_latch is not None:
                        jam, pin = bay.pending_latch
                        bay.pending_latch = None
                        bay.rig.reset_latch(flap_jammed=jam, bolt_pinned=pin)
                continue
            bay.belt_mode = "paused"
            bay.pack_dy_m = 0.0

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
        if bay.belt_transfer:
            # Lift to hover and wait above the belt while the cleared pack glides to +PACK_PITCH_M
            # and the fresh upstream pack glides into 0.0 (preserving the open bolt until wrap).
            if plan.index < 2:
                bay.hold_pose = plan.step()
            self._command(
                bay, bay.hold_pose or bay.rig.wrist_pose(), self.routine.fingers, v_cmd=0.0
            )
            return
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
            if bay.pending_latch is not None:  # never arm on a stale bolt from the last cycle
                jam, pin = bay.pending_latch
                bay.pending_latch = None
                bay.rig.reset_latch(flap_jammed=jam, bolt_pinned=pin)
            bay.routine_active = True
            bay.routine_t = 0.0
            bay.detector.reset()
            bay.touch.reset()
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
            if not bay.clutch_held:
                self._start_episode(bay, "operator")
            cmd = bay.indexer.update(True, operator.glove_wrist, bay.rig.wrist_pose())
            if cmd is not None:
                self._episode_tick(bay, cmd, operator.hand_targets)
                self._command(bay, cmd, operator.hand_targets)
                bay.hold_pose = cmd
                if bay.buffer is not None:
                    bay.buffer.record(
                        True, self.time_s, cmd, _fingers_array(bay.fingers), bay.tau_inf
                    )
                    self._calibrate_demo_touch(bay)
            bay.clutch_held = True
            return
        if bay.clutch_held:
            self._release_clutch(bay)
            return
        self._hold(bay)

    def _calibrate_demo_touch(self, bay: _Bay) -> None:
        """Record the demo's own index contact offset at its first bolt-encoder movement.

        Reads only the encoder, the index tip forward kinematics and the bay's (stall-touch
        corrected) estimate: the same frame the rescue trajectory is recorded in.
        """
        rig = bay.rig
        if bay.buffer is not None and len(bay.buffer) == 1:  # a new clutch hold starts
            bay.demo_touch.reset()
            bay.demo_offset_m = None
        x_bolt = rig.bolt_x()
        if bay.demo_offset_m is None and bay.demo_touch.update(x_bolt, None):
            bay.demo_offset_m = contact_offset(rig.estimate, rig.index_tip_world(), x_bolt)

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
        cleared = bay.rig.bolt_x() >= BOLT_CLEARED_M
        self._end_episode(bay, cleared, None if cleared else "bolt not cleared")
        if traj is not None and bay.signature is not None:
            bay.state = BayState.VALIDATING
            self._submit_gate(bay, traj, bay.signature)
        else:
            logger.info("bay %d: rescue discarded (bolt not cleared), staying in hover", bay.bay_id)

    def _control_auto(self, bay: _Bay) -> None:
        rec = bay.auto
        assert rec is not None
        try:
            cmd = None if rec.error else rec.controller.command(bay.rig, self._signals(bay))
        except Exception as exc:
            logger.exception("bay %d: %s recovery controller failed", bay.bay_id, rec.kind)
            rec.error = f"recovery error: {exc}"
            cmd = None
        if cmd is None:
            self._hold(bay)
            assert bay.hold_pose is not None
            self._episode_tick(bay, bay.hold_pose, None)
        else:
            self._episode_tick(bay, cmd.wrist, cmd.fingers)
            self._command(bay, cmd.wrist, cmd.fingers, v_cmd=cmd.v_cmd)
        rec.t += CONTROL_DT

    def _signals(self, bay: _Bay) -> BaySignals:
        return BaySignals(
            bay_id=bay.bay_id,
            t_s=bay.auto.t if bay.auto is not None else 0.0,
            tau_hand_inf=bay.tau_inf,
            confidence=bay.confidence,
            fingers=bay.fingers,
        )

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
            # The routine target is recomposed from the estimate every tick, so a correction
            # lands through the push-lead clamp without an extra blend. The target may therefore
            # step backward by up to the correction size in one tick (plan-sanctioned).
            if bay.touch.update(x_bolt, rig.wrist_push_torque()):
                self._apply_touch(bay, x_bolt)
            if x_bolt >= BOLT_CLEARED_M:
                bay.routine_active = False
                bay.outcome = "cleared"
                bay.belt_transfer = True
                # Lift slightly clear of the bolt hook so the conveyor can advance the cleared pack.
                wrist = rig.wrist_pose()
                bay.hold_pose = wrist
                bay.move = SmoothMove(
                    wrist, translate_pose(wrist, np.array([0.0, 0.0, 0.08])), 0.25
                )
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
        rec = bay.auto
        assert rec is not None
        if x_bolt >= BOLT_CLEARED_M:
            self._complete_autonomous(bay)
            return
        controller = rec.controller
        if (
            rec.error is None
            and isinstance(controller, DmpRecovery)
            and controller.in_replay
            and controller.touch_offset_m is not None
            and bay.touch.update(x_bolt, None)
            and self._apply_touch(bay, x_bolt, controller.touch_offset_m)
        ):
            # Replay touches are encoder-only and use the skill's own demo contact offset; a
            # skill without one gets no replay touch.
            controller.rebase(bay.rig.world_from_latch())
        reason = rec.error
        if reason is None:
            try:
                reason = rec.controller.check(bay.rig, self._signals(bay))
            except Exception as exc:
                logger.exception("bay %d: %s recovery check failed", bay.bay_id, rec.kind)
                reason = f"recovery error: {exc}"
        if reason is not None:
            self._abort_autonomous(bay, reason)

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
            self._end_episode(bay, False, "arm torque limit")
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

    # ------------------------------------------------------------------ touch re-localization

    def _apply_touch(self, bay: _Bay, x_bolt: float, pad_offset_m: float | None = None) -> bool:
        """Correct the estimate's push axis from the index tip at a touch. True if accepted.

        Reads only proprioception (index tip forward kinematics) and the bolt encoder. An
        outlier correction is rejected and logged, and the estimate is kept.
        """
        rig = bay.rig
        bay.touch.disarm()
        est = touch_correct(rig.estimate, rig.index_tip_world(), x_bolt, pad_offset_m)
        bay.last_touch = est
        rig.set_estimate(est)  # a rejected touch keeps the pose and records the outlier
        if est.rejected_correction_m is not None:
            logger.warning(
                "bay %d touch rejected: correction %.1f mm is an outlier, estimate kept",
                bay.bay_id,
                est.rejected_correction_m * 1e3,
            )
            return False
        shift_mm = (est.last_correction_m or 0.0) * 1e3
        logger.info("bay %d touched latch: pose corrected %.1f mm", bay.bay_id, shift_mm)
        self._touch_log.append(TouchEvent(bay.bay_id, shift_mm, self.time_s))
        return True

    # ------------------------------------------------------------------ stall handling

    def _on_stall(self, bay: _Bay, trigger: StallTrigger) -> None:
        cfg = self.config
        if bay.touch.armed:  # the stall is itself a touch: correct before retract or back-off
            self._apply_touch(bay, bay.rig.bolt_x())
        bay.state = BayState.STALLED
        bay.routine_active = False
        bay.signature = trigger.signature
        bay.trigger_reason = trigger.reason
        t0 = time.monotonic_ns()
        match, skill_id = self.unoq.classify_stall(cfg.part_sku_id, trigger.signature)
        self.last_classify_ms = (time.monotonic_ns() - t0) * 1e-6
        bay.match = match
        skill = self.library.get(skill_id) if skill_id is not None else None
        bay.fallback_skill = skill if not match.is_ood else None
        kind = self._pick_recovery(trigger.signature, dmp_available=bay.fallback_skill is not None)
        if kind == "act":
            self._start_act(bay, trigger.signature)
            return
        if kind == "dmp" and bay.fallback_skill is not None:
            self._start_autonomous(bay, bay.fallback_skill)
            return
        refused = match.is_ood and math.isfinite(match.distance)
        self._escalate(bay, refused=refused, reason=trigger.reason)

    def _pick_recovery(self, z: StallSignature, *, dmp_available: bool) -> RecoveryKind | None:
        act = self._act
        covers = healthy = False
        if self._backend is not RecoveryBackend.DMP:
            why = "no ACT checkpoint loaded"
            if act is not None:
                try:
                    healthy = act.healthy()
                    covers = healthy and act.covers(z)
                    why = act.unhealthy_reason or "unhealthy"
                except Exception as exc:
                    logger.exception("atc: ACT factory failed")
                    healthy, why = False, f"error: {exc}"
            if not healthy:
                self._downgrade(why)
        return choose_recovery(
            self._backend, act_covers=covers, act_healthy=healthy, dmp_available=dmp_available
        )

    def _downgrade(self, why: str) -> None:
        self._backend = RecoveryBackend.DMP
        self.backend_note = f"ACT unavailable ({why}): back to dmp"
        logger.warning("atc: %s", self.backend_note)

    def _start_autonomous(self, bay: _Bay, skill: SkillPrimitive) -> None:
        cfg = self.config
        controller = DmpRecovery(
            skill,
            bay.rig,
            backoff_m=cfg.backoff_m,
            backoff_s=cfg.backoff_s,
            blend_s=cfg.blend_s,
            hold_s=cfg.skill_hold_s,
            tau_margin=cfg.tau_margin,
            tau_hard_max=cfg.tau_hard_max,
            timeout_s=cfg.skill_timeout_s,
            on_replay_start=bay.touch.reset,  # backed off the hook: arm for the replay's touch
        )
        bay.touch.disarm()  # re-armed once the back-off has unloaded the wrist
        bay.auto = _Recovery(controller, "dmp", skill)
        bay.skill_id = skill.skill_id
        bay.state = BayState.AUTONOMOUS_SKILL
        bay.prev_cmd_pos = None
        self._start_episode(bay, "dmp")

    def _start_act(self, bay: _Bay, z: StallSignature) -> None:
        assert self._act is not None
        try:
            controller = self._act.create(bay.rig, z)
        except Exception as exc:
            logger.exception("bay %d: ACT recovery could not start", bay.bay_id)
            self._downgrade(f"error: {exc}")
            if bay.fallback_skill is not None:
                self._start_autonomous(bay, bay.fallback_skill)
            else:
                self._escalate(bay, refused=False, reason="ACT could not start")
            return
        bay.auto = _Recovery(controller, "act")
        bay.skill_id = "act"
        bay.state = BayState.AUTONOMOUS_SKILL
        bay.hold_pose = bay.rig.wrist_pose()
        bay.prev_cmd_pos = None
        self._start_episode(bay, "act")

    def _retract_to_hover(self, bay: _Bay) -> None:
        wrist = bay.rig.wrist_pose()
        ax = local_axis_world(bay.rig.world_from_latch(), 0)
        bay.move = SmoothMove(
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
        rec = bay.auto
        kind = rec.kind if rec is not None else "dmp"
        if rec is not None:
            rec.controller.close()
        self._end_episode(bay, False, reason)
        bay.auto = None
        bay.skill_id = None
        if (
            kind == "act"
            and self._backend is not RecoveryBackend.ACT
            and bay.fallback_skill is not None
        ):
            logger.warning(
                "bay %d: ACT recovery aborted (%s), falling back to DMP", bay.bay_id, reason
            )
            self._start_autonomous(bay, bay.fallback_skill)
            return
        logger.warning("bay %d: autonomous skill aborted (%s)", bay.bay_id, reason)
        bay.outcome = f"aborted: {reason}"
        self._retract_to_hover(bay)
        bay.refused = False
        bay.state = BayState.QUEUED
        self._queue.append(bay.bay_id)
        metadata: dict[str, Any] = {"reason": reason, "queue": list(self._queue)}
        if kind == "act":
            metadata["backend"] = "act"
        self._emit(
            "STALL_ALERT",
            bay,
            f"bay {bay.bay_id} autonomous skill aborted ({reason}), escalated to operator",
            metadata,
        )
        self._submit_reasoning("STALL_ALERT", bay)

    def _complete_autonomous(self, bay: _Bay) -> None:
        rec = bay.auto
        if rec is not None:
            rec.controller.close()
        self._end_episode(bay, True, None)
        skill = rec.skill if rec is not None else None
        act = rec is not None and rec.kind == "act"
        bay.auto = None
        bay.outcome = "cleared autonomously"
        bay.belt_transfer = True
        bay.state = BayState.NOMINAL
        bay.routine_active = False
        wrist = bay.rig.wrist_pose()
        bay.hold_pose = wrist
        bay.move = SmoothMove(wrist, translate_pose(wrist, np.array([0.0, 0.0, 0.08])), 0.25)
        bay.prev_cmd_pos = None
        by = "the ACT policy" if act else (skill.skill_id if skill else "?")
        metadata: dict[str, Any] = {
            "autonomous": True,
            "skill_id": skill.skill_id if skill else None,
        }
        if act:
            metadata["backend"] = "act"
        self._emit(
            "RESCUE_LOGGED",
            bay,
            f"bay {bay.bay_id} cleared autonomously by {by}",
            metadata,
        )

    # ------------------------------------------------------------------ episode observer

    def _start_episode(self, bay: _Bay, source: EpisodeStartSource) -> None:
        if self._observer is None:
            return
        bay.recording = True
        try:
            self._observer.on_episode_start(
                bay.bay_id,
                source,
                None if bay.match is None else bay.match.signature_id,
                bay.signature,
            )
        except Exception:
            logger.exception("episode observer failed")

    def _episode_tick(self, bay: _Bay, wrist: Pose, fingers: Mapping[str, float] | None) -> None:
        if self._observer is None or not bay.recording:
            return
        # Record the hand targets MuJoCo actually applies: a DMP or policy output can overshoot
        # an actuator's ctrlrange (MuJoCo clamps ctrl), and QC would reject the raw value.
        try:
            applied = bay.rig.clip_fingers({**bay.fingers, **(fingers or {})})
            self._observer.on_tick(bay.bay_id, bay.rig, wrist, applied)
        except Exception:
            logger.exception("episode observer failed")

    def _end_episode(self, bay: _Bay, success: bool, reason: str | None) -> None:
        if not bay.recording:
            return
        bay.recording = False
        if self._observer is None:
            return
        try:
            self._observer.on_episode_end(bay.bay_id, success, reason)
        except Exception:
            logger.exception("episode observer failed")

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

    def gate_pending(self, bay_id: int) -> bool:
        """True while the bay's skill gate is running or waiting for its sim-time deadline."""
        return self._bays[bay_id].gate is not None

    def close(self) -> None:
        """Shut the gate worker down (idempotent); a running gate finishes, queued ones drop."""
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def _apply_gates(self) -> None:
        """Apply every gate whose sim-time deadline has been reached, waiting for the worker."""
        for bay in self._bays.values():
            gate = bay.gate
            if (
                gate is not None
                and bay.state == BayState.VALIDATING
                and (self.time_s >= gate.deadline)
            ):
                bay.gate = None
                self._apply_gate(bay, gate)

    def _submit_gate(self, bay: _Bay, traj: RescueTrajectory, sig: StallSignature) -> None:
        sku = self.config.part_sku_id
        skill = self.unoq.fit_dmp(
            sku, sig, traj.timestamps_s, traj.wrist_latch_se3, traj.fingers_q, traj.tau_ext_inf
        )
        # The demo's own contact offset rides with the skill (replay touch re-localization).
        skill = replace(skill, touch_offset_m=bay.demo_offset_m)
        bay.demo_offset_m = None
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
        executor = self._executor
        if executor is None:
            raise RuntimeError("coordinator is closed")
        bay.gate = _Gate(
            skill,
            executor.submit(self.validator.validate, skill),
            self.time_s + self.config.gate_deadline_s,
        )

    def _apply_gate(self, bay: _Bay, gate: _Gate) -> None:
        skill = gate.skill
        try:
            report = gate.future.result()  # blocks if the worker is still running
        except Exception:
            logger.exception("bay %d: skill gate failed for %s", bay.bay_id, skill.skill_id)
            report = ValidationReport((), 0.0, self.validator.pass_min)
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
                "max_tau_ext_inf_nm": max(
                    (r.max_tau_ext_inf_nm for r in report.results), default=0.0
                ),
            },
        )
        self._submit_reasoning(
            "VALIDATION_GATE",
            bay,
            rollout_passed=report.n_passed,
            rollout_total=report.total,
            rollout_threshold=report.threshold,
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
        bay.belt_transfer = True
        wrist = bay.rig.wrist_pose()
        bay.hold_pose = wrist
        bay.move = SmoothMove(wrist, translate_pose(wrist, np.array([0.0, 0.0, 0.08])), 0.25)
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
        rollout_threshold: int | None = None,
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
                    rollout_threshold=rollout_threshold,
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
    "TouchDetector",
    "TouchEvent",
    "compute_confidence",
]
