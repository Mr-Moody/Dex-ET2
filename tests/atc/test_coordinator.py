"""T-140: ATC fleet coordinator, confidence C(t), FIFO queue, torque tube and 20-rollout gate.

The scripted operator drives the coordinator through its public rescue/clutch API
(``ConsoleInputState`` + ``OperatorInput``). Tests read ``flap_hinge`` only to assert simulator
ground truth, never to steer anything.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from teleop.atc.coordinator import (
    DEFAULT_REALISM,
    ROUTINE,
    AtcCoordinator,
    CoordinatorConfig,
    OperatorInput,
    StallDetector,
    TouchDetector,
    compute_confidence,
)
from teleop.atc.estimate import HOOK_BACK_X_M, INDEX_PAD_RADIUS_M, POSE_SIGMA_XY_M
from teleop.atc.stage import RESCUE_PRESS, RESCUE_WAYPOINTS
from teleop.atc.types import (
    BOLT_CLEARED_M,
    BayState,
    ConsoleInputState,
    SkillPrimitive,
)
from teleop.atc.validator import (
    CONTROL_DT,
    GATE_PASS_MIN,
    GATE_ROLLOUTS,
    PERTURBATIONS_DEG,
    ApproachPlan,
    BayRig,
    RolloutConditions,
    RolloutResult,
    SkillReplay,
    SkillValidator,
    ValidationReport,
    lifted,
    pose_position,
    step_physics,
    torque_tube_ok,
    translate_pose,
)
from teleop.compliance.protocol import ComplianceEvent
from teleop.reasoning.protocol import ReasoningContext, ReasoningVerdict
from teleop.sim.bays import CANONICAL_LATCH, PANDA_XML, build_single_bay_scene
from teleop.sim.hands import HandSpec, load_hand
from teleop.types import Pose
from teleop.unoq.client import UnoQClient
from teleop.unoq.dmp import (
    HAND_ACTUATORS,
    fit_rbf_dmp,
    mat_to_quat,
    quat_to_mat,
    relative_pose,
)
from teleop.unoq.server import UnoQServer

SKU = "ev_latch_v1"
FLAP_UP_DEG = 30.0  # simulator ground truth only


def _models_present() -> bool:
    try:
        right = load_hand("inspire").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


pytestmark = pytest.mark.skipif(
    not _models_present(), reason="Panda/Inspire MJCFs missing; run scripts/fetch_models.py"
)


# --------------------------------------------------------------------------- fakes + fixtures


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[ComplianceEvent] = []

    def emit(self, event: ComplianceEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass

    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]


class RecordingReasoner:
    def __init__(self) -> None:
        self.contexts: list[ReasoningContext] = []

    def submit(self, ctx: ReasoningContext) -> None:
        self.contexts.append(ctx)

    def latest_verdict(self, bay_id: int) -> ReasoningVerdict | None:
        return None

    def close(self) -> None:
        pass


@pytest.fixture(scope="module")
def hand_spec() -> HandSpec:
    return load_hand("inspire")


@pytest.fixture(scope="module")
def validator(hand_spec: HandSpec) -> SkillValidator:
    return SkillValidator(hand_spec, realism=DEFAULT_REALISM)  # the coordinator's default


@pytest.fixture
def unoq() -> Iterator[UnoQClient]:
    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="test-node").start()
    client = UnoQClient(
        "127.0.0.1", server.udp_port, server.tcp_port, local_server=server
    ).connect()
    yield client
    client.close()


_RIGS: list[Rig] = []


class Rig:
    def __init__(
        self,
        unoq: UnoQClient,
        hand_spec: HandSpec,
        validator: SkillValidator,
        config: CoordinatorConfig | None = None,
    ) -> None:
        _RIGS.append(self)
        self.sink = RecordingSink()
        self.reasoner = RecordingReasoner()
        self.unoq = unoq
        self.co = AtcCoordinator.build(
            unoq,
            self.sink,
            self.reasoner,
            hand_spec=hand_spec,
            validator=validator,
            config=config,
        )

    def flap_deg(self, bay_id: int) -> float:
        """Ground truth for assertions only."""
        m, d = self.co.model, self.co.data
        adr = int(m.joint(f"bay{bay_id}/flap_hinge").qposadr[0])
        return math.degrees(float(d.qpos[adr]))

    def run_until(self, bay_id: int, states: set[BayState], max_ticks: int = 1500) -> int:
        for k in range(max_ticks):
            snaps = self.co.tick()
            if snaps[bay_id - 1].state in states:
                return k
        raise AssertionError(f"bay {bay_id} never reached {states}")

    def stall_bay(self, bay_id: int, **fault: bool) -> None:
        self.co.start_cycle(bay_id, **fault)
        self.run_until(bay_id, {BayState.RESCUE, BayState.OOD_ESCALATED, BayState.QUEUED})
        for _ in range(40):  # let the 3 cm retract finish
            self.co.tick()


@pytest.fixture(autouse=True)
def _close_rigs() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Shut every coordinator's gate worker down after each test (no leaked threads)."""
    yield
    while _RIGS:
        _RIGS.pop().co.close()


@pytest.fixture
def rig(unoq: UnoQClient, hand_spec: HandSpec, validator: SkillValidator) -> Rig:
    return Rig(unoq, hand_spec, validator)


# --------------------------------------------------------------------------- scripted operator

# The stage's scripted rescue (duration s, wrist delta from the safe hover in the latch frame in
# metres, finger press fraction): one source, so the tests rescue exactly as the show does.
DEMO_PRESS = RESCUE_PRESS
DEMO_WAYPOINTS = list(RESCUE_WAYPOINTS)


def demo_operator(rig: Rig, bay_id: int) -> Iterator[OperatorInput]:
    """Glove samples (one per 60 Hz tick) of the scripted rescue, anchored at the hover."""
    bay_rig = rig.co.rig(bay_id)
    hover = bay_rig.wrist_pose()
    assert hover.position is not None
    r_world = quat_to_mat(bay_rig.world_from_latch().orientation)
    origin = np.asarray(hover.position)
    base = dict(ROUTINE.fingers)
    t_wp = 0.0
    for i in range(1, len(DEMO_WAYPOINTS)):
        dur, p1, f1 = DEMO_WAYPOINTS[i]
        _, p0, f0 = DEMO_WAYPOINTS[i - 1]
        n = max(int(round(dur / CONTROL_DT)), 1)
        for k in range(1, n + 1):
            a = k / n
            delta = np.asarray(p0) + a * (np.asarray(p1) - np.asarray(p0))
            f = f0 + a * (f1 - f0)
            q = dict(base)
            for name, v in DEMO_PRESS.items():
                q[name] = base[name] + f * v
            pos = origin + r_world @ delta
            yield OperatorInput(
                Pose(orientation=hover.orientation, position=(pos[0], pos[1], pos[2])), q
            )
        t_wp += dur


def run_rescue(rig: Rig, bay_id: int, max_extra_ticks: int | None = None) -> None:
    """Hold the clutch through the scripted rescue, release it, and let validation finish.

    ``max_extra_ticks=0`` returns as soon as the bay is VALIDATING. The default waits
    out the sim-time gate deadline plus margin.
    """
    if max_extra_ticks is None:
        max_extra_ticks = round(rig.co.config.gate_deadline_s / CONTROL_DT) + 60
    console = ConsoleInputState(selected_bay=bay_id, clutch=True)
    for sample in demo_operator(rig, bay_id):
        snaps = rig.co.tick(console, sample)
        if snaps[bay_id - 1].x_bolt_m >= BOLT_CLEARED_M + 0.0015:
            break
    released = ConsoleInputState(selected_bay=bay_id, clutch=False)
    stop = BayState.VALIDATING if max_extra_ticks == 0 else BayState.NOMINAL
    for _ in range(max_extra_ticks or 30):
        rig.co.tick(released, None)
        if rig.co.state(bay_id) == stop:
            return


def _bay_pos(rig: Rig, bay_id: int) -> np.ndarray:
    pos = rig.co.rig(bay_id).wrist_pose().position
    assert pos is not None
    return np.asarray(pos)


def _latch_rel(rig: Rig, bay_id: int) -> np.ndarray:
    r = rig.co.rig(bay_id)
    rel = relative_pose(r.world_from_latch(), r.wrist_pose())
    assert rel.position is not None
    return np.asarray(rel.position)


def _rescued_rig(rig: Rig) -> None:
    """Bay 1 jams, the scripted operator rescues it and the skill is deployed."""
    rig.stall_bay(1, flap_jammed=True)
    run_rescue(rig, 1)
    assert len(rig.co.library) == 1


# --------------------------------------------------------------------------- tests


def test_confidence_high_in_nominal_slide_and_below_0_3_on_300ms_flap_stall(rig: Rig) -> None:
    # Spec 4.1 formula, exact values.
    assert compute_confidence(0.0, 0.05, 0.05) == pytest.approx(1.0)
    assert compute_confidence(3.0, 0.05, 0.0) == pytest.approx(math.exp(-(0.8 * 0.5 + 0.9)))
    assert compute_confidence(6.0, 0.0, 0.05) == pytest.approx(math.exp(-0.8))  # no lag penalty
    assert 0.0 < compute_confidence(1e3, 1.0, 0.0) <= 1.0

    # Detector: 17 low ticks are not enough, the 18th (300 ms) fires, a high tick resets.
    det = StallDetector()
    tau = np.zeros(3)
    fired = [det.update(0.1, 0.034, 1.0, tau, 0.5) for _ in range(17)]
    assert all(f is None for f in fired)
    trig = det.update(0.1, 0.034, 1.0, tau, 0.5)
    assert trig is not None and trig.reason == "confidence"
    det.reset()
    assert all(det.update(0.1, 0.03, 1.0, tau, 0.5) is None for _ in range(10))
    assert det.update(0.9, 0.03, 1.0, tau, 0.5) is None
    assert all(det.update(0.1, 0.03, 1.0, tau, 0.5) is None for _ in range(17))
    # 3.0 s watchdog fires even with high confidence, but not once the bolt cleared.
    det.reset()
    assert det.update(0.9, 0.02, 0.0, tau, 3.01) is not None
    det.reset()
    assert det.update(0.9, 0.053, 0.0, tau, 3.01) is None

    # Nominal slide on the healthy Bay 1: C stays above 0.85 and no stall is declared.
    co = rig.co
    co.start_cycle(1)
    confidence: list[float] = []
    for _ in range(500):
        snap = co.tick()[0]
        if co.in_routine(1):
            confidence.append(snap.confidence)
        assert snap.state == BayState.NOMINAL
        if co.outcome(1) == "cleared":
            break
    assert co.outcome(1) == "cleared"
    assert co.rig(1).bolt_x() >= BOLT_CLEARED_M
    assert min(confidence[25:-3]) > 0.85
    assert rig.flap_deg(1) < 5.0  # ground truth: flap was held flat

    # Jammed Bay 1: C < 0.3 continuously for >= 300 ms before the stall is declared.
    co.start_cycle(1, flap_jammed=True)
    confidence = []
    trigger_tick = -1
    for k in range(900):
        snap = co.tick()[0]
        if co.in_routine(1) or snap.state != BayState.NOMINAL:
            confidence.append(snap.confidence)
        if snap.state != BayState.NOMINAL:
            trigger_tick = k
            break
    assert trigger_tick > 0
    assert rig.flap_deg(1) > FLAP_UP_DEG  # ground truth: the flap really was up
    low_run = 0
    for c in reversed(confidence):
        if c >= 0.3:
            break
        low_run += 1
    assert low_run >= 18
    sig = co.last_signature(1)
    assert sig is not None and 0.030 <= sig.x_bolt_m <= 0.040
    assert max(confidence[:20]) > 0.3  # not a start-up false trigger


def test_additive_torque_tube_allows_free_space_zero_demo_torque_and_aborts_on_excess(
    rig: Rig, validator: SkillValidator, unoq: UnoQClient, hand_spec: HandSpec
) -> None:
    # Free space: zero demo torque, a 3.4 Nm residual is inside the 3.5 Nm tube (no
    # percentage-of-zero abort), 3.6 Nm is not; the hard max is independent of the residual.
    assert torque_tube_ok(0.0, 0.0)
    assert torque_tube_ok(3.4, 0.0)
    assert not torque_tube_ok(3.6, 0.0)
    assert torque_tube_ok(11.0, 8.0)  # residual 3.0, below the hard max
    assert not torque_tube_ok(11.6, 8.0)  # residual 3.6
    assert not torque_tube_ok(12.0, 10.0)  # residual 2.0 but at the 12 Nm hard max
    assert not torque_tube_ok(0.0, 5.0)  # demo torque missing from the replay is also a miss

    # A synthetic zero-demo-torque skill that pushes straight through the jam.
    rig.stall_bay(2, flap_jammed=True)
    z = rig.co.last_signature(2)
    assert z is not None
    n = 90
    t = np.linspace(0.0, 1.5, n)
    q = mat_to_quat(ROUTINE.rotation())
    wrist = np.zeros((n, 7))
    wrist[:, 0] = ROUTINE.start_x - 0.01 + 0.06 * t / 1.5  # first 0.6 s stay short of the hook
    wrist[:, 1], wrist[:, 2] = ROUTINE.y, ROUTINE.z
    wrist[:, 3:] = q
    base = np.array([ROUTINE.fingers.get(a, 0.0) for a in HAND_ACTUATORS])
    skill = fit_rbf_dmp(
        t, wrist, np.tile(base, (n, 1)), np.zeros(n), prototype=z, signature_id="Jam_Sig_1"
    )

    # Free-space replay in the headless single-bay rig stays inside the tube.
    bay = validator.rig  # one-time home keyframe was applied when the validator was built
    bay.reset_latch(flap_jammed=False)
    replay = SkillReplay(skill, bay.world_from_latch(), hold_s=0.0)
    start = replay.command(0.0)
    plan = ApproachPlan(bay.wrist_pose(), [lifted(bay.wrist_pose(), 0.22), start.wrist])
    bay.ik.reset()
    for _ in range(600):  # physical approach under IK + servos, then converge
        bay.command(plan.step(), start.fingers)
        step_physics(validator.model, validator.data)
        if plan.done and bay.reached(start.wrist):
            break
    assert bay.reached(start.wrist)
    worst = 0.0
    for k in range(int(0.6 / CONTROL_DT)):  # first 0.6 s: well short of the bolt hook
        cmd = replay.command(k * CONTROL_DT)
        bay.command(cmd.wrist, cmd.fingers)
        step_physics(validator.model, validator.data)
        worst = max(worst, bay.tau_hand_inf())
        assert torque_tube_ok(bay.tau_hand_inf(), cmd.tau_demo)
    assert worst < 3.5

    # In the live coordinator the same skill hits the jam, exceeds a tight tube and aborts to
    # the operator queue instead of forcing the latch.
    rig.co.library[skill.skill_id] = skill
    unoq.register_skill(skill)
    tight = Rig(unoq, hand_spec, validator)
    # The Inspire finger servos are force-limited (1 Nm), so a jam shows as ~0.4 Nm on the hand.
    tight.co.config = type(tight.co.config)(tau_margin=0.1)
    tight.co.library[skill.skill_id] = skill
    tight.co.start_cycle(2, flap_jammed=True)
    states: set[BayState] = set()
    for _ in range(1500):
        states.add(tight.co.tick()[1].state)
        if tight.co.outcome(2):
            break
    assert BayState.AUTONOMOUS_SKILL in states
    assert "torque tube" in tight.co.outcome(2)
    assert tight.co.state(2) in (BayState.QUEUED, BayState.RESCUE)
    assert tight.co.rig(2).bolt_x() < BOLT_CLEARED_M
    assert "STALL_ALERT" in tight.sink.kinds()


def test_single_active_fifo_queue_serializes_concurrent_bay_stalls(rig: Rig) -> None:
    co = rig.co
    co.start_cycle(1, flap_jammed=True)
    for _ in range(1500):  # Bay 2 starts once Bay 1 is on its routine, so Bay 1 stalls first
        co.tick()
        if co.in_routine(1):
            break
    for _ in range(40):  # Bay 1 is mid-push when Bay 2 starts its own jammed cycle
        co.tick()
    co.start_cycle(2, flap_jammed=True)
    seen_queued = False
    stall_wrist_x: float | None = None
    for _ in range(1500):
        snaps = co.tick()
        assert len(co.rescue_slot_holders()) <= 1
        if snaps[1].state == BayState.QUEUED:
            if stall_wrist_x is None:  # first queued tick: the retract has not moved it yet
                stall_wrist_x = float(_latch_rel(rig, 2)[0])
            seen_queued = True
            if snaps[0].state == BayState.RESCUE:
                break
    assert seen_queued
    assert co.state(1) == BayState.RESCUE and co.state(2) == BayState.QUEUED
    assert co.queue == (2,) and co.snapshot(2).queue_position == 0

    # The queued bay is at zero-velocity safe hover, 3 cm behind its stall pose.
    for _ in range(40):
        co.tick()
    p0 = _bay_pos(rig, 2)
    for _ in range(20):
        co.tick()
    assert np.linalg.norm(_bay_pos(rig, 2) - p0) < 1e-3
    stall_x = co.last_signature(2)
    assert stall_x is not None and 0.030 <= stall_x.x_bolt_m <= 0.040
    assert stall_wrist_x is not None
    assert _latch_rel(rig, 2)[0] < stall_wrist_x - 0.02  # backed off along latch -X
    assert rig.sink.kinds().count("STALL_ALERT") == 2

    # Operator clutches the queued bay: ignored. Bay 1 rescue completes, then Bay 2 is served.
    ignored = ConsoleInputState(selected_bay=2, clutch=True)
    sample = next(demo_operator(rig, 2))
    for _ in range(10):
        co.tick(ignored, sample)
    assert co.state(2) == BayState.QUEUED
    run_rescue(rig, 1)
    assert co.state(1) == BayState.NOMINAL
    co.tick()
    assert co.state(2) == BayState.RESCUE and co.queue == ()
    assert len(co.rescue_slot_holders()) == 1


def test_end_to_end_bay1_rescue_fits_dmp_passes_20_rollouts_and_bay2_clears_35deg_12cm(
    rig: Rig, validator: SkillValidator
) -> None:
    co = rig.co
    # Bay 1: first jam is unseen, classified OOD, alerted and rescued by the operator.
    rig.stall_bay(1, flap_jammed=True)
    assert co.state(1) == BayState.RESCUE
    assert rig.flap_deg(1) > FLAP_UP_DEG  # ground truth
    first = co.snapshot(1).last_match
    assert first is not None and first.is_ood and first.signature_id is None
    assert rig.sink.kinds() == ["STALL_ALERT"]
    assert rig.reasoner.contexts[0].is_ood and rig.reasoner.contexts[0].bay_id == 1
    run_rescue(rig, 1)

    assert co.state(1) == BayState.NOMINAL
    assert co.rig(1).bolt_x() >= BOLT_CLEARED_M
    report = co.last_report
    assert report is not None and report.total == 20 and report.passed
    assert report.threshold == GATE_PASS_MIN
    assert all(r.max_tau_ext_inf_nm < 12.0 for r in report.results)
    assert rig.sink.kinds() == [
        "STALL_ALERT",
        "RESCUE_LOGGED",
        "VALIDATION_GATE",
        "SKILL_DEPLOYED",
    ]
    assert len(co.library) == 1
    skill: SkillPrimitive = next(iter(co.library.values()))
    assert skill.n_kernels == 30 and skill.signature_id == "Jam_Sig_1"
    assert [c.event_kind for c in rig.reasoner.contexts][-2:] == [
        "RESCUE_LOGGED",
        "VALIDATION_GATE",
    ]
    assert (rig.reasoner.contexts[-1].rollout_passed or 0) >= GATE_PASS_MIN
    assert rig.reasoner.contexts[-1].rollout_total == GATE_ROLLOUTS

    # The 20-rollout gate measures ~6-9 s on an idle booth laptop and ~14 s with other test
    # suites running, so assert a regression ceiling rather than a target. It runs in a worker;
    # past the sim-time deadline the apply blocks on it, so wall time never breaks determinism.
    assert report.elapsed_s < 20.0, f"validation gate took {report.elapsed_s:.3f}s"

    # Bay 2 sits at +12 cm / +35 deg relative to Bay 1's latch frame.
    l1, l2 = co.rig(1).true_world_latch(), co.rig(2).true_world_latch()  # scene layout
    assert math.degrees(l2.yaw_rad - l1.yaw_rad) == pytest.approx(35.0)
    assert math.hypot(0.08, 0.09) == pytest.approx(0.12, abs=0.001)

    co.start_cycle(2, flap_jammed=True)
    states: list[BayState] = []
    taus: list[float] = []
    wrist_at_stall = None
    for _ in range(1500):
        snap = co.tick()[1]
        states.append(snap.state)
        taus.append(snap.tau_ext_inf_nm)
        if snap.state == BayState.AUTONOMOUS_SKILL and wrist_at_stall is None:
            wrist_at_stall = _latch_rel(rig, 2)
            stall_snap = snap
            back_min = wrist_at_stall[0]
        elif wrist_at_stall is not None:
            back_min = min(back_min, _latch_rel(rig, 2)[0])
        if co.outcome(2):
            break
    assert wrist_at_stall is not None
    assert 0.030 <= stall_snap.x_bolt_m <= 0.040  # Jam_Sig_1 at x ~ 3.5 cm
    assert stall_snap.last_match is not None
    assert not stall_snap.last_match.is_ood
    assert stall_snap.last_match.signature_id == "Jam_Sig_1"
    assert stall_snap.last_match.distance < 3.0
    assert co.last_classify_ms < 300.0
    assert back_min <= wrist_at_stall[0] - 0.01  # backed off >= 1 cm along local -X
    assert co.outcome(2) == "cleared autonomously"
    assert co.rig(2).bolt_x() >= BOLT_CLEARED_M
    assert max(taus) < 12.0  # hard max held the whole replay
    assert BayState.QUEUED not in states and BayState.RESCUE not in states
    assert rig.sink.kinds()[-1] == "RESCUE_LOGGED"
    assert rig.sink.events[-1].metadata["autonomous"] is True
    assert "OOD_REFUSAL" not in rig.sink.kinds()
    for bay_id in (1, 2):  # arm torque guard never came close in any phase
        assert co.peak_arm_ratio(bay_id) < co.config.arm_torque_margin


def test_negative_control_pinned_bolt_at_x0_rejected_as_ood_and_escalates(rig: Rig) -> None:
    co = rig.co
    _rescued_rig(rig)
    n_events = len(rig.sink.events)

    co.start_cycle(2, bolt_pinned=True)  # Negative Control: bolt locked at x = 0
    k = rig.run_until(2, {BayState.OOD_ESCALATED, BayState.AUTONOMOUS_SKILL, BayState.RESCUE})
    assert k > 0
    assert co.state(2) == BayState.OOD_ESCALATED  # refused, handed to the human
    snap = co.snapshot(2)
    assert snap.x_bolt_m < 0.002
    assert snap.last_match is not None and snap.last_match.is_ood
    assert snap.last_match.signature_id is None and snap.last_match.distance > 3.0
    sig = co.last_signature(2)
    assert sig is not None and sig.x_bolt_m < 0.002

    new_events = rig.sink.events[n_events:]
    assert [e.kind for e in new_events] == ["OOD_REFUSAL"]
    assert new_events[0].bay_id == 2
    ctx = rig.reasoner.contexts[-1]
    assert ctx.event_kind == "OOD_REFUSAL" and ctx.is_ood and ctx.bay_id == 2
    assert ctx.mahalanobis_distance is not None and ctx.mahalanobis_distance > 3.0

    # No autonomy was attempted and the operator cannot clear a pinned bolt: the recording is
    # discarded and the bay stays escalated, library unchanged.
    for _ in range(40):
        co.tick()
    run_rescue(rig, 2, max_extra_ticks=10)
    assert co.state(2) == BayState.OOD_ESCALATED
    assert co.rig(2).bolt_x() < BOLT_CLEARED_M
    assert len(co.library) == 1
    assert rig.sink.kinds().count("SKILL_DEPLOYED") == 1


def test_clean_cycles_after_a_rescue_keep_clearing(rig: Rig) -> None:
    """Clutch release pins every hand target; the next cycles must not inherit that posture."""
    co = rig.co
    _rescued_rig(rig)
    outcomes: list[str] = []
    for _ in range(3):
        co.start_cycle(1)
        for _ in range(1500):
            co.tick()
            if co.state(1) != BayState.NOMINAL or (co.outcome(1) and not co.in_routine(1)):
                break
        outcomes.append(co.outcome(1) or co.state(1).value)
    assert outcomes == ["cleared", "cleared", "cleared"]
    assert rig.sink.kinds().count("OOD_REFUSAL") == 0


def test_arm_torque_ratio_stays_below_limit_through_jammed_routine_to_stall(rig: Rig) -> None:
    """Per-joint ratio |tau_ext| / actuator forcerange stays under the margin up to the stall."""
    co = rig.co
    margin = co.config.arm_torque_margin
    for bay_id in (1, 2):
        # limits come from the model by joint name: 87 Nm joints 1-4, 12 Nm joints 5-7
        np.testing.assert_allclose(co.rig(bay_id).arm_force_limit, [87, 87, 87, 87, 12, 12, 12])
        co.start_cycle(bay_id, flap_jammed=True)
    worst = {1: 0.0, 2: 0.0}
    for _ in range(1500):
        snaps = co.tick()
        for bay_id in (1, 2):
            worst[bay_id] = max(worst[bay_id], co.rig(bay_id).arm_torque_ratio())
        if all(s.state != BayState.NOMINAL for s in snaps):
            break
    assert all(s.state != BayState.NOMINAL for s in snaps)
    for bay_id in (1, 2):
        assert worst[bay_id] < margin
        assert co.peak_arm_ratio(bay_id) < margin
        assert co.last_trigger_reason(bay_id) == "confidence"  # the guard was only a backstop


def test_arm_torque_breach_fast_path_triggers_stall_without_300ms_window(
    unoq: UnoQClient, hand_spec: HandSpec, validator: SkillValidator
) -> None:
    from teleop.atc.coordinator import CoordinatorConfig

    tight = Rig(unoq, hand_spec, validator)
    tight.co.config = CoordinatorConfig(arm_torque_margin=0.06)  # wrist limit 0.72 Nm
    tight.co.start_cycle(1, flap_jammed=True)
    ticks_in_routine = 0
    for _ in range(1500):
        snaps = tight.co.tick()
        if tight.co.in_routine(1):
            ticks_in_routine += 1
        if snaps[0].state != BayState.NOMINAL:
            break
    assert tight.co.state(1) in (BayState.RESCUE, BayState.QUEUED)
    assert tight.co.last_trigger_reason(1) == "arm_torque"
    # Fired at the first contact tick: well before the 18-tick C(t) window could complete.
    free = Rig(unoq, hand_spec, validator)
    free.co.start_cycle(1, flap_jammed=True)
    free_ticks = 0
    for _ in range(1500):
        snaps = free.co.tick()
        if free.co.in_routine(1):
            free_ticks += 1
        if snaps[0].state != BayState.NOMINAL:
            break
    assert free.co.last_trigger_reason(1) == "confidence"
    assert ticks_in_routine <= free_ticks - 10
    assert tight.co.rig(1).bolt_x() < BOLT_CLEARED_M


# --------------------------------------------------------------------------- arm torque breach


def _force_breach_tick(
    rig: Rig, console: ConsoleInputState | None = None, operator: OperatorInput | None = None
) -> None:
    """One tick with a zero margin (every ratio breaches), then the normal margin is restored."""
    co = rig.co
    normal = co.config
    co.config = dataclasses.replace(normal, arm_torque_margin=0.0)
    try:
        co.tick(console, operator)
    finally:
        co.config = normal


def test_arm_torque_breach_in_hover_bleeds_off_to_zero_velocity_hold(rig: Rig) -> None:
    co = rig.co
    co.start_cycle(1, flap_jammed=True)
    rig.run_until(1, {BayState.RESCUE, BayState.QUEUED})
    for _ in range(5):  # the 3 cm retract is under way but far from finished
        co.tick()
    bay = co._bays[1]  # noqa: SLF001 - internal retract/hold state, no public accessor
    assert bay.move is not None and not bay.move.done
    assert bay.state == BayState.RESCUE and not bay.clutch_held  # unclutched hold = hover
    n_events = len(rig.sink.events)
    x_breach = _latch_rel(rig, 1)[0]

    _force_breach_tick(rig)
    assert co.peak_arm_ratio(1, "hover") > 0.0
    assert bay.move is None  # retract cancelled
    assert bay.hold_pose is not None
    assert co.state(1) == BayState.RESCUE  # hover breach never changes the state
    for _ in range(10):  # let the servos' own damping settle the arm
        co.tick()
    p0 = _bay_pos(rig, 1)
    for _ in range(30):
        co.tick()
    assert np.linalg.norm(_bay_pos(rig, 1) - p0) < 1e-3  # zero-velocity hold
    assert _latch_rel(rig, 1)[0] > x_breach - 0.01  # did not finish the 3 cm retract
    assert len(rig.sink.events) == n_events  # a hover breach is not an escalation


def test_arm_torque_breach_in_approach_aborts_the_approach(rig: Rig) -> None:
    co = rig.co
    co.start_cycle(1)
    for _ in range(10):
        co.tick()
    assert co._bays[1].approach is not None  # noqa: SLF001
    _force_breach_tick(rig)
    assert co.peak_arm_ratio(1, "approach") > 0.0
    assert co._bays[1].approach is None  # noqa: SLF001
    assert co._bays[1].pending_latch is None  # noqa: SLF001
    assert co.outcome(1) == "approach aborted: arm torque limit"
    for _ in range(10):
        co.tick()
    p0 = _bay_pos(rig, 1)
    for _ in range(60):
        co.tick()
        assert not co.in_routine(1)  # the routine never starts
    assert co.state(1) == BayState.NOMINAL
    assert np.linalg.norm(_bay_pos(rig, 1) - p0) < 1e-3


def test_arm_torque_breach_in_rescue_discards_recording_and_locks_out_clutch(rig: Rig) -> None:
    co = rig.co
    rig.stall_bay(1, flap_jammed=True)
    assert co.state(1) == BayState.RESCUE
    bay = co._bays[1]  # noqa: SLF001
    console = ConsoleInputState(selected_bay=1, clutch=True)
    samples = demo_operator(rig, 1)
    for _ in range(30):
        co.tick(console, next(samples))
    assert bay.clutch_held and bay.buffer is not None and len(bay.buffer) > 0
    n_events = len(rig.sink.events)

    _force_breach_tick(rig, console, next(samples))
    assert co.peak_arm_ratio(1, "rescue") > 0.0
    assert not bay.clutch_held and bay.breach_lock
    assert len(bay.buffer) == 0  # recording discarded
    assert co.state(1) == BayState.RESCUE

    # Clutch still held: re-engagement is locked out, the arm stays put, nothing is recorded.
    for _ in range(10):
        co.tick(console, next(samples))
    p0 = _bay_pos(rig, 1)
    for _ in range(20):
        co.tick(console, next(samples))
        assert not bay.clutch_held and len(bay.buffer) == 0
    assert np.linalg.norm(_bay_pos(rig, 1) - p0) < 1e-3

    # Releasing the clutch lifts the lock without a validation attempt; the next hold records.
    co.tick(ConsoleInputState(selected_bay=1, clutch=False), None)
    assert not bay.breach_lock and co.state(1) == BayState.RESCUE
    for _ in range(10):
        co.tick(console, next(samples))
    assert bay.clutch_held and len(bay.buffer) > 0
    assert len(rig.sink.events) == n_events  # no discarded-rescue event, no validation


def test_arm_torque_breach_in_autonomous_replay_aborts_retracts_and_escalates(
    rig: Rig,
) -> None:
    co = rig.co
    _rescued_rig(rig)
    n_events = len(rig.sink.events)
    n_ctx = len(rig.reasoner.contexts)
    co.start_cycle(2, flap_jammed=True)
    rig.run_until(2, {BayState.AUTONOMOUS_SKILL})
    for _ in range(5):  # mid back-off
        co.tick()
    assert co.state(2) == BayState.AUTONOMOUS_SKILL
    x_breach = _latch_rel(rig, 2)[0]

    _force_breach_tick(rig)
    assert co.peak_arm_ratio(2, "autonomous") > 0.0
    assert co.outcome(2) == "aborted: arm torque limit"
    assert co._bays[2].auto is None  # noqa: SLF001
    assert co.state(2) in (BayState.QUEUED, BayState.RESCUE)  # QUEUED, promoted at once
    alerts = rig.sink.events[n_events:]
    assert [e.kind for e in alerts] == ["STALL_ALERT"]
    assert alerts[0].bay_id == 2 and alerts[0].metadata["reason"] == "arm torque limit"
    assert len(rig.reasoner.contexts) == n_ctx + 1
    assert rig.reasoner.contexts[-1].event_kind == "STALL_ALERT"

    for _ in range(60):  # the 3 cm retract completes and the arm rests there
        co.tick()
    assert _latch_rel(rig, 2)[0] == pytest.approx(x_breach - 0.03, abs=0.01)
    assert co.rig(2).bolt_x() < BOLT_CLEARED_M
    assert co.outcome(2) == "aborted: arm torque limit"
    assert len(rig.sink.events) == n_events + 1  # no further alerts


def test_control_uses_estimate_not_truth(unoq, hand_spec, validator):
    rig = Rig(unoq, hand_spec, validator)
    rig.co.start_cycle(1)
    for _ in range(300):
        rig.co.tick()
        if rig.co.in_routine(1):
            break
    est, truth = rig.co.estimate(1).pose, rig.co.rig(1).true_world_latch()
    assert est != truth  # seeded error is applied
    assert rig.co.rig(1).world_latch() == est


def test_zero_sigma_reproduces_true_frame_cycles(unoq, hand_spec, validator):
    cfg = CoordinatorConfig(pose_sigma_xy_m=0.0, pose_sigma_yaw_rad=0.0)
    rig = Rig(unoq, hand_spec, validator, config=cfg)  # the rig fixture closes the coordinator
    co = rig.co
    for _ in range(3):
        for bay in (1, 2):
            co.start_cycle(bay)
        for _ in range(600):
            co.tick()
            if co.outcome(1) and co.outcome(2):
                break
        assert co.outcome(1) == co.outcome(2) == "cleared"
        for bay in (1, 2):
            touch = co.last_touch(bay)
            assert touch is not None and touch.last_correction_m is not None
            assert abs(touch.last_correction_m) < 0.0005  # exact frame: nothing to correct


def test_same_seed_stall_rescue_gate_is_deterministic(
    hand_spec: HandSpec, validator: SkillValidator
) -> None:
    def run() -> tuple[Any, Any, Any]:
        # A fresh UNO Q node per run: its prototype library is stateful (Jam_Sig_N).
        server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="det-node").start()
        client = UnoQClient(
            "127.0.0.1", server.udp_port, server.tcp_port, local_server=server
        ).connect()
        try:
            return _run_once(client)
        finally:
            client.close()

    def _run_once(client: UnoQClient) -> tuple[Any, Any, Any]:
        r = Rig(client, hand_spec, validator)
        _rescued_rig(r)
        report = r.co.last_report
        assert report is not None
        events = [(e.kind, e.bay_id, e.summary) for e in r.sink.events]
        results = [(x.passed, x.conditions) for x in report.results]
        return events, results, np.array(r.co.data.qpos)

    ev_a, res_a, q_a = run()
    ev_b, res_b, q_b = run()
    assert ev_a == ev_b
    assert res_a == res_b
    np.testing.assert_array_equal(q_a, q_b)


# --------------------------------------------------------------------------- touch re-localization


def _x_err(co: AtcCoordinator, bay_id: int) -> float:
    """|estimate - truth| along the latch push axis (test-only ground-truth read)."""
    est, truth = co.estimate(bay_id).pose, co.rig(bay_id).true_world_latch()
    axis = np.array([math.cos(truth.yaw_rad), math.sin(truth.yaw_rad), 0.0])
    return abs(float(np.dot(np.subtract(est.position, truth.position), axis)))


def _routine_index_gap_m() -> float:
    """Index-pad-surface to hook-back-face gap along latch +X at the routine start pose.

    Kinematic: the fingers are set to ``ROUTINE.fingers`` (servoed proximal joints at their
    targets, coupled distal joints through their joint equalities, compliant wrist at rest) and
    the index tip offset from the wrist site is carried into the latch frame by the routine
    start pose, with the bolt at 0.
    """
    import mujoco as _mujoco

    mujoco: Any = _mujoco
    m = build_single_bay_scene(CANONICAL_LATCH)
    d = mujoco.MjData(m)
    for name, q in ROUTINE.fingers.items():
        jnt = int(m.actuator_trnid[int(m.actuator(f"bay1/{name}").id)][0])
        d.qpos[int(m.jnt_qposadr[jnt])] = q
    for i in range(int(m.neq)):
        if int(m.eq_type[i]) == int(mujoco.mjtEq.mjEQ_JOINT) and int(m.eq_obj2id[i]) >= 0:
            c0, c1 = float(m.eq_data[i][0]), float(m.eq_data[i][1])
            q_driver = float(d.qpos[int(m.jnt_qposadr[int(m.eq_obj2id[i])])])
            d.qpos[int(m.jnt_qposadr[int(m.eq_obj1id[i])])] = c0 + c1 * q_driver
    mujoco.mj_kinematics(m, d)
    wrist_id, tip_id = int(m.site("bay1/wrist_site").id), int(m.site("bay1/right_index_tip").id)
    r_wrist = np.asarray(d.site_xmat[wrist_id]).reshape(3, 3)
    tip_in_wrist = r_wrist.T @ (np.asarray(d.site_xpos[tip_id]) - np.asarray(d.site_xpos[wrist_id]))
    start = ROUTINE.pose_at(0.0)
    assert start.position is not None
    tip_x = float(np.asarray(start.position)[0] + (ROUTINE.rotation() @ tip_in_wrist)[0])
    return HOOK_BACK_X_M - (tip_x + INDEX_PAD_RADIUS_M)


def test_touch_detector_encoder_and_wrist_paths() -> None:
    d = TouchDetector(wrist_nm=0.5)
    assert not d.update(0.0, 0.1) and not d.update(0.0004, 0.2)
    assert d.update(0.0006, 0.0)
    assert not d.update(0.01, 0.0)  # fires once per arming
    d.reset()
    assert not d.update(0.0, 0.6) and d.update(0.0, 0.6)  # two consecutive wrist ticks


def test_touch_detector_encoder_is_relative_to_the_bolt_at_arming() -> None:
    d = TouchDetector(wrist_nm=0.5)
    assert not d.update(0.035, 0.0)  # armed on a stalled bolt: its offset is not a touch
    assert not d.update(0.0354, 0.0)
    assert d.update(0.0356, 0.0)
    d.reset()
    assert not d.update(0.0, 0.6) and not d.update(0.0, 0.1) and not d.update(0.0, 0.6)


def test_no_false_touch_in_approach_or_lift(rig: Rig) -> None:  # Review Focus 2
    co = rig.co
    co.rig(1).set_bolt_x(0.06)  # the previous cycle left the bolt pushed out
    co.start_cycle(1)
    for _ in range(80):  # still approaching, the bolt was just reset
        co.tick()
        assert co.last_touch(1) is None or co.in_routine(1)
    for _ in range(600):  # the routine arms the detector only on the reset bolt
        co.tick()
        if co.in_routine(1):
            break
    assert co.in_routine(1) and co.rig(1).bolt_x() < 0.0005 and co.last_touch(1) is None


def test_routine_touch_shrinks_estimate_error_to_mm(rig: Rig) -> None:
    co = rig.co
    co.start_cycle(1)
    err0 = _x_err(co, 1)  # |estimate - truth| along the latch axis; test-only truth read
    for _ in range(600):
        co.tick()
        if co.outcome(1):
            break
    assert co.outcome(1) == "cleared" and co.last_touch(1) is not None
    assert _x_err(co, 1) < 0.0015 and _x_err(co, 1) < err0


def test_routine_touch_target_jump_is_bounded_by_lead_plus_correction(rig: Rig) -> None:
    co = rig.co
    bay = co._bays[1]  # noqa: SLF001 - commanded target history, no public accessor
    co.start_cycle(1)
    worst = 0.0
    prev: np.ndarray | None = None
    for _ in range(600):
        co.tick()
        if co.in_routine(1) and bay.prev_cmd_pos is not None:
            if prev is not None:
                worst = max(worst, float(np.linalg.norm(bay.prev_cmd_pos - prev)))
            prev = np.array(bay.prev_cmd_pos)
        if co.outcome(1):
            break
    touch = co.last_touch(1)
    assert touch is not None and touch.last_correction_m is not None
    assert worst <= co.config.push_lead_limit_m + abs(touch.last_correction_m)


def _spy_stall_entry(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> list[tuple[bool, int]]:
    """Record (touch already taken, armed) at the entry of every ``_on_stall``."""
    co = rig.co
    seen: list[tuple[bool, int]] = []
    orig = co._on_stall  # pyright: ignore[reportPrivateUsage]

    def spy(bay: Any, trigger: Any) -> None:
        seen.append((co.last_touch(1) is not None, len(co.touch_events())))
        orig(bay, trigger)

    monkeypatch.setattr(co, "_on_stall", spy)
    return seen


def test_pinned_bolt_touch_via_wrist_still_escalates(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:  # Review Focus 3
    _rescued_rig(rig)  # a known jam prototype, so the pinned signature is refused as OOD
    seen = _spy_stall_entry(rig, monkeypatch)
    rig.stall_bay(1, bolt_pinned=True)
    assert rig.co.state(1) == BayState.OOD_ESCALATED
    assert seen and seen[0][0], "the wrist path must fire before the stall tick"
    touch = rig.co.last_touch(1)
    assert touch is not None and touch.last_correction_m is not None
    assert touch.rejected_correction_m is None
    assert abs(touch.last_correction_m) < 0.004  # bolt pinned at x ~ 0: a small correction
    assert abs(rig.co.rig(1).bolt_x()) < 0.002


def test_stall_is_itself_a_touch_when_the_wrist_never_fired(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _rescued_rig(rig)
    monkeypatch.setattr(BayRig, "wrist_push_torque", lambda self: 0.0)
    # The pinned bolt also creeps enough to trip the encoder path; silence it so only the
    # stall-as-touch fallback in _on_stall (still armed) can correct the estimate.
    monkeypatch.setattr(TouchDetector, "update", lambda self, x, w: False)
    seen = _spy_stall_entry(rig, monkeypatch)
    rig.stall_bay(1, bolt_pinned=True)
    assert rig.co.state(1) == BayState.OOD_ESCALATED
    assert seen and not seen[0][0], "no touch before the stall: the wrist path is disabled"
    touch = rig.co.last_touch(1)
    assert touch is not None, "the stall-as-touch fallback must correct the estimate"
    assert len(rig.co.touch_events()) > seen[0][1]


def test_routine_standoff_clears_hook_at_3_sigma() -> None:
    gap = _routine_index_gap_m()  # kinematic, from tip offsets at ROUTINE.fingers
    assert gap - 3 * POSE_SIGMA_XY_M >= 0.003


def test_replay_rebase_blends_to_the_corrected_frame(rig: Rig) -> None:
    _rescued_rig(rig)
    skill = next(iter(rig.co.library.values()))
    wfl = rig.co.rig(1).world_from_latch()
    shifted = translate_pose(wfl, (0.004, 0.0, 0.0))
    replay, ref = SkillReplay(skill, wfl), SkillReplay(skill, shifted)

    def at(r: SkillReplay, t: float) -> np.ndarray:
        return pose_position(r.command(t).wrist)

    t0 = 0.5
    p_old = at(replay, t0)
    replay.rebase(shifted, t0, blend_s=0.2)
    np.testing.assert_allclose(at(replay, t0), p_old, atol=1e-9)  # no jump at the touch
    mid, old_mid = at(replay, t0 + 0.1), at(SkillReplay(skill, wfl), t0 + 0.1)
    assert np.linalg.norm(mid - old_mid) > 1e-4 and np.linalg.norm(mid - at(ref, t0 + 0.1)) > 1e-4
    np.testing.assert_allclose(at(replay, t0 + 0.25), at(ref, t0 + 0.25), atol=1e-9)


# --------------------------------------------------------------------------- randomized gate


def _res(passed: bool) -> RolloutResult:
    cond = RolloutConditions((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0, 35.0, 1.0)
    return RolloutResult(
        (0.0, 0.0, 0.0), 0.06 if passed else 0.03, 1.0, 0.1, True, passed, passed, cond
    )


@pytest.fixture(scope="module")
def rescued_skill(hand_spec: HandSpec, validator: SkillValidator) -> Iterator[SkillPrimitive]:
    """The scripted Bay 1 rescue skill, fitted once per module."""
    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="test-node").start()
    client = UnoQClient(
        "127.0.0.1", server.udp_port, server.tcp_port, local_server=server
    ).connect()
    try:
        rig = Rig(client, hand_spec, validator)
        _rescued_rig(rig)
        rig.co.close()
        yield next(iter(rig.co.library.values()))
    finally:
        client.close()


def test_gate_conditions_are_seeded_and_cover_spec_offsets(validator: SkillValidator) -> None:
    a, b = validator.draw_conditions("s/1"), validator.draw_conditions("s/1")
    assert a == b and len(a) == 20
    assert {round(c.perturbation[2] / 30) for c in a} == {-1, 0, 1}  # all three spec yaws
    assert all(0.8 <= c.flap_stiffness_scale <= 1.2 for c in a)
    assert all(abs(c.flap_springref_deg - 35.0) <= 3.0 for c in a)
    assert all(0.75 <= c.friction_scale <= 1.25 for c in a)
    assert a != validator.draw_conditions("s/2")  # the skill id seeds the gate RNG


def test_realism_scales_every_gate_condition(hand_spec: HandSpec) -> None:
    full = SkillValidator(hand_spec).draw_conditions("s/1")
    half = SkillValidator(hand_spec, realism=0.5).draw_conditions("s/1")
    off = SkillValidator(hand_spec, realism=0.0)
    nominal = off.nominal_springref_deg
    for f, h, o in zip(full, half, off.draw_conditions("s/1"), strict=True):
        assert o.est_error == (0.0, 0.0, 0.0)
        assert o.flap_stiffness_scale == 1.0 and o.friction_scale == 1.0
        assert o.flap_springref_deg == nominal
        assert h.est_error == pytest.approx(tuple(0.5 * e for e in f.est_error))
        assert h.friction_scale - 1.0 == pytest.approx(0.5 * (f.friction_scale - 1.0))
        jitter_full = np.subtract(f.perturbation, o.perturbation)
        assert np.subtract(h.perturbation, o.perturbation) == pytest.approx(0.5 * jitter_full)
    with pytest.raises(ValueError, match="realism"):
        SkillValidator(hand_spec, realism=1.5)


def test_rollout_restores_model_parameters(
    validator: SkillValidator, rescued_skill: SkillPrimitive
) -> None:
    m = validator.model
    before = m.geom_friction.copy(), m.jnt_stiffness.copy(), m.qpos_spring.copy()
    report = validator.validate(rescued_skill)
    assert report.total == GATE_ROLLOUTS
    np.testing.assert_array_equal(m.geom_friction, before[0])
    np.testing.assert_array_equal(m.jnt_stiffness, before[1])
    np.testing.assert_array_equal(m.qpos_spring, before[2])


def test_rollout_restores_model_parameters_when_a_rollout_raises(
    validator: SkillValidator, rescued_skill: SkillPrimitive, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = validator.model
    before = m.geom_friction.copy(), m.jnt_stiffness.copy(), m.qpos_spring.copy()

    def boom(*_: object, **__: object) -> None:
        raise RuntimeError("rollout crashed")

    monkeypatch.setattr(validator.rig, "command", boom)
    with pytest.raises(RuntimeError, match="rollout crashed"):
        validator.validate(rescued_skill)
    np.testing.assert_array_equal(m.geom_friction, before[0])
    np.testing.assert_array_equal(m.jnt_stiffness, before[1])
    np.testing.assert_array_equal(m.qpos_spring, before[2])


def test_threshold_rule() -> None:
    r = ValidationReport(
        results=(_res(True),) * 18 + (_res(False),) * 2, elapsed_s=1.0, threshold=18
    )
    assert r.passed and r.pass_rate == pytest.approx(0.9)
    assert not dataclasses.replace(r, threshold=19).passed


def _run_conditions(
    validator: SkillValidator,
    skill: SkillPrimitive,
    conds: tuple[RolloutConditions, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> ValidationReport:
    monkeypatch.setattr(validator, "draw_conditions", lambda _skill_id: conds)
    return validator.validate(skill)


def test_gate_rollouts_are_independent_trials(
    validator: SkillValidator, rescued_skill: SkillPrimitive, monkeypatch: pytest.MonkeyPatch
) -> None:
    jammed = RolloutConditions((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.2, 38.0, 1.25)
    nominal = RolloutConditions((0.08, 0.06, 30.0), (0.0, 0.0, 0.0), 1.0, 35.0, 1.0)
    report = _run_conditions(validator, rescued_skill, (nominal, jammed, nominal), monkeypatch)
    first, _, again = report.results
    assert first == again  # the rollout in between leaves nothing behind


def test_rescued_skill_carries_its_demo_contact_offset(rescued_skill: SkillPrimitive) -> None:
    assert rescued_skill.touch_offset_m is not None
    assert -0.03 < rescued_skill.touch_offset_m < 0.0  # the hook face sits ahead of the tip


def test_replay_touch_uses_the_skill_offset(
    validator: SkillValidator, rescued_skill: SkillPrimitive, monkeypatch: pytest.MonkeyPatch
) -> None:
    exact = [RolloutConditions(p, (0.0, 0.0, 0.0), 1.0, 35.0, 1.0) for p in PERTURBATIONS_DEG]
    ahead = [dataclasses.replace(c, est_error=(0.001, 0.0, 0.0)) for c in exact]
    results = _run_conditions(validator, rescued_skill, (*exact, *ahead), monkeypatch).results
    shifts = [r.touch_correction_m for r in results]
    assert all(s is not None for s in shifts) and not any(r.touch_rejected for r in results)
    nom, plus = np.array(shifts[:3], dtype=float), np.array(shifts[3:], dtype=float)
    assert np.all(np.abs(nom) < 0.0015)  # the demo's own offset: no posture bias
    # +1 mm along the push axis is corrected by about -1 mm on average. A single contact's encoder
    # trigger scatters by about +-0.5 mm, depending on the rescued demo: corrections of -1.55,
    # -1.47 and -0.85 mm at realism 0.25, and -0.9, +0.1 and -0.9 mm at 0.5. None goes the wrong
    # way by more than the trigger's resolution.
    delta = plus - nom
    assert float(np.mean(delta)) == pytest.approx(-0.001, abs=6e-4), delta
    assert np.all(delta < 4e-4), delta


def test_gate_result_applies_exactly_at_deadline(rig: Rig) -> None:
    rig.stall_bay(1, flap_jammed=True)
    run_rescue(rig, 1, max_extra_ticks=0)  # release only, don't wait for validation
    co = rig.co
    assert co.state(1) == BayState.VALIDATING and co.gate_pending(1)
    t_submit = co.time_s
    while co.state(1) == BayState.VALIDATING:
        co.tick()
    assert co.time_s - t_submit == pytest.approx(co.config.gate_deadline_s, abs=2 * CONTROL_DT)
    assert not co.gate_pending(1)


def test_gate_worker_exception_fails_cleanly(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rig.co.validator, "validate", lambda s: 1 / 0)
    rig.stall_bay(1, flap_jammed=True)
    run_rescue(rig, 1)
    assert rig.co.state(1) != BayState.VALIDATING and len(rig.co.library) == 0
    assert rig.co.last_report is not None and not rig.co.last_report.passed


def test_close_is_idempotent(rig: Rig) -> None:
    rig.co.close()
    rig.co.close()
