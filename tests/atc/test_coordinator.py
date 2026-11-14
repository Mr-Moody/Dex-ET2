"""T-140: ATC fleet coordinator, confidence C(t), FIFO queue, torque tube and 3-rollout gate.

The scripted operator drives the coordinator through its public rescue/clutch API
(``ConsoleInputState`` + ``OperatorInput``). Tests read ``flap_hinge`` only to assert simulator
ground truth, never to steer anything.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pytest

from teleop.atc.coordinator import (
    ROUTINE,
    AtcCoordinator,
    OperatorInput,
    StallDetector,
    compute_confidence,
)
from teleop.atc.types import (
    BOLT_CLEARED_M,
    BayState,
    ConsoleInputState,
    SkillPrimitive,
)
from teleop.atc.validator import (
    CONTROL_DT,
    ApproachPlan,
    SkillReplay,
    SkillValidator,
    lifted,
    step_physics,
    torque_tube_ok,
)
from teleop.compliance.protocol import ComplianceEvent
from teleop.reasoning.protocol import ReasoningContext, ReasoningVerdict
from teleop.sim.bays import PANDA_XML
from teleop.sim.hands import HandSpec, load_hand
from teleop.types import Pose
from teleop.unoq.client import UnoQClient
from teleop.unoq.dmp import (
    LEAP_ACTUATORS,
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
        right = load_hand("leap").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


pytestmark = pytest.mark.skipif(
    not _models_present(), reason="Panda/LEAP MJCFs missing; run scripts/fetch_models.py"
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
    return load_hand("leap")


@pytest.fixture(scope="module")
def validator(hand_spec: HandSpec) -> SkillValidator:
    return SkillValidator(hand_spec)


@pytest.fixture
def unoq() -> Iterator[UnoQClient]:
    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="test-node").start()
    client = UnoQClient(
        "127.0.0.1", server.udp_port, server.tcp_port, local_server=server
    ).connect()
    yield client
    client.close()


class Rig:
    def __init__(self, unoq: UnoQClient, hand_spec: HandSpec, validator: SkillValidator) -> None:
        self.sink = RecordingSink()
        self.reasoner = RecordingReasoner()
        self.unoq = unoq
        self.co = AtcCoordinator.build(
            unoq, self.sink, self.reasoner, hand_spec=hand_spec, validator=validator
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


@pytest.fixture
def rig(unoq: UnoQClient, hand_spec: HandSpec, validator: SkillValidator) -> Rig:
    return Rig(unoq, hand_spec, validator)


# --------------------------------------------------------------------------- scripted operator

# Waypoints (duration s, wrist delta from the safe hover in the latch frame in metres, finger
# press fraction). The index finger rests behind the bolt hook, the middle finger presses the
# raised stopper flap flat (press fraction 1), then the wrist pushes the bolt clear.
DEMO_PRESS = {"mf_mcp_act": 0.7813, "mf_pip_act": -0.0436, "mf_dip_act": -0.2812}
DEMO_WAYPOINTS: list[tuple[float, tuple[float, float, float], float]] = [
    (0.0, (0.0, 0.0, 0.0), 0.0),
    (0.4981, (0.0056, 0.004, 0.0021), 0.0),
    (0.8199, (0.0056, 0.004, 0.0016), 1.0),
    (0.3333, (0.0056, 0.004, 0.0016), 1.0),
    (1.4385, (0.0056 + 0.0458, 0.004, 0.0016), 1.0),
]


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


def run_rescue(rig: Rig, bay_id: int, max_extra_ticks: int = 30) -> None:
    """Hold the clutch through the scripted rescue, release it, and let validation finish."""
    console = ConsoleInputState(selected_bay=bay_id, clutch=True)
    for sample in demo_operator(rig, bay_id):
        snaps = rig.co.tick(console, sample)
        if snaps[bay_id - 1].x_bolt_m >= BOLT_CLEARED_M + 0.0015:
            break
    released = ConsoleInputState(selected_bay=bay_id, clutch=False)
    for _ in range(max_extra_ticks):
        rig.co.tick(released, None)
        if rig.co.state(bay_id) == BayState.NOMINAL:
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
    wrist[:, 0] = -0.0375 + 0.06 * t / 1.5
    wrist[:, 1], wrist[:, 2] = ROUTINE.y, ROUTINE.z
    wrist[:, 3:] = q
    base = np.array([ROUTINE.fingers.get(a, 0.0) for a in LEAP_ACTUATORS])
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
    tight.co.config = type(tight.co.config)(tau_margin=0.4)
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
    for _ in range(1500):
        snaps = co.tick()
        assert len(co.rescue_slot_holders()) <= 1
        if snaps[1].state == BayState.QUEUED:
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
    assert _latch_rel(rig, 2)[0] < -0.02  # backed off along latch -X (stall wrist was ~ -0.007)
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


def test_end_to_end_bay1_rescue_fits_dmp_passes_3_rollouts_and_bay2_clears_35deg_12cm(
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
    assert report is not None and report.passed and report.total == 3
    assert [r.perturbation for r in report.results] == [
        (-0.08, -0.06, -30.0),
        (0.0, 0.0, 0.0),
        (0.08, 0.06, 30.0),
    ]
    assert all(r.bolt_x_m >= BOLT_CLEARED_M and r.joint_limits_ok for r in report.results)
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
    assert rig.reasoner.contexts[-1].rollout_passed == 3

    # The gate itself is fast: 3 headless rollouts in < 0.4 s (best of three warm runs).
    elapsed = min(validator.validate(skill).elapsed_s for _ in range(3))
    assert elapsed < 0.4, f"validation gate took {elapsed:.3f}s"

    # Bay 2 sits at +12 cm / +35 deg relative to Bay 1's latch frame.
    l1, l2 = co.rig(1).world_latch(), co.rig(2).world_latch()
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
