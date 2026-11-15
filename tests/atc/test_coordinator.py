"""ATC fleet coordinator: C(t), FIFO queue, torque tube, 20-rollout gate, OOD, and touch/filter."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pytest

from conftest import NullReasoner, RecordingComplianceSink, needs_models
from teleop.atc.coordinator import (
    DEFAULT_REALISM,
    ROUTINE,
    AtcCoordinator,
    CoordinatorConfig,
    OperatorInput,
    StallDetector,
    compute_confidence,
)
from teleop.atc.estimate import (
    HOOK_BACK_X_M,
    INDEX_PAD_RADIUS_M,
    TOUCH_SIGMA_M,
    LatchEstimate,
    contact_offset,
    touch_correct,
)
from teleop.atc.glove_filter import GloveInputFilter
from teleop.atc.stage import RESCUE_PRESS, RESCUE_WAYPOINTS
from teleop.atc.types import BOLT_CLEARED_M, BayState, ConsoleInputState, LatchPose, SkillPrimitive
from teleop.atc.validator import (
    CONTROL_DT,
    GATE_PASS_MIN,
    GATE_ROLLOUTS,
    SkillValidator,
    torque_tube_ok,
)
from teleop.sim.bays import CANONICAL_LATCH
from teleop.sim.hands import HandSpec
from teleop.types import HandFrame, Pose
from teleop.unoq.client import UnoQClient
from teleop.unoq.dmp import HAND_ACTUATORS, fit_rbf_dmp, mat_to_quat, quat_to_mat, relative_pose

FLAP_UP_DEG = 30.0


@pytest.fixture(scope="module")
def validator(inspire_spec: HandSpec) -> SkillValidator:
    return SkillValidator(inspire_spec, realism=DEFAULT_REALISM)


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
        self.sink = RecordingComplianceSink()
        self.reasoner = NullReasoner()
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
        for _ in range(40):
            self.co.tick()


@pytest.fixture(autouse=True)
def _close_rigs() -> Iterator[None]:
    yield
    while _RIGS:
        _RIGS.pop().co.close()


@pytest.fixture
def rig(unoq_client: UnoQClient, inspire_spec: HandSpec, validator: SkillValidator) -> Rig:
    return Rig(unoq_client, inspire_spec, validator)


def _demo_operator(rig: Rig, bay_id: int) -> Iterator[OperatorInput]:
    bay_rig = rig.co.rig(bay_id)
    hover = bay_rig.wrist_pose()
    assert hover.position is not None
    r_world = quat_to_mat(bay_rig.world_from_latch().orientation)
    origin = np.asarray(hover.position)
    base = dict(ROUTINE.fingers)
    waypoints = list(RESCUE_WAYPOINTS)
    for i in range(1, len(waypoints)):
        dur, p1, f1 = waypoints[i]
        _, p0, f0 = waypoints[i - 1]
        n = max(round(dur / CONTROL_DT), 1)
        for k in range(1, n + 1):
            a = k / n
            delta = np.asarray(p0) + a * (np.asarray(p1) - np.asarray(p0))
            f = f0 + a * (f1 - f0)
            q = {name: base[name] + f * RESCUE_PRESS.get(name, 0.0) for name in base}
            pos = origin + r_world @ delta
            yield OperatorInput(
                Pose(orientation=hover.orientation, position=(pos[0], pos[1], pos[2])), q
            )


def _run_rescue(rig: Rig, bay_id: int, max_extra_ticks: int | None = None) -> None:
    if max_extra_ticks is None:
        max_extra_ticks = round(rig.co.config.gate_deadline_s / CONTROL_DT) + 60
    console = ConsoleInputState(selected_bay=bay_id, clutch=True)
    for sample in _demo_operator(rig, bay_id):
        snaps = rig.co.tick(console, sample)
        if snaps[bay_id - 1].x_bolt_m >= BOLT_CLEARED_M + 0.0015:
            break
    released = ConsoleInputState(selected_bay=bay_id, clutch=False)
    stop = BayState.VALIDATING if max_extra_ticks == 0 else BayState.NOMINAL
    for _ in range(max_extra_ticks or 30):
        rig.co.tick(released, None)
        if rig.co.state(bay_id) == stop:
            return


def _latch_rel(rig: Rig, bay_id: int) -> np.ndarray:
    r = rig.co.rig(bay_id)
    rel = relative_pose(r.world_from_latch(), r.wrist_pose())
    assert rel.position is not None
    return np.asarray(rel.position)


# --------------------------------------------------------------------------- pure unit invariants


def test_touch_pose_correction_and_outlier_gate() -> None:
    truth = LatchPose((0.5, 0.01, 0.02), 0.3)
    c, s = math.cos(truth.yaw_rad), math.sin(truth.yaw_rad)
    axis_x = np.array([c, s, 0.0])
    tip_expect = np.array(
        [
            truth.position[0] + c * (HOOK_BACK_X_M + 0.01 - INDEX_PAD_RADIUS_M),
            truth.position[1] + s * (HOOK_BACK_X_M + 0.01 - INDEX_PAD_RADIUS_M),
            truth.position[2] + 0.05,
        ]
    )
    est = LatchEstimate(truth, 0.003, 0.003, 0.026)
    moved = touch_correct(est, tip_expect + 0.004 * axis_x, 0.01)
    assert moved.touch_corrected and moved.last_correction_m == pytest.approx(0.004)
    assert moved.sigma_x_m == TOUCH_SIGMA_M and moved.pose.yaw_rad == est.pose.yaw_rad

    outlier = touch_correct(
        LatchEstimate(CANONICAL_LATCH, 0.003, 0.003, 0.026),
        np.array([CANONICAL_LATCH.position[0] + HOOK_BACK_X_M - INDEX_PAD_RADIUS_M + 0.015, 0, 0]),
        0.0,
    )
    assert not outlier.touch_corrected and outlier.rejected_correction_m == pytest.approx(0.015)

    off = contact_offset(est, tip_expect, 0.01)
    assert touch_correct(
        est, tip_expect, 0.01, pad_offset_m=off
    ).last_correction_m == pytest.approx(0.0, abs=1e-12)


def test_glove_input_filter_smooths_jitter_and_restarts_after_gap() -> None:
    rng = np.random.default_rng(0)
    flt = GloveInputFilter()
    ident = (0.0, 0.0, 0.0, 1.0)
    dt_us = 12_000
    raw_x: list[float] = []
    out_x: list[float] = []
    for i in range(250):
        x = 0.2 + float(rng.normal(0.0, 0.002))
        frame = HandFrame("right", i * 1_000_000, i, "test", {}, device_time_us=i * dt_us)
        wrist, _ = flt.step(frame, Pose(orientation=ident, position=(x, 0.0, 0.0)), {"idx": 0.5})
        assert wrist.position is not None
        raw_x.append(x)
        out_x.append(wrist.position[0])
    assert np.std(out_x[80:]) < 0.35 * np.std(raw_x[80:])

    # Stream gap > gap_reset_s restarts without lag.
    gap_frame = HandFrame("right", 0, 999, "test", {}, device_time_us=250 * dt_us + 500_000)
    wrist, targets = flt.step(
        gap_frame, Pose(orientation=ident, position=(0.45, 0.0, 0.0)), {"idx": 1.0}
    )
    assert wrist.position is not None and wrist.position[0] == pytest.approx(0.45)
    assert targets["idx"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- coordinator physics


@needs_models
def test_confidence_metric_nominal_slide_and_300ms_stall_detector(rig: Rig) -> None:
    assert compute_confidence(0.0, 0.05, 0.05) == pytest.approx(1.0)
    assert compute_confidence(3.0, 0.05, 0.0) == pytest.approx(math.exp(-(0.8 * 0.5 + 0.9)))

    det = StallDetector()
    tau = np.zeros(3)
    assert all(det.update(0.1, 0.034, 1.0, tau, 0.5) is None for _ in range(17))
    trig = det.update(0.1, 0.034, 1.0, tau, 0.5)
    assert trig is not None and trig.reason == "confidence"

    co = rig.co
    co.start_cycle(1)
    confidence: list[float] = []
    for _ in range(500):
        snap = co.tick()[0]
        if co.in_routine(1):
            confidence.append(snap.confidence)
        if co.outcome(1) == "cleared":
            break
    assert co.outcome(1) == "cleared" and min(confidence[25:-3]) > 0.85

    co.start_cycle(1, flap_jammed=True)
    confidence = []
    for _ in range(900):
        snap = co.tick()[0]
        if co.in_routine(1) or snap.state != BayState.NOMINAL:
            confidence.append(snap.confidence)
        if snap.state != BayState.NOMINAL:
            break
    assert rig.flap_deg(1) > FLAP_UP_DEG
    low_run = 0
    for c in reversed(confidence):
        if c >= 0.3:
            break
        low_run += 1
    assert low_run >= 18


@needs_models
def test_additive_torque_tube_aborts_jammed_autonomous_replay(
    rig: Rig, validator: SkillValidator, unoq_client: UnoQClient, inspire_spec: HandSpec
) -> None:
    assert torque_tube_ok(3.4, 0.0) and not torque_tube_ok(3.6, 0.0)
    assert torque_tube_ok(11.0, 8.0) and not torque_tube_ok(12.0, 10.0)

    rig.stall_bay(2, flap_jammed=True)
    z = rig.co.last_signature(2)
    assert z is not None
    n = 90
    t = np.linspace(0.0, 1.5, n)
    wrist = np.zeros((n, 7))
    wrist[:, 0] = ROUTINE.start_x - 0.01 + 0.06 * t / 1.5
    wrist[:, 1], wrist[:, 2] = ROUTINE.y, ROUTINE.z
    wrist[:, 3:] = mat_to_quat(ROUTINE.rotation())
    base = np.array([ROUTINE.fingers.get(a, 0.0) for a in HAND_ACTUATORS])
    skill = fit_rbf_dmp(
        t, wrist, np.tile(base, (n, 1)), np.zeros(n), prototype=z, signature_id="Jam_Sig_1"
    )
    unoq_client.register_skill(skill)

    tight = Rig(unoq_client, inspire_spec, validator, config=CoordinatorConfig(tau_margin=0.1))
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


@needs_models
def test_single_active_fifo_queue_serializes_concurrent_bay_stalls(rig: Rig) -> None:
    co = rig.co
    co.start_cycle(1, flap_jammed=True)
    for _ in range(1500):
        co.tick()
        if co.in_routine(1):
            break
    for _ in range(40):
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
    assert seen_queued and co.state(1) == BayState.RESCUE and co.state(2) == BayState.QUEUED
    assert co.queue == (2,)

    _run_rescue(rig, 1)
    assert co.state(1) == BayState.NOMINAL
    co.tick()
    assert co.state(2) == BayState.RESCUE and co.queue == ()


@needs_models
def test_bay1_rescue_passes_20_rollouts_bay2_clears_autonomously_and_pinned_bolt_refuses_ood(
    rig: Rig,
) -> None:
    co = rig.co
    rig.stall_bay(1, flap_jammed=True)
    assert co.state(1) == BayState.RESCUE
    _run_rescue(rig, 1)

    report = co.last_report
    assert report is not None and report.total == GATE_ROLLOUTS and report.passed
    assert report.threshold == GATE_PASS_MIN and len(co.library) == 1
    skill: SkillPrimitive = next(iter(co.library.values()))
    assert skill.n_kernels == 30 and skill.signature_id == "Jam_Sig_1"
    assert rig.sink.kinds() == ["STALL_ALERT", "RESCUE_LOGGED", "VALIDATION_GATE", "SKILL_DEPLOYED"]

    # Bay 2 (+12 cm / +35 deg) clears autonomously with the validated DMP skill.
    co.start_cycle(2, flap_jammed=True)
    states: list[BayState] = []
    wrist_at_stall = None
    back_min = 0.0
    for _ in range(1500):
        snap = co.tick()[1]
        states.append(snap.state)
        if snap.state == BayState.AUTONOMOUS_SKILL and wrist_at_stall is None:
            wrist_at_stall = _latch_rel(rig, 2)
            back_min = float(wrist_at_stall[0])
        elif wrist_at_stall is not None:
            back_min = min(back_min, float(_latch_rel(rig, 2)[0]))
        if co.outcome(2):
            break
    assert wrist_at_stall is not None and back_min <= float(wrist_at_stall[0]) - 0.01
    assert co.outcome(2) == "cleared autonomously" and co.rig(2).bolt_x() >= BOLT_CLEARED_M
    assert BayState.RESCUE not in states

    # Negative Control: pinned bolt at x = 0 is rejected as OOD and escalated without autonomy.
    n_events = len(rig.sink.events)
    co.start_cycle(2, bolt_pinned=True)
    rig.run_until(2, {BayState.OOD_ESCALATED, BayState.AUTONOMOUS_SKILL, BayState.RESCUE})
    assert co.state(2) == BayState.OOD_ESCALATED
    assert [e.kind for e in rig.sink.events[n_events:]] == ["OOD_REFUSAL"]
