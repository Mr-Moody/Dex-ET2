"""T-195: live ACT recovery in the coordinator, driven by the ReplayPolicy stub (no torch)."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from teleop.compliance.protocol import ComplianceEvent
from teleop.policy.schema import CAMERAS, IMAGE_HW
from teleop.reasoning.protocol import ReasoningContext, ReasoningVerdict
from teleop.sim.bays import PANDA_XML
from teleop.sim.hands import load_hand


def _models_present() -> bool:
    try:
        right = load_hand("inspire").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


needs_models = pytest.mark.skipif(
    not _models_present(), reason="Panda/Inspire MJCFs missing; run scripts/fetch_models.py"
)


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[ComplianceEvent] = []

    def emit(self, event: ComplianceEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass


class QuietReasoner:
    def submit(self, ctx: ReasoningContext) -> None:
        pass

    def latest_verdict(self, bay_id: int) -> ReasoningVerdict | None:
        return None

    def close(self) -> None:
        pass


class ZeroRenderer:
    """Stands in for ``BayCameraRenderer`` where no GL context is needed."""

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.closed = False

    def render(self, data: Any, bay_id: int) -> dict[str, NDArray[np.uint8]]:
        self.calls.append(bay_id)
        return {c: np.zeros((*IMAGE_HW, 3), dtype=np.uint8) for c in CAMERAS}

    def close(self) -> None:
        self.closed = True


@pytest.fixture(scope="module")
def unoq() -> Iterator[Any]:
    from teleop.unoq.client import UnoQClient
    from teleop.unoq.server import UnoQServer

    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="test-node").start()
    client = UnoQClient(
        "127.0.0.1", server.udp_port, server.tcp_port, local_server=server
    ).connect()
    yield client
    client.close()


def _controller(unoq: Any, **kw: Any) -> tuple[Any, RecordingSink]:
    from teleop.atc.coordinator import AtcCoordinator
    from teleop.atc.stage import EventTap, ReasonerTap, StageController

    sink = RecordingSink()
    events = EventTap(sink)
    co = AtcCoordinator.build(unoq, events, QuietReasoner(), **kw)
    return StageController(co, events, ReasonerTap(QuietReasoner()), cycle_dwell_s=0.5), sink


def _tick_until(controller: Any, pred: Any, op: Any = None, max_ticks: int = 3000) -> None:
    for _ in range(max_ticks):
        controller.tick(op.latest() if op is not None else None)
        if pred():
            return
    raise AssertionError("condition never reached")


class ActionRecorder:
    """Observer: the latch-frame actions of one bay's operator rescue, one row per 60 Hz tick."""

    def __init__(self) -> None:
        self.bay: int | None = None
        self.rows: list[NDArray[np.float64]] = []

    def on_episode_start(self, bay_id: int, source: str, signature_id: Any, signature: Any) -> None:
        pass

    def on_tick(
        self, bay_id: int, rig: Any, wrist_world: Any, fingers: Mapping[str, float]
    ) -> None:
        from teleop.policy.schema import action_from_command

        if bay_id == self.bay:
            self.rows.append(action_from_command(rig.world_from_latch(), wrist_world, fingers))

    def on_episode_end(self, bay_id: int, success: bool, aborted_reason: str | None) -> None:
        pass


@dataclass(frozen=True)
class Demo:
    actions: NDArray[np.float32]  # 30 fps latch-frame rows of one scripted flap rescue
    signatures: dict[int, tuple[float, float, float, float]]


def _jam_both(co: Any) -> None:
    co.start_cycle(1, flap_jammed=True)
    co.start_cycle(2, flap_jammed=True)


@pytest.fixture(scope="module")
def demo(unoq: Any) -> Demo:
    """Jam both flaps; rescue the first queued bay with the scripted glove and record it.

    Both stall signatures are kept as the release's ``flap`` prototypes. Actions are latch-frame,
    so one bay's rescue drives either bay.
    The clutch is never released, so no skill gate runs and no DMP is deployed.
    """
    if not _models_present():
        pytest.skip("Panda/Inspire MJCFs missing; run scripts/fetch_models.py")
    from teleop.atc.stage import KEY_SPACE, ScriptedGlove
    from teleop.atc.types import BOLT_CLEARED_M

    recorder = ActionRecorder()
    controller, _ = _controller(unoq, observer=recorder)
    co = controller.coordinator
    try:
        _jam_both(co)
        _tick_until(
            controller,
            lambda: (
                controller.rescue_bay() is not None
                and all(co.last_signature(b) is not None for b in (1, 2))
            ),
        )
        bay = controller.rescue_bay()
        recorder.bay = bay
        glove = ScriptedGlove(controller)
        controller.on_key(ord(str(bay)))
        controller.on_key(KEY_SPACE)
        rig = co.rig(bay)
        _tick_until(controller, lambda: rig.bolt_x() >= BOLT_CLEARED_M + 0.002, glove)
        for _ in range(15):
            controller.tick(glove.latest())
        sigs: dict[int, tuple[float, float, float, float]] = {}
        for b in (1, 2):
            z = co.last_signature(b)
            assert z is not None
            a, bb, c, d = (float(v) for v in z.as_vector())
            sigs[b] = (a, bb, c, d)
    finally:
        controller.close()
    rows = np.asarray(recorder.rows[::2], dtype=np.float32)  # 60 Hz ticks -> 30 fps rows
    assert len(rows) > 60
    return Demo(rows, sigs)


def _manifest(demo: Demo) -> Any:
    from teleop.policy.manifest import FaultResult, GateManifest
    from teleop.policy.schema import SCHEMA_VERSION

    return GateManifest(
        schema_version=SCHEMA_VERSION,
        checkpoint_sha256="replay",
        bar=(9, 10),
        results={"flap": FaultResult("flap", 10, 10, 3.0, 4.0, None)},
        prototypes={"flap": list(demo.signatures.values())},
        inference_p95_ms=None,
    )


def _factory(demo: Demo, cls: Any = None, **kw: Any) -> Any:
    from teleop.atc.act_recovery import ActRecoveryFactory
    from teleop.policy.act_runtime import InferenceWorker, ReplayPolicy

    worker = InferenceWorker(ReplayPolicy(demo.actions))
    return (cls or ActRecoveryFactory)(_manifest(demo), worker, ZeroRenderer(), None, **kw)


def _hybrid(unoq: Any, factory: Any) -> tuple[Any, RecordingSink]:
    from teleop.atc.recovery import RecoveryBackend

    return _controller(unoq, recovery_backend=RecoveryBackend.HYBRID, act=factory)


# --------------------------------------------------------------------------- end to end


@needs_models
def test_act_recovery_clears_flap_with_replay_stub(unoq: Any, demo: Demo) -> None:
    from teleop.atc.types import BayState

    factory = _factory(demo)
    controller, sink = _hybrid(unoq, factory)
    co = controller.coordinator
    seen: set[str | None] = set()
    try:
        co.start_cycle(1, flap_jammed=True)
        _tick_until(controller, lambda: co.state(1) == BayState.AUTONOMOUS_SKILL)
        seen.add(co.recovery_name(1))
        _tick_until(
            controller,
            lambda: seen.add(co.recovery_name(1)) or co.state(1) != BayState.AUTONOMOUS_SKILL,
            max_ticks=900,
        )
        assert co.outcome(1) == "cleared autonomously", co.outcome(1)
        assert seen == {"act", None}
        logged = [e for e in sink.events if e.kind == "RESCUE_LOGGED"]
        assert logged and logged[-1].metadata.get("backend") == "act"
        rec = factory.created[0]
        assert factory.worker.calls > 0 and factory.renderer.calls
        assert rec.phase == "policy"
        assert co.recovery_backend.value == "hybrid"  # no downgrade
    finally:
        controller.close()
        factory.close()


@needs_models
def test_idle_replay_does_not_clear(unoq: Any, demo: Demo) -> None:
    """Negative control: the same run with a replay that only holds the hover never clears."""
    from teleop.atc.types import BayState

    idle = Demo(np.repeat(demo.actions[:1], len(demo.actions), axis=0), demo.signatures)
    factory = _factory(idle, timeout_s=3.0)
    controller, _ = _hybrid(unoq, factory)
    co = controller.coordinator
    try:
        co.start_cycle(1, flap_jammed=True)
        _tick_until(controller, lambda: co.state(1) == BayState.AUTONOMOUS_SKILL)
        _tick_until(controller, lambda: co.state(1) != BayState.AUTONOMOUS_SKILL, max_ticks=400)
        assert co.outcome(1) == "aborted: act timeout", co.outcome(1)
        assert co.state(1) in (BayState.QUEUED, BayState.RESCUE)
    finally:
        controller.close()
        factory.close()


@needs_models
def test_one_bay_abort_does_not_stop_other(unoq: Any, demo: Demo) -> None:
    """Review Focus 4: one batched worker serves two bays; Bay 1's abort leaves Bay 2 alone."""
    from teleop.atc.act_recovery import ActRecovery, ActRecoveryFactory
    from teleop.atc.types import BayState

    class Bay1Fragile(ActRecoveryFactory):
        def create(self, rig: Any, z: Any) -> ActRecovery:
            rec = super().create(rig, z)
            if rig.bay_id == 1:
                rec.tau_hard_max = 0.0  # any hand torque is over the limit
            return rec

    factory = _factory(demo, Bay1Fragile)
    controller, sink = _hybrid(unoq, factory)
    co = controller.coordinator
    try:
        _jam_both(co)
        _tick_until(
            controller,
            lambda: co.recovery_name(1) == "act" and co.recovery_name(2) == "act",
        )
        _tick_until(controller, lambda: co.state(1) != BayState.AUTONOMOUS_SKILL)
        assert "hand torque limit" in co.outcome(1)
        assert co.state(1) in (BayState.QUEUED, BayState.RESCUE)
        alert = [e for e in sink.events if e.kind == "STALL_ALERT" and e.bay_id == 1][-1]
        assert alert.metadata.get("backend") == "act"
        assert co.recovery_name(2) == "act"
        _tick_until(controller, lambda: co.state(2) != BayState.AUTONOMOUS_SKILL, max_ticks=900)
        assert co.outcome(2) == "cleared autonomously", co.outcome(2)
        assert factory.worker.healthy()
    finally:
        controller.close()
        factory.close()


@needs_models
def test_two_bays_share_one_batched_call(unoq: Any, demo: Demo) -> None:
    from teleop.atc.types import BayState

    factory = _factory(demo, settle_s=0.0)
    controller, _ = _hybrid(unoq, factory)
    co = controller.coordinator
    worker = factory.worker
    try:
        _jam_both(co)
        _tick_until(
            controller,
            lambda: (
                co.state(1) == BayState.AUTONOMOUS_SKILL
                and co.state(2) == BayState.AUTONOMOUS_SKILL
            ),
        )
        _tick_until(controller, lambda: 2 in worker.batch_sizes, max_ticks=600)
    finally:
        controller.close()
        factory.close()


# --------------------------------------------------------------------------- abort reasons


class FakeRig:
    bay_id = 1

    def __init__(self) -> None:
        from teleop.types import Pose

        self.pose = Pose(orientation=(0.0, 0.0, 0.0, 1.0), position=(0.5, 0.0, 0.3))

    def wrist_pose(self) -> Any:
        return self.pose

    def world_from_latch(self) -> Any:
        from teleop.types import Pose

        return Pose(orientation=(0.0, 0.0, 0.0, 1.0), position=(0.6, 0.0, 0.3))


class StubWorker:
    timeout_ms = 250.0

    def __init__(self) -> None:
        self.ok = True
        self.submits: list[int] = []
        self.chunk: tuple[int, NDArray[np.float32]] | None = None

    def forget(self, bay_id: int) -> None:
        pass

    def submit(self, bay_id: int, state: Any, images: Any, tick: int) -> None:
        self.submits.append(tick)

    def latest(self, bay_id: int) -> tuple[int, NDArray[np.float32]] | None:
        return self.chunk

    def healthy(self) -> bool:
        return self.ok


def _signals(tau: float = 0.0, confidence: float = 1.0) -> Any:
    from teleop.atc.recovery import BaySignals

    return BaySignals(bay_id=1, t_s=0.0, tau_hand_inf=tau, confidence=confidence, fingers={})


def _act(worker: StubWorker, **kw: Any) -> Any:
    from teleop.atc.act_recovery import ActRecovery

    base: dict[str, Any] = {
        "query_every": 5,
        "retract_m": 0.03,
        "retract_s": 0.5,
        "settle_s": 0.7,
        "timeout_s": 8.0,
        "tau_hard_max": 12.0,
        "stall_confidence": 0.3,
        "stall_ticks": 18,
    }
    base.update(kw)
    return ActRecovery(FakeRig(), worker, ZeroRenderer(), None, **base)  # pyright: ignore


@pytest.fixture
def no_observe(monkeypatch: pytest.MonkeyPatch) -> None:
    import teleop.atc.act_recovery as mod

    monkeypatch.setattr(mod, "observe", lambda rig: np.zeros(29))


def _run_to_policy(rec: Any) -> None:
    rig, s = FakeRig(), _signals()
    while rec.phase != "policy":
        rec.command(rig, s)


def test_act_recovery_retracts_3cm_then_settles(no_observe: None) -> None:
    worker = StubWorker()
    rec = _act(worker)
    rig, s = FakeRig(), _signals()
    cmds = [rec.command(rig, s)]
    while rec.phase == "retract":
        cmds.append(rec.command(rig, s))
    assert len(cmds) in (30, 31)  # 0.5 s at 60 Hz (float accumulation may add one tick)
    end = np.asarray(cmds[-1].wrist.position)
    assert end == pytest.approx([0.47, 0.0, 0.3], abs=1e-9)  # 3 cm along latch -X
    for _ in range(42):  # 0.7 s settle: hold the hover, no inference yet
        cmd = rec.command(rig, s)
        assert np.asarray(cmd.wrist.position) == pytest.approx(end)
    assert rec.phase == "policy" and worker.submits == []


def test_act_recovery_plays_chunk_rows_at_30fps(no_observe: None) -> None:
    from teleop.policy.schema import ACTION_NAMES

    worker = StubWorker()
    rec = _act(worker)
    _run_to_policy(rec)
    rig, s = FakeRig(), _signals()
    assert rec.command(rig, s) is None  # submitted, no chunk yet: hold
    assert worker.submits == [0]
    chunk = np.zeros((4, len(ACTION_NAMES)), dtype=np.float32)
    chunk[:, 0] = [0.01, 0.02, 0.03, 0.04]
    worker.chunk = (0, chunk)
    xs = [float(rec.command(rig, s).wrist.position[0]) for _ in range(8)]
    # policy ticks 1..8 after the obs at tick 0: rows 0,1,1,2,2,3,3,3 (clamped)
    assert xs == pytest.approx([0.61, 0.62, 0.62, 0.63, 0.63, 0.64, 0.64, 0.64], abs=1e-9)
    assert worker.submits == [0, 5]  # every query_every ticks


def test_check_reasons_in_priority(no_observe: None, monkeypatch: pytest.MonkeyPatch) -> None:
    import teleop.atc.act_recovery as mod

    worker = StubWorker()
    rec = _act(worker)
    rig = FakeRig()
    assert rec.check(rig, _signals()) is None
    assert rec.check(rig, _signals(tau=12.0)) == "hand torque limit"
    worker.ok = False
    assert rec.check(rig, _signals(tau=12.0)) == "inference unavailable"
    worker.ok = True

    _run_to_policy(rec)
    now = [0]
    monkeypatch.setattr(mod.time, "monotonic_ns", lambda: now[0])
    rec.command(rig, _signals())  # first submit at t=0
    now[0] = int(0.2e9)
    assert rec.check(rig, _signals()) is None
    now[0] = int(0.26e9)
    assert rec.check(rig, _signals(tau=12.0)) == "inference timeout"


def test_chunk_arrival_clears_timeout(no_observe: None, monkeypatch: pytest.MonkeyPatch) -> None:
    import teleop.atc.act_recovery as mod
    from teleop.policy.schema import ACTION_NAMES

    worker = StubWorker()
    rec = _act(worker)
    _run_to_policy(rec)
    now = [0]
    monkeypatch.setattr(mod.time, "monotonic_ns", lambda: now[0])
    rig = FakeRig()
    rec.command(rig, _signals())
    worker.chunk = (0, np.zeros((3, len(ACTION_NAMES)), dtype=np.float32))
    now[0] = int(0.1e9)
    assert rec.check(rig, _signals()) is None
    now[0] = int(5e9)
    assert rec.check(rig, _signals()) is None


def test_no_progress_after_18_low_confidence_policy_ticks(no_observe: None) -> None:
    from teleop.policy.schema import ACTION_NAMES

    worker = StubWorker()
    worker.chunk = (0, np.zeros((3, len(ACTION_NAMES)), dtype=np.float32))
    rec = _act(worker)
    rig = FakeRig()
    for _ in range(30):  # low confidence during retract/settle does not count
        rec.command(rig, _signals())
        assert rec.check(rig, _signals(confidence=0.1)) is None
    _run_to_policy(rec)
    reasons = []
    for _ in range(18):
        rec.command(rig, _signals())
        reasons.append(rec.check(rig, _signals(confidence=0.1)))
    assert reasons[:17] == [None] * 17 and reasons[17] == "no progress"


def test_act_timeout(no_observe: None) -> None:
    from teleop.policy.schema import ACTION_NAMES

    worker = StubWorker()
    worker.chunk = (0, np.zeros((3, len(ACTION_NAMES)), dtype=np.float32))
    rec = _act(worker, timeout_s=1.5)
    rig = FakeRig()
    reason = None
    for _ in range(200):
        rec.command(rig, _signals())
        reason = rec.check(rig, _signals())
        if reason:
            break
    assert reason == "act timeout"
    assert rec.t == pytest.approx(1.5, abs=0.02)


# --------------------------------------------------------------------------- loading


def _write_gate(root: Path, demo_sig: tuple[float, float, float, float], version: int) -> None:
    import json

    root.mkdir(parents=True, exist_ok=True)
    gate = {
        "schema_version": version,
        "checkpoint_sha256": "x",
        "bar": [9, 10],
        "results": {
            "flap": {
                "fault": "flap",
                "rollouts": 10,
                "passed_rollouts": 10,
                "mean_time_to_clear_s": 3.0,
                "max_tau_hand_inf": 4.0,
                "dmp_passed_rollouts": None,
                "note": "",
            }
        },
        "prototypes": {"flap": [list(demo_sig)]},
        "inference_p95_ms": None,
    }
    (root / "gate.json").write_text(json.dumps(gate), encoding="utf-8")


def test_load_act_factory_rejects_bad_gate(tmp_path: Path) -> None:
    from teleop.atc.act_recovery import load_act_factory
    from teleop.policy.manifest import ManifestError
    from teleop.policy.schema import SCHEMA_VERSION

    with pytest.raises(ManifestError, match="gate.json"):
        load_act_factory(tmp_path, "cpu", None, None)
    _write_gate(tmp_path, (0.03, 1.0, 2.0, 0.3), SCHEMA_VERSION + 1)
    with pytest.raises(ManifestError, match="schema_version"):
        load_act_factory(tmp_path, "cpu", None, None)


def test_load_act_factory_checks_policy_schema_after_gate(tmp_path: Path) -> None:
    import json

    from teleop.atc.act_recovery import load_act_factory
    from teleop.policy.manifest import ManifestError
    from teleop.policy.schema import SCHEMA_VERSION

    _write_gate(tmp_path, (0.03, 1.0, 2.0, 0.3), SCHEMA_VERSION)
    model = tmp_path / "checkpoints" / "last" / "pretrained_model"
    model.mkdir(parents=True)
    cfg = {
        "input_features": {"observation.state": {"type": "STATE", "shape": [20]}},
        "output_features": {"action": {"type": "ACTION", "shape": [12]}},
    }
    (model / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ManifestError, match="observation.state"):
        load_act_factory(tmp_path, "cpu", None, None)


def test_factory_covers_only_passed_prototypes() -> None:
    from teleop.atc.types import StallSignature

    sig = (0.035, 1.2, 2.4, 0.3)
    factory = _factory(Demo(np.zeros((2, 12), dtype=np.float32), {1: sig}))
    try:
        assert factory.covers(StallSignature(*sig))
        assert not factory.covers(StallSignature(0.035, 9.0, 11.0, 4.0))
        assert factory.healthy() and factory.name == "act"
        factory.worker.close()
        assert not factory.healthy() and factory.unhealthy_reason
    finally:
        factory.close()
    assert factory.renderer.closed


# --------------------------------------------------------------------------- teleop atc


_ATC_ARGS = [
    "--headless",
    "--no-audio",
    "--source",
    "synthetic",
    "--unoq-host",
    "127.0.0.1",
    "--unoq-tcp-port",
    "1",  # nothing listens: falls back to the local loopback node
]


@needs_models
def test_atc_downgrades_on_bad_checkpoint(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from teleop.commands import atc

    monkeypatch.chdir(tmp_path)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    caplog.set_level(logging.INFO, logger="teleop.commands.atc")
    rc = atc.main(
        [
            *_ATC_ARGS,
            "--steps",
            "30",
            "--env-file",
            str(tmp_path / "missing.env"),
            "--recovery-backend",
            "hybrid",
            "--act-checkpoint",
            str(ckpt),
        ]
    )
    assert rc == 0
    warn = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("ACT checkpoint not loaded" in r.getMessage() for r in warn)
    assert any("recovery backend dmp" in r.getMessage() for r in caplog.records)
