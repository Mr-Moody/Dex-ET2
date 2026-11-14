"""T-190: recovery backend seam, DMP-mode regression guard, backend toggle and observer hook."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from teleop.compliance.protocol import ComplianceEvent
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


@pytest.fixture
def unoq() -> Iterator[Any]:
    from teleop.unoq.client import UnoQClient
    from teleop.unoq.server import UnoQServer

    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="test-node").start()
    client = UnoQClient(
        "127.0.0.1", server.udp_port, server.tcp_port, local_server=server
    ).connect()
    yield client
    client.close()


# --------------------------------------------------------------------------- DMP golden run

# First captured on main (5a40c38) before the recovery seam existed; re-captured for T-179
# (estimated latch pose, touch re-localization, randomized gate) on the show's own config.
# ``recovery_backend: dmp`` must reproduce it exactly: same compliance events in the same order
# on the same bays, after the same number of 60 Hz ticks, ending at the same wrist positions.
# The event order is unchanged from 5a40c38. The float path differs by platform (MuJoCo/NumPy
# builds), so ticks and wrist positions are pinned per platform.
GOLDEN_EVENTS: list[tuple[str, int]] = [
    ("STALL_ALERT", 1),
    ("RESCUE_LOGGED", 1),
    ("VALIDATION_GATE", 1),
    ("SKILL_DEPLOYED", 1),
    ("RESCUE_LOGGED", 2),
    ("OOD_REFUSAL", 2),
]
Wrist = tuple[tuple[float, float, float], tuple[float, float, float]]
GOLDEN_BY_PLATFORM: dict[str, tuple[int, Wrist]] = {
    "win32": (
        1903,
        ((-1.498363013, -1.977468948, 0.136654973), (2.578220427, -1.880974606, 0.135551549)),
    ),
    "linux": (  # CI (ubuntu-latest) and WSL Ubuntu; recaptured after the f5c6154 warehouse layout
        2386,
        ((-2.636515486, -3.378664865, 0.215894326), (3.77897135, -3.281898967, 0.13554331)),
    ),
}


def _run_stage(unoq: Any) -> tuple[Any, list[ComplianceEvent], int]:
    from teleop.atc.stage import StageController, StageDirector
    from teleop.commands.atc import DEFAULT_CONFIG, load_config, resolve_seed

    sink = RecordingSink()
    # The show's own settings, as `teleop atc --headless --script` resolves them (STAGE_SEED,
    # cycle dwell, background bays, realism): T-179's randomized gate only passes reliably there.
    cfg = load_config(DEFAULT_CONFIG, {"script": True})
    controller = StageController.build(
        unoq,
        sink,
        QuietReasoner(),
        cycle_dwell_s=cfg.cycle_dwell_s,
        include_background_bays=cfg.background_bays,
        seed=resolve_seed(cfg),
        realism=cfg.realism,
        gate_pass_min=cfg.gate_pass_min,
    )
    director = StageDirector(controller)
    ticks = 0
    while not director.done and ticks < 12_000:
        controller.tick(director.step())
        ticks += 1
    return director, sink.events, ticks


@needs_models
def test_dmp_mode_script_unchanged(unoq: Any) -> None:
    director, events, ticks = _run_stage(unoq)
    assert director.ok, director.results
    assert [(e.kind, e.bay_id) for e in events] == GOLDEN_EVENTS
    golden = GOLDEN_BY_PLATFORM.get(sys.platform)
    if golden is None:  # e.g. macOS: no reference captured; the events above still guard it
        return
    golden_ticks, golden_wrist = golden
    assert ticks == golden_ticks
    wrist = tuple(
        tuple(round(float(v), 9) for v in director.co.rig(b).wrist_pose().position) for b in (1, 2)
    )
    assert wrist == golden_wrist


# --------------------------------------------------------------------------- backend selection

DMP, HYBRID, ACT = "dmp", "hybrid", "act"


def _backend(name: str) -> Any:
    from teleop.atc.recovery import RecoveryBackend

    return RecoveryBackend(name)


@pytest.mark.parametrize(
    ("backend", "act_ok", "dmp_available", "expected"),
    [
        (DMP, True, True, "dmp"),
        (DMP, False, True, "dmp"),
        (DMP, True, False, None),
        (HYBRID, True, True, "act"),
        (HYBRID, True, False, "act"),
        (HYBRID, False, True, "dmp"),
        (HYBRID, False, False, None),
        (ACT, True, True, "act"),
        (ACT, True, False, "act"),
        (ACT, False, True, None),
        (ACT, False, False, None),
    ],
)
def test_choose_recovery_table(
    backend: str, act_ok: bool, dmp_available: bool, expected: str | None
) -> None:
    from teleop.atc.recovery import choose_recovery

    got = choose_recovery(
        _backend(backend), act_covers=act_ok, act_healthy=True, dmp_available=dmp_available
    )
    assert got == expected


@pytest.mark.parametrize("backend", [HYBRID, ACT])
def test_choose_recovery_act_needs_cover_health_and_no_failure(backend: str) -> None:
    from teleop.atc.recovery import choose_recovery

    fallback = "dmp" if backend == HYBRID else None
    b = _backend(backend)
    assert choose_recovery(b, act_covers=True, act_healthy=False, dmp_available=True) == fallback
    assert choose_recovery(b, act_covers=False, act_healthy=True, dmp_available=True) == fallback
    assert (
        choose_recovery(b, act_covers=True, act_healthy=True, dmp_available=True, act_failed=True)
        == fallback
    )


def test_next_backend_cycles() -> None:
    from teleop.atc.recovery import next_backend

    order = [next_backend(_backend(b)) for b in (DMP, HYBRID, ACT)]
    assert order == [_backend(HYBRID), _backend(ACT), _backend(DMP)]


# --------------------------------------------------------------------------- fakes for ACT


class HoldController:
    """Fake ACT recovery: holds pose; optionally aborts on its first check."""

    name = "act"

    def __init__(self, abort: str | None = None) -> None:
        self.abort = abort
        self.closed = False

    def command(self, rig: Any, s: Any) -> None:
        return None

    def check(self, rig: Any, s: Any) -> str | None:
        return self.abort

    def close(self) -> None:
        self.closed = True


class FakeActFactory:
    name = "act"

    def __init__(self, healthy: bool = True, abort: str | None = None) -> None:
        self._healthy = healthy
        self.abort = abort
        self.unhealthy_reason = "" if healthy else "worker crashed"
        self.created: list[HoldController] = []

    def covers(self, z: Any) -> bool:
        return True

    def healthy(self) -> bool:
        return self._healthy

    def create(self, rig: Any, z: Any) -> HoldController:
        c = HoldController(self.abort)
        self.created.append(c)
        return c

    def close(self) -> None:
        pass


class RecordingObserver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, Any]] = []

    def on_episode_start(self, bay_id: int, source: str, signature_id: Any, signature: Any) -> None:
        self.calls.append(("start", bay_id, source))

    def on_tick(self, bay_id: int, rig: Any, wrist_world: Any, fingers: Any) -> None:
        self.calls.append(("tick", bay_id, None))

    def on_episode_end(self, bay_id: int, success: bool, aborted_reason: str | None) -> None:
        self.calls.append(("end", bay_id, (success, aborted_reason)))

    def for_bay(self, bay_id: int) -> list[tuple[str, Any]]:
        return [(k, v) for k, b, v in self.calls if b == bay_id and k != "tick"]

    def ticks(self, bay_id: int) -> int:
        return sum(1 for k, b, _ in self.calls if b == bay_id and k == "tick")


def _controller(unoq: Any, **kw: Any) -> tuple[Any, RecordingSink]:
    from teleop.atc.coordinator import AtcCoordinator
    from teleop.atc.stage import EventTap, ReasonerTap, StageController

    sink = RecordingSink()
    events = EventTap(sink)
    co = AtcCoordinator.build(unoq, events, QuietReasoner(), **kw)
    return StageController(co, events, ReasonerTap(QuietReasoner()), cycle_dwell_s=0.5), sink


def _tick_until(controller: Any, pred: Any, max_ticks: int = 3000, director: Any = None) -> None:
    for _ in range(max_ticks):
        controller.tick(director.step() if director is not None else None)
        if pred():
            return
    raise AssertionError("condition never reached")


def _deploy_skill(controller: Any, sink: RecordingSink) -> None:
    """Play the stage script until Bay 1's rescue skill is in the Fleet Library."""
    from teleop.atc.stage import StageDirector

    director = StageDirector(controller)
    _tick_until(
        controller,
        lambda: any(e.kind == "SKILL_DEPLOYED" for e in sink.events),
        max_ticks=6000,
        director=director,
    )


# --------------------------------------------------------------------------- coordinator seam


@needs_models
def test_backend_switch_mid_recovery_finishes_on_old(unoq: Any) -> None:
    from teleop.atc.types import BayState

    controller, _ = _controller(unoq, recovery_backend=_backend(HYBRID), act=FakeActFactory())
    co = controller.coordinator
    co.start_cycle(1, flap_jammed=True)
    _tick_until(controller, lambda: co.recovery_name(1) == "act")
    co.set_recovery_backend(_backend(DMP))
    for _ in range(60):
        controller.tick(None)
    assert co.state(1) == BayState.AUTONOMOUS_SKILL
    assert co.recovery_name(1) == "act"
    assert co.recovery_backend == _backend(DMP)


@needs_models
def test_downgrade_when_act_unhealthy(unoq: Any) -> None:
    from teleop.atc.types import BayState

    factory = FakeActFactory(healthy=False)
    controller, _ = _controller(unoq, recovery_backend=_backend(HYBRID), act=factory)
    co = controller.coordinator
    co.start_cycle(1, flap_jammed=True)
    _tick_until(controller, lambda: co.state(1) != BayState.NOMINAL)
    assert co.recovery_backend == _backend(DMP)
    assert "back to dmp" in co.backend_note and "worker crashed" in co.backend_note
    assert co.recovery_name(1) is None


@needs_models
def test_downgrade_when_no_act_loaded(unoq: Any) -> None:
    from teleop.atc.types import BayState

    controller, _ = _controller(unoq, recovery_backend=_backend(ACT))
    co = controller.coordinator
    co.start_cycle(1, flap_jammed=True)
    _tick_until(controller, lambda: co.state(1) != BayState.NOMINAL)
    assert co.recovery_backend == _backend(DMP) and "back to dmp" in co.backend_note


class RaisingController(HoldController):
    def __init__(self, where: str) -> None:
        super().__init__()
        self.where = where

    def command(self, rig: Any, s: Any) -> None:
        if self.where == "command":
            raise RuntimeError("policy bug")
        return None

    def check(self, rig: Any, s: Any) -> str | None:
        if self.where == "check":
            raise RuntimeError("policy bug")
        return None


class RaisingFactory(FakeActFactory):
    def __init__(self, where: str) -> None:
        super().__init__()
        self.where = where

    def covers(self, z: Any) -> bool:
        if self.where == "covers":
            raise RuntimeError("manifest bug")
        return True

    def create(self, rig: Any, z: Any) -> HoldController:
        c = RaisingController(self.where)
        self.created.append(c)
        return c


@needs_models
@pytest.mark.parametrize("where", ["command", "check"])
def test_act_controller_exception_aborts_instead_of_crashing(unoq: Any, where: str) -> None:
    from teleop.atc.types import BayState

    factory = RaisingFactory(where)
    controller, _ = _controller(unoq, recovery_backend=_backend(HYBRID), act=factory)
    co = controller.coordinator
    co.start_cycle(1, flap_jammed=True)
    _tick_until(controller, lambda: co.state(1) in (BayState.QUEUED, BayState.RESCUE))
    assert "policy bug" in co.outcome(1)
    assert factory.created[0].closed


@needs_models
def test_act_factory_exception_downgrades(unoq: Any) -> None:
    from teleop.atc.types import BayState

    controller, _ = _controller(
        unoq, recovery_backend=_backend(HYBRID), act=RaisingFactory("covers")
    )
    co = controller.coordinator
    co.start_cycle(1, flap_jammed=True)
    _tick_until(controller, lambda: co.state(1) != BayState.NOMINAL)
    assert co.recovery_backend == _backend(DMP)
    assert "manifest bug" in co.backend_note


@needs_models
def test_hybrid_act_abort_without_dmp_queues(unoq: Any) -> None:
    from teleop.atc.types import BayState

    factory = FakeActFactory(abort="no progress")
    controller, sink = _controller(unoq, recovery_backend=_backend(HYBRID), act=factory)
    co = controller.coordinator
    co.start_cycle(1, flap_jammed=True)
    _tick_until(controller, lambda: co.state(1) in (BayState.QUEUED, BayState.RESCUE))
    assert factory.created and factory.created[0].closed
    alert = [e for e in sink.events if e.kind == "STALL_ALERT"][-1]
    assert alert.metadata.get("backend") == "act"
    assert "no progress" in co.outcome(1)


@needs_models
def test_hybrid_act_abort_falls_back_to_dmp(unoq: Any) -> None:
    from teleop.atc.stage import StageAction
    from teleop.atc.types import BayState

    controller, sink = _controller(unoq)
    _deploy_skill(controller, sink)
    co = controller.coordinator
    co.set_act(FakeActFactory(abort="no progress"))
    co.set_recovery_backend(_backend(HYBRID))
    controller.apply(StageAction.ARM_BAY2_FLAP)
    _tick_until(controller, lambda: co.recovery_name(2) == "dmp", max_ticks=4000)
    assert co.state(2) == BayState.AUTONOMOUS_SKILL


@needs_models
def test_observer_sees_rescue_and_auto_episodes(unoq: Any) -> None:
    from teleop.atc.stage import StageDirector

    observer = RecordingObserver()
    controller, sink = _controller(unoq, observer=observer)
    director = StageDirector(controller)
    _tick_until(
        controller,
        lambda: any(e.kind == "RESCUE_LOGGED" and e.bay_id == 2 for e in sink.events),
        max_ticks=8000,
        director=director,
    )
    assert observer.for_bay(1) == [("start", "operator"), ("end", (True, None))]
    assert observer.for_bay(2) == [("start", "dmp"), ("end", (True, None))]
    assert observer.ticks(1) > 30 and observer.ticks(2) > 30


@needs_models
def test_m_key_cycles_backend(unoq: Any) -> None:
    controller, _ = _controller(unoq)
    co = controller.coordinator
    controller.on_key(ord("M"))
    controller.tick(None)
    assert co.recovery_backend == _backend(HYBRID)
    assert "hybrid" in controller.actions[-1].text.lower()
    controller.on_key(ord("M"))
    controller.on_key(ord("M"))
    controller.tick(None)
    assert co.recovery_backend == _backend(DMP)


# --------------------------------------------------------------------------- teleop atc wiring

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
def test_dmp_mode_does_not_import_torch(tmp_path: Path) -> None:
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from teleop.commands import atc\n"
        f"rc = atc.main({[*_ATC_ARGS, '--steps', '30', '--env-file', 'missing.env']!r})\n"
        "assert rc == 0, rc\n"
        "bad = [m for m in ('torch', 'lerobot') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_unknown_backend_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    from teleop.commands import atc

    with pytest.raises(SystemExit) as exc:
        atc.main(["--headless", "--steps", "1", "--recovery-backend", "bogus"])
    assert exc.value.code == 2
    assert "invalid choice: 'bogus'" in capsys.readouterr().err


def test_load_config_backend_and_paths(tmp_path: Path) -> None:
    from teleop.commands.atc import load_config

    cfg = load_config(
        None,
        {
            "recovery_backend": "hybrid",
            "act_checkpoint": str(tmp_path),
            "record_episodes": str(tmp_path / "eps"),
        },
    )
    assert cfg.recovery_backend == "hybrid"
    assert cfg.act_checkpoint == tmp_path
    assert cfg.record_episodes == tmp_path / "eps"
    assert load_config(None, {}).recovery_backend == "dmp"
    with pytest.raises(ValueError, match="recovery_backend"):
        load_config(None, {"recovery_backend": "bogus"})


@needs_models
def test_missing_hook_exits_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib

    from teleop.commands import atc

    real = importlib.import_module

    def fake(name: str, package: str | None = None) -> Any:
        if name == "teleop.atc.episodes":
            raise ModuleNotFoundError(name=name)
        return real(name, package)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(importlib, "import_module", fake)
    argv = [*_ATC_ARGS, "--steps", "1", "--env-file", str(tmp_path / "missing.env")]
    rc = atc.main([*argv, "--record-episodes", str(tmp_path / "eps")])
    assert rc == 3
    out = capsys.readouterr().out
    assert "--record-episodes is not implemented yet (teleop.atc.episodes)" in out


@needs_models
@pytest.mark.parametrize("error", [ImportError("No module named 'torch'"), RuntimeError("no CUDA")])
def test_act_load_failure_falls_back_to_dmp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    from teleop.commands import atc

    def failing_loader(*args: Any) -> Any:
        raise error

    real = atc.load_hook
    monkeypatch.setattr(
        atc,
        "load_hook",
        lambda flag: failing_loader if flag == "--act-checkpoint" else real(flag),
    )
    monkeypatch.chdir(tmp_path)
    argv = [*_ATC_ARGS, "--steps", "1", "--env-file", str(tmp_path / "missing.env")]
    with caplog.at_level("WARNING", logger="teleop.atc"):
        rc = atc.main(
            [*argv, "--recovery-backend", "hybrid", "--act-checkpoint", str(tmp_path / "ckpt")]
        )
    assert rc == 0
    assert "ACT checkpoint not loaded" in caplog.text
