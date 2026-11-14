"""T-191: ATC recoveries recorded as LeRobot episodes by the ``EpisodeSink`` observer."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
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

REPO = "local/test"


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


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
        self.closed = False

    def render(self, data: Any, bay_id: int) -> dict[str, Any]:
        from teleop.policy.schema import CAMERAS, IMAGE_HW

        return {cam: np.zeros((*IMAGE_HW, 3), dtype=np.uint8) for cam in CAMERAS}

    def close(self) -> None:
        self.closed = True


class FakeRecorder:
    """``EpisodeRecorder`` stand-in that keeps episodes in memory (optionally slow to save)."""

    def __init__(self, save_delay_s: float = 0.0) -> None:
        self.save_delay_s = save_delay_s
        self.episodes: list[tuple[str, int, dict[str, Any]]] = []
        self.closed = False
        self._task: str | None = None
        self._frames = 0

    def start_episode(self, task: str) -> None:
        self._task, self._frames = task, 0

    def add_frame(self, state: Any, action: Any, images: Mapping[str, Any]) -> None:
        self._frames += 1

    def end_episode(self, meta: Mapping[str, Any]) -> int:
        time.sleep(self.save_delay_s)
        assert self._task is not None
        self.episodes.append((self._task, self._frames, dict(meta)))
        return len(self.episodes) - 1

    def discard_episode(self) -> None:
        self._task = None

    def close(self) -> None:
        self.closed = True


class Tee:
    """Forwards observer calls to the sink and counts each bay's episode ticks."""

    def __init__(self, sink: Any) -> None:
        self.sink = sink
        self.ticks: dict[int, int] = {}

    def on_episode_start(self, bay_id: int, source: Any, signature_id: Any, signature: Any) -> None:
        self.ticks[bay_id] = 0
        self.sink.on_episode_start(bay_id, source, signature_id, signature)

    def on_tick(self, bay_id: int, rig: Any, wrist_world: Any, fingers: Any) -> None:
        self.ticks[bay_id] = self.ticks.get(bay_id, 0) + 1
        self.sink.on_tick(bay_id, rig, wrist_world, fingers)

    def on_episode_end(self, bay_id: int, success: bool, aborted_reason: str | None) -> None:
        self.sink.on_episode_end(bay_id, success, aborted_reason)


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


def _controller(unoq: Any) -> tuple[Any, RecordingSink]:
    from teleop.atc.coordinator import AtcCoordinator
    from teleop.atc.stage import EventTap, ReasonerTap, StageController

    sink = RecordingSink()
    events = EventTap(sink)
    co = AtcCoordinator.build(unoq, events, QuietReasoner())
    return StageController(co, events, ReasonerTap(QuietReasoner()), cycle_dwell_s=0.5), sink


def _sink(controller: Any, root: Path, recorder: Any = None, source: Any = "glove") -> Any:
    from teleop.atc.episodes import EpisodeSink, fault_lookup

    co = controller.coordinator
    return EpisodeSink(
        root,
        co.model,
        co.data,
        fault_lookup(controller),
        source,
        renderer=ZeroRenderer(),
        recorder=recorder,
    )


def _tick_until(
    controller: Any, pred: Callable[[], bool], op: Any = None, max_ticks: int = 6000
) -> None:
    for _ in range(max_ticks):
        controller.tick(op.latest() if op is not None else None)
        if pred():
            return
    raise AssertionError("condition never reached")


def _rescue_bay1(controller: Any, *, release_after_s: float | None = None) -> None:
    """Jam the Bay 1 flap, then rescue it with the scripted glove (or let go early)."""
    from teleop.atc.stage import KEY_SPACE, ScriptedGlove
    from teleop.atc.types import BOLT_CLEARED_M

    glove = ScriptedGlove(controller)
    controller.on_key(ord("F"))
    _tick_until(controller, lambda: controller.rescue_bay() == 1)
    controller.on_key(ord("1"))
    controller.on_key(KEY_SPACE)
    rig = controller.coordinator.rig(1)
    if release_after_s is None:
        _tick_until(controller, lambda: rig.bolt_x() >= BOLT_CLEARED_M + 0.002, glove)
        for _ in range(15):  # hold a moment past clear, as an operator would
            controller.tick(glove.latest())
    else:
        for _ in range(int(release_after_s * 60)):
            controller.tick(glove.latest())
    controller.on_key(KEY_SPACE)
    for _ in range(3):
        controller.tick(glove.latest())


def _sidecar(root: Path) -> list[dict[str, Any]]:
    path = root / "meta" / "teleop_episodes.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------- with lerobot


@needs_models
def test_records_operator_rescue(unoq: Any, tmp_path: Path) -> None:
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from teleop.policy.schema import ACTION_NAMES, STATE_NAMES

    controller, _ = _controller(unoq)
    root = tmp_path / "ds"
    sink = _sink(controller, root)
    tee = Tee(sink)
    controller.coordinator.set_observer(tee)
    _rescue_bay1(controller)
    sink.close()

    assert sink.written == 1
    ds = LeRobotDataset(sink.repo_id, root=sink.root)
    assert ds.num_episodes == 1
    assert ds.meta.features["observation.state"]["names"] == list(STATE_NAMES)
    assert ds.meta.features["action"]["names"] == list(ACTION_NAMES)
    assert abs(len(ds) - tee.ticks[1] / 2) <= 2  # 30 fps out of 60 Hz ticks
    (line,) = _sidecar(sink.root)
    assert line["source"] == "glove"
    assert line["fault"] == "flap"
    assert line["bay"] == 1
    assert line["success"] is True
    assert ds[0]["task"] == "recover latch: flap"


@needs_models
def test_close_mid_episode_keeps_finished(unoq: Any, tmp_path: Path) -> None:
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from teleop.atc.types import BayState

    controller, events = _controller(unoq)
    sink = _sink(controller, tmp_path / "ds")
    tee = Tee(sink)
    controller.coordinator.set_observer(tee)
    _rescue_bay1(controller)
    # Jam Bay 1 again once its skill is deployed: the skill starts an autonomous recovery.
    # Close while that episode is still open.
    _tick_until(
        controller,
        lambda: (
            any(e.kind == "SKILL_DEPLOYED" for e in events.events)
            and controller.coordinator.state(1) == BayState.NOMINAL
        ),
    )
    tee.ticks.clear()
    controller.on_key(ord("R"))  # the stage re-arms a spent rig only after a reset
    controller.tick(None)
    controller.on_key(ord("F"))
    _tick_until(controller, lambda: tee.ticks.get(1, 0) >= 30)
    sink.close()

    ds = LeRobotDataset(sink.repo_id, root=sink.root)
    assert ds.num_episodes == 1
    assert len(_sidecar(sink.root)) == 1


# --------------------------------------------------------------------------- without lerobot


@needs_models
def test_discarded_rescue_is_failure(unoq: Any, tmp_path: Path) -> None:
    controller, _ = _controller(unoq)
    rec = FakeRecorder()
    sink = _sink(controller, tmp_path / "ds", recorder=rec)
    controller.coordinator.set_observer(sink)
    _rescue_bay1(controller, release_after_s=1.0)
    sink.close()

    assert len(rec.episodes) == 1
    task, frames, meta = rec.episodes[0]
    assert task == "recover latch: flap"
    assert frames > 0
    assert meta["success"] is False
    assert meta["bay"] == 1


@needs_models
def test_two_bays_two_episodes(unoq: Any, tmp_path: Path) -> None:
    from teleop.atc.stage import StageDirector

    controller, events = _controller(unoq)
    rec = FakeRecorder()
    sink = _sink(controller, tmp_path / "ds", recorder=rec)
    controller.coordinator.set_observer(sink)
    director = StageDirector(controller)
    for _ in range(8000):
        controller.tick(director.step())
        if any(e.kind == "RESCUE_LOGGED" and e.bay_id == 2 for e in events.events):
            break
    sink.close()

    metas = [m for _, _, m in rec.episodes]
    assert sorted((m["bay"], m["source"]) for m in metas) == [(1, "glove"), (2, "dmp")]
    assert all(m["success"] for m in metas)
    assert all(m["fault"] == "flap" for m in metas)
    bay2 = next(m for m in metas if m["bay"] == 2)
    assert bay2["signature_id"] is not None and len(bay2["signature"]) == 4
    assert set(bay2["latch_pose"]) == {"x", "y", "z", "yaw"}
    assert bay2["max_tau_hand_inf"] > 0.0
    assert rec.closed


@needs_models
def test_on_tick_never_blocks(unoq: Any, tmp_path: Path) -> None:
    controller, _ = _controller(unoq)
    co = controller.coordinator
    rec = FakeRecorder(save_delay_s=0.5)
    sink = _sink(controller, tmp_path / "ds", recorder=rec)
    rig = co.rig(1)
    wrist = rig.wrist_pose()
    durations: list[float] = []
    for i in range(300):
        if i % 30 == 0:
            sink.on_episode_start(1, "operator", None, None)
        t0 = time.perf_counter()
        sink.on_tick(1, rig, wrist, {})
        durations.append(time.perf_counter() - t0)
        if i % 30 == 29:
            sink.on_episode_end(1, True, None)
    assert float(np.percentile(durations, 99)) < 0.005
    sink.close()
    assert len(rec.episodes) == 10


@needs_models
def test_callable_source_is_read_per_episode(unoq: Any, tmp_path: Path) -> None:
    controller, _ = _controller(unoq)
    rig = controller.coordinator.rig(1)
    labels = iter(["scripted", "glove"])
    rec = FakeRecorder()
    sink = _sink(controller, tmp_path / "ds", recorder=rec, source=lambda: next(labels))
    for _ in range(2):
        sink.on_episode_start(1, "operator", None, None)
        sink.on_tick(1, rig, rig.wrist_pose(), {})
        sink.on_episode_end(1, True, None)
    sink.on_episode_start(1, "dmp", None, None)  # autonomous: the label is not consulted
    sink.on_tick(1, rig, rig.wrist_pose(), {})
    sink.on_episode_end(1, False, "skill timeout")
    sink.close()
    assert [m["source"] for _, _, m in rec.episodes] == ["scripted", "glove", "dmp"]
    assert rec.episodes[2][2]["aborted_reason"] == "skill timeout"


@needs_models
def test_episode_without_frames_is_dropped(unoq: Any, tmp_path: Path) -> None:
    controller, _ = _controller(unoq)
    rec = FakeRecorder()
    sink = _sink(controller, tmp_path / "ds", recorder=rec)
    sink.on_episode_start(1, "operator", None, None)
    sink.on_episode_end(1, False, "bolt not cleared")
    sink.close()
    assert rec.episodes == []
    assert sink.written == 0


def test_session_root_is_fresh(tmp_path: Path) -> None:
    from teleop.atc.episodes import session_root

    assert session_root(tmp_path / "new") == tmp_path / "new"
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "x").write_text("x", encoding="utf-8")
    fresh = session_root(tmp_path / "used")
    assert fresh.parent == tmp_path / "used"
    assert not fresh.exists()


@needs_models
def test_open_episode_sink_reads_active_fault(unoq: Any, tmp_path: Path) -> None:
    from teleop.atc.episodes import fault_lookup

    controller, _ = _controller(unoq)
    fault_of = fault_lookup(controller)
    assert fault_of(1) is None
    controller.on_key(ord("F"))
    _tick_until(controller, lambda: controller.rescue_bay() == 1)
    assert fault_of(1) == "flap"


def test_rescue_label_follows_selecting_glove() -> None:
    from teleop.atc.glove_select import SelectingGlove
    from teleop.commands.atc import rescue_label

    box: list[Any] = []
    label = rescue_label(box)
    assert label() == "glove"  # no operator built yet
    selecting = SelectingGlove(Any, Any, Any, lambda: False)  # type: ignore[arg-type]
    box.append(selecting)
    assert label() == "glove"  # not engaged
    selecting.rescue_input = "scripted"  # gloves were unhealthy at clutch engage
    assert label() == "scripted"
    selecting.rescue_input = "glove"
    assert label() == "glove"


@needs_models
def test_close_says_it_is_saving(
    unoq: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Quitting waits for encoding; the operator is told so instead of seeing a silent hang."""
    controller, _ = _controller(unoq)
    rig = controller.coordinator.rig(1)
    rec = FakeRecorder(save_delay_s=0.2)
    sink = _sink(controller, tmp_path / "ds", recorder=rec)
    sink.on_episode_start(1, "operator", None, None)
    sink.on_tick(1, rig, rig.wrist_pose(), {})
    sink.on_episode_end(1, True, None)
    sink.close()
    out = capsys.readouterr().out
    assert "saving 1 recorded episode" in out
    assert len(rec.episodes) == 1


@needs_models
def test_close_is_quiet_with_nothing_to_save(
    unoq: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    controller, _ = _controller(unoq)
    sink = _sink(controller, tmp_path / "ds", recorder=FakeRecorder())
    sink.close()
    assert "saving" not in capsys.readouterr().out
