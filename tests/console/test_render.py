"""Offscreen feed renderer and render worker (fleet ops console)."""

from __future__ import annotations

import io
import threading
import time
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest
from PIL import Image

from teleop.atc.hud import HudView, RescueStatus
from teleop.atc.types import BaySnapshot, BayState
from teleop.console.background import BackgroundPlayer, load_cycle
from teleop.console.layout import feed_layout
from teleop.console.render import FeedRenderer, RenderWorker
from teleop.console.snapshot import LatestSlot, SimSnapshot
from teleop.console.telemetry import TelemetryAdapter
from teleop.sim.bays import PANDA_XML, build_four_bay_scene
from teleop.sim.hands import load_hand

mj: Any = _mujoco


def _models_present() -> bool:
    try:
        right = load_hand("leap").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


needs_models = pytest.mark.skipif(
    not _models_present(), reason="Panda/LEAP MJCFs missing; run scripts/fetch_models.py"
)

SKU = "EV-LATCH-01"


def _view(rescue_bay: int | None = None) -> HudView:
    bays = (
        BaySnapshot(1, BayState.NOMINAL, 0.95, 0.0, 0.5, SKU),
        BaySnapshot(2, BayState.NOMINAL, 0.95, 0.0, 0.5, SKU),
    )
    return HudView(bays=bays, queue=(), background_bays=(3, 4), rescue=RescueStatus(rescue_bay))


class FakeServer:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.telemetry: list[dict[str, Any]] = []
        self.frames: list[tuple[int, int, bytes]] = []

    def publish_telemetry(self, msg: dict[str, Any]) -> None:
        with self.lock:
            self.telemetry.append(msg)

    def publish_frame(self, slot: int, layout_id: int, jpeg: bytes) -> None:
        with self.lock:
            self.frames.append((slot, layout_id, jpeg))


@pytest.fixture(scope="module")
def scene() -> tuple[Any, Any]:
    if not _models_present():
        pytest.skip("models missing")
    model = build_four_bay_scene()
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    return model, data


# ------------------------------------------------------------------ snapshot


def test_latest_slot_returns_newest() -> None:
    s = LatestSlot[int]()
    assert s.get() is None and s.version == 0
    s.put(1)
    s.put(2)
    assert s.get() == 2 and s.version == 2


@needs_models
def test_snapshot_copies(scene: tuple[Any, Any]) -> None:
    model, data = scene
    snap = SimSnapshot.capture(data, _view(), 1.25)
    before = snap.qpos.copy()
    data.qpos[0] += 0.5
    try:
        assert np.array_equal(snap.qpos, before)
        assert snap.sim_t == 1.25 and snap.qpos.shape == (model.nq,)
    finally:
        data.qpos[0] -= 0.5


# ------------------------------------------------------------------ renderer


def _decode(jpeg: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpeg))


@needs_models
def test_renders_grid_feeds_as_jpeg(offscreen_render_ok: None, scene: tuple[Any, Any]) -> None:
    model, data = scene
    r = FeedRenderer(model, BackgroundPlayer(load_cycle(), model))
    try:
        frames, conf = r.render(
            SimSnapshot.capture(data, _view(), 0.0), feed_layout([1, 2, 3, 4], None), 1.0
        )
    finally:
        r.close()
    assert [s for s, _ in frames] == [0, 1, 2, 3]
    assert all(b[:2] == b"\xff\xd8" for _, b in frames)  # JPEG SOI
    assert _decode(frames[0][1]).size == (640, 360)
    assert set(conf) == {3, 4}
    # Not a blank frame: the lit cell has real contrast.
    px = np.asarray(_decode(frames[0][1]).convert("L"), dtype=np.float64)
    assert px.std() > 8.0


@needs_models
def test_renders_rescue_feeds(offscreen_render_ok: None, scene: tuple[Any, Any]) -> None:
    model, data = scene
    r = FeedRenderer(model, None)
    try:
        frames, conf = r.render(
            SimSnapshot.capture(data, _view(1), 0.0), feed_layout([1, 2, 3, 4], 1), 0.0
        )
    finally:
        r.close()
    assert [s for s, _ in frames] == [0, 1, 2, 3, 4, 5]
    assert _decode(frames[0][1]).size == (960, 540)
    assert _decode(frames[2][1]).size == (384, 216)
    assert conf == {}


@needs_models
def test_renderer_does_not_touch_physics_model(
    offscreen_render_ok: None, scene: tuple[Any, Any]
) -> None:
    model, data = scene
    active = np.array(model.light_active).copy()
    r = FeedRenderer(model, BackgroundPlayer(load_cycle(), model))
    try:
        r.render(SimSnapshot.capture(data, _view(), 0.0), feed_layout([1, 2, 3, 4], None), 2.0)
    finally:
        r.close()
    assert np.array_equal(np.array(model.light_active), active)
    q3 = float(data.qpos[model.joint("bay3/joint1").qposadr[0]])
    assert q3 == pytest.approx(0.0, abs=1.0)  # physics data untouched by the background player


# ------------------------------------------------------------------ worker
#
# Worker logic (telemetry before frames, strip throttling, error recovery) is tested with a
# fast fake renderer: CI renders in software GL at seconds per frame, so real-GL timing tests
# would be flaky there. Real rendering is covered by the FeedRenderer tests above.


class FakeRenderer:
    """Stands in for ``FeedRenderer``: instant JPEG-ish bytes per feed."""

    def __init__(self, model: Any, background: Any) -> None:
        self.background = background
        self.closed = False

    def render(self, snap: SimSnapshot, feeds: Any, wall_t: float) -> Any:
        conf = {3: 0.9, 4: 0.95} if self.background is not None else {}
        return [(f.slot, b"\xff\xd8" + f.camera.encode()) for f in feeds], conf

    def close(self) -> None:
        self.closed = True


def _snap(rescue_bay: int | None = None, sim_t: float = 0.0) -> SimSnapshot:
    return SimSnapshot(np.zeros(3), np.zeros((0, 3)), np.zeros((0, 4)), _view(rescue_bay), sim_t)


def _worker(
    slot: LatestSlot[SimSnapshot], server: FakeServer, background: Any = None, **kw: Any
) -> RenderWorker:
    return RenderWorker(
        object(),
        slot,
        server,
        TelemetryAdapter((1, 2, 3, 4), (3, 4)),
        (1, 2, 3, 4),
        background,
        renderer_factory=FakeRenderer,
        **kw,
    )  # type: ignore[arg-type]


def _run_worker(worker: RenderWorker, seconds: float) -> None:
    worker.start()
    time.sleep(seconds)
    worker.stop()
    worker.join(timeout=5)
    assert not worker.is_alive()


def test_worker_publishes_frames_and_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("teleop.console.render.BackgroundPlayer", lambda cycle, model: cycle)
    slot: LatestSlot[SimSnapshot] = LatestSlot()
    slot.put(_snap())
    server = FakeServer()
    worker = _worker(slot, server, background=object(), fps=20, telemetry_hz=10)
    _run_worker(worker, 1.0)
    assert len(server.telemetry) >= 5
    assert {s for s, _, _ in server.frames} == {0, 1, 2, 3}
    assert worker.stats.frames >= 10 and worker.stats.errors == 0
    bay3 = next(b for b in server.telemetry[-1]["bays"] if b["id"] == 3)
    assert bay3["replayed"] is True and bay3["confidence"] == 0.9


def test_worker_frames_follow_telemetry_layout() -> None:
    slot: LatestSlot[SimSnapshot] = LatestSlot()
    slot.put(_snap())
    server = FakeServer()
    worker = _worker(slot, server, fps=30)
    worker.start()
    time.sleep(0.3)
    slot.put(_snap(1, 0.1))  # rescue starts
    time.sleep(0.4)
    worker.stop()
    worker.join(timeout=5)
    with server.lock:
        rescue_msgs = [m for m in server.telemetry if m["mode"] == "rescue"]
        assert rescue_msgs, "no rescue telemetry"
        rescue_id = rescue_msgs[0]["layout_id"]
        grid_id = server.telemetry[0]["layout_id"]
        assert rescue_id == grid_id + 1
        rescue_frames = [f for f in server.frames if f[1] == rescue_id]
        assert {s for s, _, _ in rescue_frames} == {0, 1, 2, 3, 4, 5}
        # Slot 0 means "bay1/cctv" in the grid and "bay1/top_ortho" in rescue: never mixed up.
        assert all(j.endswith(b"top_ortho") for s, _, j in rescue_frames if s == 0)
        assert all(
            j.endswith(b"bay1/cctv") for s, lid, j in server.frames if s == 0 and lid == grid_id
        )


def test_rescue_strip_renders_less_often_than_main() -> None:
    slot: LatestSlot[SimSnapshot] = LatestSlot()
    slot.put(_snap(1))
    server = FakeServer()
    _run_worker(_worker(slot, server, fps=30), 1.0)
    with server.lock:
        main = sum(1 for s, _, _ in server.frames if s == 0)
        strip = sum(1 for s, _, _ in server.frames if s == 2)
    assert main >= 10
    assert strip <= main / 2  # strip tiles refresh at a third of the main rate


def test_worker_survives_render_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any, **__: Any) -> Any:
        raise RuntimeError("GL context lost")

    monkeypatch.setattr("teleop.console.render.ERROR_BACKOFF_S", 0.05)
    slot: LatestSlot[SimSnapshot] = LatestSlot()
    slot.put(_snap())
    server = FakeServer()
    worker = RenderWorker(
        object(),
        slot,
        server,
        TelemetryAdapter((1, 2), ()),
        (1, 2),
        None,
        fps=20,
        telemetry_hz=20,
        renderer_factory=boom,
    )  # type: ignore[arg-type]
    _run_worker(worker, 0.6)
    assert worker.stats.errors >= 2
    assert len(server.telemetry) >= 3  # telemetry keeps flowing while rendering fails
    assert server.frames == []


class FlakyAdapter(TelemetryAdapter):
    """Raises on the first ``fails`` builds, like a bug tripped by an odd HudView."""

    def __init__(self, fails: int) -> None:
        super().__init__((1, 2, 3, 4), (3, 4))
        self.fails = fails

    def build(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if self.fails > 0:
            self.fails -= 1
            raise ValueError("unexpected HudView value")
        return super().build(*args, **kwargs)


def test_worker_survives_telemetry_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("teleop.console.render.ERROR_BACKOFF_S", 0.05)
    slot: LatestSlot[SimSnapshot] = LatestSlot()
    slot.put(_snap())
    server = FakeServer()
    worker = RenderWorker(
        object(),
        slot,
        server,
        FlakyAdapter(fails=2),
        (1, 2, 3, 4),
        None,
        fps=20,
        renderer_factory=FakeRenderer,
    )  # type: ignore[arg-type]
    worker.start()
    try:
        time.sleep(0.8)
        assert worker.is_alive(), "render thread died on a telemetry error"
    finally:
        worker.stop()
        worker.join(timeout=5)
    assert worker.stats.errors >= 1  # later failures may land in the back-off, also survived
    assert server.telemetry and server.frames  # recovered and kept streaming
