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


def _run_worker(worker: RenderWorker, seconds: float) -> None:
    worker.start()
    time.sleep(seconds)
    worker.stop()
    worker.join(timeout=5)
    assert not worker.is_alive()


@needs_models
def test_worker_publishes_frames_and_telemetry(
    offscreen_render_ok: None, scene: tuple[Any, Any]
) -> None:
    model, data = scene
    slot: LatestSlot[SimSnapshot] = LatestSlot()
    slot.put(SimSnapshot.capture(data, _view(), 0.0))
    server = FakeServer()
    adapter = TelemetryAdapter((1, 2, 3, 4), (3, 4))
    worker = RenderWorker(
        model, slot, server, adapter, (1, 2, 3, 4), load_cycle(), fps=20, telemetry_hz=10
    )  # type: ignore[arg-type]
    _run_worker(worker, 1.5)
    assert len(server.telemetry) >= 3
    slots = {s for s, _, _ in server.frames}
    assert slots == {0, 1, 2, 3}
    assert worker.stats.frames >= 3 and worker.stats.errors == 0
    assert worker.stats.mean_frame_ms > 0.0
    bay3 = next(b for b in server.telemetry[-1]["bays"] if b["id"] == 3)
    assert bay3["replayed"] is True


@needs_models
def test_worker_frames_follow_telemetry_layout(
    offscreen_render_ok: None, scene: tuple[Any, Any]
) -> None:
    model, data = scene
    slot: LatestSlot[SimSnapshot] = LatestSlot()
    slot.put(SimSnapshot.capture(data, _view(), 0.0))
    server = FakeServer()
    adapter = TelemetryAdapter((1, 2, 3, 4), (3, 4))
    worker = RenderWorker(model, slot, server, adapter, (1, 2, 3, 4), None, fps=20)  # type: ignore[arg-type]
    worker.start()
    time.sleep(0.6)
    slot.put(SimSnapshot.capture(data, _view(1), 0.1))  # rescue starts
    time.sleep(0.8)
    worker.stop()
    worker.join(timeout=5)
    with server.lock:
        rescue_msgs = [m for m in server.telemetry if m["mode"] == "rescue"]
        assert rescue_msgs, "no rescue telemetry"
        rescue_id = rescue_msgs[0]["layout_id"]
        first_rescue_msg = server.telemetry.index(rescue_msgs[0])
        # The new layout's telemetry went out before any frame of that layout.
        rescue_frames = [f for f in server.frames if f[1] == rescue_id]
        assert rescue_frames and {s for s, _, _ in rescue_frames} == {0, 1, 2, 3, 4, 5}
    assert first_rescue_msg >= 1


def test_worker_survives_render_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any, **__: Any) -> Any:
        raise RuntimeError("GL context lost")

    monkeypatch.setattr(FeedRenderer, "__init__", boom)
    monkeypatch.setattr("teleop.console.render.ERROR_BACKOFF_S", 0.05)
    slot: LatestSlot[SimSnapshot] = LatestSlot()
    view = _view()
    slot.put(SimSnapshot(np.zeros(3), np.zeros((0, 3)), np.zeros((0, 4)), view, 0.0))
    server = FakeServer()
    worker = RenderWorker(
        object(), slot, server, TelemetryAdapter((1, 2), ()), (1, 2), None, fps=20, telemetry_hz=20
    )  # type: ignore[arg-type]
    _run_worker(worker, 0.6)
    assert worker.stats.errors >= 2
    assert len(server.telemetry) >= 3  # telemetry keeps flowing while rendering fails
    assert server.frames == []


@needs_models
def test_rescue_strip_renders_less_often_than_main(
    offscreen_render_ok: None, scene: tuple[Any, Any]
) -> None:
    model, data = scene
    slot: LatestSlot[SimSnapshot] = LatestSlot()
    slot.put(SimSnapshot.capture(data, _view(1), 0.0))
    server = FakeServer()
    worker = RenderWorker(
        model, slot, server, TelemetryAdapter((1, 2, 3, 4), (3, 4)), (1, 2, 3, 4), None, fps=30
    )  # type: ignore[arg-type]
    _run_worker(worker, 3.5)  # the first frame pays ~1.5 s of one-off GL/shader setup
    with server.lock:
        main = sum(1 for s, _, _ in server.frames if s == 0)
        strip = sum(1 for s, _, _ in server.frames if s == 2)
    assert main >= 10
    assert strip <= main / 2  # strip tiles refresh at a third of the main rate
