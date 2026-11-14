"""Offscreen camera feeds for the fleet ops console, rendered off the physics thread.

``FeedRenderer`` owns a private copy of the model and its own ``MjData``: it poses them from
the latest ``SimSnapshot`` (Bays 3-4 from the canned background cycle), runs kinematics only,
renders each feed of the current layout and JPEG-encodes it. Per feed it enables only the
rendered bay's lights, so each frame pays for one cell's shadow map instead of four. It never
writes to the physics model or data.

``RenderWorker`` is the thread around it: it publishes telemetry first (so the page learns a
new layout before its frames arrive), then frames tagged with that layout id. A render failure
(GL context lost, driver hiccup) is logged and retried after ``ERROR_BACKOFF_S``; telemetry
keeps flowing meanwhile. The GL context is created on, and only used from, this thread.
"""

from __future__ import annotations

import copy
import io
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from PIL import Image

from teleop.console.background import BackgroundCycle, BackgroundPlayer
from teleop.console.layout import FeedSpec, feed_layout
from teleop.console.snapshot import LatestSlot, SimSnapshot
from teleop.console.telemetry import TelemetryAdapter

mj: Any = _mujoco
logger = logging.getLogger("teleop.console.render")

JPEG_QUALITY = 80
ERROR_BACKOFF_S = 1.0
STRIP_EVERY = 3  # rescue mode: the small strip tiles refresh at a third of the frame rate


class FramePublisher(Protocol):
    def publish_telemetry(self, msg: dict[str, Any]) -> None: ...

    def publish_frame(self, slot: int, layout_id: int, jpeg: bytes) -> None: ...


def _bay_lights(model: Any) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for i in range(int(model.nlight)):
        name = str(model.light(i).name)
        if name.startswith("bay") and "/" in name:
            head = name.split("/", 1)[0][3:]
            if head.isdigit():
                out.setdefault(int(head), []).append(i)
    return out


class FeedRenderer:
    """Renders feeds to JPEG. Construct and use on one thread (it owns GL contexts)."""

    def __init__(
        self, model: Any, background: BackgroundPlayer | None, quality: int = JPEG_QUALITY
    ) -> None:
        self.model: Any = copy.deepcopy(model)  # light toggles must not touch physics
        self.data: Any = mj.MjData(self.model)
        self.background = background
        self.quality = quality
        self._renderers: dict[tuple[int, int], Any] = {}
        self._bay_lights = _bay_lights(self.model)
        self._base_active = np.array(self.model.light_active, dtype=np.uint8).copy()
        self._rescue_opt: Any = None

    def _renderer(self, width: int, height: int) -> Any:
        key = (width, height)
        r = self._renderers.get(key)
        if r is None:
            r = mj.Renderer(self.model, height, width)
            self._renderers[key] = r
        return r

    def _light_only(self, bay: int) -> None:
        active = self.model.light_active
        if not self._bay_lights:
            active[:] = self._base_active
            return
        active[:] = 0
        for i in self._bay_lights.get(bay, []):
            active[i] = self._base_active[i]

    def _scene_option(self, feed: FeedSpec) -> Any:
        if feed.role != "main":
            return None
        if self._rescue_opt is None:
            from teleop.retarget.clutch import rescue_view_option

            self._rescue_opt = rescue_view_option()
        return self._rescue_opt

    def render(
        self, snap: SimSnapshot, feeds: Sequence[FeedSpec], wall_t: float
    ) -> tuple[list[tuple[int, bytes]], dict[int, float]]:
        d = self.data
        d.qpos[:] = snap.qpos
        if snap.mocap_pos.size:
            d.mocap_pos[:] = snap.mocap_pos
            d.mocap_quat[:] = snap.mocap_quat
        conf = self.background.apply(d, wall_t) if self.background is not None else {}
        mj.mj_kinematics(self.model, d)
        mj.mj_camlight(self.model, d)
        frames: list[tuple[int, bytes]] = []
        for feed in feeds:
            self._light_only(feed.bay)
            r = self._renderer(feed.width, feed.height)
            r.update_scene(d, feed.camera, scene_option=self._scene_option(feed))
            pixels = r.render()
            buf = io.BytesIO()
            Image.fromarray(pixels).save(buf, format="JPEG", quality=self.quality)
            frames.append((feed.slot, buf.getvalue()))
        return frames, conf

    def close(self) -> None:
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:  # noqa: BLE001 - best-effort GL teardown
                logger.debug("renderer close failed", exc_info=True)
        self._renderers.clear()


@dataclass
class RenderStats:
    frames: int = 0
    errors: int = 0
    total_frame_ms: float = 0.0

    @property
    def mean_frame_ms(self) -> float:
        return self.total_frame_ms / self.frames if self.frames else 0.0


class RenderWorker(threading.Thread):
    def __init__(
        self,
        model: Any,
        snapshots: LatestSlot[SimSnapshot],
        server: FramePublisher,
        adapter: TelemetryAdapter,
        bays_present: Sequence[int],
        background: BackgroundCycle | None,
        fps: int = 30,
        telemetry_hz: float = 10.0,
    ) -> None:
        super().__init__(name="console-render", daemon=True)
        self.model = model
        self.snapshots = snapshots
        self.server = server
        self.adapter = adapter
        self.bays_present = tuple(bays_present)
        self.background = background
        self.frame_period = 1.0 / max(1, fps)
        self.telemetry_period = 1.0 / telemetry_hz
        self.stats = RenderStats()
        self._stop_event = threading.Event()
        self._bg_conf: Mapping[int, float] = {}

    def stop(self) -> None:
        self._stop_event.set()

    def _publish_telemetry(self, snap: SimSnapshot, feeds: Sequence[FeedSpec]) -> None:
        msg = self.adapter.build(snap.view, snap.sim_t, feeds, self._bg_conf)
        self.server.publish_telemetry(msg)

    def run(self) -> None:
        renderer: FeedRenderer | None = None
        start = time.monotonic()
        next_tel = 0.0
        last_feeds: tuple[FeedSpec, ...] | None = None
        try:
            while not self._stop_event.is_set():
                tick = time.monotonic()
                snap = self.snapshots.get()
                if snap is None:
                    self._stop_event.wait(0.01)
                    continue
                feeds = feed_layout(self.bays_present, snap.view.rescue.bay_id)
                if feeds != last_feeds or tick >= next_tel:
                    self._publish_telemetry(snap, feeds)
                    last_feeds = feeds
                    next_tel = tick + self.telemetry_period
                try:
                    if renderer is None:
                        player = (
                            BackgroundPlayer(self.background, self.model)
                            if self.background is not None
                            else None
                        )
                        renderer = FeedRenderer(self.model, player)
                    t0 = time.perf_counter()
                    due = [
                        f
                        for f in feeds
                        if f.role != "strip" or self.stats.frames % STRIP_EVERY == 0
                    ]
                    frames, conf = renderer.render(snap, due, tick - start)
                    layout_id = self.adapter.layout_id
                    for slot, jpeg in frames:
                        self.server.publish_frame(slot, layout_id, jpeg)
                    self._bg_conf = conf
                    self.stats.frames += 1
                    self.stats.total_frame_ms += (time.perf_counter() - t0) * 1000.0
                except Exception:
                    self.stats.errors += 1
                    logger.exception("console: render failed; retrying in %.1f s", ERROR_BACKOFF_S)
                    if renderer is not None:
                        renderer.close()
                        renderer = None
                    self._wait_with_telemetry(snap, feeds, ERROR_BACKOFF_S)
                    next_tel = time.monotonic() + self.telemetry_period
                    continue
                remaining = self.frame_period - (time.monotonic() - tick)
                if remaining > 0:
                    self._stop_event.wait(remaining)
        finally:
            if renderer is not None:
                renderer.close()

    def _wait_with_telemetry(
        self, snap: SimSnapshot, feeds: Sequence[FeedSpec], seconds: float
    ) -> None:
        """Back off from rendering but keep the page's telemetry live."""
        deadline = time.monotonic() + seconds
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now >= deadline:
                return
            latest = self.snapshots.get() or snap
            self._publish_telemetry(latest, feeds)
            self._stop_event.wait(min(self.telemetry_period, deadline - now))
