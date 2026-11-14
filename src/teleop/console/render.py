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
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from PIL import Image

from teleop.console.background import BackgroundCycle, BackgroundPlayer
from teleop.console.layout import FeedSpec, ViewState, feed_layout
from teleop.console.snapshot import LatestSlot, SimSnapshot
from teleop.console.telemetry import TelemetryAdapter

mj: Any = _mujoco
logger = logging.getLogger("teleop.console.render")

JPEG_QUALITY = 80
ERROR_BACKOFF_S = 1.0
STRIP_EVERY = 3  # rescue mode: the small strip tiles refresh at a third of the frame rate


class FeedRendererLike(Protocol):
    def render(
        self, snap: SimSnapshot, feeds: Sequence[FeedSpec], wall_t: float
    ) -> tuple[list[tuple[int, bytes]], dict[int, float]]: ...

    def close(self) -> None: ...


RendererFactory = Callable[[Any, BackgroundPlayer | None], FeedRendererLike]


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


def _qmul_wxyz(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _qrot_wxyz(
    q: tuple[float, float, float, float], v: tuple[float, float, float]
) -> tuple[float, float, float]:
    qw, qx, qy, qz = q
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


class FeedRenderer:
    """Renders feeds to JPEG. Construct and use on one thread (it owns GL contexts)."""

    def __init__(
        self, model: Any, background: BackgroundPlayer | None, quality: int = JPEG_QUALITY
    ) -> None:
        self.model: Any = copy.deepcopy(model)  # light/camera tweaks must not touch physics
        self.data: Any = mj.MjData(self.model)
        self.background = background
        self.quality = quality
        self._renderers: dict[tuple[int, int], Any] = {}
        self._bay_lights = _bay_lights(self.model)
        self._base_active = np.array(self.model.light_active, dtype=np.uint8).copy()
        self._base_cam_fovy = np.array(self.model.cam_fovy, dtype=np.float64).copy()
        self._base_cam_pos = np.array(self.model.cam_pos, dtype=np.float64).copy()
        self._base_cam_quat = np.array(self.model.cam_quat, dtype=np.float64).copy()

    def _apply_zoom_pan(self, feeds: Sequence[FeedSpec]) -> None:
        if not self._base_cam_fovy.size:
            return
        self.model.cam_fovy[:] = self._base_cam_fovy
        self.model.cam_pos[:] = self._base_cam_pos
        self.model.cam_quat[:] = self._base_cam_quat
        for feed in feeds:
            if feed.role == "strip":
                continue
            if abs(feed.zoom - 1.0) <= 1e-3 and abs(feed.pan_x) <= 1e-3 and abs(feed.pan_y) <= 1e-3:
                continue
            try:
                cid = int(self.model.camera(feed.camera).id)
            except KeyError:
                continue
            fovy0 = float(self._base_cam_fovy[cid])
            q0 = (
                float(self._base_cam_quat[cid, 0]),
                float(self._base_cam_quat[cid, 1]),
                float(self._base_cam_quat[cid, 2]),
                float(self._base_cam_quat[cid, 3]),
            )
            ortho_arr = getattr(self.model, "cam_orthographic", None)
            is_ortho = bool(ortho_arr[cid]) if ortho_arr is not None else fovy0 < 2.0
            if is_ortho:
                self.model.cam_fovy[cid] = min(0.65, max(0.05, fovy0 / max(feed.zoom, 0.2)))
                dx, dy, dz = _qrot_wxyz(q0, (-feed.pan_x * 0.16, -feed.pan_y * 0.16, 0.0))
                self.model.cam_pos[cid, 0] = self._base_cam_pos[cid, 0] + dx
                self.model.cam_pos[cid, 1] = self._base_cam_pos[cid, 1] + dy
                self.model.cam_pos[cid, 2] = self._base_cam_pos[cid, 2] + dz
            else:
                self.model.cam_fovy[cid] = min(88.0, max(9.0, fovy0 / max(feed.zoom, 0.2)))
                ax = 0.5 * (feed.pan_y * 0.48)
                ay = 0.5 * (feed.pan_x * 0.48)
                cx, sx = math.cos(ax), math.sin(ax)
                cy, sy = math.cos(ay), math.sin(ay)
                q_pan = (cy * cx, cy * sx, sy * cx, -sy * sx)
                self.model.cam_quat[cid] = _qmul_wxyz(q0, q_pan)

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

    def render(
        self, snap: SimSnapshot, feeds: Sequence[FeedSpec], wall_t: float
    ) -> tuple[list[tuple[int, bytes]], dict[int, float]]:
        d = self.data
        d.qpos[:] = snap.qpos
        if snap.mocap_pos.size:
            d.mocap_pos[:] = snap.mocap_pos
            d.mocap_quat[:] = snap.mocap_quat
        conf = self.background.apply(d, wall_t) if self.background is not None else {}
        self._apply_zoom_pan(feeds)
        mj.mj_kinematics(self.model, d)
        mj.mj_camlight(self.model, d)
        frames: list[tuple[int, bytes]] = []
        for feed in feeds:
            self._light_only(feed.bay)
            r = self._renderer(feed.width, feed.height)
            r.update_scene(d, feed.camera)  # every feed shows the full arm and hand
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
        renderer_factory: RendererFactory | None = None,
        dummy_status: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(name="console-render", daemon=True)
        self.renderer_factory: RendererFactory = renderer_factory or FeedRenderer
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
        self._view_lock = threading.Lock()
        self._view_state: ViewState | None = None
        self._dummy_status = dummy_status

    def set_view_state(self, state: ViewState | None) -> None:
        with self._view_lock:
            self._view_state = state

    def _current_view_state(self) -> ViewState | None:
        with self._view_lock:
            return self._view_state

    def stop(self) -> None:
        self._stop_event.set()

    def _publish_telemetry(
        self, snap: SimSnapshot, feeds: Sequence[FeedSpec], vstate: ViewState | None = None
    ) -> None:
        dstat = self._dummy_status() if self._dummy_status is not None else None
        msg = self.adapter.build(
            snap.view,
            snap.sim_t,
            feeds,
            self._bg_conf,
            view_state=vstate,
            dummy_status=dstat,
        )
        self.server.publish_telemetry(msg)

    def run(self) -> None:
        self._renderer: FeedRendererLike | None = None
        self._start = time.monotonic()
        self._next_tel = 0.0
        self._last_feeds: tuple[FeedSpec, ...] | None = None
        try:
            while not self._stop_event.is_set():
                try:
                    pause = self._step()
                except Exception:
                    # Anything (render, layout, telemetry, publish) is logged and retried: the
                    # console must never go dark for the rest of the demo.
                    self.stats.errors += 1
                    logger.exception(
                        "console: render loop failed; retrying in %.1f s", ERROR_BACKOFF_S
                    )
                    self._drop_renderer()
                    self._backoff(ERROR_BACKOFF_S)
                    continue
                if pause > 0:
                    self._stop_event.wait(pause)
        finally:
            self._drop_renderer()

    def _drop_renderer(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:  # noqa: BLE001 - best-effort GL teardown
                logger.debug("renderer close failed", exc_info=True)
            self._renderer = None

    def _step(self) -> float:
        """One frame: telemetry if due (always before a new layout's frames), then render."""
        tick = time.monotonic()
        snap = self.snapshots.get()
        if snap is None:
            return 0.01
        vstate = self._current_view_state()
        feeds = feed_layout(self.bays_present, snap.view.rescue.bay_id, vstate)
        if feeds != self._last_feeds or tick >= self._next_tel:
            self._publish_telemetry(snap, feeds, vstate)
            self._last_feeds = feeds
            self._next_tel = tick + self.telemetry_period
        if self._renderer is None:
            player = (
                BackgroundPlayer(self.background, self.model)
                if self.background is not None
                else None
            )
            self._renderer = self.renderer_factory(self.model, player)
        t0 = time.perf_counter()
        due = [
            f
            for f in feeds
            if f.role != "strip"
            or self._last_feeds != feeds
            or self.stats.frames % STRIP_EVERY == 0
        ]
        frames, conf = self._renderer.render(snap, due, tick - self._start)
        layout_id = self.adapter.layout_id
        for slot, jpeg in frames:
            self.server.publish_frame(slot, layout_id, jpeg)
        self._bg_conf = conf
        self.stats.frames += 1
        self.stats.total_frame_ms += (time.perf_counter() - t0) * 1000.0
        return self.frame_period - (time.monotonic() - tick)

    def _backoff(self, seconds: float) -> None:
        """Back off from rendering but keep the page's telemetry live (best effort)."""
        deadline = time.monotonic() + seconds
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now >= deadline:
                break
            snap = self.snapshots.get()
            if snap is not None:
                try:
                    vstate = self._current_view_state()
                    feeds = feed_layout(self.bays_present, snap.view.rescue.bay_id, vstate)
                    self._publish_telemetry(snap, feeds, vstate)
                    self._last_feeds = feeds
                except Exception:  # noqa: BLE001 - already logged by the loop; keep waiting
                    logger.debug("console: telemetry failed during back-off", exc_info=True)
            self._stop_event.wait(min(self.telemetry_period, deadline - now))
        self._next_tel = time.monotonic() + self.telemetry_period
