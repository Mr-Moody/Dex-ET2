"""Teleop application: input source -> retarget + wrist -> sim sink, at a fixed control rate."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import logging
import math
import time
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import Any, cast

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]

from teleop.config import SimConfig
from teleop.latency import LatencyStats
from teleop.net.hotspot import HotspotWatchdog
from teleop.protocols import InputSource, Retargeter, RobotSink, Task, WristSource
from teleop.registry import RETARGETERS, SINKS, SOURCES, TASKS, WRISTS, resolve
from teleop.retarget.calibration import Calibration
from teleop.sim.hands import HandSpec, load_hand
from teleop.sim.loop import FixedRateLoop, LoopStats
from teleop.sim.scene import build_scene  # pyright: ignore[reportUnknownVariableType]
from teleop.types import SIDES, HandFrame, JointTargets, Side, WristTarget
from teleop.wrist.auto import AutoWrist
from teleop.wrist.coil import CoilWrist
from teleop.wrist.imu import ImuWrist

mj: Any = _mujoco
logger = logging.getLogger(__name__)

LOG_PERIOD_S = 2.0

# GLFW key codes, as delivered to ``mujoco.viewer.launch_passive(key_callback=...)``.
KEY: dict[str, int] = {
    "RIGHT": 262,
    "LEFT": 263,
    "DOWN": 264,
    "UP": 265,
    "PAGE_UP": 266,
    "PAGE_DOWN": 267,
    "A": 65,
    "C": 67,
    "D": 68,
    "E": 69,
    "Q": 81,
    "R": 82,
    "S": 83,
    "W": 87,
}

# keycode -> (side, unit direction) in sim world axes: +x right, +y away from the camera, +z up.
# Arrows/PgUp/PgDn move the right wrist; WASD/Q/E move the left one (applied in IMU mode).
NUDGE_KEYS: dict[int, tuple[Side, tuple[float, float, float]]] = {
    KEY["RIGHT"]: ("right", (1.0, 0.0, 0.0)),
    KEY["LEFT"]: ("right", (-1.0, 0.0, 0.0)),
    KEY["UP"]: ("right", (0.0, 1.0, 0.0)),
    KEY["DOWN"]: ("right", (0.0, -1.0, 0.0)),
    KEY["PAGE_UP"]: ("right", (0.0, 0.0, 1.0)),
    KEY["PAGE_DOWN"]: ("right", (0.0, 0.0, -1.0)),
    KEY["D"]: ("left", (1.0, 0.0, 0.0)),
    KEY["A"]: ("left", (-1.0, 0.0, 0.0)),
    KEY["W"]: ("left", (0.0, 1.0, 0.0)),
    KEY["S"]: ("left", (0.0, -1.0, 0.0)),
    KEY["E"]: ("left", (0.0, 0.0, 1.0)),
    KEY["Q"]: ("left", (0.0, 0.0, -1.0)),
}

_BOOL_FIELDS = frozenset(f.name for f in dataclasses.fields(SimConfig) if f.type in ("bool", bool))


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    """``--replay-file`` -> ``--replay_file``; a bare ``--headless`` -> ``--headless true``."""
    out: list[str] = []
    for i, tok in enumerate(argv):
        if not tok.startswith("--"):
            out.append(tok)
            continue
        key, eq, val = tok[2:].partition("=")
        key = key.replace("-", "_")
        out.append(f"--{key}{eq}{val}")
        bare = not eq and (i + 1 == len(argv) or argv[i + 1].startswith("--"))
        if bare and key in _BOOL_FIELDS:
            out.append("true")
    return out


def parse_sim_config(argv: Sequence[str]) -> SimConfig:
    """Parse ``SimConfig`` with draccus, accepting hyphenated flags and bare booleans."""
    import draccus  # pyright: ignore[reportMissingTypeStubs]

    parse: Any = draccus.parse  # pyright: ignore[reportUnknownMemberType]
    return cast("SimConfig", parse(SimConfig, args=_normalize_argv(argv)))


def make_source(cfg: SimConfig) -> InputSource:
    """Build the input source named by ``cfg.source`` through the registry."""
    cls: Any = resolve(SOURCES, cfg.source)
    if cfg.source == "replay":
        if cfg.replay_file is None:
            raise ValueError("--source replay needs --replay-file PATH")
        return cast("InputSource", cls(cfg.replay_file))
    if cfg.source in ("studio", "rokoko-sdk"):
        return cast("InputSource", cls(cfg.studio))
    return cast("InputSource", cls())


def start_hotspot_watchdog(
    env_file: Path = Path(".env"), interval_s: float = 5.0, *, inline_bring_up: bool = True
) -> HotspotWatchdog:
    """Bring the booth hotspot up and keep it up (Windows only). Fails fast on a bad password."""
    from teleop.net.hotspot import DEFAULT_SSID, Hotspot, load_password
    from teleop.net.winrt_backend import (  # pyright: ignore[reportMissingTypeStubs]
        WinRTTetheringBackend,  # pyright: ignore[reportUnknownVariableType]
    )

    backend: Any = WinRTTetheringBackend()
    hotspot = Hotspot(backend, DEFAULT_SSID, password=load_password(env_file))
    if inline_bring_up:
        for event in hotspot.ensure_up():
            logger.info("hotspot: %s", event)
    wd = HotspotWatchdog(hotspot, interval_s, on_event=lambda e: logger.warning("hotspot: %s", e))
    wd.start()
    return wd


def _frame_key(frame: HandFrame) -> tuple[int, int]:
    return (frame.host_time_ns, frame.seq)


class TeleopApp:
    """One control tick per ``step``: latest frames -> targets -> ``sink.apply``."""

    def __init__(
        self,
        cfg: SimConfig,
        source: InputSource | None = None,
        hotspot: HotspotWatchdog | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.cfg = cfg
        self._clock = clock
        self.hotspot = hotspot
        self.hand: HandSpec = load_hand(cfg.hand)
        self.task = cast("Task", resolve(TASKS, cfg.task)())
        self.model: Any = cast("Any", build_scene(self.hand, self.task, cfg.cameras))
        self.data: Any = mj.MjData(self.model)
        self.task.reset(self.model, self.data, 0)
        sink_cls: Any = resolve(SINKS, cfg.sink)
        sink = sink_cls(self.model, self.data, cfg.physics_hz, cfg.control_hz)
        self.sink = cast("RobotSink", sink)

        self.ctrlrange: dict[Side, dict[str, tuple[float, float]]] = {s: {} for s in SIDES}
        self.actuators: dict[Side, list[str]] = {s: [] for s in SIDES}
        for i in range(int(self.model.nu)):
            side_s, _, act = str(self.model.actuator(i).name).partition("/")
            if side_s not in SIDES or not act:
                continue
            side: Side = side_s
            self.actuators[side].append(act)
            if bool(self.model.actuator_ctrllimited[i]):
                lo, hi = (float(v) for v in self.model.actuator_ctrlrange[i])
                self.ctrlrange[side][act] = (lo, hi)

        self.calibration_path = cfg.calibration
        if cfg.calibration is not None:
            self.calibration = Calibration.from_json(cfg.calibration)
        else:
            logger.warning("no --calibration given; using Calibration.default (fallback only)")
            self.calibration = Calibration.default(self.hand)
        retarget_cls: Any = resolve(RETARGETERS, cfg.retargeter)
        self.retargeters: dict[Side, Retargeter] = {
            s: cast("Retargeter", retarget_cls(self.hand, self.ctrlrange, self.calibration))
            for s in SIDES
        }
        degenerate: list[Any] = sorted(getattr(self.retargeters["left"], "degenerate", set()))
        if degenerate:
            logger.warning("degenerate calibration ranges (default used): %s", degenerate)

        wrist_cls: Any = resolve(WRISTS, cfg.wrist.mode)
        self.wrist = cast("WristSource", wrist_cls(cfg.wrist))
        self.source: InputSource = source if source is not None else make_source(cfg)

        self.latency = LatencyStats()
        self.on_tick: Callable[[int], None] | None = None
        self.resets = 0
        self.overruns = 0
        self.mode_history: list[dict[str, Any]] = []
        self._modes: dict[Side, str | None] = {s: None for s in SIDES}
        self._keys: collections.deque[int] = collections.deque()
        self._last_key: dict[Side, tuple[int, int] | None] = {s: None for s in SIDES}
        self._last_frame: dict[Side, HandFrame | None] = {s: None for s in SIDES}
        self._recentered: set[Side] = set()
        self._viewer: Any = None
        self._period_ns = 1_000_000_000 // cfg.control_hz
        self._prev_tick_ns: int | None = None
        self._log_t0: int | None = None
        self._log_steps = 0
        self._gap_since_ns = clock()
        self._gap_last_ns: dict[Side, int | None] = {s: None for s in SIDES}
        self._max_gap_ns = 0

    # -- wrist helpers -------------------------------------------------------------------
    @property
    def coil(self) -> CoilWrist | None:
        if isinstance(self.wrist, AutoWrist):
            return self.wrist.coil
        return self.wrist if isinstance(self.wrist, CoilWrist) else None

    @property
    def imu(self) -> ImuWrist | None:
        if isinstance(self.wrist, AutoWrist):
            return self.wrist.imu
        return self.wrist if isinstance(self.wrist, ImuWrist) else None

    def wrist_mode(self, side: Side) -> str:
        if isinstance(self.wrist, AutoWrist):
            return self.wrist.mode_for(side)
        return self.wrist.mode

    def recenter(self) -> bool:
        """Map the latest Coil positions onto the home poses. False if none are available."""
        coil = self.coil
        frames: dict[Side, HandFrame] = {
            s: f
            for s in SIDES
            if (f := self._last_frame[s]) is not None
            and f.wrist is not None
            and f.wrist.position is not None
        }
        if coil is None or not frames:
            return False
        coil.recenter(frames.get("left"), frames.get("right"))
        self._recentered.update(frames)
        return True

    # -- input from the viewer thread ----------------------------------------------------
    def on_key(self, keycode: int) -> None:
        """Passive-viewer ``key_callback``. Queued; handled on the control thread."""
        self._keys.append(int(keycode))

    def _handle_keys(self) -> None:
        while self._keys:
            key = self._keys.popleft()
            if key in NUDGE_KEYS:
                side, (dx, dy, dz) = NUDGE_KEYS[key]
                imu = self.imu
                if imu is not None:
                    k = self.cfg.wrist.nudge_step_m
                    imu.nudge(side, dx * k, dy * k, dz * k)
            elif key == KEY["R"]:
                self.reset_task()
            elif key == KEY["C"]:
                if self.recenter():
                    logger.info("coil recentered to home")
                else:
                    logger.info("recenter skipped: no Coil positions")

    def reset_gaps(self) -> None:
        """Start a new window for :meth:`max_gap_s` (e.g. at the start of an episode)."""
        self._gap_since_ns = self._clock()
        self._gap_last_ns = {s: None for s in SIDES}
        self._max_gap_ns = 0

    def max_gap_s(self) -> float:
        """Largest gap between consecutive new frames per side since :meth:`reset_gaps`.

        Counts the window start as the reference before a side's first frame, and the
        still-open gap up to now, so a side that went (or stayed) silent is reported.
        """
        now = self._clock()
        gaps = [self._max_gap_ns]
        for s in SIDES:
            last = self._gap_last_ns[s]
            gaps.append(now - (last if last is not None else self._gap_since_ns))
        return max(gaps) / 1e9

    def reset_task(self) -> None:
        self.resets += 1
        self.task.reset(self.model, self.data, self.resets)
        logger.info("task reset (#%d)", self.resets)

    # -- control tick --------------------------------------------------------------------
    def step(self, i: int) -> bool:
        viewer = self._viewer
        if viewer is not None and not viewer.is_running():
            return False
        now = self._clock()
        # A missed tick = more than two periods since the last one. (Windows monotonic_ns ticks
        # in ~15.6 ms steps, so a 1.5-period threshold would count clock jitter as overruns.)
        if self._prev_tick_ns is not None and now - self._prev_tick_ns > 2 * self._period_ns:
            self.overruns += 1
        self._prev_tick_ns = now

        lock = viewer.lock() if viewer is not None else contextlib.nullcontext()
        with lock:
            self._handle_keys()
            self._tick()
        if viewer is not None:
            viewer.sync()
        self._maybe_log()
        if self.on_tick is not None:
            self.on_tick(i)
        return True

    def _tick(self) -> None:
        frames = self.source.latest()
        fresh: list[HandFrame] = []
        for side, frame in zip(SIDES, (frames.left, frames.right), strict=True):
            if frame is None or _frame_key(frame) == self._last_key[side]:
                continue  # no new data: the sink holds the last targets
            self._last_key[side] = _frame_key(frame)
            self._last_frame[side] = frame
            fresh.append(frame)
            prev = self._gap_last_ns[side]
            gap = frame.host_time_ns - (prev if prev is not None else self._gap_since_ns)
            self._max_gap_ns = max(self._max_gap_ns, gap)
            self._gap_last_ns[side] = frame.host_time_ns

        pending: list[Side] = [
            s
            for s in SIDES
            if s not in self._recentered
            and (f := self._last_frame[s]) is not None
            and f.wrist is not None
            and f.wrist.position is not None
        ]
        if pending and self.coil is not None:
            self.coil.recenter(
                self._last_frame["left"] if "left" in pending else None,
                self._last_frame["right"] if "right" in pending else None,
            )
            self._recentered.update(pending)
            logger.info("coil auto-recentered: %s", ", ".join(pending))

        targets: list[JointTargets] = []
        wrists: list[WristTarget] = []
        for frame in fresh:
            targets.append(self.retargeters[frame.side].retarget(frame))
            wrists.append(self.wrist.wrist(frame))
            mode = self.wrist_mode(frame.side)
            if mode != self._modes[frame.side]:
                self._modes[frame.side] = mode
                self.mode_history.append({"t_ns": self._clock(), "side": frame.side, "mode": mode})
                logger.info("wrist %s -> %s", frame.side, mode)
        self.sink.apply(targets, wrists)
        done = self._clock()
        for frame in fresh:
            self.latency.add(done - frame.host_time_ns)

    def _maybe_log(self) -> None:
        now = self._clock()
        if self._log_t0 is None:
            self._log_t0 = now
            self._log_steps = 0
            return
        self._log_steps += 1
        elapsed = (now - self._log_t0) / 1e9
        if elapsed < LOG_PERIOD_S:
            return
        self.log_status(fps=self._log_steps / elapsed)
        self._log_t0 = now
        self._log_steps = 0

    def log_status(self, fps: float = math.nan) -> None:
        lat = self.latency.summary()
        st = self.source.status()
        logger.info(
            "latency p50/p95/max %.1f/%.1f/%.1f ms (n=%d) | source %s connected=%s frames=%d "
            "dropped=%d ts_anomalies=%d | wrist left=%s right=%s | fps %.1f | overruns %d",
            lat["p50_ms"],
            lat["p95_ms"],
            lat["max_ms"],
            int(lat["n"]),
            self.source.name,
            st.connected,
            st.frames,
            st.dropped,
            st.timestamp_anomalies,
            self.wrist_mode("left"),
            self.wrist_mode("right"),
            fps,
            self.overruns,
        )
        err = getattr(self.source, "last_error", None)
        if err is not None:
            logger.error("source error: %r", err)

    # -- session -------------------------------------------------------------------------
    @contextlib.contextmanager
    def session(self) -> Generator[TeleopApp]:
        """Start the source and, unless headless, open the passive viewer."""
        self.source.start()
        try:
            if self.cfg.headless:
                yield self
                return
            import mujoco.viewer  # pyright: ignore[reportMissingTypeStubs]

            launch: Any = mujoco.viewer.launch_passive  # pyright: ignore
            with launch(self.model, self.data, key_callback=self.on_key) as viewer:
                self._viewer = viewer
                try:
                    yield self
                finally:
                    self._viewer = None
        finally:
            self.source.stop()

    def viewer_closed(self) -> bool:
        return self._viewer is not None and not self._viewer.is_running()

    def run(self) -> LoopStats:
        t0 = time.monotonic()
        with self.session():
            stats = FixedRateLoop(self.cfg.control_hz).run(self.step, self.cfg.steps)
        elapsed = time.monotonic() - t0
        self.log_status(fps=stats.steps / elapsed if elapsed > 0 else math.nan)
        return stats
