"""Latest-value buffer and the threaded input-source base class."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod

from teleop.types import BimanualFrame, HandFrame, Side, SourceStatus

logger = logging.getLogger(__name__)


class LatestBuffer:
    """Thread-safe store of the newest frame per side, with staleness and anomaly counters."""

    def __init__(self, stale_after_ns: int = 250_000_000) -> None:
        self._stale_after_ns = stale_after_ns
        self._lock = threading.Lock()
        self._frames: dict[Side, HandFrame] = {}
        self._last_device_us: dict[Side, int] = {}
        self._last_rx_ns: int | None = None
        self._count = 0
        self._anomalies = 0

    def put(self, frame: HandFrame) -> None:
        with self._lock:
            if frame.device_time_us is not None:
                prev = self._last_device_us.get(frame.side)
                if prev is not None and frame.device_time_us <= prev:
                    self._anomalies += 1
                self._last_device_us[frame.side] = frame.device_time_us
            self._frames[frame.side] = frame
            self._count += 1
            if self._last_rx_ns is None or frame.host_time_ns > self._last_rx_ns:
                self._last_rx_ns = frame.host_time_ns

    def get(self) -> BimanualFrame:
        with self._lock:
            return BimanualFrame(left=self._frames.get("left"), right=self._frames.get("right"))

    def status(self, now_ns: int | None = None) -> SourceStatus:
        now = time.monotonic_ns() if now_ns is None else now_ns
        with self._lock:
            last = self._last_rx_ns
            connected = last is not None and (now - last) <= self._stale_after_ns
            return SourceStatus(
                connected=connected,
                last_rx_ns=last,
                frames=self._count,
                dropped=0,
                timestamp_anomalies=self._anomalies,
            )


class ThreadedSource(ABC):
    """Runs ``_run`` on a daemon thread; subclasses push frames into ``self._buffer``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._buffer = LatestBuffer()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: BaseException | None = None

    @abstractmethod
    def _run(self, stop: threading.Event) -> None: ...

    def _guarded_run(self, stop: threading.Event) -> None:
        try:
            self._run(stop)
        except Exception as exc:
            self.last_error = exc
            logger.exception("input source %r thread died", self.name)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event = threading.Event()
        self.last_error = None
        self._thread = threading.Thread(
            target=self._guarded_run,
            args=(self._stop_event,),
            name=f"source-{self.name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._thread = None

    def latest(self) -> BimanualFrame:
        return self._buffer.get()

    def status(self) -> SourceStatus:
        return self._buffer.status()
