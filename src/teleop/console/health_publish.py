"""Adds the hardware health snapshot to every telemetry message on its way to the console."""

from __future__ import annotations

from typing import Any, Protocol

from teleop.health.monitor import HealthMonitor


class _Sink(Protocol):
    def publish_telemetry(self, msg: dict[str, Any]) -> None: ...

    def publish_frame(self, slot: int, layout_id: int, jpeg: bytes) -> None: ...


class HealthPublisher:
    """Stands in for the server in ``RenderWorker``; telemetry gains ``msg["health"]``."""

    def __init__(self, inner: _Sink, monitor: HealthMonitor) -> None:
        self._inner = inner
        self._monitor = monitor

    def publish_telemetry(self, msg: dict[str, Any]) -> None:
        msg["health"] = self._monitor.snapshot_dicts()
        self._inner.publish_telemetry(msg)

    def publish_frame(self, slot: int, layout_id: int, jpeg: bytes) -> None:
        self._inner.publish_frame(slot, layout_id, jpeg)


__all__ = ["HealthPublisher"]
