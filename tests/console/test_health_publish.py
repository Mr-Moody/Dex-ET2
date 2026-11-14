"""HealthPublisher decorates telemetry with the monitor's health dicts."""

from __future__ import annotations

from typing import Any

from teleop.console.health_publish import HealthPublisher


class _Inner:
    def __init__(self) -> None:
        self.telemetry: list[dict[str, Any]] = []
        self.frames: list[tuple[int, int, bytes]] = []

    def publish_telemetry(self, msg: dict[str, Any]) -> None:
        self.telemetry.append(msg)

    def publish_frame(self, slot: int, layout_id: int, jpeg: bytes) -> None:
        self.frames.append((slot, layout_id, jpeg))


class _Monitor:
    def __init__(self) -> None:
        self.dicts: list[dict[str, object]] = [{"name": "gloves", "level": "fail"}]

    def snapshot_dicts(self) -> list[dict[str, object]]:
        return self.dicts


def test_health_publisher_adds_health() -> None:
    inner, monitor = _Inner(), _Monitor()
    pub = HealthPublisher(inner, monitor)  # type: ignore[arg-type]
    pub.publish_telemetry({"type": "telemetry", "t": 1.0})
    assert inner.telemetry == [{"type": "telemetry", "t": 1.0, "health": monitor.dicts}]
    pub.publish_frame(2, 3, b"\xff\xd8x")
    assert inner.frames == [(2, 3, b"\xff\xd8x")]
    assert inner.telemetry[0]["health"] is not None
