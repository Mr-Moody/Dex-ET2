from __future__ import annotations

import threading
import time

from teleop.health.model import Health, Level, health_to_dict
from teleop.health.monitor import HealthMonitor


class FakeCheck:
    def __init__(self, name: str, *, block: bool = False) -> None:
        self.name = name
        self.block = block
        self.release = threading.Event()
        self.started = threading.Event()
        self.calls = 0
        self.thread_name = ""

    def status(self) -> Health:
        return Health(self.name, Level.OK, f"{self.name} fine", can_reconnect=True)

    def reconnect(self) -> None:
        self.calls += 1
        self.thread_name = threading.current_thread().name
        self.started.set()
        if self.block:
            self.release.wait(5.0)


class RaisingCheck:
    name = "boom"

    def status(self) -> Health:
        raise RuntimeError("kaboom")

    def reconnect(self) -> None:
        pass


def test_snapshot_reflects_last_poll() -> None:
    a, b = FakeCheck("hotspot"), FakeCheck("gloves")
    mon = HealthMonitor([a, b])
    mon.poll_once()
    assert mon.snapshot() == [a.status(), b.status()]


def test_reconnect_runs_off_caller_thread() -> None:
    fake = FakeCheck("gloves", block=True)
    mon = HealthMonitor([fake])
    try:
        assert mon.request_reconnect("gloves") is True
        assert fake.started.wait(2.0)
        assert fake.thread_name != threading.current_thread().name
        assert fake.thread_name.startswith("health-reconnect")
    finally:
        fake.release.set()
        mon.stop()


def test_busy_is_reported_while_reconnecting() -> None:
    a, b = FakeCheck("gloves", block=True), FakeCheck("panel")
    mon = HealthMonitor([a, b])
    try:
        assert mon.request_reconnect("gloves")
        assert a.started.wait(2.0)
        mon.poll_once()
        assert [h.busy for h in mon.snapshot()] == [True, False]
        a.release.set()
        deadline = time.monotonic() + 2.0
        while mon.snapshot()[0].busy and time.monotonic() < deadline:
            time.sleep(0.01)
        mon.poll_once()
        assert [h.busy for h in mon.snapshot()] == [False, False]
    finally:
        a.release.set()
        mon.stop()


def test_reconnect_ignored_while_busy() -> None:
    fake = FakeCheck("gloves", block=True)
    mon = HealthMonitor([fake])
    try:
        assert mon.request_reconnect("gloves") is True
        assert fake.started.wait(2.0)
        assert mon.request_reconnect("gloves") is False
        assert fake.calls == 1
    finally:
        fake.release.set()
        mon.stop()


def test_unknown_name_rejected() -> None:
    mon = HealthMonitor([FakeCheck("gloves")])
    assert mon.request_reconnect("nope") is False


def test_status_exception_becomes_fail() -> None:
    ok = FakeCheck("gloves")
    mon = HealthMonitor([RaisingCheck(), ok])
    mon.poll_once()
    snap = mon.snapshot()
    assert snap[0] == Health("boom", Level.FAIL, "check error", detail="kaboom")
    assert snap[1].level is Level.OK
    mon.poll_once()
    assert len(mon.snapshot()) == 2


def test_health_to_dict_shape() -> None:
    d = health_to_dict(Health("gloves", Level.WARN, "slow", detail="x", can_reconnect=True))
    assert d == {
        "name": "gloves",
        "level": "warn",
        "summary": "slow",
        "detail": "x",
        "can_reconnect": True,
        "busy": False,
    }
    assert type(d["level"]) is str
    assert list(d) == ["name", "level", "summary", "detail", "can_reconnect", "busy"]
