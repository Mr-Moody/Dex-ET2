"""HotspotCheck against the in-memory tethering backend; no WinRT, no real hotspot."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import pytest

from teleop.health.hotspot import HotspotCheck, build_hotspot_check
from teleop.health.model import Level
from teleop.net.hotspot import DEFAULT_SSID, Hotspot, HotspotConfigError, HotspotWatchdog


class CountingBackend:
    """In-memory tethering backend that counts reads."""

    def __init__(self, state: str = "on", clients: int = 0) -> None:
        self._state = state
        self.clients = clients
        self.calls = 0
        self.starts = 0

    def state(self) -> Literal["on", "off", "transitioning", "unknown"]:
        self.calls += 1
        return self._state  # type: ignore[return-value]

    def client_count(self) -> int:
        self.calls += 1
        return self.clients

    def current_config(self) -> tuple[str, str]:
        self.calls += 1
        return (DEFAULT_SSID, "2.4GHz")

    def band_supported(self, band: str) -> bool:
        return True

    def configure(self, ssid: str, password: str, band: str) -> None:
        pass

    def start(self) -> None:
        self.starts += 1
        self._state = "on"

    def stop(self) -> None:
        self._state = "off"

    def disable_idle_timeout(self) -> None:
        pass


def make(backend: CountingBackend) -> tuple[HotspotCheck, HotspotWatchdog]:
    wd = HotspotWatchdog(Hotspot(backend, password="supersecret"))
    return HotspotCheck(wd.hotspot, wd), wd


def test_ok_with_clients() -> None:
    check, wd = make(CountingBackend(clients=2))
    wd.poll_once()
    h = check.status()
    assert (h.level, h.summary) == (Level.OK, "up, 2 clients")


def test_warn_no_clients() -> None:
    check, wd = make(CountingBackend(clients=0))
    wd.poll_once()
    h = check.status()
    assert h.level is Level.WARN
    assert "gloves" in h.detail
    assert "UNO Q" in h.detail


def test_fail_off() -> None:
    backend = CountingBackend(state="on")
    check, wd = make(backend)
    wd.poll_once()
    backend.stop()
    wd.last_status = wd.hotspot.status()  # what the next poll would cache after a drop
    h = check.status()
    assert h.level is Level.FAIL
    assert h.can_reconnect


def test_fail_on_last_watchdog_error() -> None:
    check, wd = make(CountingBackend())
    wd.poll_once()
    wd.last_error = "error: reconfigure did not apply"
    h = check.status()
    assert h.level is Level.FAIL
    assert h.detail == "error: reconfigure did not apply"


def test_reconnect_calls_ensure_up() -> None:
    backend = CountingBackend(state="off")
    check, _ = make(backend)
    check.reconnect()
    assert backend.starts == 1
    assert check.status().level is not Level.FAIL


def test_status_does_not_call_backend() -> None:
    backend = CountingBackend(clients=1)
    check, wd = make(backend)
    wd.poll_once()
    before = backend.calls
    for _ in range(5):
        check.status()
    assert backend.calls == before


def test_off_when_disabled() -> None:
    check, wd = build_hotspot_check(False, Path(".env"), strict=True)
    assert wd is None
    h = check.status()
    assert (h.level, h.summary) == (Level.OFF, "disabled")


def test_build_without_password_not_strict_is_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    env = tmp_path / ".env"
    env.write_text("")
    check, wd = build_hotspot_check(True, env, strict=False)
    assert wd is None
    h = check.status()
    assert h.level is Level.OFF
    assert h.detail == "set HOTSPOT_PASSWORD in .env"


def test_build_without_password_strict_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    env = tmp_path / ".env"
    env.write_text("")
    with pytest.raises(HotspotConfigError):
        build_hotspot_check(True, env, strict=True)


def test_error_poll_then_clean_poll_is_not_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = CountingBackend(clients=1)
    check, wd = make(backend)
    real = wd.hotspot.ensure_up

    def boom() -> list[str]:
        raise RuntimeError("transient")

    monkeypatch.setattr(wd.hotspot, "ensure_up", boom)
    wd.poll_once()
    assert check.status().level is Level.FAIL
    monkeypatch.setattr(wd.hotspot, "ensure_up", real)
    wd.poll_once()
    assert check.status().level is Level.OK


def test_failed_status_read_is_fail_not_stale_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    check, wd = make(CountingBackend(clients=1))
    wd.poll_once()
    assert check.status().level is Level.OK

    def boom() -> dict[str, object]:
        raise RuntimeError("winrt gone")

    monkeypatch.setattr(wd.hotspot, "status", boom)
    wd.poll_once()
    h = check.status()
    assert h.level is Level.FAIL
    assert "winrt gone" in h.detail


def test_build_backend_error_not_strict_is_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import teleop.app

    def no_profile(*_: object, **__: object) -> HotspotWatchdog:
        raise RuntimeError("no usable connection profile")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(teleop.app, "start_hotspot_watchdog", no_profile)
    env = tmp_path / ".env"
    env.write_text("HOTSPOT_PASSWORD=booth-pass-123\n")
    check, wd = build_hotspot_check(True, env, strict=False)
    assert wd is None
    h = check.status()
    assert h.level is Level.FAIL
    assert "no usable connection profile" in h.detail
    with pytest.raises(RuntimeError, match="connection profile"):
        build_hotspot_check(True, env, strict=True)
