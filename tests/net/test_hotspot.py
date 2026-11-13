"""Hotspot manager and watchdog tests against an in-memory backend."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from teleop.net.hotspot import (
    DEFAULT_SSID,
    Hotspot,
    HotspotConfigError,
    HotspotWatchdog,
    load_password,
)

PASSWORD = "supersecret"


class FakeBackend:
    def __init__(self, state: str = "off", ssid: str = "Other", band: str = "5GHz") -> None:
        self._state = state
        self.ssid = ssid
        self.band = band
        self.supported = True
        self.idle_calls = 0
        self.configured: list[tuple[str, str, str]] = []

    def state(self) -> Literal["on", "off", "transitioning", "unknown"]:
        return self._state  # type: ignore[return-value]

    def client_count(self) -> int:
        return 0

    def current_config(self) -> tuple[str, str]:
        return (self.ssid, self.band)

    def band_supported(self, band: str) -> bool:
        return self.supported

    def configure(self, ssid: str, password: str, band: str) -> None:
        self.ssid, self.band = ssid, band
        self.configured.append((ssid, password, band))

    def start(self) -> None:
        self._state = "on"

    def stop(self) -> None:
        self._state = "off"

    def disable_idle_timeout(self) -> None:
        self.idle_calls += 1


def test_password_validation(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    with pytest.raises(HotspotConfigError, match=r"\.env"):
        load_password(tmp_path / "missing.env")
    for bad in ("1234567", "x" * 64):
        env.write_text(f"HOTSPOT_PASSWORD={bad}\n")
        with pytest.raises(HotspotConfigError, match=r"\.env"):
            load_password(env)
    env.write_text("HOTSPOT_PASSWORD=12345678\n")
    assert load_password(env) == "12345678"


def test_ensure_up_configures_and_starts() -> None:
    fake = FakeBackend()
    events = Hotspot(fake, password=PASSWORD).ensure_up()
    assert events == ["reconfigured", "started"]
    assert fake.state() == "on"
    assert fake.current_config() == (DEFAULT_SSID, "2.4GHz")


def test_ensure_up_noop_when_correct() -> None:
    fake = FakeBackend(state="on", ssid=DEFAULT_SSID, band="2.4GHz")
    assert Hotspot(fake, password=PASSWORD).ensure_up() == []


def test_band_unsupported_raises() -> None:
    fake = FakeBackend()
    fake.supported = False
    with pytest.raises(HotspotConfigError, match=r"2\.4GHz"):
        Hotspot(fake, password=PASSWORD).ensure_up()


def _watchdog(fake: FakeBackend, polls_before_stop: int, mutate: object) -> HotspotWatchdog:
    hs = Hotspot(fake, password=PASSWORD)
    box: list[HotspotWatchdog] = []
    count = 0

    def sleep(_s: float) -> None:
        nonlocal count
        count += 1
        if count == 1:
            mutate()  # type: ignore[operator]
        if count >= polls_before_stop:
            box[0].stop()

    ticks = iter(range(10**6))
    wd = HotspotWatchdog(hs, interval_s=0.0, sleep=sleep, clock=lambda: next(ticks))
    box.append(wd)
    return wd


def test_watchdog_restarts_when_off() -> None:
    fake = FakeBackend(state="on", ssid=DEFAULT_SSID, band="2.4GHz")
    wd = _watchdog(fake, 3, fake.stop)
    wd.run()
    assert "started" in [e for _, e in wd.events]
    assert fake.state() == "on"


def test_reconfigures_on_drift() -> None:
    fake = FakeBackend(state="on", ssid=DEFAULT_SSID, band="2.4GHz")

    def drift() -> None:
        fake.ssid = "Evil"

    wd = _watchdog(fake, 3, drift)
    wd.run()
    assert "reconfigured" in [e for _, e in wd.events]
    assert fake.ssid == DEFAULT_SSID


def test_disable_idle_timeout_called_once() -> None:
    fake = FakeBackend()
    hs = Hotspot(fake, password=PASSWORD)
    hs.ensure_up()
    hs.ensure_up()
    assert fake.idle_calls == 1


def test_watchdog_logs_poll_errors() -> None:
    fake = FakeBackend(state="on", ssid=DEFAULT_SSID, band="2.4GHz")

    def boom() -> int:
        raise RuntimeError("kaput")

    fake.client_count = boom  # type: ignore[method-assign]
    fake.state = boom  # type: ignore[method-assign,assignment]
    wd = _watchdog(fake, 2, lambda: None)
    wd.run()
    assert any(e.startswith("error: kaput") for _, e in wd.events)
