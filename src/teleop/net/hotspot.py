"""Windows Mobile Hotspot manager and keep-alive watchdog (pure logic, no WinRT imports)."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from dotenv import dotenv_values

log = logging.getLogger(__name__)

DEFAULT_SSID = "Router? I barely know 'er"
DEFAULT_BAND = "2.4GHz"

HotspotState = Literal["on", "off", "transitioning", "unknown"]


class HotspotConfigError(RuntimeError):
    """Hotspot configuration is missing or invalid."""


class TetheringBackend(Protocol):
    def state(self) -> HotspotState: ...
    def client_count(self) -> int: ...
    def current_config(self) -> tuple[str, str]: ...
    def band_supported(self, band: str) -> bool: ...
    def configure(self, ssid: str, password: str, band: str) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def disable_idle_timeout(self) -> None: ...


def load_password(env_file: Path = Path(".env")) -> str:
    """Read ``HOTSPOT_PASSWORD`` from ``env_file``; must be 8-63 characters (WPA2)."""
    value = dotenv_values(env_file).get("HOTSPOT_PASSWORD") if env_file.exists() else None
    if not value or not 8 <= len(value) <= 63:
        raise HotspotConfigError(
            "HOTSPOT_PASSWORD missing or not 8-63 characters "
            f"(set it in .env, see .env.example; looked in {env_file.name})"
        )
    return value


class Hotspot:
    def __init__(
        self,
        backend: TetheringBackend,
        ssid: str = DEFAULT_SSID,
        *,
        password: str,
        band: str = DEFAULT_BAND,
    ) -> None:
        self._backend = backend
        self._ssid = ssid
        self._password = password
        self._band = band
        self._idle_disabled = False

    def ensure_up(self) -> list[str]:
        """Bring the hotspot to the desired config and on-state; return the actions taken."""
        events: list[str] = []
        if not self._backend.band_supported(self._band):
            raise HotspotConfigError(f"band {self._band} is not supported by this adapter")
        if not self._idle_disabled:
            self._backend.disable_idle_timeout()
            self._idle_disabled = True
        if self._backend.current_config() != (self._ssid, self._band):
            self._backend.configure(self._ssid, self._password, self._band)
            events.append("reconfigured")
        if self._backend.state() != "on":
            self._backend.start()
            events.append("started")
        return events

    def stop(self) -> None:
        self._backend.stop()

    def status(self) -> dict[str, object]:
        ssid, band = self._backend.current_config()
        return {
            "ssid": ssid,
            "band": band,
            "state": self._backend.state(),
            "client_count": self._backend.client_count(),
        }


class HotspotWatchdog:
    """Polls the hotspot and restores config / on-state when it drifts."""

    def __init__(
        self,
        hotspot: Hotspot,
        interval_s: float = 5.0,
        on_event: Callable[[str], None] = lambda _e: None,
        *,
        sleep: Callable[[float], object] | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._hotspot = hotspot
        self._interval_s = interval_s
        self._on_event = on_event
        self._stop = threading.Event()
        self._sleep: Callable[[float], object] = sleep if sleep is not None else self._stop.wait
        self._clock = clock
        self._thread: threading.Thread | None = None
        self.events: list[tuple[int, str]] = []

    def _emit(self, event: str) -> None:
        self.events.append((self._clock(), event))
        self._on_event(event)

    def poll_once(self) -> None:
        try:
            for event in self._hotspot.ensure_up():
                self._emit(event)
        except Exception as exc:
            log.warning("hotspot watchdog poll failed: %s", exc)
            self._emit(f"error: {exc}")

    def run(self) -> None:
        """Blocking poll loop; returns after :meth:`stop`."""
        self._stop.clear()
        while not self._stop.is_set():
            self.poll_once()
            if self._stop.is_set():
                break
            self._sleep(self._interval_s)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, name="hotspot-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=self._interval_s + 5)
            self._thread = None
