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
RECONFIG_BACKOFF_NS = 60_000_000_000
RESTART_COOLDOWN_NS = 600_000_000_000

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


def wait_for_config(
    read: Callable[[], tuple[str, str]],
    want: tuple[str, str],
    timeout_s: float = 2.0,
    interval_s: float = 0.2,
    sleep: Callable[[float], object] = time.sleep,
) -> tuple[str, str]:
    """Poll ``read`` until it returns ``want`` or the timeout elapses; return the last value."""
    got = read()
    waited = 0.0
    while got != want and waited < timeout_s:
        sleep(interval_s)
        waited += interval_s
        got = read()
    return got


class Hotspot:
    def __init__(
        self,
        backend: TetheringBackend,
        ssid: str = DEFAULT_SSID,
        *,
        password: str,
        band: str = DEFAULT_BAND,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._clock = clock
        self._next_reconfig_ns = 0
        self._last_restart_ns: int | None = None
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
        want = (self._ssid, self._band)
        if self._backend.current_config() != want and self._clock() >= self._next_reconfig_ns:
            events.extend(self._reconfigure(want))
        if self._backend.state() != "on":
            self._backend.start()
            events.append("started")
        return events

    def _try_configure(self, want: tuple[str, str]) -> bool:
        try:
            self._backend.configure(self._ssid, self._password, self._band)
            return self._backend.current_config() == want
        except Exception as exc:
            log.warning("hotspot configure failed: %s", exc)
            return False

    def _reconfigure(self, want: tuple[str, str]) -> list[str]:
        if self._try_configure(want):
            self._next_reconfig_ns = 0
            return ["reconfigured"]
        now = self._clock()
        bounce_ok = self._last_restart_ns is None or now - self._last_restart_ns >= (
            RESTART_COOLDOWN_NS
        )
        if self._backend.state() == "on" and bounce_ok:
            # Some Windows builds reject reconfiguration while tethering is on.
            # A bounce drops clients, so it is rate limited.
            self._last_restart_ns = now
            self._backend.stop()
            applied = self._try_configure(want)
            self._backend.start()
            if applied:
                self._next_reconfig_ns = 0
                return ["restarted-for-reconfigure"]
        self._next_reconfig_ns = self._clock() + RECONFIG_BACKOFF_NS
        return ["error: reconfigure did not apply"]

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
        self.last_status: dict[str, object] | None = None
        self.last_error: str | None = None  # outcome of the latest poll's bring-up
        self.status_error: str | None = None  # set when the latest status read failed
        self._poll_lock = threading.Lock()

    @property
    def hotspot(self) -> Hotspot:
        return self._hotspot

    def _emit(self, event: str) -> None:
        self.events.append((self._clock(), event))
        self._on_event(event)

    def poll_once(self) -> None:
        with self._poll_lock:
            error: str | None = None
            try:
                for event in self._hotspot.ensure_up():
                    self._emit(event)
                    if event.startswith("error:"):
                        error = event
            except Exception as exc:
                log.warning("hotspot watchdog poll failed: %s", exc)
                error = f"error: {exc}"
                self._emit(error)
            self.last_error = error
            try:
                self.last_status = self._hotspot.status()
                self.status_error = None
            except Exception as exc:
                log.warning("hotspot status read failed: %s", exc)
                self.last_status = None
                self.status_error = f"status read failed: {exc}"

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
