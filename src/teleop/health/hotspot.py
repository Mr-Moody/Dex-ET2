"""Hotspot health check: reads the status the watchdog thread cached, never the WinRT backend."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from teleop.health.model import Health, Level
from teleop.net.hotspot import Hotspot, HotspotConfigError, HotspotWatchdog, load_password

logger = logging.getLogger(__name__)


class HotspotCheck:
    name = "hotspot"

    def __init__(
        self,
        hotspot: Hotspot | None,
        watchdog: HotspotWatchdog | None,
        off_reason: str = "",
        off_detail: str = "",
        off_level: Level = Level.OFF,
    ) -> None:
        self._hotspot = hotspot
        self._watchdog = watchdog
        self._off_reason = off_reason
        self._off_detail = off_detail
        self._off_level = off_level

    def status(self) -> Health:
        wd = self._watchdog
        if self._hotspot is None or wd is None:
            return Health(self.name, self._off_level, self._off_reason, self._off_detail)
        if wd.last_error is not None:
            return Health(self.name, Level.FAIL, "error", wd.last_error, can_reconnect=True)
        st = wd.last_status
        if st is None and wd.status_error is not None:
            return Health(self.name, Level.FAIL, "error", wd.status_error, can_reconnect=True)
        if st is None:
            return Health(self.name, Level.WARN, "starting", "waiting for the first poll")
        state = st["state"]
        if state == "on":
            n = st["client_count"]
            if isinstance(n, int) and n > 0:
                return Health(self.name, Level.OK, f"up, {n} clients")
            return Health(
                self.name,
                Level.WARN,
                "up, no clients",
                "no clients joined: check the gloves and UNO Q are on the hotspot",
                can_reconnect=True,
            )
        if state == "off":
            return Health(self.name, Level.FAIL, "off", "hotspot is off", can_reconnect=True)
        return Health(self.name, Level.WARN, str(state), "hotspot is changing state")

    def reconnect(self) -> None:
        if self._watchdog is not None:
            # poll_once runs ensure_up, records any error and refreshes the cached status.
            self._watchdog.poll_once()


def build_hotspot_check(
    enabled: bool, env_file: Path, *, strict: bool
) -> tuple[HotspotCheck, HotspotWatchdog | None]:
    if not enabled:
        return HotspotCheck(None, None, "disabled"), None
    if sys.platform != "win32":
        return HotspotCheck(None, None, "Windows only"), None
    try:
        load_password(env_file)
    except HotspotConfigError:
        if strict:
            raise
        return HotspotCheck(None, None, "no password", "set HOTSPOT_PASSWORD in .env"), None
    from teleop.app import start_hotspot_watchdog

    # Bring-up happens on the watchdog thread (its first poll), not inline.
    try:
        wd = start_hotspot_watchdog(env_file, inline_bring_up=False)
    except RuntimeError as exc:  # WinRTBackendError, e.g. no usable connection profile
        if strict:
            raise
        logger.warning("hotspot: unavailable (%s)", exc)
        return HotspotCheck(None, None, "unavailable", str(exc), Level.FAIL), None
    return HotspotCheck(wd.hotspot, wd), wd
