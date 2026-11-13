"""Real Windows Mobile Hotspot backend over WinRT (Windows only; excluded from pyright)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from teleop.net.hotspot import wait_for_config

log = logging.getLogger(__name__)


class WinRTBackendError(RuntimeError):
    pass


def _run(awaitable: Any) -> Any:
    async def _wrap() -> Any:
        return await awaitable

    return asyncio.run(_wrap())


class WinRTTetheringBackend:
    def __init__(self, sleep: Callable[[float], object] = time.sleep) -> None:
        self._sleep = sleep
        try:
            import winrt.windows.networking.connectivity as conn
            import winrt.windows.networking.networkoperators as ops
        except ImportError as exc:  # non-Windows or winrt missing
            raise WinRTBackendError(f"WinRT tethering is unavailable: {exc}") from exc
        self._ops = ops
        profile = conn.NetworkInformation.get_internet_connection_profile()
        if profile is None:
            for cand in conn.NetworkInformation.get_connection_profiles():
                if cand.is_wlan_connection_profile:
                    profile = cand
                    break
        if profile is None:
            raise WinRTBackendError(
                "no usable connection profile (connect this laptop to a network first)"
            )
        self._mgr = ops.NetworkOperatorTetheringManager.create_from_connection_profile(profile)

    def _band_enum(self, band: str) -> Any:
        b = self._ops.TetheringWiFiBand
        mapping = {
            "2.4GHz": b.TWO_POINT_FOUR_GIGAHERTZ,
            "5GHz": b.FIVE_GIGAHERTZ,
            "auto": b.AUTO,
        }
        if band not in mapping:
            raise WinRTBackendError(f"unknown band {band!r}")
        return mapping[band]

    def _band_name(self, value: Any) -> str:
        b = self._ops.TetheringWiFiBand
        if value == b.TWO_POINT_FOUR_GIGAHERTZ:
            return "2.4GHz"
        if value == b.FIVE_GIGAHERTZ:
            return "5GHz"
        if value == b.AUTO:
            return "auto"
        return "unknown"

    def state(self) -> Any:
        s = self._ops.TetheringOperationalState
        cur = self._mgr.tethering_operational_state
        if cur == s.ON:
            return "on"
        if cur == s.OFF:
            return "off"
        if cur == s.IN_TRANSITION:
            return "transitioning"
        return "unknown"

    def client_count(self) -> int:
        return int(self._mgr.client_count)

    def current_config(self) -> tuple[str, str]:
        cfg = self._mgr.get_current_access_point_configuration()
        return (cfg.ssid, self._band_name(cfg.band))

    def band_supported(self, band: str) -> bool:
        cfg = self._mgr.get_current_access_point_configuration()
        return bool(cfg.is_band_supported(self._band_enum(band)))

    def configure(self, ssid: str, password: str, band: str) -> None:
        cfg = self._mgr.get_current_access_point_configuration()
        cfg.ssid = ssid
        cfg.passphrase = password
        cfg.band = self._band_enum(band)
        result = _run(self._mgr.configure_access_point_async(cfg))
        if result is not None and hasattr(result, "status"):
            self._check(result, "configure access point")
        # IAsyncAction is void: read back to verify the apply took effect.
        got = wait_for_config(self.current_config, (ssid, band), sleep=self._sleep)
        if got != (ssid, band):
            raise WinRTBackendError(f"configure did not apply (now ssid={got[0]!r} band={got[1]})")

    def _check(self, result: Any, what: str) -> None:
        status = result.status
        if status != self._ops.TetheringOperationStatus.SUCCESS:
            raise WinRTBackendError(
                f"{what} failed: {status.name} {result.additional_error_message}"
            )

    def start(self) -> None:
        self._check(_run(self._mgr.start_tethering_async()), "start tethering")

    def stop(self) -> None:
        self._check(_run(self._mgr.stop_tethering_async()), "stop tethering")

    def disable_idle_timeout(self) -> None:
        mgr_cls = self._ops.NetworkOperatorTetheringManager
        fn = getattr(mgr_cls, "disable_no_connections_timeout_async", None)
        if fn is None:
            log.warning("disable_no_connections_timeout_async unavailable; hotspot may idle off")
            return
        try:
            _run(fn())
        except Exception as exc:
            log.warning("could not disable hotspot idle timeout: %s", exc)
