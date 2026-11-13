"""``teleop hotspot up|status|down|watch``."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from teleop.net.hotspot import (
    DEFAULT_SSID,
    Hotspot,
    HotspotConfigError,
    HotspotWatchdog,
    TetheringBackend,
    load_password,
)


def _backend() -> TetheringBackend:
    from teleop.net.winrt_backend import WinRTTetheringBackend

    return WinRTTetheringBackend()


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="teleop hotspot")
    p.add_argument("action", choices=["up", "status", "down", "watch"])
    p.add_argument("--env-file", type=Path, default=Path(".env"))
    p.add_argument("--interval", type=float, default=5.0)
    ns = p.parse_args(argv)
    try:
        backend = _backend()
        if ns.action == "status":
            # status never needs the password
            for key, value in Hotspot(backend, password="").status().items():
                print(f"{key}: {value}")
            return 0
        if ns.action == "down":
            backend.stop()
            print("hotspot stopped")
            return 0
        hotspot = Hotspot(backend, DEFAULT_SSID, password=load_password(ns.env_file))
        if ns.action == "up":
            events = hotspot.ensure_up()
            print("events:", ", ".join(events) or "none (already up)")
            return 0
        wd = HotspotWatchdog(hotspot, ns.interval, on_event=lambda e: print(f"[watchdog] {e}"))
        wd.start()
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            wd.stop()
        return 0
    except (HotspotConfigError, RuntimeError) as exc:
        print(f"teleop hotspot: {exc}")
        return 1
