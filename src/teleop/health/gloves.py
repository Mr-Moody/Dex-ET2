"""Rokoko glove health: is Studio JSON arriving, and does it carry a right hand."""

from __future__ import annotations

import time
from collections.abc import Callable

from teleop.health.model import Health, Level
from teleop.input.studio import StudioSource

_OK_NS = 500_000_000
_FAIL_NS = 2_000_000_000


class GloveCheck:
    name = "gloves"

    def __init__(
        self,
        source: StudioSource | None,
        port: int,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._source = source
        self._port = port
        self._clock = clock

    def _h(self, level: Level, summary: str, detail: str = "", *, reconnect: bool = True) -> Health:
        return Health(self.name, level, summary, detail, can_reconnect=reconnect)

    def status(self) -> Health:
        if self._source is None:
            return self._h(Level.OFF, "synthetic glove", reconnect=False)
        live = self._source.liveness()
        port = live.port if live.port is not None else self._port
        hint = f"Start streaming in Rokoko Studio to 127.0.0.1:{port} (JSON v3)."
        if live.bind_error is not None or not live.bound:
            err = live.bind_error or "socket not bound"
            return self._h(
                Level.FAIL, "cannot listen", f"port {port} in use or unavailable ({err})"
            )
        now = self._clock()
        if live.last_datagram_ns is None:
            return self._h(Level.FAIL, "waiting for Studio", hint)
        age = now - live.last_datagram_ns
        if age > _FAIL_NS:
            return self._h(Level.FAIL, f"no packets for {int(age / 1e9)} s", hint)
        right = live.last_right_frame_ns
        if right is None:
            return self._h(
                Level.WARN, "no right hand", "Packets arrive but carry no right hand data."
            )
        if now - right <= _OK_NS:
            return self._h(Level.OK, f"streaming {live.datagram_hz:.0f} Hz")
        return self._h(Level.WARN, "glove stream stalling", "Right hand data is arriving late.")

    def reconnect(self) -> None:
        if self._source is None:
            return
        self._source.stop()
        self._source.start()
