"""Control panel (UNO Q) health check: board link plus physical-button heartbeat (booth health).

Its own thread probes the board every ``ping_period_s`` on a separate connection, so the
coordinator's loop-thread requests never share a socket with it. Two consecutive misses swap the
:class:`~teleop.unoq.switch.SwitchableUnoQ` to the loopback node proactively, so the next stall is
classified locally instead of waiting out a dead board. While on loopback it retries the board
every ``retry_period_s`` and swaps it back in on success.

The light only goes green when the laptop itself hears the buttons. With a laptop serial console
that is the bridge's last line; with a board console it is the laptop's own console poller (the
``buttons`` source), not the probe: the board can hear its MCU while the laptop's poller is gone
(its connection failed after a swap, or died with a board service restart). A missing or stale
poller is rebuilt from this thread through ``rebuild_consoles``, as on every Reconnect.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

from teleop.health.model import Health, Level
from teleop.unoq import protocol as P
from teleop.unoq.client import UnoQClient, UnoQError
from teleop.unoq.mcu_bridge import McuConsoleBridge, open_console
from teleop.unoq.switch import SwitchableUnoQ

logger = logging.getLogger(__name__)

MISSES_TO_FAIL = 2
HEARTBEAT_OK_MS = 500.0
SILENT_DETAIL = "board up but buttons not heard: is the console sketch running?"
MOCK_DETAIL = "using local mock; keyboard works"
NO_LINK_DETAIL = "board up but the laptop has no button connection to it; retrying"
LINK_LOST_DETAIL = "the laptop lost its button connection to the board; retrying"


class ButtonsLink(Protocol):
    """The laptop's board console poller, as the panel check reads it (never blocks)."""

    @property
    def heartbeat_age_ms(self) -> int | None: ...

    @property
    def last_ok_ns(self) -> int | None: ...


class PanelCheck:
    """``status()`` reads cached probe results only; ``step()``/``reconnect()`` do the I/O.

    ``clock`` must be the clock of ``local_bridge`` and of the ``buttons`` poller when given
    (all default to ``time.monotonic_ns``). ``buttons`` returns the installed board console
    poller (None when none is installed); ``rebuild_consoles`` reinstalls it and may block.
    """

    name = "panel"

    def __init__(
        self,
        unoq: SwitchableUnoQ,
        host: str,
        udp_port: int,
        tcp_port: int,
        *,
        ping_period_s: float = 1.0,
        ping_timeout_s: float = 0.3,
        retry_period_s: float = 10.0,
        clock: Callable[[], int] = time.monotonic_ns,
        local_bridge: McuConsoleBridge | None = None,
        serial_port: str | None = None,
        buttons: Callable[[], ButtonsLink | None] | None = None,
        rebuild_consoles: Callable[[], None] | None = None,
    ) -> None:
        self._unoq = unoq
        self._host = host
        self._udp_port = udp_port
        self._tcp_port = tcp_port
        self._ping_period_s = ping_period_s
        self._ping_timeout_s = ping_timeout_s
        self._retry_ns = int(retry_period_s * 1e9)
        self._clock = clock
        self._local_bridge = local_bridge
        self._serial_port = serial_port
        self._buttons = buttons
        self._rebuild_consoles = rebuild_consoles
        self._link_misses = 0
        # A probe result older than this means probes stopped succeeding (not just between them).
        self._probe_stale_ns = int((2 * ping_period_s + ping_timeout_s) * 1e9)
        self._io_lock = threading.Lock()  # serialises step() and reconnect()
        self._lock = threading.Lock()  # guards the cached heartbeat read by status()
        self._probe: UnoQClient | None = None
        self._misses = 0
        self._last_attempt_ns = clock()  # startup already tried the board once
        self._hb_age_ms: int | None = None
        self._hb_at_ns = 0
        self._serial_error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="health-panel", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._io_lock:
            self._close_probe()

    def _run(self) -> None:
        while not self._stop.wait(self._ping_period_s):
            try:
                self.step()
            except Exception:
                logger.exception("panel check step failed")

    # ------------------------------------------------------------------ I/O (never the loop)

    def step(self) -> None:
        """One thread iteration: probe the board, or retry it when due while on loopback."""
        with self._io_lock:
            if not self._unoq.is_mock:
                if self._probe_board():
                    self._check_button_link()
            elif self._clock() - self._last_attempt_ns >= self._retry_ns:
                self._try_board()

    def reconnect(self) -> None:
        """Reopen the local serial console (if any) and try the board now."""
        with self._io_lock:
            if self._local_bridge is not None:
                self._reopen_serial(self._local_bridge)
            if self._unoq.is_mock:
                self._try_board()
            else:
                self._close_probe()
                self._misses = 0
                self._probe_board()
                self._rebuild()  # a dead button poller is not fixed by re-probing alone

    def _reopen_serial(self, bridge: McuConsoleBridge) -> None:
        """Open the new transport first; only on success hand it to the live bridge."""
        try:
            transport, kind = open_console(self._serial_port)
        except Exception as exc:  # pyserial raises its own SerialException
            logger.warning("panel: cannot reopen the serial console (%s)", exc)
            with self._lock:
                self._serial_error = f"serial reopen failed: {exc}"
            return
        try:
            bridge.reopen(transport)  # the old bridge object stays the stage's console
        except Exception as exc:  # resending the LED states failed on the new port
            logger.warning("panel: serial console reopened but LED resend failed (%s)", exc)
        with self._lock:
            self._serial_error = ""
        logger.info("panel: console reopened (%s)", kind)

    def _close_probe(self) -> None:
        if self._probe is not None:
            with contextlib.suppress(Exception):
                self._probe.close()
            self._probe = None

    def _probe_board(self) -> bool:
        """``console_state`` on our own connection: proves the link and reads the heartbeat."""
        try:
            if self._probe is None:
                probe = UnoQClient(
                    self._host, self._udp_port, self._tcp_port, timeout_s=self._ping_timeout_s
                )
                self._probe = probe
                probe.connect()
            _, age = self._probe.console_status()
        except (OSError, UnoQError) as exc:
            self._close_probe()
            self._misses += 1
            with self._lock:
                self._hb_age_ms = None
            logger.debug("panel: ping %d missed (%s)", self._misses, exc)
            if self._misses >= MISSES_TO_FAIL:
                self._misses = 0
                self._last_attempt_ns = self._clock()
                self._unoq.swap_to_loopback()
            return False
        self._misses = 0
        with self._lock:
            self._hb_age_ms, self._hb_at_ns = age, self._clock()
        return True

    def _button_link_live(self, link: ButtonsLink | None) -> bool:
        if link is None:
            return False
        last = link.last_ok_ns
        return last is not None and self._clock() - last <= HEARTBEAT_OK_MS * 1e6

    def _check_button_link(self) -> None:
        """Board reachable: rebuild the laptop's button poller if it is missing or stale."""
        if self._local_bridge is not None or self._buttons is None:
            return
        if self._button_link_live(self._buttons()):
            self._link_misses = 0
            return
        self._link_misses += 1
        if self._link_misses >= MISSES_TO_FAIL:
            logger.warning("panel: button connection to the board lost; reconnecting it")
            self._rebuild()

    def _rebuild(self) -> None:
        self._link_misses = 0
        if self._rebuild_consoles is None or self._local_bridge is not None:
            return
        try:
            self._rebuild_consoles()
        except Exception:
            logger.warning("panel: rebuilding the board console failed", exc_info=True)

    def _try_board(self) -> None:
        self._last_attempt_ns = self._clock()
        client = UnoQClient(self._host, self._udp_port, self._tcp_port)
        try:
            client.connect(self._ping_timeout_s)
            client.request({"op": P.OP_PING}, timeout_s=self._ping_timeout_s)
        except (OSError, UnoQError) as exc:
            client.close()
            logger.debug("panel: board %s still unreachable (%s)", self._host, exc)
            return
        try:
            self._unoq.swap(client)  # replays the skills; closes the loopback node
        except (OSError, UnoQError) as exc:
            logger.warning("panel: board %s found but skill replay failed (%s)", self._host, exc)
            return
        self._misses = 0
        self._close_probe()
        logger.info("panel: UNO Q %s connected", self._host)

    # ------------------------------------------------------------------ status (non-blocking)

    def _buttons_age_ms(self) -> float | None:
        if self._local_bridge is not None:
            heard = self._local_bridge.last_heard_ns
            return None if heard is None else (self._clock() - heard) / 1e6
        with self._lock:
            age, at = self._hb_age_ms, self._hb_at_ns
        if age is None or self._clock() - at > self._probe_stale_ns:
            return None  # probes stopped succeeding: do not trust the last one
        # As the board reported it at probe time: adding the time since the probe would turn a
        # healthy heartbeat stale between two probes 1 s apart (OK/WARN flicker).
        return float(age)

    def status(self) -> Health:
        health = self._status()
        with self._lock:
            error = self._serial_error
        return replace(health, detail=f"{health.detail}; {error}") if error else health

    def _status(self) -> Health:
        if self._unoq.is_mock:
            return Health(
                self.name,
                Level.FAIL,
                f"UNO Q not found at {self._host}",
                MOCK_DETAIL,
                can_reconnect=True,
            )
        if self._local_bridge is None and self._buttons is not None:
            link = self._buttons()
            if link is None:
                return self._warn(f"UNO Q {self._host}, buttons not connected", NO_LINK_DETAIL)
            if not self._button_link_live(link):
                return self._warn(f"UNO Q {self._host}, buttons not heard", LINK_LOST_DETAIL)
            hb = link.heartbeat_age_ms
            age = None if hb is None else float(hb)
        else:
            age = self._buttons_age_ms()
        if age is not None and age < HEARTBEAT_OK_MS:
            return Health(
                self.name,
                Level.OK,
                f"UNO Q {self._host}, buttons live",
                f"MCU heartbeat {age:.0f} ms ago",
            )
        return self._warn(f"UNO Q {self._host}, buttons not heard", SILENT_DETAIL)

    def _warn(self, summary: str, detail: str) -> Health:
        return Health(self.name, Level.WARN, summary, detail, can_reconnect=True)
