"""Control panel (UNO Q) health check: board link plus physical-button heartbeat (booth health).

Its own thread probes the board every ``ping_period_s`` on a separate connection, so the
coordinator's loop-thread requests never share a socket with it. Two consecutive misses swap the
:class:`~teleop.unoq.switch.SwitchableUnoQ` to the loopback node proactively, so the next stall is
classified locally instead of waiting out a dead board. While on loopback it retries the board
every ``retry_period_s`` and swaps it back in on success.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import replace

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


class PanelCheck:
    """``status()`` reads cached probe results only; ``step()``/``reconnect()`` do the I/O.

    ``clock`` must be the clock of ``local_bridge`` when one is given (both default to
    ``time.monotonic_ns``).
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
                self._probe_board()
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

    def _probe_board(self) -> None:
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
            return
        self._misses = 0
        with self._lock:
            self._hb_age_ms, self._hb_at_ns = age, self._clock()

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
        return None if age is None else age + (self._clock() - at) / 1e6

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
        age = self._buttons_age_ms()
        if age is not None and age < HEARTBEAT_OK_MS:
            return Health(
                self.name,
                Level.OK,
                f"UNO Q {self._host}, buttons live",
                f"MCU heartbeat {age:.0f} ms ago",
            )
        return Health(
            self.name,
            Level.WARN,
            f"UNO Q {self._host}, buttons not heard",
            SILENT_DETAIL,
            can_reconnect=True,
        )
