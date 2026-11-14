"""Bridge to the UNO Q STM32U585 operator console (T-131, spec 5.2).

The MCU sketch (``firmware/unoq_console/unoq_console.ino``) speaks newline-terminated text:

MCU -> MPU (edge events, plus a 20 Hz heartbeat)::

    BTN:BAY:<1-4>            bay-select button pressed
    CLUTCH:<0|1>             hold-to-control dead-man's clutch
    ESTOP:<0|1>              E-stop active
    FAULT_RIG:LATCHED        judge fault rig rising edge (once, until CMD:RESET_RIG)
    HB:<bay 0-4>:<clutch>:<estop>:<rig>   heartbeat / full-state resync (0 = no bay)

MPU -> MCU::

    CMD:BAY_STATE:<1-4>:<NOMINAL|STALLED|QUEUED|RESCUE|VALIDATING|AUTO|OOD>
    CMD:RESET_RIG

``AUTO`` (autonomous skill replay) extends the spec's list so the matrix can show Bay 2
replaying a fleet skill. On the UNO Q the MCU-MPU UART belongs to ``arduino-router``, so the
sketch carries these lines as ``Bridge.notify("console_line", line)`` and serves
``console_cmd``; :class:`RouterBridgeTransport` adapts that to the same line interface.
Any pyserial-like object works too, and :class:`SoftwareMcu` emulates the sketch for tests
and the ``teleop atc`` keyboard mirror.

Safety: the effective clutch is released while E-stop is active or when nothing has been
heard from the MCU for ``heartbeat_timeout_s``.

Pure stdlib (no mujoco, numpy or pyserial) so it runs unmodified on the UNO Q Linux side.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol

from teleop.atc.types import N_BAYS, BayState, ConsoleInputState

log = logging.getLogger(__name__)

Clock = Callable[[], int]

HEARTBEAT_PERIOD_S = 0.05  # 20 Hz, matches the sketch
HEARTBEAT_TIMEOUT_S = 0.25

LED_STATE_TOKENS: Mapping[BayState, str] = {
    BayState.NOMINAL: "NOMINAL",
    BayState.STALLED: "STALLED",
    BayState.QUEUED: "QUEUED",
    BayState.RESCUE: "RESCUE",
    BayState.VALIDATING: "VALIDATING",
    BayState.AUTONOMOUS_SKILL: "AUTO",
    BayState.OOD_ESCALATED: "OOD",
}
RESET_RIG_CMD = "CMD:RESET_RIG"


class SerialLike(Protocol):
    """Subset of ``serial.Serial``: ``readline()`` returns ``b""`` when no line is waiting."""

    def readline(self) -> bytes: ...

    def write(self, data: bytes, /) -> int | None: ...


# ---------------------------------------------------------------- events


@dataclass(frozen=True, slots=True)
class BayButton:
    bay: int


@dataclass(frozen=True, slots=True)
class Clutch:
    pressed: bool


@dataclass(frozen=True, slots=True)
class Estop:
    active: bool


@dataclass(frozen=True, slots=True)
class FaultRigLatched:
    pass


@dataclass(frozen=True, slots=True)
class Heartbeat:
    selected_bay: int | None
    clutch: bool
    estop: bool
    fault_rig_latched: bool


McuEvent = BayButton | Clutch | Estop | FaultRigLatched | Heartbeat

_BITS = {"0": False, "1": True}


def _bay(token: str, allow_zero: bool = False) -> int | None:
    if not token.isdigit():
        raise ValueError(token)
    bay = int(token)
    if allow_zero and bay == 0:
        return None
    if not 1 <= bay <= N_BAYS:
        raise ValueError(token)
    return bay


def parse_mcu_line(line: str) -> McuEvent | None:
    """Parse one MCU line; returns None for anything malformed or unknown (e.g. debug text)."""
    parts = line.strip().split(":")
    try:
        match parts:
            case ["BTN", "BAY", bay]:
                b = _bay(bay)
                return BayButton(b) if b is not None else None
            case ["CLUTCH", bit]:
                return Clutch(_BITS[bit])
            case ["ESTOP", bit]:
                return Estop(_BITS[bit])
            case ["FAULT_RIG", "LATCHED"]:
                return FaultRigLatched()
            case ["HB", bay, clutch, estop, rig]:
                return Heartbeat(
                    _bay(bay, allow_zero=True), _BITS[clutch], _BITS[estop], _BITS[rig]
                )
            case _:
                return None
    except (KeyError, ValueError):
        return None


def format_bay_state(bay_id: int, state: BayState) -> str:
    if not 1 <= bay_id <= N_BAYS:
        raise ValueError(f"bay_id must be in 1..{N_BAYS}, got {bay_id}")
    return f"CMD:BAY_STATE:{bay_id}:{LED_STATE_TOKENS[state]}"


def console_state_to_dict(state: ConsoleInputState) -> dict[str, Any]:
    """JSON-safe form for the host <-> UNO Q ``console_state`` op (T-130 server)."""
    return {
        "selected_bay": state.selected_bay,
        "clutch": state.clutch,
        "estop": state.estop,
        "fault_rig_latched": state.fault_rig_latched,
        "timestamp_ns": state.timestamp_ns,
    }


def console_state_from_dict(raw: Mapping[str, Any]) -> ConsoleInputState:
    bay = raw.get("selected_bay")
    return ConsoleInputState(
        selected_bay=None if bay is None else int(bay),
        clutch=bool(raw.get("clutch", False)),
        estop=bool(raw.get("estop", False)),
        fault_rig_latched=bool(raw.get("fault_rig_latched", False)),
        timestamp_ns=int(raw.get("timestamp_ns", 0)),
    )


# ---------------------------------------------------------------- bridge


class McuConsoleBridge:
    """Tracks console state from MCU lines and sends per-bay LED states back.

    Either call :meth:`pump` from your own loop (non-blocking transports) or :meth:`start` a
    reader thread (blocking ``readline`` with a timeout, e.g. pyserial ``timeout=0.05``).
    """

    def __init__(
        self,
        transport: SerialLike,
        clock: Clock = time.monotonic_ns,
        heartbeat_timeout_s: float = HEARTBEAT_TIMEOUT_S,
    ) -> None:
        self._transport = transport
        self._clock = clock
        self._timeout_ns = int(heartbeat_timeout_s * 1e9)
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._raw: ConsoleInputState = ConsoleInputState()
        self._last_rx_ns: int | None = None
        self._rig_edge_pending = False
        self._events: list[McuEvent] = []
        self._led: dict[int, str] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.bad_lines = 0

    # -- inbound

    def handle_line(self, line: str) -> McuEvent | None:
        event = parse_mcu_line(line)
        now = self._clock()
        with self._lock:
            if event is None:
                if line.strip():
                    self.bad_lines += 1
                return None
            self._last_rx_ns = now
            s = self._raw
            match event:
                case BayButton(bay):
                    self._raw = replace(s, selected_bay=bay, timestamp_ns=now)
                case Clutch(pressed):
                    self._raw = replace(s, clutch=pressed, timestamp_ns=now)
                case Estop(active):
                    self._raw = replace(s, estop=active, timestamp_ns=now)
                case FaultRigLatched():
                    if not s.fault_rig_latched:  # one-way: duplicates ignored until reset
                        self._rig_edge_pending = True
                        self._raw = replace(s, fault_rig_latched=True, timestamp_ns=now)
                case Heartbeat(bay, clutch, estop, _rig):
                    # Resync levels a lost edge could have missed. The rig is never latched
                    # from a heartbeat: one sent before CMD:RESET_RIG arrived would re-arm it.
                    if (s.selected_bay, s.clutch, s.estop) != (bay, clutch, estop):
                        self._raw = replace(
                            s,
                            selected_bay=bay if bay is not None else s.selected_bay,
                            clutch=clutch,
                            estop=estop,
                            timestamp_ns=now,
                        )
                    return event
            self._events.append(event)
            return event

    def pump(self, max_lines: int = 256) -> int:
        """Read and handle every waiting line (up to ``max_lines``); returns lines handled."""
        n = 0
        while n < max_lines:
            raw = self._transport.readline()
            if not raw:
                break
            self.handle_line(raw.decode("ascii", errors="replace"))
            n += 1
        return n

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, name="mcu-console", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        close = getattr(self._transport, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.pump() == 0:
                    time.sleep(0.001)
            except Exception:
                log.exception("mcu console: transport read failed")
                time.sleep(0.1)

    # -- state

    @property
    def last_heard_ns(self) -> int | None:
        """Monotonic ns (this bridge's clock) of the last valid MCU line, None before any."""
        with self._lock:
            return self._last_rx_ns

    @property
    def link_ok(self) -> bool:
        with self._lock:
            last = self._last_rx_ns
        return last is not None and self._clock() - last <= self._timeout_ns

    def state(self) -> ConsoleInputState:
        """Effective console state: clutch forced off on E-stop or a stale link."""
        link = self.link_ok
        with self._lock:
            s = self._raw
        if s.clutch and (s.estop or not link):
            return replace(s, clutch=False)
        return s

    def take_fault_rig_edge(self) -> bool:
        """True exactly once per latch: call ``arm_flap_fault()`` when this fires."""
        with self._lock:
            pending, self._rig_edge_pending = self._rig_edge_pending, False
        return pending

    def drain_events(self) -> list[McuEvent]:
        """Edge events since the last call (heartbeats excluded)."""
        with self._lock:
            events, self._events = self._events, []
        return events

    # -- outbound

    def _send(self, line: str) -> None:
        with self._write_lock:
            self._transport.write((line + "\n").encode("ascii"))

    def set_bay_state(self, bay_id: int, state: BayState) -> None:
        """Send an LED state, only when it changed (safe to call every 60 Hz tick)."""
        token = LED_STATE_TOKENS[state]
        if self._led.get(bay_id) == token:
            return
        line = format_bay_state(bay_id, state)
        self._led[bay_id] = token
        self._send(line)

    def resend_bay_states(self) -> None:
        """Push every known LED state again, e.g. after the MCU reboots."""
        for bay_id, token in sorted(self._led.items()):
            self._send(f"CMD:BAY_STATE:{bay_id}:{token}")

    def reset_rig(self) -> None:
        """Re-arm the physical fault rig for the next pallet cycle (sends CMD:RESET_RIG)."""
        with self._lock:
            self._raw = replace(self._raw, fault_rig_latched=False, timestamp_ns=self._clock())
            self._rig_edge_pending = False
        self._send(RESET_RIG_CMD)


# ---------------------------------------------------------------- transports


class SoftwareMcu:
    """In-process emulation of the console sketch, as a :class:`SerialLike`.

    Used by tests and as the keyboard mirror in ``teleop atc``: key handlers call
    :meth:`press_bay`, :meth:`set_clutch`, :meth:`set_estop` and :meth:`set_fault_rig`, and the
    same :class:`McuConsoleBridge` logic runs as with the real board.
    """

    def __init__(self, clock: Clock = time.monotonic_ns) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._out: list[str] = []
        self._last_hb_ns: int | None = None
        self.selected_bay: int | None = None
        self.clutch = False
        self.estop = False
        self._rig_closed = False
        self._rig_latched = False
        self.emitted: list[str] = []  # every non-heartbeat line sent, for inspection
        self.received: list[str] = []
        self.led_states: dict[int, str] = {}

    def _emit(self, line: str) -> None:
        self._out.append(line)
        if not line.startswith("HB:"):
            self.emitted.append(line)

    def press_bay(self, bay: int) -> None:
        if not 1 <= bay <= N_BAYS:
            raise ValueError(f"bay must be in 1..{N_BAYS}")
        with self._lock:
            self.selected_bay = bay
            self._emit(f"BTN:BAY:{bay}")

    def set_clutch(self, pressed: bool) -> None:
        with self._lock:
            if pressed != self.clutch:
                self.clutch = pressed
                self._emit(f"CLUTCH:{int(pressed)}")

    def toggle_clutch(self) -> None:
        self.set_clutch(not self.clutch)

    def set_estop(self, active: bool) -> None:
        with self._lock:
            if active != self.estop:
                self.estop = active
                self._emit(f"ESTOP:{int(active)}")

    def set_fault_rig(self, closed: bool) -> None:
        """Physical rig switch level. A rising edge latches once until ``CMD:RESET_RIG``."""
        with self._lock:
            rising = closed and not self._rig_closed
            self._rig_closed = closed
            if rising and not self._rig_latched:
                self._rig_latched = True
                self._emit("FAULT_RIG:LATCHED")

    def trigger_fault_rig(self) -> None:
        """Keyboard shortcut: one full press of the rig switch."""
        self.set_fault_rig(True)
        self.set_fault_rig(False)

    def readline(self) -> bytes:
        with self._lock:
            now = self._clock()
            if self._last_hb_ns is None or now - self._last_hb_ns >= HEARTBEAT_PERIOD_S * 1e9:
                self._last_hb_ns = now
                self._out.append(
                    f"HB:{self.selected_bay or 0}:{int(self.clutch)}:{int(self.estop)}:"
                    f"{int(self._rig_latched)}"
                )
            if not self._out:
                return b""
            return self._out.pop(0).encode("ascii") + b"\n"

    def write(self, data: bytes, /) -> int:
        with self._lock:
            for line in data.decode("ascii").splitlines():
                self.received.append(line)
                parts = line.split(":")
                match parts:
                    case ["CMD", "RESET_RIG"]:
                        self._rig_latched = False
                    case ["CMD", "BAY_STATE", bay, token] if bay.isdigit():
                        self.led_states[int(bay)] = token
                    case _:
                        pass
        return len(data)


class RouterBridgeTransport:
    """``arduino-router`` RPC (UNO Q App Lab ``arduino.app_utils.Bridge``) as a SerialLike.

    Only importable on the UNO Q Linux side. The sketch notifies ``console_line`` for every
    MCU line and provides ``console_cmd`` for MPU -> MCU commands.
    """

    def __init__(self, readline_timeout_s: float = 0.05) -> None:
        app_utils: Any = importlib.import_module("arduino.app_utils")
        self._bridge: Any = app_utils.Bridge
        self._timeout = readline_timeout_s
        self._rx: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._bridge.provide("console_line", self._on_line)

    def _on_line(self, line: str) -> None:
        self._rx.put(str(line))

    def readline(self) -> bytes:
        try:
            return self._rx.get(timeout=self._timeout).encode("ascii") + b"\n"
        except queue.Empty:
            return b""

    def write(self, data: bytes, /) -> int:
        for line in data.decode("ascii").splitlines():
            self._bridge.call("console_cmd", line)
        return len(data)


def open_console(serial_port: str | None = None) -> tuple[SerialLike, str]:
    """Pick a transport: explicit/``UNOQ_SERIAL_PORT`` pyserial port, then the UNO Q router
    bridge, then a :class:`SoftwareMcu` (keyboard mirror). Returns ``(transport, kind)``."""
    port = serial_port or os.environ.get("UNOQ_SERIAL_PORT") or None
    if port:
        serial: Any = importlib.import_module("serial")  # optional pyserial, not a dependency
        return serial.Serial(port, 115200, timeout=0.05), f"serial:{port}"
    try:
        return RouterBridgeTransport(), "router"
    except ImportError:
        pass
    log.info("mcu console: no UNO Q MCU found; using software console (keyboard mirror)")
    return SoftwareMcu(), "software"
