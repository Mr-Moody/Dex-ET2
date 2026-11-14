"""UNO Q STM32U585 console bridge (T-131). No hardware: the firmware is emulated by SoftwareMcu."""

from __future__ import annotations

import pytest

from teleop.atc.types import BayState, ConsoleInputState
from teleop.unoq.mcu_bridge import (
    BayButton,
    Clutch,
    Estop,
    FaultRigLatched,
    Heartbeat,
    McuConsoleBridge,
    SoftwareMcu,
    console_state_from_dict,
    console_state_to_dict,
    format_bay_state,
    parse_mcu_line,
)


class FakeClock:
    def __init__(self) -> None:
        self.ns = 1_000_000_000

    def __call__(self) -> int:
        return self.ns

    def advance(self, s: float) -> None:
        self.ns += int(s * 1e9)


class FakeSerial:
    """pyserial-shaped transport: readline() returns b"" when no line is waiting."""

    def __init__(self, lines: list[str] | None = None) -> None:
        self.rx = [line.encode() + b"\n" for line in lines or []]
        self.tx: list[str] = []

    def push(self, *lines: str) -> None:
        self.rx.extend(line.encode() + b"\n" for line in lines)

    def readline(self) -> bytes:
        return self.rx.pop(0) if self.rx else b""

    def write(self, data: bytes) -> int:
        self.tx.append(data.decode())
        return len(data)


def _bridge(clock: FakeClock, mcu: SoftwareMcu) -> McuConsoleBridge:
    return McuConsoleBridge(mcu, clock=clock)


# ---------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("line", "event"),
    [
        ("BTN:BAY:3", BayButton(3)),
        ("CLUTCH:1", Clutch(True)),
        ("CLUTCH:0\r", Clutch(False)),
        ("ESTOP:1", Estop(True)),
        ("FAULT_RIG:LATCHED", FaultRigLatched()),
        ("HB:2:1:0:1", Heartbeat(2, True, False, True)),
        ("HB:0:0:0:0", Heartbeat(None, False, False, False)),
    ],
)
def test_parse_valid_lines(line: str, event: object):
    assert parse_mcu_line(line) == event


@pytest.mark.parametrize(
    "line", ["", "BTN:BAY:5", "BTN:BAY:x", "CLUTCH:2", "HB:1:1", "DEBUG hello", "FAULT_RIG:"]
)
def test_parse_rejects_garbage(line: str):
    assert parse_mcu_line(line) is None


# ---------------------------------------------------------------- fault rig


def test_fault_rig_latches_once_and_ignores_toggles_until_reset_cmd():
    clock = FakeClock()
    mcu = SoftwareMcu(clock=clock)
    bridge = _bridge(clock, mcu)

    # Judge raises the flap, then flicks the switch repeatedly.
    for closed in (True, False, True, False, True):
        mcu.set_fault_rig(closed)
    bridge.pump()
    assert bridge.state().fault_rig_latched
    assert bridge.take_fault_rig_edge() is True
    assert bridge.take_fault_rig_edge() is False  # one arm_flap_fault() per pallet cycle
    assert mcu.emitted.count("FAULT_RIG:LATCHED") == 1

    # A duplicate LATCHED line (glitch) must not re-arm the bay either.
    bridge.handle_line("FAULT_RIG:LATCHED")
    assert bridge.take_fault_rig_edge() is False

    bridge.reset_rig()
    bridge.pump()
    assert "CMD:RESET_RIG" in mcu.received
    assert not bridge.state().fault_rig_latched

    mcu.set_fault_rig(False)
    mcu.set_fault_rig(True)  # next pallet cycle: a fresh rising edge latches again
    bridge.pump()
    assert bridge.take_fault_rig_edge() is True
    assert mcu.emitted.count("FAULT_RIG:LATCHED") == 2


# ---------------------------------------------------------------- clutch, bays, safety


def test_clutch_and_bay_select_transitions_parsed():
    clock = FakeClock()
    mcu = SoftwareMcu(clock=clock)
    bridge = _bridge(clock, mcu)
    assert bridge.state() == ConsoleInputState()  # nothing heard yet: link down, safe

    mcu.press_bay(2)
    mcu.set_clutch(True)
    bridge.pump()
    s = bridge.state()
    assert s.selected_bay == 2 and s.clutch and not s.estop
    assert s.timestamp_ns == clock.ns

    mcu.press_bay(4)
    mcu.set_clutch(False)
    bridge.pump()
    assert bridge.state().selected_bay == 4 and not bridge.state().clutch
    assert [type(e) for e in bridge.drain_events()] == [BayButton, Clutch, BayButton, Clutch]
    assert bridge.drain_events() == []


def test_estop_overrides_clutch():
    clock = FakeClock()
    mcu = SoftwareMcu(clock=clock)
    bridge = _bridge(clock, mcu)
    mcu.set_clutch(True)
    mcu.set_estop(True)
    bridge.pump()
    s = bridge.state()
    assert s.estop and not s.clutch  # dead-man: no motion while E-stop is active


def test_stale_heartbeat_releases_clutch():
    clock = FakeClock()
    mcu = SoftwareMcu(clock=clock)
    bridge = _bridge(clock, mcu)
    mcu.set_clutch(True)
    bridge.pump()
    assert bridge.link_ok and bridge.state().clutch

    clock.advance(0.04)
    bridge.pump()  # SoftwareMcu heartbeats every 50 ms while polled
    clock.advance(0.04)
    bridge.pump()
    assert bridge.link_ok and bridge.state().clutch

    clock.advance(1.0)  # link lost (cable pulled, MCU crashed): fail safe
    assert not bridge.link_ok
    assert not bridge.state().clutch


def test_heartbeat_resyncs_missed_edges_but_never_latches_rig():
    clock = FakeClock()
    serial = FakeSerial(["HB:3:1:0:1"])
    bridge = McuConsoleBridge(serial, clock=clock)
    bridge.pump()
    s = bridge.state()
    assert s.selected_bay == 3 and s.clutch
    assert not s.fault_rig_latched  # only FAULT_RIG:LATCHED latches
    assert bridge.take_fault_rig_edge() is False


# ---------------------------------------------------------------- LED matrix commands


def test_led_state_commands_formatted_per_bay():
    assert format_bay_state(1, BayState.NOMINAL) == "CMD:BAY_STATE:1:NOMINAL"
    assert format_bay_state(2, BayState.STALLED) == "CMD:BAY_STATE:2:STALLED"
    assert format_bay_state(3, BayState.QUEUED) == "CMD:BAY_STATE:3:QUEUED"
    assert format_bay_state(4, BayState.RESCUE) == "CMD:BAY_STATE:4:RESCUE"
    assert format_bay_state(1, BayState.VALIDATING) == "CMD:BAY_STATE:1:VALIDATING"
    assert format_bay_state(2, BayState.AUTONOMOUS_SKILL) == "CMD:BAY_STATE:2:AUTO"
    assert format_bay_state(2, BayState.OOD_ESCALATED) == "CMD:BAY_STATE:2:OOD"
    with pytest.raises(ValueError):
        format_bay_state(5, BayState.NOMINAL)

    serial = FakeSerial()
    bridge = McuConsoleBridge(serial, clock=FakeClock())
    bridge.set_bay_state(1, BayState.STALLED)
    bridge.set_bay_state(1, BayState.STALLED)  # unchanged: not resent at 60 Hz
    bridge.set_bay_state(2, BayState.AUTONOMOUS_SKILL)
    assert serial.tx == ["CMD:BAY_STATE:1:STALLED\n", "CMD:BAY_STATE:2:AUTO\n"]
    bridge.resend_bay_states()
    assert serial.tx[-2:] == serial.tx[:2]


def test_software_mcu_mirrors_led_states():
    clock = FakeClock()
    mcu = SoftwareMcu(clock=clock)
    bridge = _bridge(clock, mcu)
    bridge.set_bay_state(3, BayState.VALIDATING)
    assert mcu.led_states == {3: "VALIDATING"}


def test_console_state_wire_roundtrip():
    s = ConsoleInputState(selected_bay=2, clutch=True, estop=False, fault_rig_latched=True)
    assert console_state_from_dict(console_state_to_dict(s)) == s


class _ClosableMcu(SoftwareMcu):
    """A serial port that refuses writes and reads once closed, like pyserial."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def readline(self) -> bytes:
        if self.closed:
            raise OSError("port closed")
        return super().readline()

    def write(self, data: bytes, /) -> int:
        if self.closed:
            raise OSError("port closed")
        return super().write(data)


def test_reopen_switches_transport_and_resends_leds() -> None:
    old, new = _ClosableMcu(), _ClosableMcu()
    bridge = McuConsoleBridge(old)
    bridge.set_bay_state(1, BayState.STALLED)
    bridge.reopen(new)
    assert old.closed and not new.closed
    assert new.led_states == {1: "STALLED"}
    bridge.pump()
    assert bridge.last_heard_ns is not None  # reads come from the new transport


def test_reopen_during_led_writes_never_raises() -> None:
    import threading

    bridge = McuConsoleBridge(_ClosableMcu())
    bridge.start()
    errors: list[BaseException] = []
    done = threading.Event()

    def writer() -> None:
        states = (BayState.STALLED, BayState.NOMINAL)
        i = 0
        while not done.is_set():
            try:
                bridge.set_bay_state(1, states[i % 2])
            except BaseException as exc:
                errors.append(exc)
            i += 1

    th = threading.Thread(target=writer)
    th.start()
    try:
        for _ in range(50):
            bridge.reopen(_ClosableMcu())
    finally:
        done.set()
        th.join()
        bridge.close()
    assert errors == []
