"""Console joystick calibration and per-robot E-stop (T-216). No hardware: SoftwareMcu emulates."""

from __future__ import annotations

from pathlib import Path

import pytest

from teleop.atc.types import ConsoleInputState
from teleop.commands.unoq import describe_joy_map
from teleop.unoq import protocol as P
from teleop.unoq.mcu_bridge import (
    DEFAULT_JOY_MAP,
    Boot,
    Heartbeat,
    JoyCal,
    JoyMap,
    JoyRaw,
    McuConsoleBridge,
    Power,
    SoftwareMcu,
    StickDirection,
    bays_from_mask,
    console_state_from_dict,
    console_state_to_dict,
    is_joy_map,
    load_joy_map,
    neighbour_bay,
    parse_mcu_line,
    power_mask,
    save_joy_map,
)
from teleop.unoq.server import UnoQServer


class FakeClock:
    def __init__(self) -> None:
        self.ns = 1_000_000_000

    def __call__(self) -> int:
        return self.ns

    def advance(self, s: float) -> None:
        self.ns += int(s * 1e9)


def _rig() -> tuple[FakeClock, SoftwareMcu, McuConsoleBridge]:
    clock = FakeClock()
    mcu = SoftwareMcu(clock=clock)
    return clock, mcu, McuConsoleBridge(mcu, clock=clock)


# ---------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("line", "event"),
    [
        ("BOOT", Boot()),
        ("POWER:3:0", Power(3, False)),
        ("POWER:1:1", Power(1, True)),
        ("JOYMAP:2431", JoyMap("2431")),
        ("JOYCAL:START", JoyCal("start")),
        ("JOYCAL:2", JoyCal("step", bay=2)),
        ("JOYCAL:DUP:3", JoyCal("dup", bay=3)),
        ("JOYCAL:OK:1234", JoyCal("ok", code="1234")),
        ("JOYCAL:FAIL:TIMEOUT", JoyCal("fail", reason="TIMEOUT")),
        ("JOY:512:498", JoyRaw(512, 498)),
        ("HB:2:1:0:0:5", Heartbeat(2, True, False, False, frozenset({1, 3}))),
        ("HB:2:1:0:0", Heartbeat(2, True, False, False, None)),
    ],
)
def test_parse_new_lines(line: str, event: object) -> None:
    assert parse_mcu_line(line) == event


@pytest.mark.parametrize(
    "line", ["POWER:5:0", "POWER:1:2", "JOYMAP:1123", "JOYMAP:12345", "HB:1:0:0:0:16", "JOYCAL:9"]
)
def test_parse_rejects_bad_new_lines(line: str) -> None:
    assert parse_mcu_line(line) is None


def test_joy_map_and_mask_helpers() -> None:
    assert is_joy_map(DEFAULT_JOY_MAP) and is_joy_map("4321")
    assert not is_joy_map("1124") and not is_joy_map("123") and not is_joy_map("12a4")
    assert power_mask({1, 4}) == 9 and bays_from_mask(9) == frozenset({1, 4})
    assert describe_joy_map("2431") == "X+ -> right, X- -> down, Y+ -> up, Y- -> left"
    assert describe_joy_map(DEFAULT_JOY_MAP) == "X+ -> right, X- -> left, Y+ -> down, Y- -> up"


def test_console_dict_round_trip_carries_powered_off() -> None:
    s = ConsoleInputState(selected_bay=2, clutch=True, powered_off=frozenset({2, 4}))
    raw = console_state_to_dict(s)
    assert raw["powered_off"] == [2, 4]
    assert console_state_from_dict(raw) == s
    assert console_state_from_dict({"clutch": True}).powered_off == frozenset()


def test_console_state_rejects_bad_powered_off_bay() -> None:
    with pytest.raises(ValueError):
        ConsoleInputState(powered_off=frozenset({5}))


# ---------------------------------------------------------------- per-robot E-stop


def test_estop_press_toggles_power_of_the_selected_bay_only() -> None:
    _, mcu, bridge = _rig()
    mcu.press_bay(2)
    mcu.press_estop()
    bridge.pump()
    assert bridge.state().powered_off == frozenset({2})
    assert "POWER:2:0" in mcu.emitted

    mcu.press_bay(3)
    mcu.set_clutch(True)
    bridge.pump()
    assert bridge.state().clutch  # Bay 3 still has power: the clutch drives it

    mcu.press_bay(2)
    bridge.pump()
    assert not bridge.state().clutch  # a cut bay never takes the clutch

    mcu.press_estop()  # second press on Bay 2 restores it
    bridge.pump()
    assert bridge.state().powered_off == frozenset()
    assert bridge.state().clutch


def test_estop_with_no_bay_selected_cuts_then_restores_every_bay() -> None:
    _, mcu, bridge = _rig()
    mcu.press_estop()
    bridge.pump()
    assert bridge.state().powered_off == frozenset({1, 2, 3, 4})
    mcu.press_estop()
    bridge.pump()
    assert bridge.state().powered_off == frozenset()


def test_heartbeat_resyncs_a_lost_power_edge() -> None:
    _, _, bridge = _rig()
    bridge.handle_line("HB:1:0:0:0:4")  # POWER:3:0 was lost on the wire
    assert bridge.state().powered_off == frozenset({3})
    bridge.handle_line("HB:1:0:0:0")  # old firmware heartbeat: keeps what we know
    assert bridge.state().powered_off == frozenset({3})


def test_mcu_reboot_restores_power_mask_and_joystick_map() -> None:
    _, mcu, bridge = _rig()
    bridge.set_joy_map("1234")
    mcu.press_bay(4)
    mcu.press_estop()
    bridge.pump()
    assert bridge.state().powered_off == frozenset({4})

    mcu.reboot()  # MCU forgot both; its heartbeats now say "nothing cut"
    bridge.pump()
    assert "CMD:POWER_MASK:8" in mcu.received
    assert mcu.received.count("CMD:JOY_MAP:1234") == 2
    assert mcu.powered_off == {4} and mcu.joy_map == "1234"
    bridge.pump()
    assert bridge.state().powered_off == frozenset({4})  # never briefly re-powered


def test_restore_pending_ignores_stale_heartbeats() -> None:
    clock = FakeClock()
    sent: list[str] = []

    class Wire:
        def readline(self) -> bytes:
            return b""

        def write(self, data: bytes) -> int:
            sent.append(data.decode())
            return len(data)

    bridge = McuConsoleBridge(Wire(), clock=clock)
    bridge.handle_line("POWER:2:0")
    bridge.handle_line("BOOT")
    assert "CMD:POWER_MASK:2\n" in sent
    bridge.handle_line("HB:0:0:0:0:0")  # sent before the MCU applied the restore
    assert bridge.state().powered_off == frozenset({2})
    bridge.handle_line("HB:0:0:0:0:2")
    bridge.handle_line("HB:0:0:0:0:0")  # restore confirmed: a later 0 is a real change
    assert bridge.state().powered_off == frozenset()


# ---------------------------------------------------------------- joystick map


def test_joy_map_pushed_at_start_and_calibration_result_saved(tmp_path: Path) -> None:
    clock = FakeClock()
    mcu = SoftwareMcu(clock=clock)
    saved: list[str] = []
    bridge = McuConsoleBridge(mcu, clock=clock, joy_map="4321", on_joy_map=saved.append)
    assert mcu.received == ["CMD:JOY_MAP:4321"]

    bridge.start_joystick_calibration()
    assert "CMD:JOY_CAL" in mcu.received
    for line in ("JOYCAL:START", "JOYCAL:1", "JOYCAL:2", "JOYCAL:DUP:2", "JOYCAL:OK:3142"):
        bridge.handle_line(line)
    bridge.handle_line("JOYMAP:3142")
    status = bridge.joystick_status()
    assert status["map"] == "3142" and status["wanted"] == "3142"
    assert status["calibration"] == {"phase": "ok", "bay": None, "map": "3142", "reason": None}
    assert saved == ["3142"]

    with pytest.raises(ValueError):
        bridge.set_joy_map("1111")


def test_raw_stream_reported_in_status() -> None:
    _, mcu, bridge = _rig()
    bridge.stream_joystick_raw(True)
    assert "CMD:JOY_RAW:1" in mcu.received
    bridge.handle_line("JOY:900:120")
    assert bridge.joystick_status()["raw"] == [900, 120]
    assert bridge.drain_events() == []  # raw samples are not edge events


def test_joy_map_file_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "nested" / "map"
    monkeypatch.setenv("UNOQ_JOYMAP_FILE", str(path))
    assert load_joy_map() is None
    save_joy_map("1342")
    assert load_joy_map() == "1342"
    path.write_text("junk\n")
    assert load_joy_map() is None
    with pytest.raises(ValueError):
        save_joy_map("1")


# ---------------------------------------------------------------- edge service op


def test_server_joystick_op_drives_the_attached_bridge() -> None:
    server = UnoQServer()
    assert server.handle({"op": P.OP_JOYSTICK})["ok"] is False  # no MCU console

    _, mcu, bridge = _rig()
    server._bridge = bridge  # pyright: ignore[reportPrivateUsage]
    assert server.handle({"op": P.OP_JOYSTICK, "action": "calibrate"})["ok"]
    assert "CMD:JOY_CAL" in mcu.received
    reply = server.handle({"op": P.OP_JOYSTICK, "action": "set", "map": "2143"})
    assert reply["ok"] and reply["wanted"] == "2143"
    assert server.handle({"op": P.OP_JOYSTICK, "action": "set", "map": "22"})["ok"] is False
    assert server.handle({"op": P.OP_JOYSTICK, "action": "wiggle"})["ok"] is False
    assert server.handle({"op": P.OP_JOYSTICK, "action": "abort"})["ok"]
    assert "CMD:JOY_CAL_ABORT" in mcu.received


def test_console_state_op_carries_powered_off() -> None:
    server = UnoQServer()
    server.set_console_state(ConsoleInputState(selected_bay=1, powered_off=frozenset({3})))
    reply = server.handle({"op": P.OP_CONSOLE_STATE})
    assert reply["console"]["powered_off"] == [3]
    P.encode(reply)  # JSON-safe


# ---------------------------------------------------------------- keyboard / button parity (T-225)


def test_heartbeat_without_a_bay_does_not_restamp_every_beat() -> None:
    clock, _, bridge = _rig()
    bridge.handle_line("BTN:BAY:2")
    stamp = bridge.state().timestamp_ns
    clock.advance(0.05)
    bridge.handle_line("HB:0:0:0:0:0")  # MCU rebooted: no bay, we keep Bay 2
    clock.advance(0.05)
    bridge.handle_line("HB:0:0:0:0:0")
    assert bridge.state().selected_bay == 2
    assert bridge.state().timestamp_ns == stamp  # else this console always wins the merge


def test_set_power_mask_reaches_the_mcu_and_waits_for_its_echo() -> None:
    _, mcu, bridge = _rig()
    bridge.set_power_mask({1, 3})
    assert "CMD:POWER_MASK:5" in mcu.received and mcu.powered_off == {1, 3}
    assert bridge.state().powered_off == frozenset()  # until the MCU echoes it
    bridge.pump()
    assert bridge.state().powered_off == frozenset({1, 3})


def test_server_set_power_mask_op_drives_the_attached_bridge() -> None:
    mcu = SoftwareMcu()
    server = UnoQServer()
    server.attach_console(McuConsoleBridge(mcu))
    try:
        assert server.handle({"op": P.OP_SET_POWER_MASK, "powered_off": [2]})["ok"]
        assert mcu.powered_off == {2}
        assert server.handle({"op": P.OP_SET_POWER_MASK, "powered_off": [5]})["ok"] is False
        assert UnoQServer().handle({"op": P.OP_SET_POWER_MASK, "powered_off": []})["ok"]
    finally:
        server.stop()


def test_console_state_op_never_waits_for_the_main_lock() -> None:
    """A DMP fit holds the main lock; the clutch poll must still answer at once."""
    server = UnoQServer()
    server.set_console_state(ConsoleInputState(selected_bay=1, clutch=True))
    with server._lock:  # pyright: ignore[reportPrivateUsage]
        reply = server.handle({"op": P.OP_CONSOLE_STATE})
    assert reply["console"]["clutch"] is True


# ---------------------------------------------------------------- panel navigation (T-227)

_SKETCHES = [
    Path(__file__).resolve().parents[2] / "firmware" / "unoq_console" / "unoq_console.ino",
    Path(__file__).resolve().parents[2] / "firmware" / "unoq_app" / "sketch" / "sketch.ino",
]
L, R, U, D = StickDirection.LEFT, StickDirection.RIGHT, StickDirection.UP, StickDirection.DOWN


@pytest.mark.parametrize(
    ("bay", "direction", "expected"),
    [
        # Bay 1 | Bay 2 over Bay 3 | Bay 4; the grid edges stop the move
        (1, R, 2), (1, D, 3), (1, L, 1), (1, U, 1),
        (2, L, 1), (2, D, 4), (2, R, 2), (2, U, 2),
        (3, R, 4), (3, U, 1), (3, L, 3), (3, D, 3),
        (4, L, 3), (4, U, 2), (4, R, 4), (4, D, 4),
        # nothing selected yet: the move starts from Bay 1
        (None, R, 2), (None, D, 3), (None, L, 1), (None, U, 1),
    ],
)  # fmt: skip
def test_stick_moves_one_panel_on_the_camera_grid(
    bay: int | None, direction: StickDirection, expected: int
) -> None:
    assert neighbour_bay(bay, direction) == expected


def test_push_stick_navigates_through_the_bridge() -> None:
    _clock, mcu, bridge = _rig()
    path = [mcu.push_stick(d) for d in (R, D, L, L, U)]
    assert path == [2, 4, 3, 3, 1]
    bridge.pump()
    assert bridge.state().selected_bay == 1
    # every flick is announced, an edge bump too (it re-announces the selection)
    assert [ln for ln in mcu.emitted if ln.startswith("BTN:")] == [
        "BTN:BAY:2", "BTN:BAY:4", "BTN:BAY:3", "BTN:BAY:3", "BTN:BAY:1",
    ]  # fmt: skip


def test_estop_cuts_the_panel_the_stick_moved_to() -> None:
    _clock, mcu, bridge = _rig()
    mcu.push_stick(D)  # Bay 1 -> Bay 3
    mcu.press_estop()
    bridge.pump()
    assert bridge.state().powered_off == frozenset({3})


def test_sketch_copies_match_and_mirror_the_python_navigation() -> None:
    console, app = (p.read_text(encoding="utf-8") for p in _SKETCHES)
    assert console == app, "firmware/unoq_app/sketch/sketch.ino must stay a copy"
    assert "JOY_LEFT = 1, JOY_RIGHT = 2, JOY_UP = 3, JOY_DOWN = 4" in console
    assert [int(d) for d in DEFAULT_JOY_MAP] == [R, L, D, U]
    assert "joyMap[4] = {JOY_RIGHT, JOY_LEFT, JOY_DOWN, JOY_UP}" in console
    assert "moveSelection(selectedBay, joyMap[dir])" in console
    assert "selectedBay = joyMap[dir]" not in console  # no direct direction -> bay jumps left
