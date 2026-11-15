"""Physical UNO Q console buttons behave like the keyboard mirror (SPACE, E, 1-4).

No hardware: the "board" is a :class:`SoftwareMcu` (the sketch's emulation) behind its own
:class:`McuConsoleBridge` (the serial path), or a fake UNO Q client behind the network poller.
"""

from __future__ import annotations

import time
from typing import Any, cast

import pytest

from teleop.atc.coordinator import AtcCoordinator
from teleop.atc.stage import KEY_SPACE, EventTap, ReasonerTap, StageController, UnoQConsolePoller
from teleop.atc.types import BayState, ConsoleInputState
from teleop.unoq.client import UnoQClient
from teleop.unoq.mcu_bridge import McuConsoleBridge, SoftwareMcu


class _Coordinator:
    """Just enough of AtcCoordinator for the stage tick: two idle active bays."""

    time_s = 0.0

    def __init__(self) -> None:
        self.consoles: list[ConsoleInputState] = []

    def snapshots(self) -> tuple[()]:
        return ()

    def tick(self, console: ConsoleInputState, operator: object = None) -> tuple[()]:
        self.consoles.append(console)
        return ()

    def state(self, bay: int) -> BayState:
        return BayState.NOMINAL

    def outcome(self, bay: int) -> str:
        return ""

    def in_routine(self, bay: int) -> bool:
        return False

    def rescue_slot_holders(self) -> list[int]:
        return []


class _Events:
    def drain(self) -> list[Any]:
        return []


def _stage(*hardware: Any) -> StageController:
    return StageController(
        cast(AtcCoordinator, _Coordinator()),
        cast(EventTap, _Events()),
        cast(ReasonerTap, None),
        hardware=list(hardware),
    )


class Board:
    """The physical console on the serial path: buttons are levels, as on the real board."""

    def __init__(self) -> None:
        self.mcu = SoftwareMcu()
        self.bridge = McuConsoleBridge(self.mcu)

    def tick(self, ctl: StageController, n: int = 1) -> None:
        for _ in range(n):
            self.bridge.pump()  # the real bridge runs a reader thread
            ctl.tick()
            self.bridge.pump()

    def press_estop(self, ctl: StageController) -> None:
        self.mcu.set_estop(True)
        self.tick(ctl)
        self.mcu.set_estop(False)
        self.tick(ctl)


def _key(ctl: StageController, key: str | int) -> None:
    ctl.on_key(ord(key) if isinstance(key, str) else key)
    ctl.tick()


# ------------------------------------------------------------------ E-stop


def test_physical_estop_restores_a_bay_the_keyboard_cut() -> None:
    board = Board()
    ctl = _stage(board.bridge)
    _key(ctl, "1")
    _key(ctl, "E")
    board.tick(ctl, 2)
    assert ctl.console.powered_off == {1}
    assert board.mcu.powered_off == {1}  # the board's LEDs show the cut too

    board.mcu.press_bay(1)
    board.press_estop(ctl)
    board.tick(ctl, 2)
    assert ctl.console.powered_off == frozenset()  # one press restores, as E would


def test_keyboard_estop_restores_a_bay_the_physical_button_cut() -> None:
    board = Board()
    ctl = _stage(board.bridge)
    board.mcu.press_bay(2)
    board.press_estop(ctl)
    board.tick(ctl, 2)
    assert ctl.console.powered_off == {2}

    _key(ctl, "E")
    board.tick(ctl, 2)
    assert ctl.console.powered_off == frozenset()
    assert board.mcu.powered_off == set()


def test_keyboard_estop_acts_on_the_joystick_selected_bay() -> None:
    board = Board()
    ctl = _stage(board.bridge)
    board.mcu.press_bay(2)
    board.tick(ctl)
    assert ctl.console.selected_bay == 2

    _key(ctl, "E")
    board.tick(ctl, 2)
    assert ctl.console.powered_off == {2}  # not every robot (the keyboard had no bay)


def test_physical_estop_and_keyboard_estop_cut_the_same_way() -> None:
    board = Board()
    ctl = _stage(board.bridge)
    board.mcu.press_bay(1)
    board.press_estop(ctl)
    board.tick(ctl, 2)
    hw = ctl.console.powered_off

    ref = _stage()
    _key(ref, "1")
    _key(ref, "E")
    assert hw == ref.console.powered_off == {1}


# ------------------------------------------------------------------ clutch


def test_keyboard_clutch_drives_the_joystick_selected_bay() -> None:
    board = Board()
    ctl = _stage(board.bridge)
    board.mcu.press_bay(2)
    board.tick(ctl)
    _key(ctl, KEY_SPACE)
    assert ctl.console.clutch and ctl.console.selected_bay == 2


# ------------------------------------------------------------------ operator feedback


def test_physical_button_presses_reach_the_action_log_like_keys() -> None:
    board = Board()
    ctl = _stage(board.bridge)
    board.tick(ctl)  # the console is installed and idle
    board.mcu.press_bay(1)
    board.tick(ctl)
    board.mcu.set_clutch(True)
    board.tick(ctl)
    board.mcu.set_clutch(False)
    board.tick(ctl)
    board.press_estop(ctl)
    board.tick(ctl)
    texts = [n.text for n in ctl.actions]
    assert texts == [
        "Bay 1 selected",
        "Clutch engaged, but no bay is waiting for rescue",
        "Clutch released",
        "E-stop: Bay 1 robot powered off. Press it again to restore",
    ]
    assert {n.key for n in ctl.actions} <= {"Joystick", "Clutch button", "E-stop button"}


def test_keyboard_presses_are_not_logged_twice() -> None:
    board = Board()
    ctl = _stage(board.bridge)
    _key(ctl, "1")
    _key(ctl, KEY_SPACE)
    _key(ctl, "E")
    board.tick(ctl, 3)
    assert [n.key for n in ctl.actions] == ["1", "SPACE", "E"]


# ------------------------------------------------------------------ network poller


class _Clock:
    def __init__(self, ns: int) -> None:
        self.ns = ns

    def __call__(self) -> int:
        return self.ns


class _BoardClient:
    """Fake UNO Q client: console states stamped with the board's own monotonic clock."""

    def __init__(self) -> None:
        self.state = ConsoleInputState()
        self.calls: list[str] = []
        self.power_masks: list[frozenset[int]] = []

    def console_status(self) -> tuple[ConsoleInputState, int | None]:
        self.calls.append("console_status")
        return self.state, 10

    def set_bay_state(self, bay_id: int, state: BayState) -> None:
        self.calls.append("set_bay_state")

    def set_power_mask(self, off: frozenset[int]) -> None:
        self.calls.append("set_power_mask")
        self.power_masks.append(off)

    def close(self) -> None:
        pass


def _poller(client: _BoardClient, clock: _Clock) -> UnoQConsolePoller:
    return UnoQConsolePoller(cast(UnoQClient, client), period_s=3600.0, clock=clock)


def test_poller_restamps_board_changes_on_the_laptop_clock() -> None:
    clock = _Clock(5_000_000_000_000)  # laptop up for longer than the board
    client = _BoardClient()
    poller = _poller(client, clock)
    try:
        client.state = ConsoleInputState(selected_bay=2, timestamp_ns=1_000)  # board clock
        poller.poll_once()
        assert poller.state().timestamp_ns == clock.ns
        clock.ns += 10**9
        poller.poll_once()  # unchanged: keeps the time of the change
        assert poller.state().timestamp_ns == clock.ns - 10**9
    finally:
        poller.close()


def test_joystick_selection_beats_an_older_keyboard_selection_across_clocks() -> None:
    client = _BoardClient()
    later = _Clock(0)
    poller = UnoQConsolePoller(cast(UnoQClient, client), period_s=3600.0, clock=later)
    try:
        ctl = _stage(poller)
        _key(ctl, "1")
        later.ns = time.monotonic_ns() + 10**9  # the laptop clock, a second after the key
        client.state = ConsoleInputState(selected_bay=2, timestamp_ns=1)  # board just booted
        poller.poll_once()
        ctl.tick()
        assert ctl.console.selected_bay == 2
    finally:
        poller.close()


def test_poller_polls_the_buttons_before_sending_led_states() -> None:
    clock = _Clock(10**12)
    client = _BoardClient()
    poller = _poller(client, clock)
    try:
        for bay in (1, 2, 3, 4):
            poller.set_bay_state(bay, BayState.STALLED)
        poller.poll_once()
        assert client.calls[0] == "console_status"
    finally:
        poller.close()


def test_held_clutch_survives_one_late_network_poll() -> None:
    clock = _Clock(10**12)
    client = _BoardClient()
    poller = _poller(client, clock)
    try:
        client.state = ConsoleInputState(selected_bay=1, clutch=True)
        poller.poll_once()
        clock.ns += 300_000_000  # one TCP retransmit / Wi-Fi power-save gap
        assert poller.state().clutch
        clock.ns += 400_000_000  # a dead link still releases the dead-man
        assert not poller.state().clutch
    finally:
        poller.close()


def test_poller_forwards_the_power_mask_to_the_board() -> None:
    clock = _Clock(10**12)
    client = _BoardClient()
    poller = _poller(client, clock)
    try:
        poller.set_power_mask(frozenset({3}))
        poller.poll_once()
        assert client.power_masks == [frozenset({3})]
        poller.poll_once()
        assert client.power_masks == [frozenset({3})]  # sent once
    finally:
        poller.close()


class _DeafBoard:
    """A board service without the power-mask op: it never echoes a pushed mask."""

    def __init__(self) -> None:
        self.s = ConsoleInputState()
        self.pushed: list[frozenset[int]] = []

    def state(self) -> ConsoleInputState:
        return self.s

    def take_fault_rig_edge(self) -> bool:
        return False

    def set_bay_state(self, bay_id: int, state: BayState) -> None:
        pass

    def set_power_mask(self, off: frozenset[int]) -> None:
        self.pushed.append(off)

    def reset_rig(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_a_board_that_never_echoes_does_not_undo_a_keyboard_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import teleop.atc.stage as stage

    monkeypatch.setattr(stage, "POWER_ECHO_TIMEOUT_NS", 0)  # give up waiting at once
    deaf = _DeafBoard()
    ctl = _stage(deaf)
    _key(ctl, "1")
    _key(ctl, "E")
    ctl.tick()
    ctl.tick()
    assert ctl.console.powered_off == {1}
    assert deaf.pushed and deaf.pushed[0] == {1}


def test_a_button_press_on_another_bay_counts_while_a_push_is_in_flight() -> None:
    deaf = _DeafBoard()
    ctl = _stage(deaf)
    _key(ctl, "1")
    _key(ctl, "E")  # pushes {1}; the board has not echoed it yet
    deaf.s = ConsoleInputState(selected_bay=2, powered_off=frozenset({2}))  # its own press
    ctl.tick()
    assert ctl.console.powered_off == {1, 2}


def test_physical_clutch_drives_keyboard_selected_bay_without_switching_bay() -> None:
    client = _BoardClient()
    clock = _Clock(time.monotonic_ns())
    poller = _poller(client, clock)
    try:
        ctl = _stage(poller)
        client.state = ConsoleInputState(selected_bay=2)
        poller.poll_once()
        ctl.tick()
        assert ctl.console.selected_bay == 2

        _key(ctl, "1")
        ctl.tick()
        assert ctl.console.selected_bay == 1

        clock.ns = time.monotonic_ns() + 10**9
        client.state = ConsoleInputState(selected_bay=2, clutch=True)
        poller.poll_once()
        ctl.tick()
        assert ctl.console.selected_bay == 1
        assert ctl.console.clutch

        clock.ns += 10**8
        client.state = ConsoleInputState(selected_bay=2, clutch=False)
        poller.poll_once()
        ctl.tick()
        assert ctl.console.selected_bay == 1
        assert not ctl.console.clutch

        clock.ns += 10**8
        client.state = ConsoleInputState(selected_bay=3, clutch=False)
        poller.poll_once()
        ctl.tick()
        assert ctl.console.selected_bay == 3
    finally:
        poller.close()
