"""Control panel health check: proactive board ping, auto retry, buttons heartbeat (no hardware)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from teleop.atc.types import BayState, ConsoleInputState, StallSignature
from teleop.health.model import Level
from teleop.health.panel import PanelCheck
from teleop.unoq import protocol as P
from teleop.unoq.client import UnoQClient
from teleop.unoq.mcu_bridge import McuConsoleBridge, SoftwareMcu
from teleop.unoq.server import UnoQServer, console_to_dict
from teleop.unoq.switch import SwitchableUnoQ, start_loopback

Z = StallSignature(0.035, 4.0, 8.0, 2.0)


class FakeClock:
    def __init__(self) -> None:
        self.ns = 0

    def __call__(self) -> int:
        return self.ns

    def advance(self, seconds: float) -> None:
        self.ns += int(seconds * 1e9)


class _HeartbeatServer(UnoQServer):
    """A board whose MCU heartbeat age is fixed, so the level does not depend on wall time."""

    def __init__(self, age_ms: int | None) -> None:
        super().__init__(host="127.0.0.1", udp_port=0, tcp_port=0)
        self.age_ms = age_ms

    def _console_state(self, _: dict[str, Any]) -> dict[str, Any]:
        return P.ok(console=console_to_dict(ConsoleInputState()), mcu_heartbeat_age_ms=self.age_ms)


def _client(server: UnoQServer) -> UnoQClient:
    return UnoQClient("127.0.0.1", server.udp_port, server.tcp_port).connect()


def _panel(unoq: SwitchableUnoQ, server: UnoQServer, **kw: Any) -> PanelCheck:
    return PanelCheck(unoq, "127.0.0.1", server.udp_port, server.tcp_port, **kw)


def _wait(pred: Callable[[], bool], timeout_s: float = 2.0) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


@pytest.fixture
def board() -> Iterator[UnoQServer]:
    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0).start()
    try:
        yield server
    finally:
        server.stop()


def test_board_loss_swaps_before_next_request(board: UnoQServer) -> None:
    unoq = SwitchableUnoQ(_client(board))
    panel = _panel(unoq, board, clock=FakeClock())
    try:
        board.stop()
        panel.step()
        assert unoq.is_mock is False  # one miss is not enough
        panel.step()
        assert unoq.is_mock is True
        t0 = time.monotonic()
        unoq.classify_stall("ev_latch_v1", Z)
        assert time.monotonic() - t0 < 0.1
        assert panel.status().level == Level.FAIL
    finally:
        panel.stop()
        unoq.close()


def test_auto_retry_returns_to_board() -> None:
    board = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0).start()
    udp_port, tcp_port = board.udp_port, board.tcp_port
    board.stop()
    clock = FakeClock()
    unoq = SwitchableUnoQ(start_loopback())
    panel = PanelCheck(unoq, "127.0.0.1", udp_port, tcp_port, clock=clock)
    revived = UnoQServer(host="127.0.0.1", udp_port=udp_port, tcp_port=tcp_port).start()
    try:
        clock.advance(5.0)
        panel.step()
        assert unoq.is_mock is True  # not yet: retries every retry_period_s
        clock.advance(5.0)
        panel.step()
        assert unoq.is_mock is False
        assert unoq.host == "127.0.0.1"
        assert unoq.current.tcp_port == tcp_port
    finally:
        panel.stop()
        unoq.close()
        revived.stop()


def test_reconnect_tries_the_board_now(board: UnoQServer) -> None:
    unoq = SwitchableUnoQ(start_loopback())
    panel = _panel(unoq, board, clock=FakeClock())
    try:
        panel.reconnect()
        assert unoq.is_mock is False
        assert unoq.current.tcp_port == board.tcp_port
    finally:
        panel.stop()
        unoq.close()


def test_panel_levels() -> None:
    with _HeartbeatServer(120).start() as live:
        unoq = SwitchableUnoQ(_client(live))
        panel = _panel(unoq, live, clock=FakeClock())
        try:
            panel.step()
            h = panel.status()
            assert h.name == "panel"
            assert h.level == Level.OK
            assert "buttons live" in h.summary
            assert "127.0.0.1" in h.summary
        finally:
            panel.stop()
            unoq.close()

    with _HeartbeatServer(None).start() as silent:
        unoq = SwitchableUnoQ(_client(silent))
        panel = _panel(unoq, silent, clock=FakeClock())
        try:
            panel.step()
            h = panel.status()
            assert h.level == Level.WARN
            assert "console sketch" in h.detail
        finally:
            panel.stop()
            unoq.close()

    unoq = SwitchableUnoQ(start_loopback())
    panel = PanelCheck(unoq, "unoq.local", 1, 1, clock=FakeClock())
    try:
        h = panel.status()
        assert h.level == Level.FAIL
        assert "not found" in h.summary
        assert "keyboard works" in h.detail
        assert h.can_reconnect is True
    finally:
        panel.stop()
        unoq.close()


def test_stale_board_heartbeat_warns() -> None:
    clock = FakeClock()
    with _HeartbeatServer(120).start() as live:
        unoq = SwitchableUnoQ(_client(live))
        panel = _panel(unoq, live, clock=clock)
        try:
            panel.step()
            assert panel.status().level == Level.OK
            clock.advance(0.5)  # no fresh probe since: the heartbeat is now 620 ms old
            assert panel.status().level == Level.WARN
        finally:
            panel.stop()
            unoq.close()


def test_panel_uses_local_bridge_heartbeat(board: UnoQServer) -> None:
    mcu = SoftwareMcu()
    bridge = McuConsoleBridge(mcu)
    bridge.start()
    unoq = SwitchableUnoQ(_client(board))  # the board reports no MCU heartbeat at all
    panel = _panel(unoq, board, local_bridge=bridge)
    try:
        panel.step()
        assert _wait(lambda: panel.status().level == Level.OK)
        assert "buttons live" in panel.status().summary
    finally:
        panel.stop()
        bridge.close()
        unoq.close()


def test_start_stop_runs_the_thread(board: UnoQServer) -> None:
    unoq = SwitchableUnoQ(_client(board))
    panel = _panel(unoq, board, ping_period_s=0.01)
    try:
        board.stop()  # first: a connection accepted while stop() runs can outlive it
        panel.start()
        assert _wait(lambda: unoq.is_mock)
    finally:
        panel.stop()
        unoq.close()


def test_failed_serial_reopen_leaves_bridge_usable(
    board: UnoQServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(_: str | None) -> tuple[SoftwareMcu, str]:
        raise OSError("COM7 busy")

    monkeypatch.setattr("teleop.health.panel.open_console", broken)
    mcu = SoftwareMcu()
    bridge = McuConsoleBridge(mcu)
    bridge.start()
    unoq = SwitchableUnoQ(_client(board))
    panel = _panel(unoq, board, local_bridge=bridge, serial_port="COM7")
    try:
        panel.reconnect()
        assert "COM7 busy" in panel.status().detail
        bridge.set_bay_state(1, BayState.STALLED)  # still the live port
        assert mcu.led_states == {1: "STALLED"}
        assert _wait(lambda: panel.status().level == Level.OK)
    finally:
        panel.stop()
        bridge.close()
        unoq.close()


def test_serial_reopen_moves_bridge_to_new_port(
    board: UnoQServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    fresh = SoftwareMcu()
    monkeypatch.setattr("teleop.health.panel.open_console", lambda _: (fresh, "serial:COM7"))
    bridge = McuConsoleBridge(SoftwareMcu())
    bridge.set_bay_state(2, BayState.STALLED)
    unoq = SwitchableUnoQ(_client(board))
    panel = _panel(unoq, board, local_bridge=bridge, serial_port="COM7")
    try:
        panel.reconnect()
        assert fresh.led_states == {2: "STALLED"}  # LED states resent on the new port
        assert panel.status().detail.count("serial") == 0
    finally:
        panel.stop()
        bridge.close()
        unoq.close()


def test_concurrent_reconnects_are_serialised(
    board: UnoQServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    active, peak = [0], [0]
    guard = threading.Lock()

    def slow_open(_: str | None) -> tuple[SoftwareMcu, str]:
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)  # a second open of the same COM port would fail on Windows
        with guard:
            active[0] -= 1
        return SoftwareMcu(), "serial:COM7"

    monkeypatch.setattr("teleop.health.panel.open_console", slow_open)
    bridge = McuConsoleBridge(SoftwareMcu())
    unoq = SwitchableUnoQ(_client(board))
    panel = _panel(unoq, board, local_bridge=bridge, serial_port="COM7")
    try:
        threads = [threading.Thread(target=panel.reconnect) for _ in range(2)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert peak[0] == 1
    finally:
        panel.stop()
        bridge.close()
        unoq.close()
