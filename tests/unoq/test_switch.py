"""Hot-swappable UNO Q client: skill replay, failover to loopback, console poller LED forwarding."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import cast

import pytest

from teleop.atc.coordinator import AtcCoordinator
from teleop.atc.stage import StageController, UnoQConsolePoller
from teleop.atc.types import BayState, ConsoleInputState, SkillPrimitive, StallSignature
from teleop.unoq.client import UnoQClient
from teleop.unoq.mcu_bridge import McuConsoleBridge, SoftwareMcu
from teleop.unoq.server import UnoQServer
from teleop.unoq.switch import SwitchableUnoQ

SKU = "ev_latch_v1"
Z = StallSignature(0.035, 4.0, 8.0, 2.0)


def _skill(skill_id: str = "ev_latch_v1/Jam_Sig_1/1") -> SkillPrimitive:
    d = 7  # 6 wrist dims + one actuator
    return SkillPrimitive(
        skill_id=skill_id,
        part_sku_id=SKU,
        signature_id="Jam_Sig_1",
        duration_s=1.0,
        alpha_s=4.0,
        actuator_names=("a",),
        goal_orientation=(0.0, 0.0, 0.0, 1.0),
        y0=(0.0,) * d,
        y_goal=(1.0,) * d,
        weights=((0.0,) * d,),
        tau_demo=(0.1,),
        prototype=Z,
    )


def _server() -> UnoQServer:
    return UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0).start()


def _client(server: UnoQServer) -> UnoQClient:
    return UnoQClient("127.0.0.1", server.udp_port, server.tcp_port).connect()


@pytest.fixture
def board() -> Iterator[UnoQServer]:
    server = _server()
    try:
        yield server
    finally:
        server.stop()


def test_swap_replays_skills(board: UnoQServer) -> None:
    unoq = SwitchableUnoQ(_client(board))
    with _server() as other:
        try:
            assert unoq.register_skill(_skill()) == 1
            unoq.swap(_client(other))
            assert unoq.current.tcp_port == other.tcp_port
            match, skill_id = unoq.classify_stall(SKU, Z)
            assert match.signature_id == "Jam_Sig_1"
            assert skill_id == "ev_latch_v1/Jam_Sig_1/1"
            assert "ev_latch_v1/Jam_Sig_1/1" in other.skills
        finally:
            unoq.close()


def test_swap_closes_old_client_and_notifies(board: UnoQServer) -> None:
    old = _client(board)
    unoq = SwitchableUnoQ(old)
    seen: list[int] = []
    unoq.on_swap.append(lambda sw: seen.append(sw.current.tcp_port))
    with _server() as other:
        try:
            unoq.swap(_client(other))
            assert seen == [other.tcp_port]
            assert old.is_mock is False
            with pytest.raises(Exception, match="not connected"):
                old.ping()
        finally:
            unoq.close()


def test_failover_retries_on_loopback(board: UnoQServer) -> None:
    unoq = SwitchableUnoQ(_client(board))
    try:
        unoq.register_skill(_skill())
        assert unoq.is_mock is False
        board.stop()
        match, skill_id = unoq.classify_stall(SKU, Z)
        assert unoq.is_mock is True
        assert match.signature_id == "Jam_Sig_1"  # skills were replayed onto the loopback node
        assert skill_id == "ev_latch_v1/Jam_Sig_1/1"
    finally:
        unoq.close()


def test_swap_to_loopback_is_idempotent(board: UnoQServer) -> None:
    unoq = SwitchableUnoQ(_client(board))
    try:
        unoq.swap_to_loopback()
        first = unoq.current
        unoq.swap_to_loopback()
        assert unoq.current is first
        assert unoq.is_mock and unoq.host == "127.0.0.1"
    finally:
        unoq.close()


def test_poller_forwards_bay_state_on_change_only() -> None:
    mcu = SoftwareMcu()
    bridge = McuConsoleBridge(mcu)
    with _server() as server:
        server.attach_console(bridge)
        poller = UnoQConsolePoller(_client(server), period_s=3600.0)  # driven by poll_once()
        try:
            poller.set_bay_state(1, BayState.STALLED)
            poller.set_bay_state(1, BayState.STALLED)
            poller.poll_once()
            poller.set_bay_state(1, BayState.STALLED)
            poller.poll_once()
            assert mcu.received.count("CMD:BAY_STATE:1:STALLED") == 1
            poller.set_bay_state(1, BayState.NOMINAL)
            poller.poll_once()
            assert mcu.received[-1] == "CMD:BAY_STATE:1:NOMINAL"
        finally:
            poller.close()


def test_poller_records_heartbeat_age() -> None:
    mcu = SoftwareMcu()
    bridge = McuConsoleBridge(mcu)
    bridge.pump()  # one heartbeat heard
    with _server() as server:
        server.attach_console(bridge)
        poller = UnoQConsolePoller(_client(server), period_s=3600.0)
        try:
            assert poller.heartbeat_age_ms is None
            for _ in range(100):  # the server copies the bridge state every 50 ms
                poller.poll_once()
                if poller.heartbeat_age_ms is not None:
                    break
                time.sleep(0.01)
            assert poller.heartbeat_age_ms is not None
        finally:
            poller.close()


def test_poller_drops_bay_state_failures() -> None:
    with _server() as server:
        client = _client(server)
        poller = UnoQConsolePoller(client, period_s=3600.0)
        server.stop()
        poller.set_bay_state(1, BayState.STALLED)
        poller.poll_once()  # no exception
        poller.close()


class _Console:
    def __init__(self) -> None:
        self.closed = False

    def state(self) -> ConsoleInputState:
        return ConsoleInputState()

    def take_fault_rig_edge(self) -> bool:
        return False

    def set_bay_state(self, bay_id: int, state: BayState) -> None:
        pass

    def reset_rig(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _Coordinator:
    def snapshots(self) -> tuple[()]:
        return ()


def test_replace_hardware_closes_old_consoles() -> None:
    from teleop.atc.stage import EventTap, ReasonerTap

    old, new = _Console(), _Console()
    controller = StageController(
        cast(AtcCoordinator, _Coordinator()),
        cast(EventTap, None),
        cast(ReasonerTap, None),
        hardware=[old],
    )
    controller.replace_hardware([new])
    assert old.closed and not new.closed
    assert controller.consoles == [controller.keyboard, new]
