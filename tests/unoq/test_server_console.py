"""UNO Q edge service serving the MCU console and bay-state LEDs (no hardware: SoftwareMcu)."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import pytest

from teleop.atc.types import BayState
from teleop.unoq.client import UnoQClient, UnoQError
from teleop.unoq.mcu_bridge import McuConsoleBridge, SoftwareMcu
from teleop.unoq.server import UnoQServer

Rig = tuple[SoftwareMcu, UnoQClient]


def _wait(pred: Callable[[], bool], timeout_s: float = 0.5) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


@pytest.fixture
def rig() -> Iterator[Rig]:
    mcu = SoftwareMcu()
    bridge = McuConsoleBridge(mcu)
    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0).start()
    server.attach_console(bridge)
    bridge.start()
    client = UnoQClient("127.0.0.1", server.udp_port, server.tcp_port).connect()
    try:
        yield mcu, client
    finally:
        client.close()
        bridge.close()
        server.stop()


@pytest.fixture
def bare_client() -> Iterator[UnoQClient]:
    with UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0) as server:
        client = UnoQClient("127.0.0.1", server.udp_port, server.tcp_port).connect()
        try:
            yield client
        finally:
            client.close()


def test_console_state_reflects_mcu_buttons(rig: Rig) -> None:
    mcu, client = rig
    mcu.press_bay(2)
    mcu.set_clutch(True)

    def synced() -> bool:
        state = client.console_status()[0]
        return state.selected_bay == 2 and state.clutch is True

    assert _wait(synced)


def test_heartbeat_age_reported(rig: Rig) -> None:
    _, client = rig
    assert _wait(lambda: client.console_status()[1] is not None)
    age = client.console_status()[1]
    assert age is not None
    assert age < 500


def test_heartbeat_age_none_without_bridge(bare_client: UnoQClient) -> None:
    assert bare_client.console_status()[1] is None


def test_set_bay_state_reaches_mcu(rig: Rig) -> None:
    mcu, client = rig
    client.set_bay_state(1, BayState.STALLED)
    assert "CMD:BAY_STATE:1:STALLED" in mcu.received


def test_set_bay_state_without_console_is_ok(bare_client: UnoQClient) -> None:
    bare_client.set_bay_state(1, BayState.STALLED)


def test_set_bay_state_rejects_bad_state(rig: Rig) -> None:
    _, client = rig
    with pytest.raises(UnoQError):
        client.request({"op": "set_bay_state", "bay_id": 1, "state": "BOGUS"})
    assert client.ping()
