"""UNO Q STM32U585 serial console bridge and SwitchableUnoQ skill replay/failover."""

from __future__ import annotations

import pytest

from teleop.atc.types import BayState, ConsoleInputState, SkillPrimitive, StallSignature
from teleop.unoq.client import UnoQClient
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
from teleop.unoq.server import UnoQServer
from teleop.unoq.switch import SwitchableUnoQ

SKU = "ev_latch_v1"
Z = StallSignature(0.035, 4.0, 8.0, 2.0)


class _Clock:
    def __init__(self) -> None:
        self.ns = 1_000_000_000

    def __call__(self) -> int:
        return self.ns

    def advance(self, s: float) -> None:
        self.ns += int(s * 1e9)


def _skill(skill_id: str = "ev_latch_v1/Jam_Sig_1/1") -> SkillPrimitive:
    d = 7
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
def test_parse_valid_mcu_lines_and_reject_garbage(line: str, event: object) -> None:
    assert parse_mcu_line(line) == event
    for bad in ("", "BTN:BAY:5", "CLUTCH:2", "HB:1:1", "FAULT_RIG:"):
        assert parse_mcu_line(bad) is None


def test_fault_rig_latches_once_per_cycle_until_reset() -> None:
    clock = _Clock()
    mcu = SoftwareMcu(clock=clock)
    bridge = McuConsoleBridge(mcu, clock=clock)

    for closed in (True, False, True, False, True):
        mcu.set_fault_rig(closed)
    bridge.pump()
    assert bridge.state().fault_rig_latched
    assert bridge.take_fault_rig_edge() is True
    assert bridge.take_fault_rig_edge() is False
    bridge.handle_line("FAULT_RIG:LATCHED")
    assert bridge.take_fault_rig_edge() is False

    bridge.reset_rig()
    bridge.pump()
    assert not bridge.state().fault_rig_latched
    mcu.set_fault_rig(False)
    mcu.set_fault_rig(True)
    bridge.pump()
    assert bridge.take_fault_rig_edge() is True


def test_clutch_event_does_not_update_selected_bay_timestamp() -> None:
    clock = _Clock()
    bridge = McuConsoleBridge(SoftwareMcu(clock=clock), clock=clock)
    bridge.handle_line("BTN:BAY:2")
    bay_stamp = bridge.state().selected_bay_timestamp_ns
    assert bay_stamp is not None
    clock.advance(0.5)
    bridge.handle_line("CLUTCH:1")
    s = bridge.state()
    assert s.clutch
    assert s.timestamp_ns > bay_stamp
    assert s.selected_bay_timestamp_ns == bay_stamp


def test_estop_overrides_clutch_and_stale_heartbeat_fails_safe() -> None:
    clock = _Clock()
    mcu = SoftwareMcu(clock=clock)
    bridge = McuConsoleBridge(mcu, clock=clock)

    mcu.press_bay(2)
    mcu.set_clutch(True)
    bridge.pump()
    assert bridge.state().selected_bay == 2 and bridge.state().clutch

    mcu.set_estop(True)
    bridge.pump()
    assert bridge.state().estop and not bridge.state().clutch
    assert bridge.state().powered_off == frozenset({2})

    # Releasing E-stop keeps clutch inhibited on powered-off Bay 2, but allows powered Bay 1.
    mcu.set_estop(False)
    mcu.set_clutch(True)
    bridge.pump()
    assert not bridge.state().clutch
    mcu.press_bay(1)
    bridge.pump()
    assert bridge.link_ok and bridge.state().clutch

    clock.advance(1.0)
    assert not bridge.link_ok and not bridge.state().clutch

    s = ConsoleInputState(selected_bay=2, clutch=True, estop=False, fault_rig_latched=True)
    assert console_state_from_dict(console_state_to_dict(s)) == s
    assert format_bay_state(2, BayState.AUTONOMOUS_SKILL) == "CMD:BAY_STATE:2:AUTO"


def test_switchable_unoq_replays_skills_idempotently_and_fails_over_to_loopback() -> None:
    board = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0).start()
    other = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0).start()

    def connect(srv: UnoQServer) -> UnoQClient:
        return UnoQClient("127.0.0.1", srv.udp_port, srv.tcp_port).connect()

    unoq = SwitchableUnoQ(connect(board))
    try:
        assert unoq.register_skill(_skill()) == 1
        unoq.swap(connect(other))
        unoq.swap(connect(board))
        unoq.swap(connect(other))
        assert board.classifier.sample_count(SKU, "Jam_Sig_1") == 1
        assert other.classifier.sample_count(SKU, "Jam_Sig_1") == 1

        other.stop()
        match, skill_id = unoq.classify_stall(SKU, Z)
        assert unoq.is_mock is True
        assert match.signature_id == "Jam_Sig_1" and skill_id == "ev_latch_v1/Jam_Sig_1/1"
    finally:
        unoq.close()
        board.stop()
        other.stop()
