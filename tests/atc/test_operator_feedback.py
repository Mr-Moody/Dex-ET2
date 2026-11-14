"""Every operator key press is answered: what it did, or why it did nothing."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from teleop.atc.hud import hud_lines
from teleop.atc.stage import StageController, key_name
from teleop.atc.types import BayState
from teleop.compliance.protocol import ComplianceEvent
from teleop.reasoning.protocol import ReasoningContext, ReasoningVerdict
from teleop.sim.bays import PANDA_XML
from teleop.sim.hands import load_hand

SPACE = 32


def _models_present() -> bool:
    try:
        right = load_hand("leap").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


needs_models = pytest.mark.skipif(
    not _models_present(), reason="Panda/LEAP MJCFs missing; run scripts/fetch_models.py"
)


class NullSink:
    def emit(self, event: ComplianceEvent) -> None:
        return None

    def close(self) -> None:
        return None


class NullReasoner:
    def submit(self, ctx: ReasoningContext) -> None:
        return None

    def latest_verdict(self, bay_id: int) -> ReasoningVerdict | None:
        return None

    def close(self) -> None:
        return None


@pytest.fixture
def controller() -> Iterator[StageController]:
    from teleop.unoq.client import UnoQClient
    from teleop.unoq.server import UnoQServer

    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="test-node").start()
    client = UnoQClient("127.0.0.1", server.udp_port, server.tcp_port, local_server=server)
    client.connect()
    ctl = StageController.build(client, NullSink(), NullReasoner(), cycle_dwell_s=0.2)  # type: ignore[arg-type]
    yield ctl
    ctl.close()
    client.close()


def _press(ctl: StageController, key: str | int) -> tuple[str, str, str]:
    """Press one key, run one tick, return the newest ``(key, text, level)``."""
    seq = ctl.actions[-1].seq if ctl.actions else 0
    ctl.on_key(key if isinstance(key, int) else ord(key))
    ctl.tick(None)
    note = ctl.actions[-1]
    assert note.seq == seq + 1, "every press must produce exactly one note"
    return note.key, note.text, note.level


def test_key_names() -> None:
    assert key_name(SPACE) == "SPACE"
    assert key_name(ord("F")) == "F"
    assert key_name(ord("1")) == "1"
    assert key_name(300) == "key 300"


@needs_models
def test_presses_outside_a_rescue_are_explained(controller: StageController) -> None:
    ctl = controller

    key, text, level = _press(ctl, "X")
    assert (key, level) == ("X", "warn") and "Not a control key" in text

    _, text, level = _press(ctl, "3")
    assert level == "warn" and "Bay 3" in text  # no background bays in this build

    _, text, level = _press(ctl, SPACE)
    assert level == "warn" and "no bay is waiting for rescue" in text
    _, text, level = _press(ctl, SPACE)
    assert (text, level) == ("Clutch released", "info")

    _, text, level = _press(ctl, "F")
    assert level == "ok" and "Bay 1 flap jam armed" in text
    assert any(f[:2] == (1, "flap") for f in ctl.hud_view().faults)
    _, text, level = _press(ctl, "F")
    assert level == "warn" and "already armed" in text

    _, text, level = _press(ctl, "P")
    assert level == "ok" and "negative control" in text
    assert any(f[:2] == (2, "pin") for f in ctl.hud_view().faults)

    _, text, level = _press(ctl, "R")
    assert level == "ok" and text.startswith("Reset")
    assert ctl.hud_view().faults == ()

    lines = hud_lines(ctl.hud_view())
    assert any(line.startswith("KEY R: Reset") for line in lines)


@needs_models
def test_rescue_presses_point_at_the_bay_waiting(controller: StageController) -> None:
    ctl = controller
    co = ctl.coordinator
    _press(ctl, "F")
    for _ in range(3_000):  # Bay 1 finishes its clean cycle, then stalls on the jam
        ctl.tick(None)
        if co.state(1) == BayState.RESCUE:
            break
    assert co.state(1) == BayState.RESCUE

    _, text, level = _press(ctl, "2")
    assert level == "warn" and "Bay 1 is the one waiting for rescue" in text
    _, text, level = _press(ctl, SPACE)
    assert level == "warn" and "Press 1 to drive Bay 1" in text
    _, text, level = _press(ctl, SPACE)  # release again
    assert level == "info"

    _, text, level = _press(ctl, "1")
    assert level == "ok" and "Engage the clutch" in text
    _, text, level = _press(ctl, SPACE)
    assert level == "ok" and "gloves now drive Bay 1" in text
    _, text, level = _press(ctl, SPACE)
    assert level == "info" and "checking whether the bolt cleared" in text
