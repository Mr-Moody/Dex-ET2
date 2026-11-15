"""``teleop atc --demo-rescue``: hand shake model, flag wiring and the full jam -> rescue take."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from teleop.atc.demo_rescue import HandNoise, Shake
from teleop.commands.atc import DEFAULT_CONFIG, load_config, main, resolve_seed
from teleop.compliance.protocol import ComplianceEvent
from teleop.reasoning.protocol import ReasoningContext, ReasoningVerdict
from teleop.sim.bays import PANDA_XML
from teleop.sim.hands import load_hand


def _models_present() -> bool:
    try:
        right = load_hand("inspire").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


needs_models = pytest.mark.skipif(
    not _models_present(), reason="Panda/Inspire MJCFs missing; run scripts/fetch_models.py"
)


def test_hand_noise_is_small_and_zero_mean() -> None:
    shake = Shake()
    noise = HandNoise(shake, np.random.default_rng(0))
    pos = np.array([noise.step()[0] for _ in range(60 * 30)])
    assert np.abs(pos).max() < 4 * (shake.tremor_m + shake.drift_m)
    assert np.abs(pos.mean(axis=0)).max() < shake.drift_m
    assert pos.std(axis=0).min() > 0.1 * shake.drift_m  # it does shake


def test_zero_shake_is_steady() -> None:
    noise = HandNoise(Shake().scaled(0.0), np.random.default_rng(0))
    for _ in range(120):
        d_pos, d_finger = noise.step()
        assert not d_pos.any()
        assert d_finger == 0.0


def test_demo_rescue_uses_stage_seed_and_rejects_script() -> None:
    cfg = load_config(DEFAULT_CONFIG, {"demo_rescue": True})
    assert resolve_seed(cfg) == 7
    assert main(["--demo-rescue", "--script", "--headless"]) == 2


class _Sink:
    def __init__(self) -> None:
        self.events: list[ComplianceEvent] = []

    def emit(self, event: ComplianceEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass


class _Reasoner:
    def submit(self, ctx: ReasoningContext) -> None:
        pass

    def latest_verdict(self, bay_id: int) -> ReasoningVerdict | None:
        return None

    def close(self) -> None:
        pass


@pytest.fixture
def unoq() -> Iterator[Any]:
    from teleop.unoq.client import UnoQClient
    from teleop.unoq.server import UnoQServer

    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="test-node").start()
    client = UnoQClient(
        "127.0.0.1", server.udp_port, server.tcp_port, local_server=server
    ).connect()
    yield client
    client.close()


@needs_models
def test_demo_take_jams_then_rescues_bay1(unoq: Any) -> None:
    from teleop.atc.demo_rescue import DemoRescueDirector
    from teleop.atc.stage import StageController

    cfg = load_config(DEFAULT_CONFIG, {"demo_rescue": True})
    sink = _Sink()
    controller = StageController.build(
        unoq,
        sink,
        _Reasoner(),
        cycle_dwell_s=cfg.cycle_dwell_s,
        include_background_bays=cfg.background_bays,
        seed=resolve_seed(cfg),
        realism=cfg.realism,
        gate_pass_min=cfg.gate_pass_min,
    )
    director = DemoRescueDirector(controller, seed=resolve_seed(cfg))
    ticks = 0
    while not director.done and ticks < 12_000:
        controller.tick(director.step())
        ticks += 1
    assert director.rescued
    kinds = [(e.kind, e.bay_id) for e in sink.events]
    assert ("STALL_ALERT", 1) in kinds
    assert ("RESCUE_LOGGED", 1) in kinds
    assert kinds.index(("STALL_ALERT", 1)) < kinds.index(("RESCUE_LOGGED", 1))
