"""T-192: the headless scripted rescue episode generator."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from teleop.atc.generate import GeneratorConfig, latch_perturbation, run_generation
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


class _Compliance:
    def emit(self, event: ComplianceEvent) -> None:
        pass

    def close(self) -> None:
        pass


class _QuietReasoner:
    def submit(self, ctx: ReasoningContext) -> None:
        pass

    def latest_verdict(self, bay_id: int) -> ReasoningVerdict | None:
        return None

    def close(self) -> None:
        pass


class _ZeroRenderer:
    def render(self, data: Any, bay_id: int) -> dict[str, Any]:
        from teleop.policy.schema import CAMERAS, IMAGE_HW

        return {cam: np.zeros((*IMAGE_HW, 3), dtype=np.uint8) for cam in CAMERAS}

    def close(self) -> None:
        pass


class _MemoryRecorder:
    """``EpisodeRecorder`` stand-in: keeps each episode's sidecar meta in memory."""

    def __init__(self) -> None:
        self.metas: list[dict[str, Any]] = []

    def start_episode(self, task: str) -> None:
        pass

    def add_frame(self, state: Any, action: Any, images: Mapping[str, Any]) -> None:
        pass

    def end_episode(self, meta: Mapping[str, Any]) -> int:
        self.metas.append(dict(meta))
        return len(self.metas) - 1

    def discard_episode(self) -> None:
        pass

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


def _stage(unoq: Any, root: Path) -> tuple[Any, Any, _MemoryRecorder]:
    from teleop.atc.episodes import EpisodeSink, fault_lookup
    from teleop.atc.stage import StageController

    controller = StageController.build(
        unoq, _Compliance(), _QuietReasoner(), seed=0, cycle_dwell_s=0.5
    )
    co = controller.coordinator
    recorder = _MemoryRecorder()
    sink = EpisodeSink(
        root,
        co.model,
        co.data,
        fault_lookup(controller),
        "scripted",
        renderer=_ZeroRenderer(),
        recorder=recorder,
    )
    return controller, sink, recorder


def test_perturbation_bounds_and_seed() -> None:
    config = GeneratorConfig()
    rng = np.random.default_rng(7)
    draws = [latch_perturbation(rng, config) for _ in range(1000)]
    assert all(abs(dx) <= 0.10 and abs(dy) <= 0.10 and abs(dyaw) <= 40.0 for dx, dy, dyaw in draws)
    assert len({round(d[0], 6) for d in draws}) > 900  # really random, not a constant
    again = np.random.default_rng(7)
    assert draws == [latch_perturbation(again, config) for _ in range(1000)]
    other = np.random.default_rng(8)
    assert draws[0] != latch_perturbation(other, config)


def test_perturbation_follows_config() -> None:
    config = GeneratorConfig(perturb_xy_m=0.01, perturb_yaw_deg=5.0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        dx, dy, dyaw = latch_perturbation(rng, config)
        assert abs(dx) <= 0.01 and abs(dy) <= 0.01 and abs(dyaw) <= 5.0


@needs_models
def test_generates_n_successes(unoq: Any, tmp_path: Path) -> None:
    controller, sink, recorder = _stage(unoq, tmp_path / "ds")
    n = run_generation(controller, sink, 3, seed=0)
    sink.close()

    assert n >= 3
    successes = [m for m in recorder.metas if m["success"]]
    assert len(successes) >= 3
    assert {m["source"] for m in successes} <= {"scripted", "dmp"}
    assert {m["fault"] for m in successes} <= {"flap", "pin"}
    assert all(m["bolt_x_final"] > 0.05 for m in successes)


@needs_models
def test_gives_up_after_3n_attempts(unoq: Any, tmp_path: Path) -> None:
    # A pinned bolt cannot be cleared by any rescue, so every attempt fails.
    controller, sink, recorder = _stage(unoq, tmp_path / "ds")
    config = GeneratorConfig(faults=("bay2_pin",), max_sim_s_per_episode=12.0)
    n = run_generation(controller, sink, 1, seed=0, config=config)
    sink.close()

    assert n == 0
    assert not any(m["success"] for m in recorder.metas)


@needs_models
def test_same_seed_gives_same_latch_poses(unoq: Any, tmp_path: Path) -> None:
    runs: list[list[tuple[float, ...]]] = []
    for _ in range(2):
        controller, sink, recorder = _stage(unoq, tmp_path / f"ds{len(runs)}")
        run_generation(controller, sink, 1, seed=3)
        sink.close()
        runs.append([tuple(sorted(m["latch_pose"].items())) for m in recorder.metas][:1])
    assert runs[0] == runs[1]
