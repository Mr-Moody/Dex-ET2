from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from pathlib import Path

import mujoco
import pytest

from teleop.config import CameraConfig
from teleop.sim.hands import HandSpec, load_hand
from teleop.sim.scene import build_scene
from teleop.sim.tasks.lift_cube import LiftCube

ROOT = Path(__file__).resolve().parents[2]


def available_hand() -> HandSpec | None:
    for name in ("leap", "inspire"):
        try:
            spec = load_hand(name, ROOT / "configs" / "hands")
        except FileNotFoundError:
            continue
        if all(p.is_file() for p in spec.mjcf.values()):
            return spec
    return None


@pytest.fixture(scope="session")
def hand() -> HandSpec:
    spec = available_hand()
    if spec is None:
        pytest.skip("hand MJCFs missing; run `uv run python scripts/fetch_models.py`")
    return spec


@pytest.fixture
def scene(hand: HandSpec):
    task = LiftCube()
    model = build_scene(hand, task, [CameraConfig()])
    data = mujoco.MjData(model)
    task.reset(model, data, 0)
    return model, data, task
