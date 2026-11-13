from pathlib import Path

import mujoco
import pytest

from teleop.sim.hands import HandSpec, actuator_names, joint_names, load_hand
from teleop.types import OPENXR_JOINTS, SIDES

ROOT = Path(__file__).resolve().parents[2]
HANDS = ["inspire", "leap"]

pytestmark = pytest.mark.skipif(
    not (ROOT / "third_party" / "menagerie").exists()
    or not (ROOT / "third_party" / "inspire").exists(),
    reason="third_party/ models missing; run `uv run python scripts/fetch_models.py`",
)


def spec_for(name: str) -> HandSpec:
    return load_hand(name, ROOT / "configs" / "hands")


@pytest.mark.parametrize("name", HANDS)
def test_hand_loads_in_mujoco(name: str) -> None:
    spec = spec_for(name)
    for side in SIDES:
        path = spec.mjcf[side]
        assert path.is_file(), path
        model = mujoco.MjModel.from_xml_path(str(path))  # pyright: ignore
        assert model.nu > 0
        assert len(actuator_names(spec, side)) == model.nu
        assert len(joint_names(spec, side)) == model.njnt
        assert spec.base_body in [model.body(i).name for i in range(model.nbody)]


@pytest.mark.parametrize("name", HANDS)
def test_flexmap_actuators_exist(name: str) -> None:
    spec = spec_for(name)
    for side in SIDES:
        names = actuator_names(spec, side)
        for fm in spec.flex:
            assert fm.actuator in names, (side, fm.actuator)


@pytest.mark.parametrize("name", HANDS)
def test_flexmap_joints_are_openxr(name: str) -> None:
    for fm in spec_for(name).flex:
        assert fm.joints
        assert fm.axis in ("x", "y", "z")
        for joint in fm.joints:
            assert joint in OPENXR_JOINTS, joint
