"""Sustainable EV-battery recycling warehouse: conveyors, separated bays, cameras, lighting."""

from __future__ import annotations

from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest

from teleop.retarget.clutch import rescue_camera_names
from teleop.sim.bays import BAY_BASES, OVERVIEW_CAMERA, PANDA_XML, build_four_bay_scene
from teleop.sim.cells import (
    CONVEYOR_BAYS,
    HIDDEN_GROUP,
    WAREHOUSE_BOUNDS,
    WAREHOUSE_FLOOR_Z,
    WAREHOUSE_ROOF_Z,
    cctv_camera_name,
)
from teleop.sim.hands import load_hand

mj: Any = _mujoco


def _models_present() -> bool:
    try:
        right = load_hand("inspire").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


pytestmark = pytest.mark.skipif(
    not _models_present(), reason="Panda/Inspire MJCFs missing; run scripts/fetch_models.py"
)


@pytest.fixture(scope="module")
def scene() -> tuple[Any, Any]:
    model = build_four_bay_scene()
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    return model, data


def test_bays_are_spread_across_warehouse() -> None:
    assert BAY_BASES == {
        1: (-3.2, -3.4, 0.0),
        2: (3.2, -3.4, 0.0),
        3: (-3.2, 3.4, 0.0),
        4: (3.2, 3.4, 0.0),
    }


def test_cctv_camera_names() -> None:
    assert cctv_camera_name(3) == "bay3/cctv"


def test_cameras_sit_inside_warehouse(scene: tuple[Any, Any]) -> None:
    model, data = scene
    x0, x1, y0, y1 = WAREHOUSE_BOUNDS
    for name in (*(cctv_camera_name(k) for k in (1, 2, 3, 4)), OVERVIEW_CAMERA):
        x, y, z = data.cam_xpos[model.camera(name).id]
        assert x0 < x < x1 and y0 < y < y1, name
        assert WAREHOUSE_FLOOR_Z < z < WAREHOUSE_ROOF_Z, name


def test_warehouse_has_conveyors_battery_packs_and_sustainable_elements(
    scene: tuple[Any, Any],
) -> None:
    model, _ = scene
    geom_names = {str(model.geom(i).name) for i in range(model.ngeom)}
    body_names = {str(model.body(i).name) for i in range(model.nbody)}
    for k in CONVEYOR_BAYS:
        assert f"warehouse/conveyor_bay{k}" in body_names
        assert f"conveyor_bay{k}_belt" in geom_names
        assert f"conveyor_bay{k}_feeder_hood" in geom_names
        assert f"conveyor_bay{k}_extractor_hood" in geom_names
        assert f"conveyor_bay{k}_extractor_duct" in geom_names
        assert any(g.startswith(f"conveyor_cleat_bay{k}_") for g in geom_names)
        assert f"bay{k}/conveyor_beacon" in geom_names
        assert f"bay{k}/cell_pedestal_col" in geom_names
        assert f"bay{k}/cell_jig_housing" in geom_names
    assert any("rack_w_" in g for g in geom_names)
    assert any("agv_" in g for g in geom_names)
    assert any("greenwall" in g for g in geom_names)
    assert any("planter_bush" in g for g in geom_names)
    assert any("bin_cells" in g for g in geom_names)


def test_open_warehouse_has_no_enclosing_cell_walls(scene: tuple[Any, Any]) -> None:
    model, _ = scene
    geom_names = [str(model.geom(i).name) for i in range(model.ngeom)]
    assert not any("cell_wall_" in g for g in geom_names)


def test_cctv_looks_at_its_own_latch(scene: tuple[Any, Any]) -> None:
    model, data = scene
    for k in (1, 2, 3, 4):
        cam_id = model.camera(cctv_camera_name(k)).id
        pos = np.asarray(data.cam_xpos[cam_id])
        forward = -np.asarray(data.cam_xmat[cam_id]).reshape(3, 3)[:, 2]
        latch = np.asarray(data.xpos[model.body(f"bay{k}/latch_base").id])
        to_latch = (latch - pos) / np.linalg.norm(latch - pos)
        assert float(forward @ to_latch) > np.cos(np.radians(22.0))


def test_rescue_cameras_in_live_scene(scene: tuple[Any, Any]) -> None:
    model, _ = scene
    for k in (1, 2):
        for name in rescue_camera_names(f"bay{k}/"):
            assert model.camera(name).id >= 0


def test_warehouse_and_cell_geoms_do_not_collide(scene: tuple[Any, Any]) -> None:
    model, _ = scene
    env_geoms = [
        i
        for i in range(model.ngeom)
        if (
            "/cell_" in str(model.geom(i).name)
            or str(model.geom(i).name).startswith(("warehouse/", "conveyor_"))
        )
    ]
    assert len(env_geoms) > 50
    for i in env_geoms:
        assert model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0
        assert model.geom_group[i] == 0


def test_cells_cast_shadows_within_shader_budget(scene: tuple[Any, Any]) -> None:
    model, _ = scene
    assert sum(bool(model.light_castshadow[i]) for i in range(model.nlight)) >= 4
    assert int(model.nlight) <= 7  # +1 headlight <= MuJoCo's 8-light OpenGL cap
    assert int(model.vis.quality.shadowsize) >= 4096
    assert int(model.vis.quality.offsamples) >= 8


def test_no_background_scene_has_two_cells() -> None:
    model = build_four_bay_scene(include_background_bays=False)
    with pytest.raises(KeyError):
        model.camera(cctv_camera_name(3))
    assert model.camera(cctv_camera_name(2)).id >= 0


def test_track_twin_and_hidden_physics_floor(scene: tuple[Any, Any]) -> None:
    model, _ = scene
    floor = model.geom("floor")
    assert model.geom_contype[floor.id] == 1 and model.geom_group[floor.id] == HIDDEN_GROUP
    for k in (1, 2):
        phys = model.geom(f"bay{k}/base_track")
        twin = model.geom(f"bay{k}/base_track_visual")
        assert model.geom_contype[phys.id] == 1 and model.geom_group[phys.id] == HIDDEN_GROUP
        assert model.geom_contype[twin.id] == 0 and model.geom_group[twin.id] == 0
        phys_top = model.geom_pos[phys.id][2] + model.geom_size[phys.id][2]
        twin_top = model.geom_pos[twin.id][2] + model.geom_size[twin.id][2]
        assert twin_top == pytest.approx(phys_top - 0.001)
