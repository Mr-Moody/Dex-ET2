"""Enclosed bay cells (fleet ops console): isolation, cameras, lighting."""

from __future__ import annotations

from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest

from teleop.retarget.clutch import rescue_camera_names
from teleop.sim.bays import BAY_BASES, PANDA_XML, build_four_bay_scene
from teleop.sim.cells import WALL_HEIGHT_M, cctv_camera_name, cell_bounds
from teleop.sim.hands import load_hand

mj: Any = _mujoco


def _models_present() -> bool:
    try:
        right = load_hand("leap").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


pytestmark = pytest.mark.skipif(
    not _models_present(), reason="Panda/LEAP MJCFs missing; run scripts/fetch_models.py"
)


@pytest.fixture(scope="module")
def scene() -> tuple[Any, Any]:
    model = build_four_bay_scene()
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    return model, data


def test_bays_are_four_metres_apart() -> None:
    assert BAY_BASES == {
        1: (-2.0, -2.0, 0.0),
        2: (2.0, -2.0, 0.0),
        3: (-2.0, 2.0, 0.0),
        4: (2.0, 2.0, 0.0),
    }


def test_cctv_camera_names() -> None:
    assert cctv_camera_name(3) == "bay3/cctv"


def test_every_bay_has_cctv_inside_its_cell(scene: tuple[Any, Any]) -> None:
    model, data = scene
    for k in (1, 2, 3, 4):
        x0, x1, y0, y1 = cell_bounds(k)
        x, y, z = data.cam_xpos[model.camera(cctv_camera_name(k)).id]
        assert x0 < x < x1 and y0 < y < y1
        assert 0.5 < z < WALL_HEIGHT_M


def test_cctv_cannot_see_other_bays(scene: tuple[Any, Any]) -> None:
    model, data = scene
    geomid = np.zeros(1, dtype=np.int32)
    for k in (1, 2, 3, 4):
        cam = np.array(data.cam_xpos[model.camera(cctv_camera_name(k)).id], dtype=np.float64)
        for j in (1, 2, 3, 4):
            if j == k:
                continue
            target = np.array(BAY_BASES[j], dtype=np.float64) + np.array([0.3, 0.0, 0.5])
            dist = mj.mj_ray(model, data, cam, target - cam, None, 1, -1, geomid)
            assert 0.0 <= dist < 1.0, f"bay {k} cctv has a clear line to bay {j}"
            assert str(model.geom(int(geomid[0])).name).startswith(f"bay{k}/cell_wall")


def test_cctv_looks_at_its_own_latch(scene: tuple[Any, Any]) -> None:
    model, data = scene
    for k in (1, 2, 3, 4):
        cam_id = model.camera(cctv_camera_name(k)).id
        pos = np.asarray(data.cam_xpos[cam_id])
        forward = -np.asarray(data.cam_xmat[cam_id]).reshape(3, 3)[:, 2]
        latch = np.asarray(data.xpos[model.body(f"bay{k}/latch_base").id])
        to_latch = (latch - pos) / np.linalg.norm(latch - pos)
        assert float(forward @ to_latch) > np.cos(np.radians(20.0))


def test_rescue_cameras_in_live_scene(scene: tuple[Any, Any]) -> None:
    model, _ = scene
    for k in (1, 2):
        for name in rescue_camera_names(f"bay{k}/"):
            assert model.camera(name).id >= 0


def test_cell_geoms_do_not_collide(scene: tuple[Any, Any]) -> None:
    model, _ = scene
    cell_geoms = [i for i in range(model.ngeom) if "/cell_" in str(model.geom(i).name)]
    assert cell_geoms
    for i in cell_geoms:
        assert model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0
        assert model.geom_group[i] == 0


def test_cells_cast_shadows(scene: tuple[Any, Any]) -> None:
    model, _ = scene
    assert sum(bool(model.light_castshadow[i]) for i in range(model.nlight)) >= 4
    assert int(model.vis.quality.shadowsize) >= 4096
    assert int(model.vis.quality.offsamples) >= 8


def test_no_background_scene_has_two_cells() -> None:
    model = build_four_bay_scene(include_background_bays=False)
    with pytest.raises(KeyError):
        model.camera(cctv_camera_name(3))
    assert model.camera(cctv_camera_name(2)).id >= 0


def test_track_twin_is_visual_and_physics_track_unchanged(scene: tuple[Any, Any]) -> None:
    model, _ = scene
    for k in (1, 2):
        phys = model.geom(f"bay{k}/base_track")
        twin = model.geom(f"bay{k}/base_track_visual")
        assert model.geom_contype[phys.id] == 1 and model.geom_group[phys.id] == 3
        assert model.geom_contype[twin.id] == 0 and model.geom_group[twin.id] == 0
        # The twin's top face sits 1 mm below the physics track's top (no flap z-fighting).
        phys_top = model.geom_pos[phys.id][2] + model.geom_size[phys.id][2]
        twin_top = model.geom_pos[twin.id][2] + model.geom_size[twin.id][2]
        assert twin_top == pytest.approx(phys_top - 0.001)
