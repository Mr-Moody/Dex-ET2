"""T-191: the per-bay wrist camera and the policy camera renderer."""

from __future__ import annotations

from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest

from teleop.sim.bays import PANDA_XML
from teleop.sim.hands import load_hand

mj: Any = _mujoco


def _models_present() -> bool:
    try:
        right = load_hand("inspire").mjcf["right"]
    except FileNotFoundError:
        return False
    return PANDA_XML.is_file() and right.is_file()


needs_models = pytest.mark.skipif(
    not _models_present(), reason="Panda/Inspire MJCFs missing; run scripts/fetch_models.py"
)


@needs_models
def test_wrist_camera_exists_in_both_scenes() -> None:
    from teleop.policy.schema import camera_name
    from teleop.sim.bays import build_four_bay_scene, build_single_bay_scene

    four = build_four_bay_scene(include_background_bays=False)
    single = build_single_bay_scene()
    for model in (four, single):
        assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, camera_name(1, "wrist")) >= 0
    assert mj.mj_name2id(four, mj.mjtObj.mjOBJ_CAMERA, camera_name(2, "wrist")) >= 0


@needs_models
def test_render_shapes(offscreen_render_ok: None) -> None:
    from teleop.policy.cameras import BayCameraRenderer
    from teleop.policy.schema import CAMERAS, IMAGE_HW
    from teleop.sim.bays import build_four_bay_scene, reset_robot_home

    model: Any = build_four_bay_scene(include_background_bays=False)
    data: Any = mj.MjData(model)
    reset_robot_home(model, data, 1)
    mj.mj_forward(model, data)
    renderer = BayCameraRenderer(model)
    try:
        images = renderer.render(data, 1)
    except mj.FatalError as exc:  # pragma: no cover - GL context missing on this host
        pytest.skip(f"no GL context: {exc}")
    finally:
        renderer.close()
    assert set(images) == set(CAMERAS)
    for img in images.values():
        assert img.shape == (*IMAGE_HW, 3)
        assert img.dtype == np.uint8
    assert int(images["wrist"].max()) > int(images["wrist"].min())  # not a uniform frame


def _visible_bodies(model: Any, data: Any, camera: str) -> set[str]:
    """Body names with at least 20 pixels in a segmentation render of ``camera``."""
    from teleop.policy.cameras import policy_clip

    r = mj.Renderer(model, 128, 128)
    try:
        r.enable_segmentation_rendering()
        with policy_clip(model, camera):
            r.update_scene(data, camera)
            seg = r.render()
    finally:
        r.close()
    ids, types = seg[..., 0], seg[..., 1]
    geoms = ids[(types == int(mj.mjtObj.mjOBJ_GEOM)) & (ids >= 0)]
    counts = np.bincount(model.geom_bodyid[geoms], minlength=model.nbody)
    return {str(model.body(int(b)).name) for b in np.flatnonzero(counts >= 20)}


@needs_models
def test_wrist_camera_sees_fingertip_and_bolt(offscreen_render_ok: None) -> None:
    """At the routine start pose the wrist view shows the index finger and the slide bolt.

    The warehouse scene is ~37 m across, so MuJoCo's extent-relative near plane would clip
    everything within ~0.37 m of the camera: the whole hand and latch.
    """
    from teleop.atc.coordinator import ROUTINE
    from teleop.atc.validator import BayRig, step_physics
    from teleop.policy.schema import camera_name
    from teleop.sim.bays import build_four_bay_scene, reset_robot_home
    from teleop.unoq.dmp import compose_pose

    model: Any = build_four_bay_scene(include_background_bays=False)
    data: Any = mj.MjData(model)
    reset_robot_home(model, data, 1)
    rig = BayRig(model, data, 1)
    start = compose_pose(rig.world_from_latch(), ROUTINE.pose_at(0.0))
    for _ in range(180):  # 3 s of 60 Hz control: settle onto the routine start pose
        rig.command(start, ROUTINE.fingers)
        step_physics(model, data)
    assert rig.reached(start, pos_tol=0.01, rot_tol=0.1)
    znear = float(model.vis.map.znear)
    seen = _visible_bodies(model, data, camera_name(1, "wrist"))
    assert float(model.vis.map.znear) == znear  # the shared model is left as it was
    assert "bay1/index_intermediate" in seen
    assert "bay1/slide_bolt" in seen
