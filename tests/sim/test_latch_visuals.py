"""Visual twin of the printed EV latch: meshes dress the latch boxes without changing physics."""

from __future__ import annotations

from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest

from teleop.atc.types import LatchPose
from teleop.sim.latch import MESH_DIR, VISUAL_MESHES, add_latch_to_spec

mj: Any = _mujoco

pytestmark = pytest.mark.skipif(
    not all((MESH_DIR / f"{stem}.stl").is_file() for stem, _, _ in VISUAL_MESHES),
    reason="assets/latch meshes missing",
)


def _build(visual_meshes: bool, n: int = 1) -> Any:
    spec: Any = mj.MjSpec()
    for k in range(n):
        add_latch_to_spec(
            spec,
            f"bay{k + 1}",
            LatchPose(position=(0.3 * k, 0.0, 0.02), yaw_rad=0.0),
            visual_meshes=visual_meshes,
        )
    return spec.compile()


def _mesh_extent_in_body(model: Any, geom: int) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned extent of a mesh geom's vertices in its body frame."""
    mesh = int(model.geom_dataid[geom])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    verts = np.asarray(model.mesh_vert[start : start + count], dtype=float)
    rot = np.zeros(9)
    mj.mju_quat2Mat(rot, model.geom_quat[geom])
    pts = verts @ rot.reshape(3, 3).T + np.asarray(model.geom_pos[geom], dtype=float)
    return pts.min(axis=0), pts.max(axis=0)


def test_meshes_are_visual_only_and_physics_is_unchanged() -> None:
    plain, dressed = _build(False), _build(True)
    assert int(dressed.nmesh) == len(VISUAL_MESHES)
    for stem, body, _ in VISUAL_MESHES:
        g = dressed.geom(f"bay1/{stem}_visual")
        assert int(dressed.geom_type[g.id]) == int(mj.mjtGeom.mjGEOM_MESH)
        assert dressed.geom_contype[g.id] == 0 and dressed.geom_conaffinity[g.id] == 0
        assert dressed.body(int(dressed.geom_bodyid[g.id])).name == f"bay1/{body}"
    for name in ("bay1/latch_base", "bay1/slide_bolt", "bay1/stopper_flap"):
        assert dressed.body(name).mass[0] == pytest.approx(plain.body(name).mass[0])
        np.testing.assert_allclose(dressed.body(name).inertia, plain.body(name).inertia)
    for box in ("base_track", "bolt_bar", "bolt_hook", "flap_paddle"):
        g = dressed.geom(f"bay1/{box}")
        assert dressed.geom_contype[g.id] == 1 and dressed.geom_rgba[g.id][3] == 0.0


def test_mesh_assets_are_shared_between_latches() -> None:
    assert int(_build(True, n=3).nmesh) == len(VISUAL_MESHES)


def test_touched_meshes_match_their_boxes() -> None:
    model = _build(True)
    tol = 1.5e-3
    lo, hi = _mesh_extent_in_body(model, model.geom("bay1/slide_bolt_visual").id)
    # bar x [-0.09, -0.01] plus hook x [-0.0215, -0.0055]; bar z from 0.008, hook top 0.059
    np.testing.assert_allclose(lo, [-0.09, -0.014, 0.008], atol=tol)
    np.testing.assert_allclose(hi, [-0.0055, 0.014, 0.059], atol=tol)
    lo, hi = _mesh_extent_in_body(model, model.geom("bay1/safety_flap_visual").id)
    # paddle x [-0.04, 0], z [0, 0.008] in the hinge frame, plus a 4 mm hinge barrel
    np.testing.assert_allclose(lo, [-0.04, -0.02, -0.004], atol=tol)
    np.testing.assert_allclose(hi, [0.004, 0.02, 0.008], atol=tol)
