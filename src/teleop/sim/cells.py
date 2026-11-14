"""Enclosed bay cells for the fleet ops console: one walled work cell per bay.

Every bay of ``teleop.sim.bays`` sits in its own open-top cell (dark floor tile, jig plate
under the latch, four walls taller than the cell's CCTV camera, safety-yellow floor strips)
with its own lights and a ``bay{k}/cctv`` camera in a high inner corner. All cell geometry is
visual only (``contype=conaffinity=0``, geom group 0) so it adds no contact cost and never
touches the physics. Bays are 4 m apart, so a camera inside one cell can never see another.

``apply_visual_style`` sets the scene-wide look: dark gradient skybox, dimmed headlight,
4096 shadow maps, 8x offscreen MSAA, haze and specular latch materials.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from numpy.typing import NDArray

from teleop.atc.types import LatchPose

mj: Any = _mujoco

CELL_HALF_M = 1.5  # inner half-width of a cell
CELL_OFFSET_X_M = 0.3  # cell centre relative to the robot base (the latch is at +0.5 m)
WALL_HEIGHT_M = 2.2
WALL_THICK_M = 0.06
STRIP_WIDTH_M = 0.08

CCTV_SUFFIX = "cctv"
CCTV_POS_M = (1.05, -1.0, 1.4)  # in the base frame: high front-right inner corner
CCTV_TARGET_M = (0.30, 0.0, 0.34)  # frames full Panda shoulder/elbow clearance + latch
CCTV_FOVY_DEG = 50.0

FLOOR_RGBA = (0.11, 0.115, 0.125, 1.0)
WALL_RGBA = (0.2, 0.21, 0.23, 1.0)
WALL_BAND_RGBA = (0.3, 0.31, 0.33, 1.0)
JIG_RGBA = (0.16, 0.17, 0.19, 1.0)
SAFETY_YELLOW = (0.95, 0.72, 0.05, 1.0)


def _bases() -> dict[int, tuple[float, float, float]]:
    from teleop.sim.bays import BAY_BASES

    return BAY_BASES


def cctv_camera_name(bay_id: int) -> str:
    return f"bay{bay_id}/{CCTV_SUFFIX}"


def cell_center(bay_id: int) -> tuple[float, float]:
    bx, by, _ = _bases()[bay_id]
    return bx + CELL_OFFSET_X_M, by


def cell_bounds(bay_id: int) -> tuple[float, float, float, float]:
    """Inner wall faces ``(x_min, x_max, y_min, y_max)`` of a bay's cell."""
    cx, cy = cell_center(bay_id)
    return cx - CELL_HALF_M, cx + CELL_HALF_M, cy - CELL_HALF_M, cy + CELL_HALF_M


def look_at_xyaxes(
    pos: Sequence[float] | NDArray[np.float64], target: Sequence[float] | NDArray[np.float64]
) -> list[float]:
    """MuJoCo camera ``xyaxes`` looking from ``pos`` at ``target`` with world +Z up."""
    fwd: NDArray[np.float64] = np.asarray(target, dtype=np.float64) - np.asarray(
        pos, dtype=np.float64
    )
    fwd = fwd / float(np.linalg.norm(fwd))
    right = np.asarray(np.cross(fwd, np.array([0.0, 0.0, 1.0])), dtype=np.float64)
    right = right / float(np.linalg.norm(right))
    up = np.asarray(np.cross(right, fwd), dtype=np.float64)
    return [float(v) for v in (*right, *up)]


def _visual(geom: Any) -> None:
    """Visual-only cell geometry. It casts no shadows: the 1-2 mm floor layers and the walls
    would otherwise self-shadow into acne; only the robot and latch should cast."""
    geom.contype = 0
    geom.conaffinity = 0
    geom.group = 0
    geom.castshadow = False


def _has_material(spec: Any, name: str) -> bool:
    return any(str(m.name) == name for m in spec.materials)


def _ensure_material(spec: Any, name: str, rgba: Sequence[float], **kw: float) -> str:
    if not _has_material(spec, name):
        mat = spec.add_material(name=name, rgba=list(rgba))
        for key, value in kw.items():
            setattr(mat, key, value)
    return name


def add_cell(spec: Any, bay_id: int, latch: LatchPose) -> None:
    """Add bay ``bay_id``'s cell: floor tile, jig plate, walls, strips, lights, CCTV camera.

    ``latch`` is the bay's latch pose in the world frame (``bays.latch_world_pose``).
    """
    p = f"bay{bay_id}/"
    bx, by, bz = _bases()[bay_id]
    cx, cy = cell_center(bay_id)
    box = mj.mjtGeom.mjGEOM_BOX
    floor_mat = _ensure_material(spec, "cell_floor", FLOOR_RGBA, specular=0.15, shininess=0.3)
    wall_mat = _ensure_material(spec, "cell_wall", WALL_RGBA, specular=0.05, shininess=0.1)
    band_mat = _ensure_material(spec, "cell_band", WALL_BAND_RGBA, specular=0.1)
    jig_mat = _ensure_material(spec, "cell_jig", JIG_RGBA, specular=0.4, shininess=0.6)
    strip_mat = _ensure_material(spec, "cell_safety", SAFETY_YELLOW, specular=0.2)

    cell = spec.worldbody.add_body(name=f"{p}cell", pos=[cx, cy, bz])
    h = CELL_HALF_M

    # Floor tile 1-2 mm above the physics plane (no z-fighting with it).
    _visual(
        cell.add_geom(
            name=f"{p}cell_floor",
            type=box,
            size=[h, h, 0.0005],
            pos=[0.0, 0.0, 0.0015],
            material=floor_mat,
        )
    )
    # Safety-yellow strips just inside the walls.
    inner = h - STRIP_WIDTH_M / 2 - 0.05
    for side, (sx, sy, ex, ey) in {
        "n": (0.0, inner, h - 0.05, STRIP_WIDTH_M / 2),
        "s": (0.0, -inner, h - 0.05, STRIP_WIDTH_M / 2),
        "e": (inner, 0.0, STRIP_WIDTH_M / 2, h - 0.05),
        "w": (-inner, 0.0, STRIP_WIDTH_M / 2, h - 0.05),
    }.items():
        _visual(
            cell.add_geom(
                name=f"{p}cell_strip_{side}",
                type=box,
                size=[ex, ey, 0.0005],
                pos=[sx, sy, 0.0025],
                material=strip_mat,
            )
        )
    # Walls (outside the inner faces) with a lighter band at eye height.
    t = WALL_THICK_M / 2
    hz = WALL_HEIGHT_M / 2
    for side, (wx, wy, ex, ey) in {
        "n": (0.0, h + t, h + 2 * t, t),
        "s": (0.0, -h - t, h + 2 * t, t),
        "e": (h + t, 0.0, t, h),
        "w": (-h - t, 0.0, t, h),
    }.items():
        _visual(
            cell.add_geom(
                name=f"{p}cell_wall_{side}",
                type=box,
                size=[ex, ey, hz],
                pos=[wx, wy, hz],
                material=wall_mat,
            )
        )
        bx_in = wx - math.copysign(t + 0.002, wx) if wx else 0.0
        by_in = wy - math.copysign(t + 0.002, wy) if wy else 0.0
        _visual(
            cell.add_geom(
                name=f"{p}cell_wall_band_{side}",
                type=box,
                size=[ex if wx == 0.0 else 0.002, ey if wy == 0.0 else 0.002, 0.06],
                pos=[bx_in, by_in, 1.05],
                material=band_mat,
            )
        )

    # Jig plate under the latch (follows its yaw).
    lx, ly, _ = latch.position
    half = 0.5 * latch.yaw_rad
    jig = spec.worldbody.add_body(
        name=f"{p}cell_jig", pos=[lx, ly, 0.0], quat=[math.cos(half), 0.0, 0.0, math.sin(half)]
    )
    _visual(
        jig.add_geom(
            name=f"{p}cell_jig_plate",
            type=box,
            size=[0.12, 0.08, 0.002],
            pos=[0.0, 0.0, 0.004],
            material=jig_mat,
        )
    )

    # Lights: one shadow-casting key spot and one soft fill, both short-ranged to the cell.
    target = np.array([bx + CCTV_TARGET_M[0], by, 0.15])
    key_pos = np.array([bx - 0.4, by - 0.9, 2.05])
    fill_pos = np.array([bx + 1.2, by + 1.0, 1.9])
    for name, pos, diffuse, shadow, cutoff in (
        ("key", key_pos, 0.7, True, 45.0),
        ("fill", fill_pos, 0.3, False, 70.0),
    ):
        d = target - pos
        light = cell.add_light(
            name=f"{p}cell_light_{name}",
            pos=(pos - np.array([cx, cy, bz])).tolist(),
            dir=(d / np.linalg.norm(d)).tolist(),
            diffuse=[diffuse, diffuse * 0.98, diffuse * 0.94],
            specular=[0.3 * diffuse] * 3,
            ambient=[0.0, 0.0, 0.0],
            castshadow=shadow,
            cutoff=cutoff,
            exponent=2.0,
        )
        light.range = 4.0

    cam_pos = np.array([bx, by, bz]) + np.array(CCTV_POS_M)
    cam_target = np.array([bx, by, bz]) + np.array(CCTV_TARGET_M)
    spec.worldbody.add_camera(
        name=cctv_camera_name(bay_id),
        pos=cam_pos.tolist(),
        xyaxes=look_at_xyaxes(cam_pos, cam_target),
        fovy=CCTV_FOVY_DEG,
    )


# Specular materials for the latch geoms. Their explicit geom rgba (grey track, steel bolt,
# orange flap from ``teleop.sim.latch``) still sets the colour; the material adds the shine.
_WHITE = (1.0, 1.0, 1.0, 1.0)
LATCH_MATERIALS = {
    "base_track_visual": ("latch_track", _WHITE, 0.3, 0.4),
    "bolt_bar": ("latch_bolt", _WHITE, 0.35, 0.5),
    "bolt_hook": ("latch_bolt", _WHITE, 0.35, 0.5),
    "flap_paddle": ("latch_flap", _WHITE, 0.3, 0.4),
}


def apply_visual_style(spec: Any) -> None:
    """Scene-wide look: skybox, dim headlight, shadows, MSAA, haze, latch materials."""
    spec.add_texture(
        name="skybox",
        type=mj.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mj.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.09, 0.1, 0.12],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=3072,
    )
    vis = spec.visual
    vis.headlight.ambient = [0.12, 0.12, 0.13]
    vis.headlight.diffuse = [0.1, 0.1, 0.1]
    vis.headlight.specular = [0.05, 0.05, 0.05]
    vis.quality.shadowsize = 4096
    vis.quality.offsamples = 8
    vis.map.haze = 0.25

    for geom in spec.geoms:
        if str(geom.name) == "floor":
            geom.castshadow = False
    _add_track_twins(spec)
    for suffix, (name, rgba, specular, shininess) in LATCH_MATERIALS.items():
        _ensure_material(spec, name, rgba, specular=specular, shininess=shininess)
        for geom in spec.geoms:
            if str(geom.name).endswith(f"/{suffix}"):
                geom.material = name
                geom.castshadow = False  # flat latch boxes self-shadow into acne


HIDDEN_GROUP = 3  # not drawn by the default ``MjvOption`` (groups 0-2 are visible)
TRACK_SINK_M = 0.001


def _add_track_twins(spec: Any) -> None:
    """Fix flap/track z-fighting without touching physics.

    The latched flap's top face is coplanar with the track's top face. Each physics track geom
    moves to a hidden group and gets a visual-only twin whose top sits ``TRACK_SINK_M`` lower.
    """
    for geom in list(spec.geoms):
        name = str(geom.name)
        if not name.endswith("/base_track"):
            continue
        prefix = name[: -len("base_track")]
        geom.group = HIDDEN_GROUP
        size = [float(v) for v in geom.size]
        pos = [float(v) for v in geom.pos]
        twin = spec.body(f"{prefix}latch_base").add_geom(
            name=f"{prefix}base_track_visual",
            type=geom.type,
            size=[size[0], size[1], size[2] - TRACK_SINK_M / 2],
            pos=[pos[0], pos[1], pos[2] - TRACK_SINK_M / 2],
            rgba=[float(v) for v in geom.rgba],
            material="latch_track",
        )
        twin.contype = 0
        twin.conaffinity = 0
        twin.group = 0
        twin.castshadow = False
        twin.mass = 0.0
