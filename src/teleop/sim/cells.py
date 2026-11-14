"""Sustainable EV-battery disassembly & e-waste recycling warehouse scene (fleet ops console).

Places all four robot arms at separate workstation islands inside a single bright, sustainable
recycling warehouse served by twin industrial conveyor belts:

* **Line A (West conveyor, x = -1.50 m):** carries spent EV battery modules past Bay 1 (South)
  and Bay 3 (North).
* **Line B (East conveyor, x = +2.54 m):** carries spent EV battery modules past Bay 2 (South)
  and Bay 4 (North).

The physics ground plane at ``z = 0`` moves to a hidden render group so all physics contacts stay
identical while the visual warehouse opens down to ``WAREHOUSE_FLOOR_Z = -0.75 m``: each Panda arm
stands on a sage-steel pedestal beside its conveyor belt, and each 3-box retention latch sits on top
of a detailed EV battery module pallet (recessed 21700 cell banks, copper busbars, HV badge) riding
the belt. All warehouse, conveyor, pedestal and battery-pack geometry is visual only
(``contype=conaffinity=0``, geom group 0) so it adds zero physics or collision cost.
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

CELL_HALF_M = 1.5
CELL_OFFSET_X_M = 0.3
WALL_HEIGHT_M = 4.60

WAREHOUSE_FLOOR_Z = -0.75
WAREHOUSE_ROOF_Z = 4.60
WAREHOUSE_BOUNDS: tuple[float, float, float, float] = (-5.80, 6.60, -6.40, 6.40)

CONVEYOR_BAYS: tuple[int, ...] = (1, 2, 3, 4)
# Dedicated segmented conveyor belt X position per bay (aligned under each bay's latch).
BAY_CONVEYOR_X: dict[int, float] = {1: -1.50, 2: 2.54, 3: -1.50, 4: 2.54}
CONVEYOR_HALF_LEN_M = 1.48  # 2.96 m per independent belt segment; clear 1.04 m aisle between S/N
CONVEYOR_BED_HALF_W_M = 0.27
CONVEYOR_TOP_Z = -0.052
CONVEYOR_CLEAT_PITCH_M = 0.28
CONVEYOR_SPEED_M_S = 0.08
CONVEYOR_FEED_TRAVEL_M = 0.14  # incoming pack travel before arm descends onto the station
BEACON_FEED_RGBA = (0.18, 0.86, 0.45, 1.0)
BEACON_PAUSE_RGBA = (0.95, 0.72, 0.08, 1.0)
BEACON_ESTOP_RGBA = (0.94, 0.16, 0.16, 1.0)

CCTV_SUFFIX = "cctv"
CCTV_POS_M = (1.30, -1.32, 0.96)
CCTV_TARGET_M = (0.28, 0.18, 0.20)
CCTV_FOVY_DEG = 55.0

CONCRETE_RGB1 = (0.43, 0.47, 0.45)
CONCRETE_RGB2 = (0.38, 0.42, 0.40)
WALL_PANEL_RGBA = (0.76, 0.82, 0.78, 1.0)
WALL_WAINSCOT_RGBA = (0.18, 0.36, 0.27, 1.0)
TIMBER_RGBA = (0.64, 0.46, 0.29, 1.0)
ECO_GREEN_RGBA = (0.19, 0.46, 0.33, 1.0)
ECO_BRIGHT_RGBA = (0.25, 0.64, 0.44, 1.0)
PLANT_DARK_RGBA = (0.15, 0.38, 0.24, 1.0)
PLANT_MID_RGBA = (0.24, 0.56, 0.35, 1.0)
PLANT_LIGHT_RGBA = (0.38, 0.72, 0.46, 1.0)
SKYLIGHT_RGBA = (0.72, 0.88, 0.97, 1.0)
SOLAR_PV_RGBA = (0.11, 0.20, 0.36, 1.0)
BELT_RUBBER_RGBA = (0.14, 0.15, 0.16, 1.0)
BELT_CLEAT_RGBA = (0.24, 0.27, 0.28, 1.0)
STEEL_DARK_RGBA = (0.24, 0.27, 0.29, 1.0)
STEEL_BRUSHED_RGBA = (0.64, 0.68, 0.70, 1.0)
BATTERY_CASE_RGBA = (0.19, 0.25, 0.27, 1.0)
PALLET_RGBA = (0.16, 0.18, 0.20, 1.0)
CELL_TEAL_RGBA = (0.18, 0.58, 0.50, 1.0)
COPPER_RGBA = (0.83, 0.49, 0.22, 1.0)
HV_ORANGE_RGBA = (0.96, 0.46, 0.08, 1.0)
SAFETY_YELLOW = (0.94, 0.74, 0.08, 1.0)
BIN_GREEN_RGBA = (0.18, 0.54, 0.36, 1.0)
BIN_BLUE_RGBA = (0.18, 0.44, 0.68, 1.0)
BIN_AMBER_RGBA = (0.84, 0.58, 0.12, 1.0)


def _bases() -> dict[int, tuple[float, float, float]]:
    from teleop.sim.bays import BAY_BASES

    return BAY_BASES


def cctv_camera_name(bay_id: int) -> str:
    return f"bay{bay_id}/{CCTV_SUFFIX}"


def cell_center(bay_id: int) -> tuple[float, float]:
    bx, by, _ = _bases()[bay_id]
    return bx + CELL_OFFSET_X_M, by


def cell_bounds(bay_id: int) -> tuple[float, float, float, float]:
    """Workstation zone bounds ``(x_min, x_max, y_min, y_max)`` around ``bay_id``."""
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


def _visual(geom: Any, *, shadow: bool = False) -> Any:
    """Visual-only prop geometry: never collides, zero physics cost."""
    geom.contype = 0
    geom.conaffinity = 0
    geom.group = 0
    geom.castshadow = shadow
    geom.mass = 0.0
    return geom


def _has_material(spec: Any, name: str) -> bool:
    return any(str(m.name) == name for m in spec.materials)


def _has_texture(spec: Any, name: str) -> bool:
    return any(str(t.name) == name for t in spec.textures)


def _ensure_material(spec: Any, name: str, rgba: Sequence[float], **kw: float) -> str:
    if not _has_material(spec, name):
        mat = spec.add_material(name=name, rgba=list(rgba))
        for key, value in kw.items():
            setattr(mat, key, value)
    return name


def _yaw_wxyz(yaw_rad: float) -> list[float]:
    h = 0.5 * yaw_rad
    return [math.cos(h), 0.0, 0.0, math.sin(h)]


def _axis_quat_x() -> list[float]:
    """Rotate cylinder local Z axis onto world X axis (90 deg about +Y)."""
    s = math.sqrt(0.5)
    return [s, 0.0, s, 0.0]


def _add_battery_module(
    parent: Any,
    prefix: str,
    pos_xy: tuple[float, float],
    yaw_rad: float,
    *,
    disassembled: bool = False,
    with_dummy_latch: bool = False,
) -> None:
    """Visual EV waste battery module + carrier pallet resting on the conveyor belt."""
    box = mj.mjtGeom.mjGEOM_BOX
    cyl = mj.mjtGeom.mjGEOM_CYLINDER
    holder = parent.add_body(
        name=f"{prefix}_pack",
        pos=[pos_xy[0], pos_xy[1], 0.0],
        quat=_yaw_wxyz(yaw_rad),
    )
    _visual(
        holder.add_geom(
            name=f"{prefix}_pallet",
            type=box,
            size=[0.225, 0.195, 0.009],
            pos=[0.0, 0.0, -0.043],
            rgba=list(PALLET_RGBA),
        )
    )
    for sy in (-1.0, 1.0):
        _visual(
            holder.add_geom(
                name=f"{prefix}_pallet_trim_{'l' if sy < 0 else 'r'}",
                type=box,
                size=[0.226, 0.004, 0.006],
                pos=[0.0, sy * 0.196, -0.042],
                rgba=list(ECO_BRIGHT_RGBA),
            )
        )
    _visual(
        holder.add_geom(
            name=f"{prefix}_housing",
            type=box,
            size=[0.195, 0.170, 0.020],
            pos=[0.0, 0.0, -0.014],
            rgba=list(BATTERY_CASE_RGBA),
        )
    )
    _visual(
        holder.add_geom(
            name=f"{prefix}_hv_badge",
            type=box,
            size=[0.026, 0.018, 0.0015],
            pos=[-0.150, -0.125, 0.007],
            rgba=list(HV_ORANGE_RGBA),
        )
    )
    for bank_idx, sy in enumerate((-1.0, 1.0)):
        y_center = sy * 0.112
        _visual(
            holder.add_geom(
                name=f"{prefix}_cell_well_{bank_idx}",
                type=box,
                size=[0.155, 0.042, 0.002],
                pos=[0.0, y_center, 0.0065],
                rgba=[0.11, 0.14, 0.15, 1.0],
            )
        )
        _visual(
            holder.add_geom(
                name=f"{prefix}_busbar_{bank_idx}",
                type=box,
                size=[0.145, 0.007, 0.0015],
                pos=[0.0, y_center, 0.009],
                rgba=list(COPPER_RGBA),
            )
        )
        for c_idx, cx in enumerate((-0.11, -0.055, 0.0, 0.055, 0.11)):
            for r_idx, dy in enumerate((-0.022, 0.022)):
                _visual(
                    holder.add_geom(
                        name=f"{prefix}_li_cell_{bank_idx}_{c_idx}_{r_idx}",
                        type=cyl,
                        size=[0.015, 0.0025, 0.0],
                        pos=[cx, y_center + dy, 0.0075],
                        rgba=list(STEEL_BRUSHED_RGBA if disassembled else CELL_TEAL_RGBA),
                    )
                )
    if with_dummy_latch:
        bolt_dx = 0.052 if disassembled else 0.0
        _visual(
            holder.add_geom(
                name=f"{prefix}_q_track",
                type=box,
                size=[0.07, 0.04, 0.009],
                pos=[0.0, 0.0, 0.010],
                rgba=[0.55, 0.55, 0.58, 1.0],
            )
        )
        _visual(
            holder.add_geom(
                name=f"{prefix}_q_bolt",
                type=box,
                size=[0.04, 0.012, 0.008],
                pos=[-0.05 + bolt_dx, 0.0, 0.026],
                rgba=[0.75, 0.78, 0.82, 1.0],
            )
        )
        _visual(
            holder.add_geom(
                name=f"{prefix}_q_flap",
                type=box,
                size=[0.02, 0.02, 0.004],
                pos=[0.036, 0.0, 0.018],
                rgba=[1.0, 0.5, 0.05, 1.0],
            )
        )


def _add_bay_conveyor(world: Any, bay_id: int) -> None:
    """Add one independent segmented conveyor belt dedicated to ``bay_id``."""
    bx, by, _ = _bases()[bay_id]
    lx = BAY_CONVEYOR_X[bay_id]
    box = mj.mjtGeom.mjGEOM_BOX
    cyl = mj.mjtGeom.mjGEOM_CYLINDER
    tag = f"bay{bay_id}"
    conv = world.add_body(name=f"warehouse/conveyor_{tag}", pos=[lx, by, 0.0])
    half_y = CONVEYOR_HALF_LEN_M
    bed_w = CONVEYOR_BED_HALF_W_M

    _visual(
        conv.add_geom(
            name=f"conveyor_{tag}_belt",
            type=box,
            size=[bed_w, half_y, 0.016],
            pos=[0.0, 0.0, CONVEYOR_TOP_Z - 0.016],
            rgba=list(BELT_RUBBER_RGBA),
        )
    )
    n_cleats = int((2.0 * half_y) / CONVEYOR_CLEAT_PITCH_M)
    y_start = -half_y + 0.14
    for i in range(n_cleats):
        cy = y_start + i * CONVEYOR_CLEAT_PITCH_M
        _visual(
            conv.add_geom(
                name=f"conveyor_cleat_{tag}_{i:02d}",
                type=box,
                size=[bed_w - 0.015, 0.012, 0.003],
                pos=[0.0, cy, CONVEYOR_TOP_Z + 0.003],
                rgba=list(BELT_CLEAT_RGBA),
            )
        )

    rail_dx = bed_w + 0.026
    for side_tag, sx in (("l", -1.0), ("r", 1.0)):
        _visual(
            conv.add_geom(
                name=f"conveyor_{tag}_rail_{side_tag}",
                type=box,
                size=[0.024, half_y + 0.04, 0.032],
                pos=[sx * rail_dx, 0.0, CONVEYOR_TOP_Z - 0.014],
                rgba=list(ECO_GREEN_RGBA),
            )
        )
        _visual(
            conv.add_geom(
                name=f"conveyor_{tag}_stripe_{side_tag}",
                type=box,
                size=[0.003, half_y + 0.04, 0.007],
                pos=[sx * (rail_dx + 0.025), 0.0, CONVEYOR_TOP_Z - 0.008],
                rgba=list(SAFETY_YELLOW),
            )
        )

    # Motor housing + E-stop / feeding status beacon mounted on the outer conveyor rail.
    beacon_sx = -1.0 if bx < 0.0 else 1.0
    beacon_x = beacon_sx * (rail_dx + 0.065)
    _visual(
        conv.add_geom(
            name=f"conveyor_{tag}_motor",
            type=box,
            size=[0.045, 0.09, 0.065],
            pos=[beacon_x, -0.36, CONVEYOR_TOP_Z - 0.02],
            rgba=list(STEEL_DARK_RGBA),
        )
    )
    _visual(
        conv.add_geom(
            name=f"conveyor_{tag}_beacon_post",
            type=cyl,
            size=[0.012, 0.11, 0.0],
            pos=[beacon_x, -0.36, CONVEYOR_TOP_Z + 0.13],
            rgba=list(STEEL_BRUSHED_RGBA),
        )
    )
    _visual(
        conv.add_geom(
            name=f"{tag}/conveyor_beacon",
            type=cyl,
            size=[0.036, 0.034, 0.0],
            pos=[beacon_x, -0.36, CONVEYOR_TOP_Z + 0.255],
            rgba=list(BEACON_FEED_RGBA),
        )
    )

    q_x = _axis_quat_x()
    for end_tag, ey in (("south", -half_y), ("north", half_y)):
        _visual(
            conv.add_geom(
                name=f"conveyor_{tag}_drum_{end_tag}",
                type=cyl,
                size=[0.036, bed_w + 0.02, 0.0],
                pos=[0.0, ey, CONVEYOR_TOP_Z - 0.018],
                quat=q_x,
                rgba=list(STEEL_BRUSHED_RGBA),
            )
        )
    for r_i, ry in enumerate(np.linspace(-half_y + 0.35, half_y - 0.35, 5)):
        _visual(
            conv.add_geom(
                name=f"conveyor_{tag}_roller_{r_i}",
                type=cyl,
                size=[0.018, bed_w - 0.01, 0.0],
                pos=[0.0, float(ry), CONVEYOR_TOP_Z - 0.044],
                quat=q_x,
                rgba=list(STEEL_BRUSHED_RGBA),
            )
        )

    leg_top = CONVEYOR_TOP_Z - 0.046
    leg_hz = 0.5 * (leg_top - WAREHOUSE_FLOOR_Z)
    leg_cz = 0.5 * (leg_top + WAREHOUSE_FLOOR_Z)
    for l_i, ly in enumerate((-half_y + 0.20, 0.0, half_y - 0.20)):
        for side_tag, sx in (("l", -1.0), ("r", 1.0)):
            _visual(
                conv.add_geom(
                    name=f"conveyor_{tag}_leg_{l_i}_{side_tag}",
                    type=box,
                    size=[0.025, 0.025, leg_hz],
                    pos=[sx * (bed_w + 0.01), ly, leg_cz],
                    rgba=list(STEEL_DARK_RGBA),
                )
            )
        _visual(
            conv.add_geom(
                name=f"conveyor_{tag}_brace_{l_i}",
                type=box,
                size=[bed_w + 0.01, 0.018, 0.018],
                pos=[0.0, ly, WAREHOUSE_FLOOR_Z + 0.28],
                rgba=list(STEEL_DARK_RGBA),
            )
        )

    # Upstream queued pack & downstream disassembled pack riding this bay's segmented belt.
    yaw = 0.25 if bay_id == 2 else 0.0
    _add_battery_module(
        conv,
        f"conveyor_{tag}_in",
        (0.0, -0.92),
        yaw,
        disassembled=False,
        with_dummy_latch=True,
    )
    _add_battery_module(
        conv,
        f"conveyor_{tag}_out",
        (0.0, 0.92),
        yaw,
        disassembled=True,
        with_dummy_latch=True,
    )


def _add_living_green_wall(
    parent: Any, name: str, pos: Sequence[float], half_xyz: Sequence[float]
) -> None:
    """Timber-framed vertical biophilic living plant wall."""
    box = mj.mjtGeom.mjGEOM_BOX
    px, py, pz = (float(v) for v in pos)
    hx, hy, hz = (float(v) for v in half_xyz)
    _visual(
        parent.add_geom(
            name=f"{name}_frame",
            type=box,
            size=[hx + 0.03, hy + 0.03, hz + 0.05],
            pos=[px, py, pz],
            rgba=list(TIMBER_RGBA),
        )
    )
    inward_x = -0.025 if px > 2.0 else (0.025 if px < -2.0 else 0.0)
    inward_y = -0.025 if py > 2.0 else (0.025 if py < -2.0 else 0.0)
    _visual(
        parent.add_geom(
            name=f"{name}_moss",
            type=box,
            size=[hx, hy, hz],
            pos=[px + inward_x, py + inward_y, pz],
            rgba=list(PLANT_DARK_RGBA),
        )
    )
    shades = (PLANT_MID_RGBA, PLANT_LIGHT_RGBA, ECO_BRIGHT_RGBA)
    for idx, frac in enumerate((-0.62, -0.20, 0.22, 0.62)):
        z_band = pz + frac * hz * 0.82
        _visual(
            parent.add_geom(
                name=f"{name}_foliage_{idx}",
                type=box,
                size=[
                    hx * (0.94 if abs(inward_y) > 0 else 1.08),
                    hy * (0.94 if abs(inward_x) > 0 else 1.08),
                    hz * 0.15,
                ],
                pos=[px + inward_x * 1.7, py + inward_y * 1.7, z_band],
                rgba=list(shades[idx % len(shades)]),
            )
        )


def _add_recycling_bins(parent: Any, prefix: str, cx: float, cy: float) -> None:
    """Trio of colour-coded EV battery material recovery bins (cells, Cu/Al, BMS e-waste)."""
    box = mj.mjtGeom.mjGEOM_BOX
    z0 = WAREHOUSE_FLOOR_Z
    specs = (
        ("cells", -0.36, BIN_GREEN_RGBA, CELL_TEAL_RGBA),
        ("metals", 0.0, BIN_BLUE_RGBA, COPPER_RGBA),
        ("ewaste", 0.36, BIN_AMBER_RGBA, STEEL_BRUSHED_RGBA),
    )
    for tag, dy, bin_rgba, fill_rgba in specs:
        _visual(
            parent.add_geom(
                name=f"{prefix}_bin_{tag}",
                type=box,
                size=[0.20, 0.15, 0.24],
                pos=[cx, cy + dy, z0 + 0.24],
                rgba=list(bin_rgba),
            )
        )
        _visual(
            parent.add_geom(
                name=f"{prefix}_bin_{tag}_rim",
                type=box,
                size=[0.215, 0.165, 0.022],
                pos=[cx, cy + dy, z0 + 0.49],
                rgba=list(STEEL_DARK_RGBA),
            )
        )
        _visual(
            parent.add_geom(
                name=f"{prefix}_bin_{tag}_fill",
                type=box,
                size=[0.17, 0.12, 0.015],
                pos=[cx, cy + dy, z0 + 0.46],
                rgba=list(fill_rgba),
            )
        )


def _add_warehouse_shell(spec: Any, wh: Any) -> None:
    """Floor slab, eco-walkway, perimeter walls, portals, timber trusses & skylights."""
    box = mj.mjtGeom.mjGEOM_BOX
    x0, x1, y0, y1 = WAREHOUSE_BOUNDS
    z_floor = WAREHOUSE_FLOOR_Z
    z_roof = WAREHOUSE_ROOF_Z
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    hx = 0.5 * (x1 - x0)
    hy = 0.5 * (y1 - y0)

    if not _has_texture(spec, "warehouse_floor_tex"):
        spec.add_texture(
            name="warehouse_floor_tex",
            type=mj.mjtTexture.mjTEXTURE_2D,
            builtin=mj.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=list(CONCRETE_RGB1),
            rgb2=list(CONCRETE_RGB2),
            mark=mj.mjtMark.mjMARK_EDGE,
            markrgb=[0.58, 0.63, 0.60],
            width=256,
            height=256,
        )
    if not _has_material(spec, "warehouse_floor_mat"):
        fmat = spec.add_material(
            name="warehouse_floor_mat",
            rgba=[1.0, 1.0, 1.0, 1.0],
            specular=0.12,
            shininess=0.25,
            reflectance=0.04,
        )
        fmat.textures = ["warehouse_floor_tex"] + [""] * (len(fmat.textures) - 1)
        fmat.texrepeat = [16.0, 16.0]

    _visual(
        wh.add_geom(
            name="warehouse/floor",
            type=box,
            size=[hx + 0.25, hy + 0.25, 0.04],
            pos=[cx, cy, z_floor - 0.04],
            rgba=[1.0, 1.0, 1.0, 1.0],
            material="warehouse_floor_mat",
        )
    )

    aisle_x = 0.42
    _visual(
        wh.add_geom(
            name="warehouse/eco_aisle",
            type=box,
            size=[0.52, hy - 0.15, 0.002],
            pos=[aisle_x, cy, z_floor + 0.002],
            rgba=list(ECO_GREEN_RGBA),
        )
    )
    for tag, sx in (("w", -1.0), ("e", 1.0)):
        _visual(
            wh.add_geom(
                name=f"warehouse/eco_aisle_line_{tag}",
                type=box,
                size=[0.03, hy - 0.15, 0.0025],
                pos=[aisle_x + sx * 0.54, cy, z_floor + 0.0025],
                rgba=[0.90, 0.94, 0.91, 1.0],
            )
        )

    wall_h = z_roof - z_floor
    wall_cz = 0.5 * (z_roof + z_floor)
    wain_h = 1.35
    wain_cz = z_floor + 0.5 * wain_h
    t = 0.08
    for side, (wx, wy, sx, sy) in {
        "north": (cx, y1 + t, hx + 2 * t, t),
        "south": (cx, y0 - t, hx + 2 * t, t),
        "east": (x1 + t, cy, t, hy + 2 * t),
        "west": (x0 - t, cy, t, hy + 2 * t),
    }.items():
        _visual(
            wh.add_geom(
                name=f"warehouse/wall_{side}",
                type=box,
                size=[sx, sy, 0.5 * wall_h],
                pos=[wx, wy, wall_cz],
                rgba=list(WALL_PANEL_RGBA),
            )
        )
        in_x = -0.015 if side == "east" else (0.015 if side == "west" else 0.0)
        in_y = -0.015 if side == "north" else (0.015 if side == "south" else 0.0)
        _visual(
            wh.add_geom(
                name=f"warehouse/wainscot_{side}",
                type=box,
                size=[
                    sx * (0.99 if in_x == 0 else 0.25),
                    sy * (0.99 if in_y == 0 else 0.25),
                    0.5 * wain_h,
                ],
                pos=[wx + in_x, wy + in_y, wain_cz],
                rgba=list(WALL_WAINSCOT_RGBA),
            )
        )
        _visual(
            wh.add_geom(
                name=f"warehouse/clerestory_{side}",
                type=box,
                size=[
                    sx * 0.88 if in_x == 0 else 0.018,
                    sy * 0.88 if in_y == 0 else 0.018,
                    0.52,
                ],
                pos=[wx + in_x * 1.4, wy + in_y * 1.4, 2.95],
                rgba=list(SKYLIGHT_RGBA),
            )
        )

    # Cross-warehouse green eco-walkway separating South conveyor segments (Bays 1-2)
    # from North conveyor segments (Bays 3-4).
    _visual(
        wh.add_geom(
            name="warehouse/cross_aisle",
            type=box,
            size=[hx - 0.30, 0.34, 0.002],
            pos=[cx, 0.0, z_floor + 0.002],
            rgba=list(ECO_GREEN_RGBA),
        )
    )

    for tag, lx in (("w", -1.50), ("e", 2.54)):
        for ptag, py in (("infeed", y0 + 0.02), ("outfeed", y1 - 0.02)):
            _visual(
                wh.add_geom(
                    name=f"warehouse/portal_{tag}_{ptag}_opening",
                    type=box,
                    size=[0.42, 0.04, 0.42],
                    pos=[lx, py, z_floor + 0.82],
                    rgba=[0.08, 0.11, 0.10, 1.0],
                )
            )
            _visual(
                wh.add_geom(
                    name=f"warehouse/portal_{tag}_{ptag}_hood",
                    type=box,
                    size=[0.48, 0.07, 0.06],
                    pos=[lx, py, z_floor + 1.28],
                    rgba=list(ECO_BRIGHT_RGBA),
                )
            )


def add_warehouse(spec: Any, include_background: bool = True) -> None:
    """Build the shared sustainable battery-recycling warehouse and twin conveyor lines."""
    world = spec.worldbody
    box = mj.mjtGeom.mjGEOM_BOX
    x0, x1, y0, y1 = WAREHOUSE_BOUNDS
    z_floor = WAREHOUSE_FLOOR_Z
    z_roof = WAREHOUSE_ROOF_Z
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    hx = 0.5 * (x1 - x0)
    wall_h = z_roof - z_floor
    wall_cz = 0.5 * (z_roof + z_floor)
    aisle_x = 0.42

    wh = world.add_body(name="warehouse/shell", pos=[0.0, 0.0, 0.0])
    _add_warehouse_shell(spec, wh)

    for c_i, col_y in enumerate((-5.6, -1.9, 1.9, 5.6)):
        for side_tag, col_x in (("w", x0 + 0.12), ("e", x1 - 0.12)):
            _visual(
                wh.add_geom(
                    name=f"warehouse/timber_col_{side_tag}_{c_i}",
                    type=box,
                    size=[0.11, 0.11, 0.5 * wall_h],
                    pos=[col_x, col_y, wall_cz],
                    rgba=list(TIMBER_RGBA),
                )
            )
        _visual(
            wh.add_geom(
                name=f"warehouse/timber_beam_{c_i}",
                type=box,
                size=[hx, 0.09, 0.14],
                pos=[cx, col_y, z_roof - 0.22],
                rgba=list(TIMBER_RGBA),
            )
        )
        _visual(
            wh.add_geom(
                name=f"warehouse/skylight_{c_i}",
                type=box,
                size=[hx * 0.82, 0.55, 0.02],
                pos=[cx, col_y + 0.9, z_roof - 0.05],
                rgba=list(SKYLIGHT_RGBA),
            )
        )
        _visual(
            wh.add_geom(
                name=f"warehouse/solar_pv_{c_i}",
                type=box,
                size=[hx * 0.72, 0.36, 0.02],
                pos=[cx, col_y - 0.85, z_roof - 0.03],
                rgba=list(SOLAR_PV_RGBA),
            )
        )

    for g_i, gx in enumerate((-3.8, 0.42, 4.5)):
        _add_living_green_wall(
            wh, f"warehouse/greenwall_n_{g_i}", (gx, y1 - 0.04, 1.25), (1.05, 0.03, 0.72)
        )
    # Sunlit mullioned eco-windows flanking the living walls on North and East walls.
    for w_i, wx in enumerate((-1.65, 2.45)):
        _visual(
            wh.add_geom(
                name=f"warehouse/window_n_{w_i}_frame",
                type=box,
                size=[0.66, 0.03, 0.62],
                pos=[wx, y1 - 0.03, 1.85],
                rgba=list(TIMBER_RGBA),
            )
        )
        _visual(
            wh.add_geom(
                name=f"warehouse/window_n_{w_i}_glass",
                type=box,
                size=[0.60, 0.035, 0.56],
                pos=[wx, y1 - 0.04, 1.85],
                rgba=list(SKYLIGHT_RGBA),
            )
        )
    for w_i, wy in enumerate((-1.9, 1.9)):
        _visual(
            wh.add_geom(
                name=f"warehouse/window_e_{w_i}_frame",
                type=box,
                size=[0.03, 0.66, 0.62],
                pos=[x1 - 0.03, wy, 1.85],
                rgba=list(TIMBER_RGBA),
            )
        )
        _visual(
            wh.add_geom(
                name=f"warehouse/window_e_{w_i}_glass",
                type=box,
                size=[0.035, 0.60, 0.56],
                pos=[x1 - 0.04, wy, 1.85],
                rgba=list(SKYLIGHT_RGBA),
            )
        )
    # Circular-economy green recycling emblem flanking the central North green wall.
    for s_i, (ang_deg, dx, dz) in enumerate(
        ((0.0, 0.0, 2.38), (60.0, -0.26, 2.68), (-60.0, 0.26, 2.68))
    ):
        rad = math.radians(ang_deg)
        _visual(
            wh.add_geom(
                name=f"warehouse/eco_emblem_{s_i}",
                type=box,
                size=[0.24, 0.025, 0.055],
                pos=[aisle_x + dx, y1 - 0.03, dz],
                quat=[math.cos(0.5 * rad), 0.0, math.sin(0.5 * rad), 0.0],
                rgba=list(ECO_BRIGHT_RGBA),
            )
        )
    for g_i, gy in enumerate((-3.8, 0.0, 3.8)):
        _add_living_green_wall(
            wh, f"warehouse/greenwall_e_{g_i}", (x1 - 0.04, gy, 1.25), (0.03, 1.05, 0.72)
        )
        _add_living_green_wall(
            wh, f"warehouse/greenwall_w_{g_i}", (x0 + 0.04, gy, 1.25), (0.03, 0.95, 0.68)
        )

    for p_i, py in enumerate((-4.4, 4.4)):
        _visual(
            wh.add_geom(
                name=f"warehouse/planter_box_{p_i}",
                type=box,
                size=[0.30, 0.55, 0.20],
                pos=[aisle_x, py, z_floor + 0.20],
                rgba=list(TIMBER_RGBA),
            )
        )
        _visual(
            wh.add_geom(
                name=f"warehouse/planter_bush_{p_i}",
                type=box,
                size=[0.26, 0.50, 0.16],
                pos=[aisle_x, py, z_floor + 0.48],
                rgba=list(PLANT_MID_RGBA),
            )
        )
        _visual(
            wh.add_geom(
                name=f"warehouse/planter_canopy_{p_i}",
                type=box,
                size=[0.20, 0.38, 0.12],
                pos=[aisle_x, py, z_floor + 0.70],
                rgba=list(PLANT_LIGHT_RGBA),
            )
        )

    _add_recycling_bins(wh, "warehouse/island_s", aisle_x, -2.1)
    _add_recycling_bins(wh, "warehouse/island_n", aisle_x, 2.1)

    for b_i, by in enumerate((-2.0, 2.0)):
        _visual(
            wh.add_geom(
                name=f"warehouse/bess_cab_{b_i}",
                type=box,
                size=[0.28, 0.62, 0.85],
                pos=[x1 - 0.34, by, z_floor + 0.85],
                rgba=[0.88, 0.92, 0.90, 1.0],
            )
        )
        _visual(
            wh.add_geom(
                name=f"warehouse/bess_led_{b_i}",
                type=box,
                size=[0.015, 0.46, 0.035],
                pos=[x1 - 0.625, by, z_floor + 1.48],
                rgba=list(ECO_BRIGHT_RGBA),
            )
        )

    # Four independent segmented conveyor belts (one per bay), all inside the same warehouse.
    active_conveyors = (1, 2, 3, 4) if include_background else (1, 2)
    for bay_id in active_conveyors:
        _add_bay_conveyor(world, bay_id)

    # Two directional daylight/bounce lights keep total scene lights <= 7 (under MuJoCo's 8-light
    # OpenGL shader limit including headlight) while evenly illuminating every wall and conveyor.
    world.add_light(
        name="warehouse/sun",
        type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[cx - 2.0, cy - 3.0, z_roof + 2.0],
        dir=[0.22, 0.34, -0.91],
        diffuse=[0.40, 0.42, 0.39],
        specular=[0.14, 0.14, 0.13],
        ambient=[0.10, 0.11, 0.10],
        castshadow=False,
    )
    world.add_light(
        name="warehouse/sky_bounce",
        type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[cx + 2.0, cy + 3.0, z_roof + 2.0],
        dir=[-0.28, -0.32, -0.90],
        diffuse=[0.24, 0.26, 0.25],
        specular=[0.06, 0.06, 0.06],
        ambient=[0.0, 0.0, 0.0],
        castshadow=False,
    )


def add_cell(spec: Any, bay_id: int, latch: LatchPose) -> None:
    """Add bay ``bay_id``'s workstation: pedestal, floor markings, EV battery pack, lights, CCTV."""
    p = f"bay{bay_id}/"
    bx, by, bz = _bases()[bay_id]
    cx, cy = cell_center(bay_id)
    box = mj.mjtGeom.mjGEOM_BOX
    cyl = mj.mjtGeom.mjGEOM_CYLINDER
    z_floor = WAREHOUSE_FLOOR_Z

    cell = spec.worldbody.add_body(name=f"{p}cell", pos=[cx, cy, bz])

    # Anti-fatigue workstation floor mat + safety-yellow floor markings at z = -0.75 m.
    _visual(
        cell.add_geom(
            name=f"{p}cell_floor",
            type=box,
            size=[0.92, 0.92, 0.002],
            pos=[-0.08, 0.0, z_floor + 0.002],
            rgba=[0.20, 0.23, 0.22, 1.0],
        )
    )
    for tag, (sx, sy, ex, ey) in {
        "n": (-0.08, 0.90, 0.92, 0.035),
        "s": (-0.08, -0.90, 0.92, 0.035),
        "w": (-0.98, 0.0, 0.035, 0.935),
    }.items():
        _visual(
            cell.add_geom(
                name=f"{p}cell_strip_{tag}",
                type=box,
                size=[ex, ey, 0.0025],
                pos=[sx, sy, z_floor + 0.003],
                rgba=list(SAFETY_YELLOW),
            )
        )

    # Robot pedestal column anchored to the warehouse floor under (bx, by, 0.0).
    px_rel = bx - cx
    py_rel = by - cy
    _visual(
        cell.add_geom(
            name=f"{p}cell_pedestal_base",
            type=cyl,
            size=[0.24, 0.025, 0.0],
            pos=[px_rel, py_rel, z_floor + 0.025],
            rgba=list(STEEL_DARK_RGBA),
        )
    )
    _visual(
        cell.add_geom(
            name=f"{p}cell_pedestal_col",
            type=cyl,
            size=[0.132, 0.348, 0.0],
            pos=[px_rel, py_rel, -0.377],
            rgba=list(ECO_GREEN_RGBA),
        )
    )
    _visual(
        cell.add_geom(
            name=f"{p}cell_pedestal_collar",
            type=cyl,
            size=[0.155, 0.014, 0.0],
            pos=[px_rel, py_rel, -0.015],
            rgba=list(STEEL_BRUSHED_RGBA),
        )
    )

    # EV waste battery module + pallet sitting on the conveyor belt beneath the bay's latch.
    lx, ly, _ = latch.position
    _add_battery_module(
        cell,
        f"{p}cell_jig",
        (lx - cx, ly - cy),
        latch.yaw_rad,
        disassembled=False,
        with_dummy_latch=False,
    )

    # Local shadow-casting luminaire (1 per bay -> 6 scene lights <= MuJoCo's 8-light OpenGL cap).
    target = np.array([lx, ly, 0.08], dtype=np.float64)
    key_pos = np.array([cx - 0.35, cy - 0.85, bz + 2.15], dtype=np.float64)
    d = target - key_pos
    cell.add_light(
        name=f"{p}cell_light_key",
        pos=(key_pos - np.array([cx, cy, bz])).tolist(),
        dir=(d / np.linalg.norm(d)).tolist(),
        diffuse=[0.38, 0.38, 0.36],
        specular=[0.18, 0.18, 0.17],
        ambient=[0.0, 0.0, 0.0],
        castshadow=True,
        cutoff=50.0,
        exponent=1.6,
    )

    # Angle West-line cameras (bx < 0) inward (+X/+Y) and East-line cameras (bx > 0) inward (-X/+Y)
    # so every station camera frames its own arm + conveyor in front and the wider warehouse behind.
    cam_offset = (-0.98, -1.28, 0.94) if bx < 0.0 else CCTV_POS_M
    tgt_offset = (0.38, 0.16, 0.18) if bx < 0.0 else CCTV_TARGET_M
    cam_pos = np.array([bx, by, bz]) + np.array(cam_offset)
    cam_target = np.array([bx, by, bz]) + np.array(tgt_offset)
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
    """Scene-wide sustainable daylight look: skybox, natural light, shadows, MSAA, latch shine."""
    spec.add_texture(
        name="skybox",
        type=mj.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mj.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.50, 0.73, 0.90],
        rgb2=[0.85, 0.92, 0.88],
        width=512,
        height=3072,
    )
    vis = spec.visual
    vis.headlight.ambient = [0.14, 0.15, 0.14]
    vis.headlight.diffuse = [0.14, 0.14, 0.13]
    vis.headlight.specular = [0.06, 0.06, 0.06]
    vis.quality.shadowsize = 4096
    vis.quality.offsamples = 8
    vis.map.haze = 0.06

    # Hide the physics ground plane at z = 0 so the visual warehouse floor (z = -0.75 m),
    # pedestals and twin conveyor belts are visible without altering any physics contact.
    for geom in spec.geoms:
        if str(geom.name) == "floor":
            geom.castshadow = False
            geom.group = HIDDEN_GROUP
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
