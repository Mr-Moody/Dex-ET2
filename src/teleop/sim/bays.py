"""4-bay EV-latch fleet scene (spec 2.2, 2.3): Franka Panda + right LEAP Hand + latch per bay.

Bays 1-2 are active physics. Bays 3-4 are a visual-only background row (every geom has
``contype=conaffinity=0``) so rendering them adds no contact cost. Each bay's robot base frame
has no rotation (the arm reaches along world +X), and latch poses passed to the builders are
**bay-local**, relative to that base. ``latch_world_pose`` converts to the world frame.

Names are prefixed per bay: ``bay{k}/joint1..7``, ``bay{k}/actuator1..7``, the 16 LEAP joints
and actuators (``bay{k}/if_mcp_act`` ...), ``bay{k}/wrist_site`` at the LEAP palm frame, and
the latch bodies, joints and equalities from ``teleop.sim.latch``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]

from teleop.atc.types import N_BAYS, LatchPose
from teleop.sim.cells import add_cell, apply_visual_style, look_at_xyaxes
from teleop.sim.hands import REPO_ROOT, load_hand
from teleop.sim.ik import ARM_JOINTS, PANDA_HOME
from teleop.sim.latch import add_latch_to_spec
from teleop.types import Pose

mj: Any = _mujoco

PANDA_XML = REPO_ROOT / "third_party" / "menagerie" / "franka_emika_panda" / "panda_nohand.xml"

# 4 m apart on a 2x2 grid: each bay sits in its own walled cell (``teleop.sim.cells``), so no
# bay's CCTV camera can see another bay.
BAY_BASES: dict[int, tuple[float, float, float]] = {
    1: (-2.0, -2.0, 0.0),
    2: (2.0, -2.0, 0.0),
    3: (-2.0, 2.0, 0.0),
    4: (2.0, 2.0, 0.0),
}
ACTIVE_BAYS = (1, 2)
BACKGROUND_BAYS = (3, 4)

CANONICAL_LATCH = LatchPose(position=(0.50, 0.0, 0.02), yaw_rad=0.0)
# Bay 2's perturbed latch: shifted ~12 cm and rotated +35 deg (spec 2.3).
BAY2_LATCH = CANONICAL_LATCH.offset(dx=0.08, dy=0.09, dyaw_rad=math.radians(35.0))

# The LEAP palm frame sits at the knuckles; its wrist end is ~0.10 m back along palm -X. Mount
# it so the fingers point along the flange's tool axis (+Z) and palm +Z faces flange -X.
PALM_MOUNT_POS = (0.0, 0.0, 0.10)
PALM_MOUNT_QUAT = (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)  # wxyz, -90 deg about flange Y

TIMESTEP = 1.0 / 600
OVERVIEW_CAMERA = "overview"


def bay_prefix(bay_id: int) -> str:
    if not 1 <= bay_id <= N_BAYS:
        raise ValueError(f"bay_id must be in 1..{N_BAYS}, got {bay_id}")
    return f"bay{bay_id}/"


def latch_world_pose(bay_id: int, local: LatchPose) -> LatchPose:
    """World pose of a bay-local latch pose (bay base frames are unrotated)."""
    bx, by, bz = BAY_BASES[bay_id]
    x, y, z = local.position
    return LatchPose(position=(bx + x, by + y, bz + z), yaw_rad=local.yaw_rad)


def wrist_pose_over_latch(
    world_latch: LatchPose, height: float, along_x: float = 0.0, along_y: float = 0.0
) -> Pose:
    """World wrist-site pose above a latch: fingers pointing down, palm normal along latch +X.

    ``along_x``/``along_y`` offset the wrist in the latch frame; ``height`` is above its origin.
    """
    c, s = math.cos(world_latch.yaw_rad), math.sin(world_latch.yaw_rad)
    x, y, z = world_latch.position
    position = (x + c * along_x - s * along_y, y + s * along_x + c * along_y, z + height)
    # Rz(yaw) * Ry(+90 deg): palm +X (fingers) -> world -Z, palm +Z -> latch +X.
    hz, hy = 0.5 * world_latch.yaw_rad, 0.25 * math.pi
    orientation = (
        -math.sin(hz) * math.sin(hy),
        math.cos(hz) * math.sin(hy),
        math.sin(hz) * math.cos(hy),
        math.cos(hz) * math.cos(hy),
    )
    return Pose(orientation=orientation, position=position)


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run `uv run python scripts/fetch_models.py`")
    return path


def _sync_options(parent: Any, child: Any) -> None:
    """Match the child's global options to the parent's so attach() raises no conflicts."""
    child.option.timestep = parent.option.timestep
    child.option.integrator = parent.option.integrator
    child.option.cone = parent.option.cone
    child.option.impratio = parent.option.impratio


def _strip_globals(child: Any) -> None:
    for key in list(child.keys):
        child.delete(key)
    for light in list(child.lights):
        child.delete(light)


def _new_spec(name: str) -> Any:
    spec: Any = mj.MjSpec()
    spec.modelname = name
    spec.compiler.degree = False
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mj.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 100.0
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    world = spec.worldbody
    world.add_geom(
        name="floor",
        type=mj.mjtGeom.mjGEOM_PLANE,
        size=[7.0, 7.0, 0.1],
        rgba=[0.32, 0.34, 0.38, 1.0],
        friction=[1.0, 0.005, 0.0001],
    )
    return spec


def _add_robot(spec: Any, bay_id: int, *, collide: bool) -> None:
    prefix = bay_prefix(bay_id)
    panda: Any = mj.MjSpec.from_file(str(_require(PANDA_XML)))
    hand: Any = mj.MjSpec.from_file(str(_require(load_hand("leap").mjcf["right"])))
    for child in (panda, hand):
        _strip_globals(child)
        _sync_options(spec, child)

    # Both models ship a material named "white"; rename the hand's (and its references).
    for mat in list(hand.materials):
        old = str(mat.name)
        mat.name = f"leap_{old}"
        for geom in hand.geoms:
            if geom.material == old:
                geom.material = mat.name

    palm = hand.body("palm")
    palm.pos = list(PALM_MOUNT_POS)
    palm.quat = list(PALM_MOUNT_QUAT)
    palm.add_site(name="wrist_site", size=[0.006, 0.006, 0.006], rgba=[0.1, 0.9, 0.2, 0.6])

    frame = spec.worldbody.add_frame(pos=list(BAY_BASES[bay_id]))
    spec.attach(panda, frame=frame, prefix=prefix)
    spec.attach(hand, site=spec.site(f"{prefix}attachment_site"), prefix=prefix)

    # Runs before the latch is added, so every prefixed body here belongs to the robot.
    for body in spec.bodies:
        if not str(body.name).startswith(prefix):
            continue
        body.gravcomp = 1.0  # torque-controlled arm and hand compensate gravity
        if not collide:
            for geom in body.geoms:
                geom.contype = 0
                geom.conaffinity = 0


def _add_bay(spec: Any, bay_id: int, local_latch: LatchPose, *, collide: bool) -> None:
    _add_robot(spec, bay_id, collide=collide)
    add_latch_to_spec(
        spec, bay_prefix(bay_id), latch_world_pose(bay_id, local_latch), collide=collide
    )


def _add_overview_camera(spec: Any, include_background: bool) -> None:
    """High bird's-eye view down into the open-top cells (the passive viewer's fallback)."""
    target = (0.3, 0.0 if include_background else -2.0, 0.0)
    pos = (0.3, target[1] - 4.5, 9.0)
    spec.worldbody.add_camera(
        name=OVERVIEW_CAMERA, pos=list(pos), xyaxes=look_at_xyaxes(pos, target), fovy=60.0
    )


def build_four_bay_spec(
    bay1_pose: LatchPose = CANONICAL_LATCH,
    bay2_pose: LatchPose = BAY2_LATCH,
    include_background_bays: bool = True,
) -> Any:
    from teleop.retarget.clutch import add_rescue_cameras

    spec = _new_spec("atc_four_bay")
    poses = {1: bay1_pose, 2: bay2_pose}
    if include_background_bays:
        poses.update({bay_id: CANONICAL_LATCH for bay_id in BACKGROUND_BAYS})
    for bay_id, pose in poses.items():
        _add_bay(spec, bay_id, pose, collide=bay_id in ACTIVE_BAYS)
        add_cell(spec, bay_id, latch_world_pose(bay_id, pose))
    for bay_id in ACTIVE_BAYS:
        add_rescue_cameras(spec, bay_prefix(bay_id), latch_world_pose(bay_id, poses[bay_id]))
    apply_visual_style(spec)
    _add_overview_camera(spec, include_background_bays)
    return spec


def build_four_bay_scene(
    bay1_pose: LatchPose = CANONICAL_LATCH,
    bay2_pose: LatchPose = BAY2_LATCH,
    include_background_bays: bool = True,
) -> Any:  # mujoco.MjModel
    """Compile the fleet scene. Latch poses are bay-local (see module docstring)."""
    return build_four_bay_spec(bay1_pose, bay2_pose, include_background_bays).compile()


def build_single_bay_scene(pose: LatchPose = CANONICAL_LATCH) -> Any:  # mujoco.MjModel
    """One active Bay 1 (base at its fleet position) for fast headless validation rollouts."""
    spec = _new_spec("atc_single_bay")
    _add_bay(spec, 1, pose, collide=True)
    return spec.compile()


def reset_robot_home(model: Any, data: Any, bay_id: int) -> None:
    """Put a bay's arm at ``PANDA_HOME`` with an open hand, at rest, ctrl matching qpos."""
    prefix = bay_prefix(bay_id)
    for name, q in zip(ARM_JOINTS, PANDA_HOME, strict=True):
        jnt = model.joint(f"{prefix}{name}")
        data.qpos[int(jnt.qposadr[0])] = q
        data.qvel[int(jnt.dofadr[0])] = 0.0
        data.ctrl[int(model.actuator(f"{prefix}actuator{name[-1]}").id)] = q
    for i in range(int(model.nu)):
        act = model.actuator(i)
        if str(act.name).startswith(prefix) and not str(act.name).startswith(f"{prefix}actuator"):
            data.ctrl[i] = 0.0
            jnt_id = int(model.actuator_trnid[i][0])
            data.qpos[int(model.jnt_qposadr[jnt_id])] = 0.0
            data.qvel[int(model.jnt_dofadr[jnt_id])] = 0.0
    mj.mj_forward(model, data)
