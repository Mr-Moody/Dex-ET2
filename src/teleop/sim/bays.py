"""4-bay EV-latch fleet scene (spec 2.2, 2.3): Franka Panda + right Inspire hand + latch per bay.

Bays 1-2 are active physics. Bays 3-4 are a visual-only background row (every geom has
``contype=conaffinity=0``) so rendering them adds no contact cost. Each bay's robot base frame
has no rotation (the arm reaches along world +X), and latch poses passed to the builders are
**bay-local**, relative to that base. ``latch_world_pose`` converts to the world frame.

Names are prefixed per bay: ``bay{k}/joint1..7``, ``bay{k}/actuator1..7``, the Inspire hand
joints and its 6 actuators (``bay{k}/index`` ...), ``bay{k}/wrist_site`` at the knuckles, and
the latch bodies, joints and equalities from ``teleop.sim.latch``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]

from teleop.atc.types import N_BAYS, LatchPose
from teleop.sim.cells import add_cell, add_warehouse, apply_visual_style, look_at_xyaxes
from teleop.sim.hands import REPO_ROOT, load_hand
from teleop.sim.ik import ARM_JOINTS, PANDA_HOME
from teleop.sim.latch import add_latch_to_spec
from teleop.types import Pose

mj: Any = _mujoco

PANDA_XML = REPO_ROOT / "third_party" / "menagerie" / "franka_emika_panda" / "panda_nohand.xml"

# Spread across the recycling warehouse on a 6.4 m x 6.8 m grid so each bay sits on its own
# segmented conveyor line with dedicated infeed/outfeed machines and wide logistics aisles.
BAY_BASES: dict[int, tuple[float, float, float]] = {
    1: (-3.2, -3.4, 0.0),
    2: (3.2, -3.4, 0.0),
    3: (-3.2, 3.4, 0.0),
    4: (3.2, 3.4, 0.0),
}
ACTIVE_BAYS = (1, 2)
BACKGROUND_BAYS = (3, 4)

CANONICAL_LATCH = LatchPose(position=(0.50, 0.0, 0.02), yaw_rad=0.0)
# Bay 2's perturbed latch: shifted ~12 cm and rotated +35 deg (spec 2.3).
BAY2_LATCH = CANONICAL_LATCH.offset(dx=0.08, dy=0.09, dyaw_rad=math.radians(35.0))

# Inspire base frame: wrist at the origin, fingers along +Z, curling towards +X (the palm side),
# index on +Y. The wrist site gives it the palm-frame convention the latch-frame routine, rescue
# and clutch use: site +X along the fingers, fingers curling towards site -Z, index on +Y,
# origin on the knuckle line between the index and middle fingers, on the back of the hand.
WRIST_SITE_POS = (-0.019, 0.0226, 0.137)  # in the hand base frame
# wxyz, -90 deg about base Y: site x = base z, site y = base y, site z = -base x
WRIST_SITE_QUAT = (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)
# The base mounts straight on the flange (fingers along the tool axis +Z), which puts the wrist
# site in the same flange pose as the LEAP palm frame it replaces: site +X along flange +Z and
# site +Z along flange -X.
HAND_MOUNT_POS = (0.0, 0.0, 0.0)
HAND_MOUNT_QUAT = (0.0, 0.0, 0.0, 1.0)  # wxyz
# A passive compliant wrist (like the compliance devices on real insertion cells): the Inspire
# fingers have no abduction joint, so without it a push that the stalled bolt stops loads the
# Panda almost rigidly. Two sprung hinges about the flange X and Y axes give the hand a few mm
# of give at the fingertips.
WRIST_COMPLIANCE_AXES = {"wrist_comp_x": (1.0, 0.0, 0.0), "wrist_comp_y": (0.0, 1.0, 0.0)}
WRIST_COMPLIANCE_STIFFNESS = 30.0  # Nm/rad
WRIST_COMPLIANCE_DAMPING = 0.5  # Nm s/rad
WRIST_COMPLIANCE_RANGE_RAD = 0.2
HAND_CONTACT_SOLREF = (0.05, 1.0)  # soft fingertip pads (the model ships stiff 2 ms contacts)
HAND_RGBA = (0.16, 0.17, 0.19, 1.0)  # dark graphite links
HAND_VISUAL_GROUP = 1  # Inspire ships visuals in group 2, which the rescue cameras hide (Panda)

TIMESTEP = 1.0 / 600
OVERVIEW_CAMERA = "overview"
WRIST_CAMERA_SUFFIX = "wrist_cam"
WRIST_CAM_POS = (0.095, 0.025, 0.020)  # palm-side wrist bracket outside the palm shell
WRIST_CAM_TARGET = (0.045, 0.028, 0.175)  # pinch zone: index hook, thumb & latch paddle
WRIST_CAM_FOVY_DEG = 64.0


def wrist_camera_name(bay_id: int) -> str:
    return f"{bay_prefix(bay_id)}{WRIST_CAMERA_SUFFIX}"


def _wrist_cam_xyaxes() -> list[float]:
    dx = WRIST_CAM_TARGET[0] - WRIST_CAM_POS[0]
    dy = WRIST_CAM_TARGET[1] - WRIST_CAM_POS[1]
    dz = WRIST_CAM_TARGET[2] - WRIST_CAM_POS[2]
    fn = math.sqrt(dx * dx + dy * dy + dz * dz)
    fx, fy, fz = dx / fn, dy / fn, dz / fn
    # up_hint = (-1, 0, 0): right = normalize(fwd x up_hint) = normalize((0, -fz, fy))
    rn = math.hypot(fz, fy)
    ry, rz = -fz / rn, fy / rn
    # up = right x fwd
    ux = ry * fz - rz * fy
    uy = rz * fx
    uz = -ry * fx
    return [0.0, ry, rz, ux, uy, uz]


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
    """World wrist-site pose above a latch: fingers pointing down, palm normal along latch -X.

    ``along_x``/``along_y`` offset the wrist in the latch frame; ``height`` is above its origin.
    """
    c, s = math.cos(world_latch.yaw_rad), math.sin(world_latch.yaw_rad)
    x, y, z = world_latch.position
    position = (x + c * along_x - s * along_y, y + s * along_x + c * along_y, z + height)
    # Rz(yaw) * Ry(+90 deg) * Rx(180 deg): site +X (fingers) -> world -Z, site +Z -> latch -X.
    # The Rx(180 deg) undoes the hand's half-turn mount, keeping the Panda in its home posture.
    hz, hy = 0.5 * world_latch.yaw_rad, 0.25 * math.pi
    orientation = (
        math.cos(hz) * math.cos(hy),
        math.sin(hz) * math.cos(hy),
        -math.cos(hz) * math.sin(hy),
        math.sin(hz) * math.sin(hy),
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
        size=[14.0, 14.0, 0.1],
        rgba=[0.32, 0.34, 0.38, 1.0],
        friction=[1.0, 0.005, 0.0001],
    )
    return spec


def _floor_coupled_servos(hand: Any) -> None:
    """Raise each finger servo's lower ctrl bound to keep its coupled joint inside its range.

    Each Inspire finger's distal joint follows its servoed proximal joint through a linear joint
    equality ``q_distal = c0 + c1 * q_proximal``. Below ``(distal_min - c0) / c1`` the coupling
    would drive the distal joint past its limit, so the servo saturates against the constraint
    and the torque estimator sees a phantom external torque on an open hand.
    """
    floors: dict[str, float] = {}
    for eq in hand.equalities:
        if eq.type != mj.mjtEq.mjEQ_JOINT or not str(eq.name2):
            continue
        c0, c1 = float(eq.data[0]), float(eq.data[1])
        if c1 <= 0.0:
            continue
        lo = float(hand.joint(str(eq.name1)).range[0])
        driver = str(eq.name2)
        floors[driver] = max(floors.get(driver, -math.inf), (lo - c0) / c1)
    for act in hand.actuators:
        floor = floors.get(str(act.target))
        if floor is not None and floor > float(act.ctrlrange[0]):
            act.ctrlrange = [floor + 1e-3, float(act.ctrlrange[1])]


def _add_robot(spec: Any, bay_id: int, *, collide: bool) -> None:
    prefix = bay_prefix(bay_id)
    panda: Any = mj.MjSpec.from_file(str(_require(PANDA_XML)))
    hand_spec = load_hand("inspire")
    hand: Any = mj.MjSpec.from_file(str(_require(hand_spec.mjcf["right"])))
    for child in (panda, hand):
        _strip_globals(child)
        _sync_options(spec, child)
    for site in list(hand.worldbody.sites):  # the model's loose "eeff" marker
        hand.delete(site)

    # Avoid material name clashes with the Panda (and rename the references).
    for mat in list(hand.materials):
        old = str(mat.name)
        mat.name = f"hand_{old}"
        for geom in hand.geoms:
            if geom.material == old:
                geom.material = mat.name
    for geom in hand.geoms:
        if int(geom.group) == 2:
            geom.group = HAND_VISUAL_GROUP
            geom.rgba = list(HAND_RGBA)
        else:  # the model's 2 ms contacts make a finger striking the stalled bolt a torque spike
            geom.solref = list(HAND_CONTACT_SOLREF)

    _floor_coupled_servos(hand)

    base = hand.body(hand_spec.base_body)
    base.pos = list(HAND_MOUNT_POS)
    base.quat = list(HAND_MOUNT_QUAT)
    # Compliant wrist: two sprung, damped hinges across the tool axis between flange and hand.
    for name, axis in WRIST_COMPLIANCE_AXES.items():
        base.add_joint(
            name=name,
            type=mj.mjtJoint.mjJNT_HINGE,
            axis=list(axis),
            stiffness=WRIST_COMPLIANCE_STIFFNESS,
            damping=WRIST_COMPLIANCE_DAMPING,
            armature=0.001,
            range=[-WRIST_COMPLIANCE_RANGE_RAD, WRIST_COMPLIANCE_RANGE_RAD],
            limited=mj.mjtLimited.mjLIMITED_TRUE,
        )
    base.add_site(
        name="wrist_site",
        pos=list(WRIST_SITE_POS),
        quat=list(WRIST_SITE_QUAT),
        size=[0.006, 0.006, 0.006],
        rgba=[0.1, 0.9, 0.2, 0.6],
    )
    base.add_camera(
        name=WRIST_CAMERA_SUFFIX,
        pos=list(WRIST_CAM_POS),
        xyaxes=_wrist_cam_xyaxes(),
        fovy=WRIST_CAM_FOVY_DEG,
    )

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
    """Elevated mezzanine view across the battery-recycling warehouse and 4 conveyor cells."""
    target = (0.45, 0.20 if include_background else -3.0, -0.05)
    pos = (0.45, -8.25, 4.35 if include_background else 3.20)
    spec.worldbody.add_camera(
        name=OVERVIEW_CAMERA, pos=list(pos), xyaxes=look_at_xyaxes(pos, target), fovy=56.0
    )


def build_four_bay_spec(
    bay1_pose: LatchPose = CANONICAL_LATCH,
    bay2_pose: LatchPose = BAY2_LATCH,
    include_background_bays: bool = True,
) -> Any:
    from teleop.retarget.clutch import add_rescue_cameras

    spec = _new_spec("atc_four_bay")
    add_warehouse(spec, include_background_bays)
    poses = {1: bay1_pose, 2: bay2_pose}
    if include_background_bays:
        poses.update({bay_id: CANONICAL_LATCH for bay_id in BACKGROUND_BAYS})
    for bay_id, pose in poses.items():
        _add_bay(spec, bay_id, pose, collide=bay_id in ACTIVE_BAYS)
        add_cell(spec, bay_id, latch_world_pose(bay_id, pose))
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
    for name in WRIST_COMPLIANCE_AXES:
        jnt = model.joint(f"{prefix}{name}")
        data.qpos[int(jnt.qposadr[0])] = 0.0
        data.qvel[int(jnt.dofadr[0])] = 0.0
    for i in range(int(model.nu)):
        act = model.actuator(i)
        if str(act.name).startswith(prefix) and not str(act.name).startswith(f"{prefix}actuator"):
            jnt_id = int(model.actuator_trnid[i][0])
            # Open hand: 0 rad, inside both the servo and the joint range (the Inspire index
            # joint starts at 0.025 rad and thumb flexion at 0.1 rad; a target past a joint
            # limit would hold the finger against it with a phantom external torque).
            lo = max(float(model.actuator_ctrlrange[i][0]), float(model.jnt_range[jnt_id][0]))
            hi = min(float(model.actuator_ctrlrange[i][1]), float(model.jnt_range[jnt_id][1]))
            q_open = min(max(0.0, lo), hi)
            data.ctrl[i] = q_open
            data.qpos[int(model.jnt_qposadr[jnt_id])] = q_open
            data.qvel[int(model.jnt_dofadr[jnt_id])] = 0.0
    mj.mj_forward(model, data)
