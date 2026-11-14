"""3-box sprung slide-bolt latch (spec 2.1) and its fault controller.

Latch frame: origin at the top-centre of the Base Track, +X is the bolt's unlatching direction,
+Z up. The three collision bodies are built only from ``box`` geoms:

* ``latch_base``: Base Track, 0.14 x 0.08 x 0.02 m, its top face at z = 0.
* ``slide_bolt``: slide joint ``bolt_slide`` along +X over [0, 0.06] m. A 0.08 m bar rides
  8 mm above the track and carries a vertical finger hook tab 0.07 m behind the flap hinge.
* ``stopper_flap``: 0.04 x 0.04 x 0.008 m thumb paddle on hinge ``flap_hinge``
  (``stiffness=0.45``, ``springref=+35 deg``). Flat (0 deg) it lies recessed in the track,
  below the bolt. Sprung up, its free end swings towards the bolt, which stalls against it at
  ``x_bolt`` near 0.035 m. The paddle tip only reaches the bolt's underside above about 12 deg,
  so a thumb holding it at 8 deg or lower lets the bolt slide clear.

The hinge axis is local +Y so that a positive angle lifts the free end that faces the bolt.
Two joint-equality constraints sit on top: ``flap_catch`` (active in routine operation, holds
the flap flat) and ``bolt_pin`` (the Negative Control, locks the bolt at x = 0).

What you see is the sim-scale twin of the 3D-printed EV-module latch (``assets/latch``): mesh
geoms on the same three bodies with ``contype=conaffinity=0`` and zero mass. The boxes still do
all the physics, only fully transparent. The meshes match the boxes exactly where they are
touched (bar, hook, paddle) and only add dressing around them, such as the pallet, module and
bolt housing.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]

from teleop.atc.types import BOLT_CLEARED_M, LatchPose
from teleop.sim.hands import REPO_ROOT

mj: Any = _mujoco

TRACK_HALF = (0.07, 0.04, 0.01)
BOLT_BAR_HALF = (0.04, 0.012, 0.008)
BOLT_BAR_POS = (-0.05, 0.0, 0.016)  # bar spans x in [-0.09, -0.01], z in [0.008, 0.024]
HOOK_HALF = (0.008, 0.014, 0.0175)
HOOK_POS = (-0.0135, 0.0, 0.0415)  # 0.07 m behind the flap hinge
BOLT_RANGE_M = (0.0, 0.06)
BOLT_DAMPING = 2.0

FLAP_HINGE_POS = (0.0565, 0.0, -0.008)  # paddle bottom edge, flat paddle spans z in [-0.008, 0]
FLAP_HALF = (0.02, 0.02, 0.004)
FLAP_GEOM_POS = (-0.02, 0.0, 0.004)  # free end extends towards the bolt (-X)
FLAP_RANGE_RAD = (0.0, math.radians(38.0))
FLAP_STIFFNESS = 0.45
FLAP_SPRINGREF_RAD = math.radians(35.0)
FLAP_DAMPING = 0.03
# Reflected inertia: a 10 g paddle alone is too light for the soft catch and limit constraints
# to hold it against the spring.
FLAP_ARMATURE = 0.002

# Stiffer than MuJoCo's 0.02 default (still >= 3 x the 600 Hz timestep) for the latch's contacts
# and joint limits, so the bolt stalls cleanly against the flap instead of sinking into it.
LATCH_SOLREF = (0.005, 1.0)

GREY = (0.55, 0.55, 0.58, 1.0)
STEEL = (0.75, 0.78, 0.82, 1.0)
ORANGE = (1.0, 0.5, 0.05, 1.0)

# Visual twin of the printed rig. Each mesh is in its body's frame: the latch frame for the base
# and the bolt (bolt at x = 0), and the hinge frame (FLAP_HINGE_POS) for the flap.
MESH_DIR = REPO_ROOT / "assets" / "latch"
GRAPHITE = (0.23, 0.24, 0.27, 1.0)
MODULE_BLUE = (0.13, 0.29, 0.5, 1.0)
SAFETY_YELLOW = (0.96, 0.77, 0.0, 1.0)
SAFETY_ORANGE = (1.0, 0.43, 0.04, 1.0)
TERMINAL_RED = (0.75, 0.08, 0.08, 1.0)
TERMINAL_BLACK = (0.06, 0.06, 0.07, 1.0)
VISUAL_MESHES: tuple[tuple[str, str, tuple[float, float, float, float]], ...] = (
    ("pallet_base", "latch_base", GRAPHITE),
    ("battery_module", "latch_base", MODULE_BLUE),
    ("terminal_pos", "latch_base", TERMINAL_RED),
    ("terminal_neg", "latch_base", TERMINAL_BLACK),
    ("slide_bolt", "slide_bolt", SAFETY_YELLOW),
    ("safety_flap", "stopper_flap", SAFETY_ORANGE),
)
HIDDEN_GROUP = 3  # not drawn by the default MjvOption


def _norm_prefix(prefix: str) -> str:
    return prefix if not prefix or prefix.endswith("/") else f"{prefix}/"


def yaw_quat_wxyz(yaw_rad: float) -> list[float]:
    """MuJoCo-boundary quaternion (wxyz) for a rotation about +Z."""
    half = 0.5 * yaw_rad
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _add_visual_twin(spec: Any, p: str, bodies: dict[str, Any]) -> None:
    """Dress the latch with the printed rig's meshes and hide the physics boxes.

    The boxes keep their geometry, contact settings and mass. They only become fully transparent
    and shadowless. The base track's alpha also carries over to the visual twin that
    ``teleop.sim.cells`` adds for it. Mesh assets are shared by every latch in the spec. If the
    mesh files are missing, the boxes stay visible.
    """
    if not all((MESH_DIR / f"{stem}.stl").is_file() for stem, _, _ in VISUAL_MESHES):
        return
    have = {str(m.name) for m in spec.meshes}
    for stem, body_key, rgba in VISUAL_MESHES:
        mesh = f"latch_{stem}"
        if mesh not in have:
            spec.add_mesh(name=mesh, file=str(MESH_DIR / f"{stem}.stl"))
            have.add(mesh)
        geom = bodies[body_key].add_geom(
            name=f"{p}{stem}_visual",
            type=mj.mjtGeom.mjGEOM_MESH,
            meshname=mesh,
            rgba=list(rgba),
        )
        geom.contype = 0
        geom.conaffinity = 0
        geom.mass = 0.0
        geom.group = 0
    for body in bodies.values():
        for geom in body.geoms:
            if geom.type == mj.mjtGeom.mjGEOM_BOX:
                geom.rgba = [float(v) for v in geom.rgba[:3]] + [0.0]
                geom.castshadow = False
                if not str(geom.name).endswith("/base_track"):
                    geom.group = HIDDEN_GROUP


def add_latch_to_spec(
    spec: Any, prefix: str, pose: LatchPose, *, collide: bool = True, visual_meshes: bool = True
) -> None:
    """Add one latch (3 box bodies, 2 joints, 2 equalities) to ``spec.worldbody``.

    ``pose`` is the latch frame in the world. ``collide=False`` builds a visual-only copy
    (``contype=conaffinity=0``) for the kinematic background bays. ``visual_meshes`` dresses the
    boxes with the printed rig's meshes (``assets/latch``). It is visual only, so physics is the
    same either way.
    """
    p = _norm_prefix(prefix)
    # Hinge angles go through the compiler's angle unit; this spec may use degrees.
    ang = 180.0 / math.pi if spec.compiler.degree else 1.0
    contype = 1 if collide else 0
    box = mj.mjtGeom.mjGEOM_BOX

    base = spec.worldbody.add_body(
        name=f"{p}latch_base", pos=list(pose.position), quat=yaw_quat_wxyz(pose.yaw_rad)
    )
    base.add_geom(
        name=f"{p}base_track",
        type=box,
        size=list(TRACK_HALF),
        pos=[0.0, 0.0, -TRACK_HALF[2]],
        rgba=list(GREY),
        contype=contype,
        conaffinity=contype,
    )
    base.add_site(name=f"{p}latch_frame", size=[0.005, 0.005, 0.005])

    bolt = base.add_body(name=f"{p}slide_bolt")
    bolt.add_joint(
        name=f"{p}bolt_slide",
        type=mj.mjtJoint.mjJNT_SLIDE,
        axis=[1.0, 0.0, 0.0],
        range=list(BOLT_RANGE_M),
        limited=mj.mjtLimited.mjLIMITED_TRUE,
        damping=BOLT_DAMPING,
    )
    bolt.add_geom(
        name=f"{p}bolt_bar",
        type=box,
        size=list(BOLT_BAR_HALF),
        pos=list(BOLT_BAR_POS),
        rgba=list(STEEL),
        mass=0.05,
        friction=[0.4, 0.005, 0.0001],
        contype=contype,
        conaffinity=contype,
    )
    bolt.add_geom(
        name=f"{p}bolt_hook",
        type=box,
        size=list(HOOK_HALF),
        pos=list(HOOK_POS),
        rgba=list(STEEL),
        mass=0.01,
        contype=contype,
        conaffinity=contype,
    )

    flap = base.add_body(name=f"{p}stopper_flap", pos=list(FLAP_HINGE_POS))
    flap.add_joint(
        name=f"{p}flap_hinge",
        type=mj.mjtJoint.mjJNT_HINGE,
        axis=[0.0, 1.0, 0.0],
        range=[a * ang for a in FLAP_RANGE_RAD],
        limited=mj.mjtLimited.mjLIMITED_TRUE,
        stiffness=FLAP_STIFFNESS,
        springref=FLAP_SPRINGREF_RAD * ang,
        damping=FLAP_DAMPING,
        armature=FLAP_ARMATURE,
    )
    flap.add_geom(
        name=f"{p}flap_paddle",
        type=box,
        size=list(FLAP_HALF),
        pos=list(FLAP_GEOM_POS),
        rgba=list(ORANGE),
        mass=0.01,
        friction=[1.0, 0.005, 0.0001],
        contype=contype,
        conaffinity=contype,
    )

    for body in (base, bolt, flap):
        for geom in body.geoms:
            geom.solref = list(LATCH_SOLREF)
        for joint in body.joints:
            joint.solref_limit = list(LATCH_SOLREF)

    # Bolt and flap are joint-guided in the track: no contacts with their own base.
    spec.add_exclude(bodyname1=f"{p}latch_base", bodyname2=f"{p}slide_bolt")
    spec.add_exclude(bodyname1=f"{p}latch_base", bodyname2=f"{p}stopper_flap")

    if visual_meshes:
        _add_visual_twin(spec, p, {"latch_base": base, "slide_bolt": bolt, "stopper_flap": flap})

    spec.add_equality(
        type=mj.mjtEq.mjEQ_JOINT,
        name=f"{p}flap_catch",
        name1=f"{p}flap_hinge",
        objtype=mj.mjtObj.mjOBJ_JOINT,
        data=[0.0] * 12,
        active=True,
    )
    spec.add_equality(
        type=mj.mjtEq.mjEQ_JOINT,
        name=f"{p}bolt_pin",
        name1=f"{p}bolt_slide",
        objtype=mj.mjtObj.mjOBJ_JOINT,
        data=[0.0] * 12,
        solref=list(LATCH_SOLREF),
        active=False,
    )


class LatchController:
    """Fault state of one latch in a live ``MjData``.

    The flap fault is one-way: ``arm_flap_fault()`` releases ``flap_catch`` once and every later
    call is ignored until ``reset()``. There is deliberately no flap-angle accessor: jam
    discovery must come from contact (``x_bolt`` and torques), never from ``flap_hinge``.
    """

    def __init__(self, model: Any, data: Any, prefix: str) -> None:
        p = _norm_prefix(prefix)
        self._model = model
        self._data = data
        bolt = model.joint(f"{p}bolt_slide")
        flap = model.joint(f"{p}flap_hinge")
        self._bolt_qadr = int(bolt.qposadr[0])
        self._bolt_dadr = int(bolt.dofadr[0])
        self._flap_qadr = int(flap.qposadr[0])
        self._flap_dadr = int(flap.dofadr[0])
        self._catch_id = int(model.equality(f"{p}flap_catch").id)
        self._pin_id = int(model.equality(f"{p}bolt_pin").id)
        self._fault_armed = False
        self._bolt_pinned = False

    @property
    def flap_fault_armed(self) -> bool:
        return self._fault_armed

    @property
    def bolt_pinned(self) -> bool:
        return self._bolt_pinned

    def reset(self, flap_jammed: bool = False, bolt_pinned: bool = False) -> None:
        """Start a new pallet cycle: bolt home at x = 0, flap flat or sprung up, faults set."""
        d = self._data
        d.qpos[self._bolt_qadr] = 0.0
        d.qvel[self._bolt_dadr] = 0.0
        d.qpos[self._flap_qadr] = FLAP_SPRINGREF_RAD if flap_jammed else 0.0
        d.qvel[self._flap_dadr] = 0.0
        self._fault_armed = flap_jammed
        self._bolt_pinned = bolt_pinned
        d.eq_active[self._catch_id] = 0 if flap_jammed else 1
        d.eq_active[self._pin_id] = 1 if bolt_pinned else 0
        mj.mj_forward(self._model, d)

    def arm_flap_fault(self) -> bool:
        """Release the flap catch for this cycle. Returns False if it was already released."""
        if self._fault_armed:
            return False
        self._fault_armed = True
        self._data.eq_active[self._catch_id] = 0
        return True

    def bolt_x(self) -> float:
        return float(self._data.qpos[self._bolt_qadr])

    def is_cleared(self) -> bool:
        return self.bolt_x() >= BOLT_CLEARED_M
