"""Bimanual scene: floor, table, two floating hands welded to mocap bodies, cameras."""
# pyright: reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np

from teleop.config import CameraConfig, WristConfig
from teleop.protocols import Task
from teleop.sim.hands import HandSpec
from teleop.types import SIDES, Side

TABLE_HALF = (0.5, 0.4, 0.02)  # table top surface is at z = 0
FLOOR_Z = -0.75
CAMERA_POS = (0.0, -0.8, 0.6)
CAMERA_TARGET = (0.0, 0.05, 0.05)


def _home(side: Side) -> tuple[float, float, float]:
    w = WristConfig()
    return w.home_left if side == "left" else w.home_right


def _camera_xyaxes(pos: Sequence[float], target: Sequence[float]) -> list[float]:
    fwd = np.asarray(target, dtype=float) - np.asarray(pos, dtype=float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return [*right.tolist(), *up.tolist()]


def _strip_globals(child: Any) -> None:
    """Remove assets/keyframes that would clash with or duplicate the scene's own."""
    for key in list(child.keys):
        child.delete(key)
    for light in list(child.lights):
        child.delete(light)
    for mat in list(child.materials):
        if mat.textures and any(mat.textures):
            mat.textures = [""] * len(mat.textures)
    for tex in list(child.textures):
        child.delete(tex)


def build_spec(hand: HandSpec, task: Task, cameras: Sequence[CameraConfig]) -> Any:
    spec: Any = mujoco.MjSpec()
    spec.modelname = "bimanual"
    spec.option.timestep = 1.0 / 600
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 100.0
    spec.visual.global_.offwidth = max([c.width for c in cameras], default=640)
    spec.visual.global_.offheight = max([c.height for c in cameras], default=480)
    world = spec.worldbody

    world.add_light(pos=[0, 0, 2.0], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[3, 3, 0.1],
        pos=[0, 0, FLOOR_Z],
        rgba=[0.5, 0.5, 0.5, 1],
    )
    world.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(TABLE_HALF),
        pos=[0, 0, -TABLE_HALF[2]],
        rgba=[0.6, 0.45, 0.3, 1],
        friction=[1.0, 0.005, 0.0001],
    )

    for side in SIDES:
        child: Any = mujoco.MjSpec.from_file(str(hand.mjcf[side]))
        _strip_globals(child)
        base_offset = np.asarray(child.worldbody.first_body().pos, dtype=float)
        home = np.asarray(_home(side), dtype=float)

        root = world.add_body(name=f"{side}/root", pos=(home - base_offset).tolist())
        root.add_freejoint()
        frame = root.add_frame()
        spec.attach(child, frame=frame, prefix=f"{side}/")

        # Hands float: cancel gravity on every hand body so they track the mocap.
        for body in spec.bodies:
            if body.name.startswith(f"{side}/"):
                body.gravcomp = 1.0

        mocap = world.add_body(name=f"{side}/mocap", pos=home.tolist(), mocap=True)
        mocap.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.005, 0, 0],
            contype=0,
            conaffinity=0,
            rgba=[1, 0, 0, 0.3],
        )
        spec.add_equality(
            type=mujoco.mjtEq.mjEQ_WELD,
            name1=f"{side}/mocap",
            name2=f"{side}/{hand.base_body}",
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            solref=[0.02, 1.0],
        )

    for cam in cameras:
        world.add_camera(
            name=cam.name,
            pos=list(CAMERA_POS),
            xyaxes=_camera_xyaxes(CAMERA_POS, CAMERA_TARGET),
        )

    task.add_to_spec(spec)
    return spec


def build_scene(hand: HandSpec, task: Task, cameras: Sequence[CameraConfig]) -> Any:
    """Compile the bimanual scene into an ``mujoco.MjModel``."""
    return build_spec(hand, task, cameras).compile()
