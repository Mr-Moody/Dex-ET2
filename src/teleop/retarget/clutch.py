"""Relative-clutch rescue teleop (spec 3): pose indexing, force bleed-off, rescue recording.

The operator holds a dead-man clutch to drive the stalled bay's wrist. On every press the
current sim wrist and glove wrist are anchored, so the robot never jumps; while held, glove
motion relative to the anchor moves the robot relative to its anchor (like lifting a mouse to
re-centre it). Releasing bleeds off force by pinning every actuator target to its measured
joint position, and either keeps the recorded rescue (bolt cleared) or discards it.

Also hosts the split rescue cameras: per active bay an orthographic top view and side view of
the latch, shown picture-in-picture next to the overview camera.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from numpy.typing import NDArray

from teleop.atc.types import LatchPose, StallSignature
from teleop.types import Pose, Quat
from teleop.unoq.dmp import (
    LEAP_ACTUATORS,
    compose_pose,
    mat_to_quat,
    pose_to_vec7,
    quat_to_mat,
    relative_pose,
)
from teleop.unoq.protocol import OP_FIT_DMP

mj: Any = _mujoco

FloatArray = NDArray[np.float64]

ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
BAY_ACTUATORS = (*ARM_ACTUATORS, *LEAP_ACTUATORS)  # 7 Panda + 16 LEAP, unprefixed

IDENTITY: Quat = (0.0, 0.0, 0.0, 1.0)


def _norm_prefix(prefix: str) -> str:
    return prefix if prefix.endswith("/") else f"{prefix}/"


def _pos(pose: Pose) -> FloatArray:
    if pose.position is None:
        raise ValueError("clutch poses need a position")
    return np.asarray(pose.position, dtype=np.float64)


# --------------------------------------------------------------------------- clutch indexing


class RelativeClutchIndexer:
    """Maps glove wrist motion onto the sim wrist, relative to the pose at clutch engage.

    While the clutch is held::

        p_cmd = p_w0 + s * R_align (p_g - p_g0)
        R_cmd = R_align R_g R_g0^T R_align^T R_w0

    ``R_align`` rotates the operator's desk frame into the sim world (identity when the glove
    source already reports the sim's Z-up axes and the operator faces sim +X); ``s`` scales
    hand travel.
    """

    def __init__(self, pos_scale: float = 1.0, align: Quat = IDENTITY) -> None:
        if pos_scale <= 0.0:
            raise ValueError("pos_scale must be positive")
        self.pos_scale = pos_scale
        self._r_align = quat_to_mat(align)
        self._anchor: tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None = None

    @classmethod
    def with_desk_yaw(cls, yaw_rad: float, pos_scale: float = 1.0) -> RelativeClutchIndexer:
        """Operator desk rotated ``yaw_rad`` about +Z relative to the sim world."""
        h = 0.5 * yaw_rad
        return cls(pos_scale=pos_scale, align=(0.0, 0.0, math.sin(h), math.cos(h)))

    @property
    def engaged(self) -> bool:
        return self._anchor is not None

    def reset(self) -> None:
        self._anchor = None

    def update(
        self, clutch_pressed: bool, glove_wrist: Pose, current_sim_wrist: Pose
    ) -> Pose | None:
        """Commanded sim wrist pose while the clutch is held, else ``None``."""
        if not clutch_pressed:
            self._anchor = None
            return None
        if self._anchor is None:
            self._anchor = (
                _pos(current_sim_wrist),
                quat_to_mat(current_sim_wrist.orientation),
                _pos(glove_wrist),
                quat_to_mat(glove_wrist.orientation),
            )
            return current_sim_wrist
        p_w0, r_w0, p_g0, r_g0 = self._anchor
        a = self._r_align
        p = p_w0 + self.pos_scale * (a @ (_pos(glove_wrist) - p_g0))
        r_delta = a @ quat_to_mat(glove_wrist.orientation) @ r_g0.T @ a.T
        return Pose(
            orientation=mat_to_quat(r_delta @ r_w0),
            position=(float(p[0]), float(p[1]), float(p[2])),
        )


# --------------------------------------------------------------------------- force bleed-off


def _actuator_joint_qadr(model: Any, actuator_id: int) -> int:
    joint_id = int(model.actuator_trnid[actuator_id][0])
    return int(model.jnt_qposadr[joint_id])


def read_leap_q(model: Any, data: Any, bay_prefix: str) -> FloatArray:
    """Measured LEAP joint positions in ``LEAP_ACTUATORS`` order (16 values)."""
    p = _norm_prefix(bay_prefix)
    qpos = np.asarray(data.qpos, dtype=np.float64)
    return np.array(
        [
            qpos[_actuator_joint_qadr(model, int(model.actuator(f"{p}{n}").id))]
            for n in LEAP_ACTUATORS
        ],
        dtype=np.float64,
    )


def bleed_off_targets(
    model: Any, data: Any, bay_prefix: str
) -> tuple[FloatArray, dict[str, float]]:
    """Pin all 23 bay actuators (7 Panda + 16 LEAP) to their measured joint positions.

    Writes ``data.ctrl`` so the position servos stop pushing at once (a held target is a zero
    commanded velocity; only the servos' own damping acts), and returns the targets as an array
    in ``BAY_ACTUATORS`` order plus a map from prefixed actuator name. Callers driving the arm
    with ``PandaDiffIK`` should also ``reset()`` it so the next clutch starts from here.
    """
    p = _norm_prefix(bay_prefix)
    qpos = np.asarray(data.qpos, dtype=np.float64)
    targets = np.empty(len(BAY_ACTUATORS), dtype=np.float64)
    named: dict[str, float] = {}
    for i, name in enumerate(BAY_ACTUATORS):
        act_id = int(model.actuator(f"{p}{name}").id)
        q = float(qpos[_actuator_joint_qadr(model, act_id)])
        lo, hi = (float(v) for v in model.actuator_ctrlrange[act_id])
        if bool(model.actuator_ctrllimited[act_id]):
            q = min(max(q, lo), hi)
        data.ctrl[act_id] = q
        targets[i] = q
        named[f"{p}{name}"] = q
    return targets, named


# --------------------------------------------------------------------------- rescue recording


@dataclass(frozen=True, slots=True)
class RescueTrajectory:
    """One successful clutch-held rescue, in the latch frame of the bay it happened in."""

    timestamps_s: FloatArray  # (N,), starts at 0
    wrist_latch_se3: FloatArray  # (N, 7): x, y, z, qx, qy, qz, qw
    fingers_q: FloatArray  # (N, 16) in actuator_names order
    tau_ext_inf: FloatArray  # (N,)
    actuator_names: tuple[str, ...] = LEAP_ACTUATORS

    def __len__(self) -> int:
        return int(self.timestamps_s.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.timestamps_s[-1])

    def fit_dmp_request(
        self, part_sku_id: str, z: StallSignature, signature_id: str | None = None
    ) -> dict[str, Any]:
        """The UNO Q ``fit_dmp`` request for this rescue (spec 5.1)."""
        req: dict[str, Any] = {
            "op": OP_FIT_DMP,
            "part_sku_id": part_sku_id,
            "z": list(z.as_vector()),
            "timestamps_s": self.timestamps_s.tolist(),
            "wrist_latch_se3": self.wrist_latch_se3.tolist(),
            "fingers_q": self.fingers_q.tolist(),
            "tau_ext_inf": self.tau_ext_inf.tolist(),
            "actuator_names": list(self.actuator_names),
        }
        if signature_id is not None:
            req["signature_id"] = signature_id
        return req


class RescueTrajectoryBuffer:
    """Records one clutch hold; ``finish()`` keeps it only if the bolt was cleared."""

    MIN_SAMPLES = 2

    def __init__(
        self, world_from_latch: LatchPose | Pose, actuator_names: Sequence[str] = LEAP_ACTUATORS
    ) -> None:
        self.world_from_latch = (
            world_from_latch.as_pose()
            if isinstance(world_from_latch, LatchPose)
            else world_from_latch
        )
        self.actuator_names = tuple(actuator_names)
        self._t: list[float] = []
        self._wrist: list[list[float]] = []
        self._q: list[list[float]] = []
        self._tau: list[float] = []

    def __len__(self) -> int:
        return len(self._t)

    def clear(self) -> None:
        self._t.clear()
        self._wrist.clear()
        self._q.clear()
        self._tau.clear()

    def record(
        self,
        clutch_pressed: bool,
        t_s: float,
        world_wrist: Pose,
        leap_q: Mapping[str, float] | Sequence[float] | FloatArray,
        tau_ext_inf: float,
    ) -> bool:
        """Append one control tick if the clutch is held. Returns whether it was recorded.

        Ticks that do not advance time are dropped so the payload stays strictly increasing.
        """
        if not clutch_pressed:
            return False
        if self._t and t_s <= self._t[-1]:
            return False
        if isinstance(leap_q, Mapping):
            q = [float(leap_q[n]) for n in self.actuator_names]
        else:
            q = [float(v) for v in np.asarray(leap_q, dtype=np.float64).ravel()]
        if len(q) != len(self.actuator_names):
            raise ValueError(f"need {len(self.actuator_names)} LEAP joints, got {len(q)}")
        local = relative_pose(self.world_from_latch, world_wrist)
        self._t.append(float(t_s))
        self._wrist.append(pose_to_vec7(local))
        self._q.append(q)
        self._tau.append(float(tau_ext_inf))
        return True

    def finish(self, bolt_cleared: bool) -> RescueTrajectory | None:
        """End the hold: the rescue if the bolt cleared, else ``None``. Always empties."""
        try:
            if not bolt_cleared or len(self._t) < self.MIN_SAMPLES:
                return None
            t = np.asarray(self._t, dtype=np.float64)
            return RescueTrajectory(
                timestamps_s=t - t[0],
                wrist_latch_se3=np.asarray(self._wrist, dtype=np.float64),
                fingers_q=np.asarray(self._q, dtype=np.float64),
                tau_ext_inf=np.asarray(self._tau, dtype=np.float64),
                actuator_names=self.actuator_names,
            )
        finally:
            self.clear()

    def world_wrist(self, latch_wrist: Pose) -> Pose:
        """Inverse of the recording transform (latch frame -> world)."""
        return compose_pose(self.world_from_latch, latch_wrist)


# --------------------------------------------------------------------------- split cameras

TOP_CAMERA = "top_ortho"
SIDE_CAMERA = "side_ortho"
ORTHO_EXTENT_M = 0.24  # view height: the 14 cm track plus the hand around it
TOP_HEIGHT_M = 0.6  # camera distance from the latch origin along its view axis
TOP_TILT_RAD = math.radians(25.0)
SIDE_STANDOFF_M = 0.6
SIDE_LIFT_M = 0.06
PANDA_VISUAL_GROUP = 2  # Menagerie Panda link meshes; LEAP visuals are group 1, the latch group 0


def rescue_camera_names(bay_prefix: str) -> tuple[str, str]:
    p = _norm_prefix(bay_prefix)
    return f"{p}{TOP_CAMERA}", f"{p}{SIDE_CAMERA}"


def add_rescue_cameras(
    spec: Any, bay_prefix: str, world_latch: LatchPose, extent_m: float = ORTHO_EXTENT_M
) -> tuple[str, str]:
    """Add orthographic top and side cameras over one bay's latch to an ``MjSpec``.

    The top view looks down with the bolt sliding left to right, showing index-to-hook
    alignment. It is tilted ``TOP_TILT_RAD`` about the slide axis (from the +Y side): looking
    straight down, the overhead light and headlight glare straight back off the flat track and
    wash the paddle out to white. The side view looks along latch +Y from the -Y side, showing
    the thumb's depth on the flap. Use with ``teleop.sim.bays.build_four_bay_spec`` before
    ``compile()``. Returns the two camera names.
    """
    top_name, side_name = rescue_camera_names(bay_prefix)
    c, s = math.cos(world_latch.yaw_rad), math.sin(world_latch.yaw_rad)
    x_axis = np.array([c, s, 0.0])
    y_axis = np.array([-s, c, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    # Camera +Z (it looks along -Z) leans from world up towards latch +Y.
    z_cam = math.cos(TOP_TILT_RAD) * up + math.sin(TOP_TILT_RAD) * y_axis
    y_cam = np.cross(z_cam, x_axis)
    origin = np.asarray(world_latch.position, dtype=np.float64)
    ortho = spec.worldbody.add_camera(
        name=top_name,
        pos=(origin + TOP_HEIGHT_M * z_cam).tolist(),
        xyaxes=[*x_axis.tolist(), *y_cam.tolist()],  # image right = latch +X
    )
    side = spec.worldbody.add_camera(
        name=side_name,
        # SIDE_STANDOFF_M out along latch -Y
        pos=(origin - SIDE_STANDOFF_M * y_axis + SIDE_LIFT_M * up).tolist(),
        xyaxes=[*x_axis.tolist(), *up.tolist()],  # looks along latch +Y: right = +X, up = +Z
    )
    for cam in (ortho, side):
        cam.proj = mj.mjtProjection.mjPROJ_ORTHOGRAPHIC
        cam.fovy = extent_m  # for orthographic cameras fovy is the view height in metres
    return top_name, side_name


def rescue_view_option() -> Any:
    """``MjvOption`` for the inset cameras: hides the Panda's link meshes.

    From straight above, the forearm hovering over the latch would otherwise fill the top view;
    with it hidden the LEAP fingers, bolt and flap stay visible. Pass it as
    ``renderer.update_scene(data, camera, scene_option=rescue_view_option())``.
    """
    opt = mj.MjvOption()
    opt.geomgroup[PANDA_VISUAL_GROUP] = 0
    return opt


def compose_split_view(
    main: NDArray[np.uint8],
    top: NDArray[np.uint8],
    side: NDArray[np.uint8],
    border_px: int = 2,
    corner: Literal["bottom_right", "top_right"] = "bottom_right",
) -> NDArray[np.uint8]:
    """Overview image with the top and side views stacked picture-in-picture at the right.

    Each inset is scaled (nearest neighbour) to a third of the main image's height and framed
    by a white border. The top view sits above the side view, against the chosen corner.
    """
    if main.ndim != 3 or top.ndim != 3 or side.ndim != 3:
        raise ValueError("images must be H x W x C")
    out = main.copy()
    h, w = main.shape[:2]
    inset_h = max(h // 3, 2 * border_px + 1)
    y = border_px if corner == "top_right" else h - 2 * (inset_h + 2 * border_px) + border_px
    y = max(y, border_px)
    for img in (top, side):
        ih, iw = img.shape[:2]
        inset_w = max(round(iw * inset_h / ih), 1)
        if inset_w + 2 * border_px > w or y + inset_h + border_px > h:
            break
        rows = (np.arange(inset_h) * ih // inset_h).clip(0, ih - 1)
        cols = (np.arange(inset_w) * iw // inset_w).clip(0, iw - 1)
        scaled = img[rows][:, cols]
        x = w - inset_w - border_px
        out[y - border_px : y + inset_h + border_px, x - border_px : x + inset_w + border_px] = 255
        out[y : y + inset_h, x : x + inset_w] = scaled
        y += inset_h + 2 * border_px
    return out
