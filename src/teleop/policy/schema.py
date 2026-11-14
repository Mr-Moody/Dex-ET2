"""ACT recovery policy schema: what the policy sees, what it outputs, what each episode records.

Everything is by name and in the bay's latch frame, so one policy serves every bay and maps onto
a real latch whose pose comes from perception. Sim-only state (``bolt_x``, ``flap_hinge``, latch
fault flags) is deliberately absent from the observation.

State (29): arm joint positions (7), hand actuator joint positions (6, ``HAND_ACTUATORS``
order), wrist pose in the latch frame (x, y, z, qx, qy, qz, qw), external torque on each hand
actuator joint (6), external torque on Panda joints 5-7 (3).

Action (12): target wrist position in the latch frame (3), target wrist rotation as a rotation
vector relative to the routine's latch-frame orientation (3), hand actuator targets (6).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from teleop.types import Pose
from teleop.unoq.dmp import (
    HAND_ACTUATORS,
    compose_pose,
    mat_to_quat,
    quat_to_mat,
    relative_pose,
    so3_exp,
    so3_log,
)

if TYPE_CHECKING:
    from teleop.atc.validator import BayRig

FloatArray = NDArray[np.float64]

SCHEMA_VERSION = 1
FPS = 30
IMAGE_HW = (128, 128)
CAMERAS: tuple[str, ...] = ("wrist", "top")

N_ARM = 7
WRIST_TAU_JOINTS = (5, 6, 7)

STATE_NAMES: tuple[str, ...] = (
    *(f"arm/joint{i}" for i in range(1, N_ARM + 1)),
    *(f"hand/{a}" for a in HAND_ACTUATORS),
    *(f"wrist/{f}" for f in ("x", "y", "z", "qx", "qy", "qz", "qw")),
    *(f"tau_hand/{a}" for a in HAND_ACTUATORS),
    *(f"tau_wrist/joint{j}" for j in WRIST_TAU_JOINTS),
)
ACTION_NAMES: tuple[str, ...] = (
    *(f"wrist/{f}" for f in ("x", "y", "z", "rx", "ry", "rz")),
    *(f"hand/{a}" for a in HAND_ACTUATORS),
)

EpisodeSource = Literal["glove", "scripted", "dmp", "act"]


def camera_name(bay_id: int, cam: str) -> str:
    """MuJoCo camera name of a policy camera (``wrist`` or ``top``) in one bay."""
    from teleop.retarget.clutch import rescue_camera_names
    from teleop.sim.bays import bay_prefix

    if cam == "wrist":
        return f"{bay_prefix(bay_id)}wrist_cam"
    if cam == "top":
        return rescue_camera_names(bay_prefix(bay_id))[0]
    raise ValueError(f"unknown policy camera {cam!r}; expected one of {CAMERAS}")


@cache
def _reference_rotation() -> FloatArray:
    """The routine's wrist orientation in the latch frame (imported lazily: no cycle)."""
    from teleop.atc.coordinator import ROUTINE

    return np.asarray(ROUTINE.rotation(), dtype=np.float64)


def observe(rig: BayRig) -> FloatArray:
    """The 29-value ``STATE_NAMES`` vector of one bay, read by name."""
    model, data, p = rig.model, rig.data, rig.prefix
    qpos = np.asarray(data.qpos, dtype=np.float64)
    arm = [qpos[int(model.joint(f"{p}joint{i}").qposadr[0])] for i in range(1, N_ARM + 1)]
    rel = relative_pose(rig.world_from_latch(), rig.wrist_pose())
    assert rel.position is not None
    tau = rig.tau_ext()
    index = {name: i for i, name in enumerate(rig.torque.joint_names)}
    tau_hand = [
        tau[
            index[
                str(
                    model.joint(
                        int(model.actuator_trnid[int(model.actuator(f"{p}{a}").id)][0])
                    ).name
                )
            ]
        ]
        for a in HAND_ACTUATORS
    ]
    tau_wrist = [tau[index[f"{p}joint{j}"]] for j in WRIST_TAU_JOINTS]
    return np.array(
        [*arm, *rig.hand_q(), *rel.position, *rel.orientation, *tau_hand, *tau_wrist],
        dtype=np.float64,
    )


def action_from_command(
    world_from_latch: Pose, wrist_world: Pose, fingers: Mapping[str, float]
) -> FloatArray:
    """World wrist command + finger targets -> the 12-value latch-frame ``ACTION_NAMES`` vector."""
    rel = relative_pose(world_from_latch, wrist_world)
    assert rel.position is not None
    rot = so3_log(_reference_rotation().T @ quat_to_mat(rel.orientation))
    return np.array(
        [*rel.position, *rot, *(float(fingers.get(a, 0.0)) for a in HAND_ACTUATORS)],
        dtype=np.float64,
    )


def command_from_action(world_from_latch: Pose, action: ArrayLike) -> tuple[Pose, dict[str, float]]:
    """Inverse of :func:`action_from_command`: world wrist pose and finger targets by name."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.shape != (len(ACTION_NAMES),):
        raise ValueError(f"action must have {len(ACTION_NAMES)} values, got {a.shape}")
    rot = _reference_rotation() @ so3_exp(a[3:6])
    rel = Pose(orientation=mat_to_quat(rot), position=(float(a[0]), float(a[1]), float(a[2])))
    fingers = {name: float(a[6 + i]) for i, name in enumerate(HAND_ACTUATORS)}
    return compose_pose(world_from_latch, rel), fingers


def lerobot_features(image_hw: tuple[int, int] = IMAGE_HW) -> dict[str, dict[str, Any]]:
    """LeRobot v3 feature spec for recovery episodes."""
    h, w = image_hw
    feats: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": list(STATE_NAMES),
        },
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": list(ACTION_NAMES)},
    }
    for cam in CAMERAS:
        feats[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    return feats


@dataclass(frozen=True)
class EpisodeMeta:
    """One sidecar line (``meta/teleop_episodes.jsonl``) per recorded recovery episode."""

    source: EpisodeSource
    fault: str | None
    bay: int
    signature_id: str | None
    signature: tuple[float, float, float, float] | None
    latch_pose: dict[str, float]
    success: bool
    bolt_x_final: float
    max_tau_hand_inf: float
    max_arm_ratio: float
    aborted_reason: str | None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["signature"] = None if self.signature is None else list(self.signature)
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> EpisodeMeta:
        sig = d.get("signature")
        return cls(
            source=cast("EpisodeSource", d["source"]),
            fault=d.get("fault"),
            bay=int(d["bay"]),
            signature_id=d.get("signature_id"),
            signature=None
            if sig is None
            else cast("tuple[float, float, float, float]", tuple(float(v) for v in sig)),
            latch_pose={k: float(v) for k, v in dict(d.get("latch_pose") or {}).items()},
            success=bool(d["success"]),
            bolt_x_final=float(d["bolt_x_final"]),
            max_tau_hand_inf=float(d["max_tau_hand_inf"]),
            max_arm_ratio=float(d["max_arm_ratio"]),
            aborted_reason=d.get("aborted_reason"),
        )
