"""Object-centric 30-kernel RBF-DMP: closed-form ridge fit and SE(3) replay (spec 4.3).

A rescue is recorded in the latch frame of the bay it happened in. Each sample becomes a
``6 + n_fingers`` vector ``y = [p_latch; Log(R_goal^T R_latch); q_fingers]`` where ``R_goal`` is
the final wrist orientation. Over the canonical phase ``s(t) = exp(-alpha_s t / T)``:

    y(s) = (1 - s) y_goal + s y0 + phi(s)^T W,   phi_i(s) = psi_i(s) s / sum_j psi_j(s)

and ``W`` (n_kernels x dims) comes from one ridge solve. Replaying in another bay only swaps the
latch frame: ``T_world_wrist(s) = T_world_latch_b * T_latch_wrist(s)``.

Wrist samples are 7-vectors ``[x, y, z, qx, qy, qz, qw]`` (xyzw quaternions).
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from teleop.atc.types import SkillPrimitive, StallSignature
from teleop.types import Pose

FloatArray = NDArray[np.float64]

ALPHA_S = 4.0
N_KERNELS = 30
RIDGE = 1e-4
WIDTH_SCALE = 0.65

# Right Inspire RH56 hand actuators in model order (6 position servos: one per finger, whose
# distal joint is coupled to it by an equality constraint, plus thumb flexion and thumb yaw).
HAND_ACTUATORS: tuple[str, ...] = (
    "pinky",
    "ring",
    "middle",
    "index",
    "thumb_proximal",
    "thumb_yaw",
)


# --------------------------------------------------------------------------- SO(3) / SE(3)


def quat_to_mat(q: ArrayLike) -> FloatArray:
    """xyzw quaternion -> 3x3 rotation matrix."""
    x, y, z, w = (float(v) for v in np.asarray(q, dtype=np.float64))
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        raise ValueError("zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(r: FloatArray) -> tuple[float, float, float, float]:
    """3x3 rotation matrix -> xyzw quaternion with w >= 0."""
    m = np.asarray(r, dtype=np.float64)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = 2.0 * math.sqrt(tr + 1.0)
        w, x = 0.25 * s, (m[2, 1] - m[1, 2]) / s
        y, z = (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q: FloatArray = np.array([float(x), float(y), float(z), float(w)], dtype=np.float64)
    q = q / float(np.linalg.norm(q))
    if q[3] < 0.0:
        q = -q
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def so3_log(r: FloatArray) -> FloatArray:
    """Rotation vector (axis * angle) of a rotation matrix."""
    cos = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cos)
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]], dtype=np.float64)
    if angle < 1e-6:
        return 0.5 * v
    if math.pi - angle < 1e-4:
        b = 0.5 * (r + np.eye(3))
        k = int(np.argmax(np.diag(b)))
        axis = b[:, k] / math.sqrt(max(float(b[k, k]), 1e-12))
        return angle * axis / float(np.linalg.norm(axis))
    return angle / (2.0 * math.sin(angle)) * v


def so3_exp(w: ArrayLike) -> FloatArray:
    """Rotation matrix of a rotation vector (Rodrigues)."""
    v = np.asarray(w, dtype=np.float64)
    angle = float(np.linalg.norm(v))
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    if angle < 1e-9:
        return np.eye(3) + k
    a = math.sin(angle) / angle
    b = (1.0 - math.cos(angle)) / (angle * angle)
    return np.eye(3) + a * k + b * (k @ k)


def _pose_rt(pose: Pose) -> tuple[FloatArray, FloatArray]:
    if pose.position is None:
        raise ValueError("pose needs a position")
    return quat_to_mat(pose.orientation), np.asarray(pose.position, dtype=np.float64)


def relative_pose(world_from_latch: Pose, world_wrist: Pose) -> Pose:
    """``T_latch_wrist = T_world_latch^-1 * T_world_wrist``."""
    r_l, t_l = _pose_rt(world_from_latch)
    r_w, t_w = _pose_rt(world_wrist)
    p = r_l.T @ (t_w - t_l)
    return Pose(
        orientation=mat_to_quat(r_l.T @ r_w), position=(float(p[0]), float(p[1]), float(p[2]))
    )


def compose_pose(world_from_latch: Pose, latch_wrist: Pose) -> Pose:
    """``T_world_wrist = T_world_latch * T_latch_wrist``."""
    r_l, t_l = _pose_rt(world_from_latch)
    r_w, t_w = _pose_rt(latch_wrist)
    p = r_l @ t_w + t_l
    return Pose(
        orientation=mat_to_quat(r_l @ r_w), position=(float(p[0]), float(p[1]), float(p[2]))
    )


def pose_to_vec7(pose: Pose) -> list[float]:
    """``[x, y, z, qx, qy, qz, qw]`` wire form of a pose."""
    if pose.position is None:
        raise ValueError("pose needs a position")
    return [*pose.position, *pose.orientation]


# --------------------------------------------------------------------------- basis


def kernel_centres(n_kernels: int = N_KERNELS, alpha_s: float = ALPHA_S) -> FloatArray:
    i = np.arange(n_kernels, dtype=np.float64)
    return np.exp(-alpha_s * i / max(n_kernels - 1, 1))


def kernel_widths(centres: FloatArray) -> FloatArray:
    gaps = np.abs(np.diff(centres))
    gaps = np.append(gaps, gaps[-1])
    return 1.0 / (WIDTH_SCALE * gaps) ** 2


def basis(s: ArrayLike, n_kernels: int = N_KERNELS, alpha_s: float = ALPHA_S) -> FloatArray:
    """Normalised, phase-gated features ``phi(s)``: shape (len(s), n_kernels)."""
    sv = np.atleast_1d(np.asarray(s, dtype=np.float64))
    c = kernel_centres(n_kernels, alpha_s)
    h = kernel_widths(c)
    psi = np.exp(-h[None, :] * (sv[:, None] - c[None, :]) ** 2)
    return psi * sv[:, None] / (psi.sum(axis=1, keepdims=True) + 1e-10)


def phase(t_s: ArrayLike, duration_s: float, alpha_s: float = ALPHA_S) -> FloatArray:
    return np.exp(-alpha_s * np.asarray(t_s, dtype=np.float64) / duration_s)


def time_of_phase(s: float, duration_s: float, alpha_s: float = ALPHA_S) -> float:
    return -math.log(max(s, 1e-12)) * duration_s / alpha_s


# --------------------------------------------------------------------------- fit / evaluate


def _tau_profile(u: FloatArray, tau: FloatArray, n_bins: int) -> FloatArray:
    """Mean ``tau`` per equal bin of normalised time ``u`` in [0, 1]; gaps interpolated."""
    idx = np.minimum((u * n_bins).astype(int), n_bins - 1)
    sums = np.bincount(idx, weights=tau, minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    centres = (np.arange(n_bins) + 0.5) / n_bins
    have = counts > 0
    means = sums[have] / counts[have]
    return np.interp(centres, centres[have], means)


def fit_rbf_dmp(
    timestamps_s: ArrayLike,
    wrist_latch_se3: ArrayLike,
    fingers_q: ArrayLike,
    tau_ext_inf: ArrayLike,
    n_kernels: int = N_KERNELS,
    ridge: float = RIDGE,
    *,
    alpha_s: float = ALPHA_S,
    part_sku_id: str = "ev_latch_v1",
    signature_id: str = "Jam_Sig_1",
    prototype: StallSignature | Sequence[float] | None = None,
    actuator_names: Sequence[str] = HAND_ACTUATORS,
    skill_id: str | None = None,
) -> SkillPrimitive:
    """Fit a latch-frame RBF-DMP to one rescue in a single ``n_kernels``-square ridge solve."""
    start = time.perf_counter()
    t = np.asarray(timestamps_s, dtype=np.float64).ravel()
    wrist = np.asarray(wrist_latch_se3, dtype=np.float64)
    fingers = np.asarray(fingers_q, dtype=np.float64)
    tau = np.asarray(tau_ext_inf, dtype=np.float64).ravel()
    n = t.shape[0]
    names = tuple(actuator_names)
    if n < 2:
        raise ValueError("need at least 2 samples")
    if wrist.shape != (n, 7):
        raise ValueError(f"wrist_latch_se3 must be {n} x 7, got {wrist.shape}")
    if fingers.shape != (n, len(names)):
        raise ValueError(f"fingers_q must be {n} x {len(names)}, got {fingers.shape}")
    if tau.shape != (n,):
        raise ValueError(f"tau_ext_inf must have {n} samples, got {tau.shape}")
    rel = t - t[0]
    duration = float(rel[-1])
    if duration <= 0.0 or np.any(np.diff(rel) < 0.0):
        raise ValueError("timestamps must be increasing")

    rots = [quat_to_mat(q) for q in wrist[:, 3:]]
    r_goal = rots[-1]
    y = np.empty((n, 6 + len(names)))
    y[:, :3] = wrist[:, :3]
    for k, r in enumerate(rots):
        y[k, 3:6] = so3_log(r_goal.T @ r)
    y[:, 6:] = fingers

    s = phase(rel, duration, alpha_s)
    y0, y_goal = y[0], y[-1]
    target = y - ((1.0 - s)[:, None] * y_goal + s[:, None] * y0)
    phi = basis(s, n_kernels, alpha_s)
    gram = phi.T @ phi + ridge * np.eye(n_kernels)
    weights = np.asarray(np.linalg.solve(gram, phi.T @ target), dtype=np.float64)
    tau_demo = _tau_profile(rel / duration, tau, n_kernels)

    if prototype is None:
        proto = StallSignature(0.0, 0.0, 0.0, 0.0)
    elif isinstance(prototype, StallSignature):
        proto = prototype
    else:
        proto = StallSignature.from_vector(prototype)
    fit_ms = (time.perf_counter() - start) * 1000.0
    return SkillPrimitive(
        skill_id=skill_id or f"{part_sku_id}/{signature_id}/{uuid.uuid4().hex[:8]}",
        part_sku_id=part_sku_id,
        signature_id=signature_id,
        duration_s=duration,
        alpha_s=alpha_s,
        actuator_names=names,
        goal_orientation=mat_to_quat(r_goal),
        y0=tuple(float(v) for v in y0),
        y_goal=tuple(float(v) for v in y_goal),
        weights=tuple(tuple(float(v) for v in row) for row in weights),
        tau_demo=tuple(float(v) for v in tau_demo),
        prototype=proto,
        fit_ms=fit_ms,
    )


def evaluate_latch_frame(skill: SkillPrimitive, s: float) -> tuple[Pose, dict[str, float], float]:
    """Wrist pose in the latch frame, named finger targets and nominal ``tau_demo(s)``."""
    s = min(max(float(s), 0.0), 1.0)
    w = np.asarray(skill.weights, dtype=np.float64)
    y0 = np.asarray(skill.y0, dtype=np.float64)
    y_goal = np.asarray(skill.y_goal, dtype=np.float64)
    phi = basis([s], skill.n_kernels, skill.alpha_s)[0]
    y = (1.0 - s) * y_goal + s * y0 + phi @ w
    r = quat_to_mat(skill.goal_orientation) @ so3_exp(y[3:6])
    wrist = Pose(orientation=mat_to_quat(r), position=(float(y[0]), float(y[1]), float(y[2])))
    fingers = {name: float(y[6 + i]) for i, name in enumerate(skill.actuator_names)}
    u = time_of_phase(s, skill.duration_s, skill.alpha_s) / skill.duration_s
    k = skill.n_kernels
    centres = (np.arange(k) + 0.5) / k
    tau = float(
        np.interp(min(max(u, 0.0), 1.0), centres, np.asarray(skill.tau_demo, dtype=np.float64))
    )
    return wrist, fingers, tau


def evaluate_rbf_dmp(
    skill: SkillPrimitive, s: float, world_from_latch: Pose
) -> tuple[Pose, dict[str, float], float]:
    """World wrist pose (warped into ``world_from_latch``), hand targets and ``tau_demo(s)``."""
    local, fingers, tau = evaluate_latch_frame(skill, s)
    return compose_pose(world_from_latch, local), fingers, tau
