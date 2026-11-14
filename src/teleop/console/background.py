"""Canned routine cycle that animates the background Bays 3-4 on the fleet ops console.

Bays 3-4 are visual-only props in the fleet scene: the coordinator never runs them. For the
console they replay one recorded routine Bay 1 unlatch cycle (joint positions by name plus its
C(t) trace), phase-offset so they don't move in sync. This is display only: the coordinator
still treats them as static, and the console marks their telemetry as replayed.

``record_cycle`` regenerates the clip headless (``scripts/record_background_cycle.py``);
``BackgroundPlayer`` writes interpolated joint positions into a render-side ``MjData``.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from teleop.atc.hud import ConveyorStatus
from teleop.sim.cells import CONVEYOR_SPEED_M_S, PACK_PITCH_M

DEFAULT_CYCLE_PATH = Path(__file__).parent / "data" / "background_cycle.npz"
BLEND_S = 0.5
PHASE_OFFSETS: dict[int, float] = {3: 0.37, 4: 0.71}
LATCH_JOINTS = frozenset({"bolt_slide", "flap_hinge"})  # snap at the wrap: a new pallet
SAMPLE_HZ = 30.0
RECORD_DWELL_S = 1.0
SOURCE_BAY = 1
_BG_WORK_END_S = 3.35

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BackgroundCycle:
    """One recorded cycle: ``qpos`` rows are the joints in ``joint_names`` order (unprefixed)."""

    t: FloatArray  # (N,) seconds, starting at 0
    joint_names: tuple[str, ...]
    qpos_widths: tuple[int, ...]
    qpos: FloatArray  # (N, sum(qpos_widths))
    confidence: FloatArray  # (N,)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1])

    def snap_mask(self) -> NDArray[np.bool_]:
        mask: list[bool] = []
        for name, width in zip(self.joint_names, self.qpos_widths, strict=True):
            mask.extend([name in LATCH_JOINTS] * width)
        return np.asarray(mask, dtype=np.bool_)

    def sample(self, t_s: float) -> tuple[FloatArray, float]:
        """Joint positions and C(t) at ``t_s`` (wrapped into the cycle).

        Linear between frames. Over the last ``BLEND_S`` seconds the arm and hand joints blend
        towards frame 0 so the loop has no jump; latch joints keep their value and snap.
        """
        dur = self.duration_s
        t = float(t_s) % dur if dur > 0.0 else 0.0
        q = np.array(
            [np.interp(t, self.t, self.qpos[:, i]) for i in range(self.qpos.shape[1])],
            dtype=np.float64,
        )
        conf = float(np.interp(t, self.t, self.confidence))
        blend_start = dur - min(BLEND_S, dur)
        if t > blend_start and dur > 0.0:
            w = (t - blend_start) / (dur - blend_start)
            blend = ~self.snap_mask()
            q[blend] = (1.0 - w) * q[blend] + w * self.qpos[0, blend]
            conf = (1.0 - w) * conf + w * float(self.confidence[0])
        return q, conf


def save_cycle(cycle: BackgroundCycle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        np.savez_compressed(
            fh,
            t=cycle.t,
            joint_names=np.asarray(cycle.joint_names),
            qpos_widths=np.asarray(cycle.qpos_widths, dtype=np.int64),
            qpos=cycle.qpos,
            confidence=cycle.confidence,
        )


def load_cycle(path: Path = DEFAULT_CYCLE_PATH) -> BackgroundCycle:
    with np.load(path, allow_pickle=False) as npz:
        return BackgroundCycle(
            t=np.asarray(npz["t"], dtype=np.float64),
            joint_names=tuple(str(n) for n in npz["joint_names"]),
            qpos_widths=tuple(int(w) for w in npz["qpos_widths"]),
            qpos=np.asarray(npz["qpos"], dtype=np.float64),
            confidence=np.asarray(npz["confidence"], dtype=np.float64),
        )


def _qpos_width(model: Any, joint_id: int) -> int:
    import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]

    mj: Any = _mujoco
    jtype = int(model.jnt_type[joint_id])
    if jtype == int(mj.mjtJoint.mjJNT_FREE):
        return 7
    if jtype == int(mj.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def _bay_joints(model: Any, bay_id: int) -> list[tuple[str, int, int]]:
    """``(unprefixed name, qpos address, width)`` for every ``bay{k}/`` joint, in id order."""
    prefix = f"bay{bay_id}/"
    out: list[tuple[str, int, int]] = []
    for j in range(int(model.njnt)):
        name = str(model.joint(j).name)
        if name.startswith(prefix):
            out.append((name[len(prefix) :], int(model.jnt_qposadr[j]), _qpos_width(model, j)))
    return out


class BackgroundPlayer:
    """Writes the canned cycle into the background bays of a render-side ``MjData``."""

    def __init__(self, cycle: BackgroundCycle, model: Any, bays: Sequence[int] = (3, 4)) -> None:
        self.cycle = cycle
        cols: dict[str, tuple[int, int]] = {}
        col = 0
        for name, width in zip(cycle.joint_names, cycle.qpos_widths, strict=True):
            cols[name] = (col, width)
            col += width
        # bay -> list of (cycle column, qpos address, width)
        self._maps: dict[int, list[tuple[int, int, int]]] = {}
        self._last_t: dict[int, float] = {}
        self._belt_dist: dict[int, float] = {}
        self._pause_s: dict[int, float] = {}  # replay time lost to E-stop freezes, per bay
        self._frozen_at: dict[int, float] = {}  # replay time a currently frozen bay holds
        self._shown_t: dict[int, float] = {}  # last replay time posed, where a cut freezes
        for bay in bays:
            mapping: list[tuple[int, int, int]] = []
            for name, adr, width in _bay_joints(model, bay):
                if name in cols and cols[name][1] == width:
                    mapping.append((cols[name][0], adr, width))
            if mapping:
                self._maps[bay] = mapping

    @property
    def bays(self) -> tuple[int, ...]:
        return tuple(self._maps)

    def loop_params(self) -> tuple[float, float, float]:
        dur = self.cycle.duration_s
        work_end = min(_BG_WORK_END_S, max(0.5 * dur, dur - 0.5))
        transfer_s = PACK_PITCH_M / CONVEYOR_SPEED_M_S if dur > 2.5 else 0.0
        return work_end, transfer_s, work_end + transfer_s

    def _replay_t(self, bay: int, wall_t: float, frozen: Collection[int]) -> float:
        """Replay clock of ``bay``: it stops while the bay is frozen and resumes from there."""
        if bay in frozen:
            if bay not in self._frozen_at:
                self._frozen_at[bay] = self._shown_t.get(bay, wall_t - self._pause_s.get(bay, 0.0))
            return self._frozen_at[bay]
        if bay in self._frozen_at:
            self._pause_s[bay] = wall_t - self._frozen_at.pop(bay)
        t = wall_t - self._pause_s.get(bay, 0.0)
        self._shown_t[bay] = t
        return t

    def apply(self, data: Any, wall_t: float, frozen: Collection[int] = ()) -> dict[int, float]:
        """Pose each background bay at ``wall_t``; returns ``{bay: C(t)}``. No ``mj_forward``.

        ``frozen`` bays (robot power cut by the console E-stop) hold their current pose.
        """
        import math

        out: dict[int, float] = {}
        work_end, transfer_s, loop_s = self.loop_params()
        if transfer_s <= 0.0 or loop_s <= 0.0:
            dur = self.cycle.duration_s
            for bay, mapping in self._maps.items():
                t = self._replay_t(bay, wall_t, frozen)
                q, conf = self.cycle.sample(t + PHASE_OFFSETS.get(bay, 0.0) * dur)
                for col, adr, width in mapping:
                    data.qpos[adr : adr + width] = q[col : col + width]
                out[bay] = conf
            return out

        q_end, c_end = self.cycle.sample(work_end)
        q_start = self.cycle.qpos[0]
        c_start = float(self.cycle.confidence[0])
        arm_mask = ~self.cycle.snap_mask()
        for bay, mapping in self._maps.items():
            phase = PHASE_OFFSETS.get(bay, 0.0) * loop_s
            u = max(0.0, float(self._replay_t(bay, wall_t, frozen)) + phase) % loop_s
            if u < work_end:
                q, conf = self.cycle.sample(u)
            else:
                tau = min(1.0, (u - work_end) / transfer_s)
                w = 0.5 - 0.5 * math.cos(math.pi * tau)
                q = q_end.copy()
                q[arm_mask] = (1.0 - w) * q_end[arm_mask] + w * q_start[arm_mask]
                conf = (1.0 - w) * c_end + w * c_start
            for col, adr, width in mapping:
                data.qpos[adr : adr + width] = q[col : col + width]
            out[bay] = conf
        return out

    def conveyor_statuses(
        self, wall_t: float, *, estop: bool = False, frozen: Collection[int] = ()
    ) -> dict[int, ConveyorStatus]:
        """Seamless modular conveyor loop (pause during unlatching -> advance full PACK_PITCH_M).

        ``estop`` stops every belt; ``frozen`` bays (robot power cut) stop their own belt.
        """
        work_end, transfer_s, loop_s = self.loop_params()
        out: dict[int, ConveyorStatus] = {}
        for bay in self._maps:
            stopped = estop or bay in frozen
            replay_t = self._replay_t(bay, wall_t, frozen)
            t_eff = self._last_t.get(bay, replay_t) if estop else replay_t
            self._last_t[bay] = t_eff
            if transfer_s <= 0.0 or loop_s <= 0.0:
                out[bay] = ConveyorStatus(bay_id=bay, mode="estop" if stopped else "paused")
                continue
            phase = PHASE_OFFSETS.get(bay, 0.0) * loop_s
            t_total = max(0.0, float(t_eff) + phase)
            n_cycles = int(t_total // loop_s)
            u = t_total % loop_s
            if u < work_end:
                mode = "paused"
                pack_dy = 0.0
                belt_dist = n_cycles * PACK_PITCH_M
            else:
                mode = "feeding"
                tau = min(1.0, (u - work_end) / transfer_s)
                pack_dy = tau * PACK_PITCH_M
                belt_dist = (n_cycles + tau) * PACK_PITCH_M
            self._belt_dist[bay] = belt_dist
            out[bay] = ConveyorStatus(
                bay_id=bay,
                mode="estop" if stopped else mode,
                belt_dist_m=belt_dist,
                pack_dy_m=pack_dy,
            )
        return out


class _NullSink:
    def emit(self, event: Any) -> None:
        pass

    def close(self) -> None:
        pass


class _NullReasoner:
    def submit(self, ctx: Any) -> None:
        pass

    def latest_verdict(self, bay_id: int) -> None:
        return None

    def close(self) -> None:
        pass


def record_cycle(max_sim_s: float = 30.0) -> BackgroundCycle:
    """Run Bay 1 headless through one routine cycle (plus ``RECORD_DWELL_S``) and record it.

    Samples every ``bay1/`` joint and Bay 1's C(t) at ``SAMPLE_HZ`` from the first tick until
    the cycle has cleared and dwelt. Raises ``RuntimeError`` if that takes over ``max_sim_s``.
    """
    from teleop.atc.coordinator import AtcCoordinator
    from teleop.unoq.client import UnoQClient

    unoq = UnoQClient.connect_or_mock("127.0.0.1", 1, 1, connect_timeout_s=0.2)
    co: AtcCoordinator | None = None
    try:
        co = AtcCoordinator.build(unoq, _NullSink(), _NullReasoner())
        joints = _bay_joints(co.model, SOURCE_BAY)
        ts: list[float] = []
        rows: list[FloatArray] = []
        confs: list[float] = []
        next_sample = 0.0
        cleared_at: float | None = None
        while co.time_s <= max_sim_s:
            snaps = co.tick()
            snap = next(s for s in snaps if s.bay_id == SOURCE_BAY)
            if co.time_s + 1e-9 >= next_sample:
                ts.append(co.time_s)
                rows.append(
                    np.concatenate([np.asarray(co.data.qpos[a : a + w]) for _, a, w in joints])
                )
                confs.append(float(snap.confidence))
                next_sample += 1.0 / SAMPLE_HZ
            if cleared_at is None and co.outcome(SOURCE_BAY) == "cleared":
                cleared_at = co.time_s
            if cleared_at is not None and co.time_s - cleared_at >= RECORD_DWELL_S:
                break
        else:
            raise RuntimeError(f"bay {SOURCE_BAY} did not clear within {max_sim_s} s")
    finally:
        if co is not None:
            co.close()
        unoq.close()  # also stops the local loopback node
    t = np.asarray(ts, dtype=np.float64)
    return BackgroundCycle(
        t=t - t[0],
        joint_names=tuple(n for n, _, _ in joints),
        qpos_widths=tuple(w for _, _, w in joints),
        qpos=np.vstack(rows).astype(np.float64),
        confidence=np.asarray(confs, dtype=np.float64),
    )
