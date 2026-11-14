"""Canned routine cycle that animates the background Bays 3-4 on the fleet ops console.

Bays 3-4 are visual-only props in the fleet scene: the coordinator never runs them. For the
console they replay one recorded routine Bay 1 unlatch cycle (joint positions by name plus its
C(t) trace), phase-offset so they don't move in sync. This is display only: the coordinator
still treats them as static, and the console marks their telemetry as replayed.

``record_cycle`` regenerates the clip headless (``scripts/record_background_cycle.py``);
``BackgroundPlayer`` writes interpolated joint positions into a render-side ``MjData``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from teleop.atc.hud import ConveyorStatus
from teleop.sim.cells import CONVEYOR_SPEED_M_S

DEFAULT_CYCLE_PATH = Path(__file__).parent / "data" / "background_cycle.npz"
BLEND_S = 0.5
PHASE_OFFSETS: dict[int, float] = {3: 0.37, 4: 0.71}
LATCH_JOINTS = frozenset({"bolt_slide", "flap_hinge"})  # snap at the wrap: a new pallet
SAMPLE_HZ = 30.0
RECORD_DWELL_S = 1.0
SOURCE_BAY = 1
_BG_FEED_IN_END_S = 1.32
_BG_FEED_OUT_START_S = 3.25

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

    def _snap_mask(self) -> NDArray[np.bool_]:
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
            blend = ~self._snap_mask()
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

    def apply(self, data: Any, wall_t: float) -> dict[int, float]:
        """Pose each background bay at ``wall_t``; returns ``{bay: C(t)}``. No ``mj_forward``."""
        out: dict[int, float] = {}
        dur = self.cycle.duration_s
        for bay, mapping in self._maps.items():
            phase = PHASE_OFFSETS.get(bay, 0.0) * dur
            q, conf = self.cycle.sample(wall_t + phase)
            for col, adr, width in mapping:
                data.qpos[adr : adr + width] = q[col : col + width]
            out[bay] = conf
        return out

    def conveyor_statuses(self, wall_t: float, *, estop: bool = False) -> dict[int, ConveyorStatus]:
        """Compute segmented conveyor belt status (feed in -> pause -> advance out) for Bays 3-4."""
        dur = self.cycle.duration_s
        out: dict[int, ConveyorStatus] = {}
        for bay in self._maps:
            t_eff = self._last_t.get(bay, wall_t) if estop else wall_t
            self._last_t[bay] = t_eff
            phase = PHASE_OFFSETS.get(bay, 0.0) * dur
            t_total = max(0.0, t_eff + phase)
            t_in = t_total % dur if dur > 0.0 else 0.0
            feed_out_dur = max(0.0, dur - _BG_FEED_OUT_START_S)
            feed_per_cycle = (_BG_FEED_IN_END_S + feed_out_dur) * CONVEYOR_SPEED_M_S
            n_cycles = int(t_total // dur) if dur > 0.0 else 0
            if t_in < _BG_FEED_IN_END_S:
                mode = "feeding"
                pack_dy = -(_BG_FEED_IN_END_S - t_in) * CONVEYOR_SPEED_M_S
                cycle_dist = t_in * CONVEYOR_SPEED_M_S
            elif t_in < _BG_FEED_OUT_START_S:
                mode = "paused"
                pack_dy = 0.0
                cycle_dist = _BG_FEED_IN_END_S * CONVEYOR_SPEED_M_S
            else:
                mode = "feeding"
                pack_dy = (t_in - _BG_FEED_OUT_START_S) * CONVEYOR_SPEED_M_S
                cycle_dist = (
                    _BG_FEED_IN_END_S + (t_in - _BG_FEED_OUT_START_S)
                ) * CONVEYOR_SPEED_M_S
            belt_dist = n_cycles * feed_per_cycle + cycle_dist
            self._belt_dist[bay] = belt_dist
            out[bay] = ConveyorStatus(
                bay_id=bay,
                mode="estop" if estop else mode,
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
        unoq.close()  # also stops the local loopback node
    t = np.asarray(ts, dtype=np.float64)
    return BackgroundCycle(
        t=t - t[0],
        joint_names=tuple(n for n, _, _ in joints),
        qpos_widths=tuple(w for _, _, w in joints),
        qpos=np.vstack(rows).astype(np.float64),
        confidence=np.asarray(confs, dtype=np.float64),
    )
