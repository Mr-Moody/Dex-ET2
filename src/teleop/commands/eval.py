"""``teleop eval --checkpoint DIR --dataset DIR [--faults flap,pin] [--rollouts 10] [--bar 9]``.

Offline ACT release gate (spec section 8): runs ACT and a comparison DMP through the same seeded
``SkillValidator`` rollouts per fault type, writes ``<checkpoint>/gate.json`` with the passed
faults and their stall fingerprint prototypes, and prints a per-fault summary table.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from teleop.atc.act_recovery import GATE_FILE, ActRecoveryFactory
from teleop.atc.coordinator import DEFAULT_REALISM
from teleop.atc.recovery import DmpRecovery, RecoveryController
from teleop.atc.types import SkillPrimitive, StallSignature
from teleop.atc.validator import (
    TAU_HARD_MAX_NM,
    TAU_MARGIN_NM,
    BayRig,
    SkillValidator,
)
from teleop.policy.act_runtime import (
    InferenceWorker,
    PolicyBackend,
    ReplayPolicy,
    load_act_policy,
    pretrained_dir,
    read_policy_config,
)
from teleop.policy.cameras import BayCameraRenderer
from teleop.policy.manifest import FaultResult, GateManifest, ManifestError, Proto
from teleop.policy.schema import (
    CAMERAS,
    FPS,
    IMAGE_HW,
    SCHEMA_VERSION,
    STATE_NAMES,
    EpisodeMeta,
    command_from_action,
)
from teleop.process.qc import episode_actions, open_dataset, read_sidecar
from teleop.sim.hands import load_hand
from teleop.types import Pose
from teleop.unoq.dmp import HAND_ACTUATORS, fit_rbf_dmp

_load_policy = load_act_policy

_HAND_IDX = slice(
    STATE_NAMES.index(f"hand/{HAND_ACTUATORS[0]}"),
    STATE_NAMES.index(f"hand/{HAND_ACTUATORS[-1]}") + 1,
)
_WRIST_IDX = slice(STATE_NAMES.index("wrist/x"), STATE_NAMES.index("wrist/qw") + 1)
_TAU_HAND_IDX = slice(
    STATE_NAMES.index(f"tau_hand/{HAND_ACTUATORS[0]}"),
    STATE_NAMES.index(f"tau_hand/{HAND_ACTUATORS[-1]}") + 1,
)
_IDENTITY_LATCH = Pose(orientation=(0.0, 0.0, 0.0, 1.0), position=(0.0, 0.0, 0.0))


class _EvalRenderer(BayCameraRenderer):
    """Uses ``BayCameraRenderer`` for vision policies; skips GL work for ``ReplayPolicy``."""

    def __init__(self, model: Any, image_hw: tuple[int, int], *, skip_gl: bool = False) -> None:
        super().__init__(model, image_hw)
        self._skip_gl = skip_gl

    def render(self, data: Any, bay_id: int) -> dict[str, NDArray[np.uint8]]:
        if not self._skip_gl:
            try:
                return super().render(data, bay_id)
            except Exception:  # noqa: BLE001 - fall back to blank frames if headless GL is absent
                self._skip_gl = True
        h, w = self.image_hw
        return {cam: np.zeros((h, w, 3), dtype=np.uint8) for cam in CAMERAS}


def _checkpoint_sha256(checkpoint: Path) -> str:
    """SHA-256 over the checkpoint's files (excluding ``gate.json``)."""
    try:
        root = pretrained_dir(checkpoint)
    except ManifestError:
        root = checkpoint
    h = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != GATE_FILE)
    if not files:
        h.update(str(checkpoint.resolve()).encode("utf-8"))
        return h.hexdigest()
    for p in files:
        h.update(p.relative_to(root).as_posix().encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def _image_hw(checkpoint: Path) -> tuple[int, int]:
    try:
        _, hw = read_policy_config(checkpoint)
        return hw
    except ManifestError:
        return IMAGE_HW


def _episode_states(ds: Any, ep: int) -> NDArray[np.float32]:
    row = ds.meta.episodes[ep]
    lo, hi = int(row["dataset_from_index"]), int(row["dataset_to_index"])
    if hi <= lo:
        return np.zeros((0, len(STATE_NAMES)), np.float32)
    col = ds.hf_dataset[lo:hi]["observation.state"]
    return np.stack([np.asarray(s, dtype=np.float32) for s in col])


def _fit_episode_dmp(
    ds: Any, ep: int, meta: EpisodeMeta, fault: str, fps: int
) -> SkillPrimitive | None:
    states = _episode_states(ds, ep)
    actions = episode_actions(ds, ep)
    n = len(states)
    if n < 2:
        return None
    t = np.arange(n, dtype=np.float64) / max(fps, 1)
    wrist = np.asarray(states[:, _WRIST_IDX], dtype=np.float64)
    fingers = np.asarray(states[:, _HAND_IDX], dtype=np.float64)
    tau = np.max(np.abs(np.asarray(states[:, _TAU_HAND_IDX], dtype=np.float64)), axis=1)
    if np.any(np.linalg.norm(wrist[:, 3:], axis=1) < 1e-6) and len(actions) == n:
        rebuilt_w = np.zeros((n, 7), dtype=np.float64)
        rebuilt_f = np.zeros((n, len(HAND_ACTUATORS)), dtype=np.float64)
        for k in range(n):
            pose, fdict = command_from_action(_IDENTITY_LATCH, actions[k])
            assert pose.position is not None
            rebuilt_w[k, :3] = pose.position
            rebuilt_w[k, 3:] = pose.orientation
            rebuilt_f[k] = [fdict[a] for a in HAND_ACTUATORS]
        wrist, fingers = rebuilt_w, rebuilt_f
    return fit_rbf_dmp(
        t,
        wrist,
        fingers,
        tau,
        prototype=meta.signature,
        signature_id=meta.signature_id or fault,
    )


def _pick_dmp_episode(episodes: list[tuple[int, EpisodeMeta]]) -> tuple[int, EpisodeMeta]:
    for source in ("glove", "scripted"):
        for ep, meta in episodes:
            if meta.source == source:
                return ep, meta
    return episodes[0]


def _make_dmp_controller(skill: SkillPrimitive, rig: BayRig) -> RecoveryController:
    return DmpRecovery(
        skill,
        rig,
        backoff_m=0.01,
        backoff_s=0.25,
        blend_s=0.2,
        hold_s=0.2,
        tau_margin=TAU_MARGIN_NM,
        tau_hard_max=TAU_HARD_MAX_NM,
        timeout_s=0.5,
    )


def _fault_kind(fault: str) -> Literal["flap", "pin"]:
    return "pin" if fault == "pin" else "flap"


def _format_row(res: FaultResult, p95_ms: float | None) -> str:
    act_str = f"{res.passed_rollouts}/{res.rollouts}"
    dmp_str = (
        "-" if res.dmp_passed_rollouts is None else f"{res.dmp_passed_rollouts}/{res.rollouts}"
    )
    ttc_str = "-" if res.mean_time_to_clear_s is None else f"{res.mean_time_to_clear_s:.2f}s"
    tau_str = f"{res.max_tau_hand_inf:.2f}Nm"
    p95_str = "-" if p95_ms is None else f"{p95_ms:.1f}ms"
    note_str = f"  ({res.note})" if res.note else ""
    return (
        f"{res.fault:<8}  ACT {act_str:<6}  DMP {dmp_str:<6}  "
        f"clear {ttc_str:<7}  max_tau {tau_str:<8}  p95 {p95_str}{note_str}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="teleop eval")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--faults", default="flap,pin")
    ap.add_argument("--rollouts", type=int, default=10)
    ap.add_argument("--bar", type=int, default=9)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--realism", type=float, default=DEFAULT_REALISM)
    ns = ap.parse_args(argv)

    faults = [f.strip() for f in str(ns.faults).split(",") if f.strip()]
    if not faults:
        print("teleop eval: --faults must list at least one fault", file=sys.stderr)
        return 2

    checkpoint: Path = Path(ns.checkpoint)
    dataset: Path = Path(ns.dataset)
    sidecar_raw = read_sidecar(dataset)
    by_fault: dict[str, list[tuple[int, EpisodeMeta]]] = {f: [] for f in faults}
    for ep_idx, raw in sorted(sidecar_raw.items()):
        if "source" not in raw:
            continue
        meta = EpisodeMeta.from_json(raw)
        if meta.success and meta.fault in by_fault:
            by_fault[meta.fault].append((ep_idx, meta))

    policy: PolicyBackend = _load_policy(checkpoint, str(ns.device))
    validator = SkillValidator(
        load_hand("inspire"),
        n_rollouts=int(ns.rollouts),
        pass_min=int(ns.bar),
        seed=int(ns.seed),
        realism=float(ns.realism),
    )
    renderer = _EvalRenderer(
        validator.model,
        _image_hw(checkpoint),
        skip_gl=isinstance(policy, ReplayPolicy),
    )
    worker = InferenceWorker(policy)
    empty_manifest = GateManifest(
        schema_version=SCHEMA_VERSION,
        checkpoint_sha256="",
        bar=(int(ns.bar), int(ns.rollouts)),
        results={},
        prototypes={},
        inference_p95_ms=None,
    )
    factory = ActRecoveryFactory(empty_manifest, worker, renderer, validator.data)
    dummy_sig = StallSignature(0.0, 0.0, 0.0, 0.0)

    results: dict[str, FaultResult] = {}
    prototypes: dict[str, list[Proto]] = {}
    ds: Any = None
    try:
        for fault in faults:
            episodes = by_fault.get(fault, [])
            if not episodes:
                results[fault] = FaultResult(
                    fault=fault,
                    rollouts=0,
                    passed_rollouts=0,
                    mean_time_to_clear_s=None,
                    max_tau_hand_inf=0.0,
                    dmp_passed_rollouts=None,
                    note="no data",
                )
                continue

            if ds is None:
                ds = open_dataset(dataset)
            fps = int(getattr(ds, "fps", FPS))
            conds = validator.draw_conditions(fault)
            fkind = _fault_kind(fault)

            act_report = validator.validate_controller(
                lambda rig: factory.create(rig, dummy_sig),
                fault=fkind,
                conditions=conds,
            )
            passed_times = [r.elapsed_s for r in act_report.results if r.passed]
            mean_ttc = float(np.mean(passed_times)) if passed_times else None
            max_tau = max((r.max_tau_ext_inf_nm for r in act_report.results), default=0.0)

            dmp_ep, dmp_meta = _pick_dmp_episode(episodes)
            dmp_skill = _fit_episode_dmp(ds, dmp_ep, dmp_meta, fault, fps)
            dmp_passed: int | None = None
            if dmp_skill is not None:
                dmp_report = validator.validate_controller(
                    lambda rig, sk=dmp_skill: _make_dmp_controller(sk, rig),
                    fault=fkind,
                    conditions=conds,
                )
                dmp_passed = dmp_report.n_passed

            results[fault] = FaultResult(
                fault=fault,
                rollouts=len(conds),
                passed_rollouts=act_report.n_passed,
                mean_time_to_clear_s=mean_ttc,
                max_tau_hand_inf=max_tau,
                dmp_passed_rollouts=dmp_passed,
            )
            if act_report.n_passed >= int(ns.bar):
                protos: list[Proto] = []
                for _, meta in episodes:
                    if meta.signature is not None and len(meta.signature) == 4:
                        a, b, c, d = (float(v) for v in meta.signature)
                        protos.append((a, b, c, d))
                prototypes[fault] = protos

        p95_ms = worker.p95_ms()
    finally:
        factory.close()

    checkpoint.mkdir(parents=True, exist_ok=True)
    manifest = GateManifest(
        schema_version=SCHEMA_VERSION,
        checkpoint_sha256=_checkpoint_sha256(checkpoint),
        bar=(int(ns.bar), int(ns.rollouts)),
        results=results,
        prototypes=prototypes,
        inference_p95_ms=p95_ms,
    )
    manifest.save(checkpoint / GATE_FILE)

    for fault in faults:
        print(_format_row(results[fault], p95_ms))
    return 0 if manifest.passed_faults() else 1


if __name__ == "__main__":
    raise SystemExit(main())
