"""``teleop eval --checkpoint DIR --dataset DIR``: the offline ACT gate and release manifest.

Every fault type in ``--faults`` is run through ``--rollouts`` seeded headless rollouts, the same
randomized conditions for the ACT checkpoint and for the DMP fitted to the dataset's first
successful operator rescue of that fault. A rollout starts in today's stall state (the live
routine push, stall detector and touch correction) and hands over to the controller, which does
its own pre-move. A fault passes when at least ``--bar`` rollouts clear the bolt inside the hand
torque, arm torque and joint limits. ``DIR/gate.json`` lists every fault with its counts, plus
the stall fingerprints of the passed ones: the live runtime runs ACT only on a stall that matches
one of those.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import mujoco as _mujoco  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from numpy.typing import NDArray

from teleop.atc.act_recovery import ActRecoveryFactory, CameraRenderer
from teleop.atc.types import SkillPrimitive, StallSignature
from teleop.policy.act_runtime import (
    InferenceWorker,
    ObsBatch,
    PolicyBackend,
    load_act_policy,
    pretrained_dir,
    read_policy_config,
)
from teleop.policy.manifest import FaultResult, GateManifest, ManifestError, Proto
from teleop.policy.schema import CAMERAS, SCHEMA_VERSION, STATE_NAMES, EpisodeMeta
from teleop.types import Pose
from teleop.unoq.dmp import HAND_ACTUATORS, compose_pose, relative_pose

mj: Any = _mujoco

FAULTS: tuple[str, ...] = ("flap", "pin")
DEFAULT_ROLLOUTS = 10
DEFAULT_BAR = 9
GATE_FILE = "gate.json"

# State columns by name, in HAND_ACTUATORS order (pinky first), so they follow the schema.
_HAND = [STATE_NAMES.index(f"hand/{a}") for a in HAND_ACTUATORS]
_WRIST = [STATE_NAMES.index(f"wrist/{f}") for f in ("x", "y", "z", "qx", "qy", "qz", "qw")]
_TAU_HAND = [STATE_NAMES.index(f"tau_hand/{a}") for a in HAND_ACTUATORS]


@dataclass(frozen=True)
class EvalEpisode:
    """What the gate needs from one recorded recovery episode."""

    fault: str | None
    source: str
    success: bool
    signature: tuple[float, float, float, float] | None
    t_s: NDArray[np.float64]
    wrist_latch: NDArray[np.float64]  # (n, 7): x, y, z, qx, qy, qz, qw in the latch frame
    hand: NDArray[np.float64]  # (n, 6) in HAND_ACTUATORS order
    tau_hand_inf: NDArray[np.float64]  # (n,)


def read_eval_episodes(root: Path) -> list[EvalEpisode]:
    """Every sidecar-labelled episode of an ATC dataset (needs ``lerobot``)."""
    from teleop.process.qc import num_episodes, open_dataset, read_sidecar

    root = Path(root)
    sidecar = read_sidecar(root)
    if not sidecar:
        raise ManifestError(f"{root}: no teleop_episodes.jsonl sidecar; not an ATC dataset")
    ds: Any = open_dataset(root)
    out: list[EvalEpisode] = []
    for ep in range(num_episodes(root)):
        row = sidecar.get(ep)
        if row is None or "source" not in row:
            continue
        meta = EpisodeMeta.from_json(row)
        ep_row = ds.meta.episodes[ep]
        lo, hi = int(ep_row["dataset_from_index"]), int(ep_row["dataset_to_index"])
        if hi - lo < 2:
            continue
        cols = ds.hf_dataset[lo:hi]
        state = np.stack([np.asarray(s, dtype=np.float64) for s in cols["observation.state"]])
        t = np.asarray([float(v) for v in cols["timestamp"]], dtype=np.float64)
        out.append(
            EvalEpisode(
                fault=meta.fault,
                source=meta.source,
                success=meta.success,
                signature=meta.signature,
                t_s=t,
                wrist_latch=state[:, _WRIST],
                hand=state[:, _HAND],
                tau_hand_inf=np.abs(state[:, _TAU_HAND]).max(axis=1),
            )
        )
    return out


def checkpoint_sha256(checkpoint: Path) -> str:
    """SHA-256 over the policy's ``config.json`` and ``model.safetensors`` (those present)."""
    path = pretrained_dir(checkpoint)
    h = hashlib.sha256()
    for name in ("config.json", "model.safetensors"):
        f = path / name
        if f.is_file():
            with f.open("rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- camera mirror


def mirror_bay_state(src_model: Any, src_data: Any, dst_model: Any, dst_data: Any) -> None:
    """Copy every joint position and body placement of ``src`` onto the same names in ``dst``.

    ``dst`` is the full fleet scene (cell, lights, both cameras) and ``src`` the validator's
    single-bay model; only names present in both are copied, so the other bays stay at rest.
    """
    obj = mj.mjtObj
    for j in range(int(src_model.njnt)):
        name = mj.mj_id2name(src_model, obj.mjOBJ_JOINT, j)
        if not name:
            continue
        dj = mj.mj_name2id(dst_model, obj.mjOBJ_JOINT, name)
        if dj < 0:
            continue
        n = {0: 7, 1: 4, 2: 1, 3: 1}[int(src_model.jnt_type[j])]
        s, d = int(src_model.jnt_qposadr[j]), int(dst_model.jnt_qposadr[dj])
        dst_data.qpos[d : d + n] = src_data.qpos[s : s + n]
    for b in range(1, int(src_model.nbody)):
        name = mj.mj_id2name(src_model, obj.mjOBJ_BODY, b)
        db = mj.mj_name2id(dst_model, obj.mjOBJ_BODY, name) if name else -1
        if db < 0:
            continue
        dst_model.body_pos[db] = src_model.body_pos[b]
        dst_model.body_quat[db] = src_model.body_quat[b]
    mj.mj_forward(dst_model, dst_data)


def _body_pose(data: Any, body: int) -> Pose:
    w, x, y, z = (float(v) for v in data.xquat[body])
    px, py, pz = (float(v) for v in data.xpos[body])
    return Pose(orientation=(x, y, z, w), position=(px, py, pz))


class MirrorRenderer:
    """Renders the policy cameras of the validator's bay 1 from the live fleet scene.

    The fleet scene's rescue cameras (the policy's ``top`` view) are world-fixed over the
    latch's build pose, while a gate rollout moves the latch by up to 8 cm and 30 deg. They
    are moved with the mirrored latch, keeping their pose relative to it, so the ``top`` view
    frames the latch the way it does in every fleet bay the policy was trained in.
    """

    def __init__(self, src_model: Any, image_hw: tuple[int, int]) -> None:
        from teleop.policy.cameras import BayCameraRenderer
        from teleop.retarget.clutch import rescue_camera_names
        from teleop.sim.bays import bay_prefix, build_four_bay_scene

        self.src_model = src_model
        self.model: Any = build_four_bay_scene()
        self.data: Any = mj.MjData(self.model)
        self._inner = BayCameraRenderer(self.model, image_hw)
        mj.mj_forward(self.model, self.data)
        p = bay_prefix(1)
        self._latch = int(self.model.body(f"{p}latch_base").id)
        latch0 = _body_pose(self.data, self._latch)
        self._cams: dict[int, Pose] = {}
        for name in rescue_camera_names(p):
            cid = int(self.model.camera(name).id)
            w, x, y, z = (float(v) for v in self.model.cam_quat[cid])
            px, py, pz = (float(v) for v in self.model.cam_pos[cid])
            cam = Pose(orientation=(x, y, z, w), position=(px, py, pz))
            self._cams[cid] = relative_pose(latch0, cam)  # world-fixed: cam_pos is world

    def render(self, data: Any, bay_id: int) -> dict[str, NDArray[np.uint8]]:
        mirror_bay_state(self.src_model, data, self.model, self.data)
        latch = _body_pose(self.data, self._latch)
        for cid, rel in self._cams.items():
            pose = compose_pose(latch, rel)
            assert pose.position is not None
            x, y, z, w = pose.orientation
            self.model.cam_pos[cid] = pose.position
            self.model.cam_quat[cid] = (w, x, y, z)
        mj.mj_camlight(self.model, self.data)
        return self._inner.render(self.data, bay_id)

    def close(self) -> None:
        self._inner.close()


class SyncWorker(InferenceWorker):
    """The live worker's interface, but ``submit`` runs the policy before it returns.

    The live worker answers from its own thread, so a rollout's hold ticks would depend on
    thread timing and device speed. Here each observation gets its chunk before the next physics
    step: rollouts are reproducible, and ``p95_ms`` is the policy's own forward-pass time.
    """

    def __init__(self, policy: PolicyBackend) -> None:
        super().__init__(policy, autostart=False)

    def submit(
        self, bay_id: int, state: NDArray[Any], images: Mapping[str, NDArray[Any]], tick: int
    ) -> None:
        if not self.healthy():
            return
        batch = ObsBatch(
            np.asarray(state, dtype=np.float32).reshape(1, len(STATE_NAMES)),
            {c: np.asarray(images[c], dtype=np.uint8)[None] for c in CAMERAS},
        )
        t0 = time.perf_counter_ns()
        try:
            chunk = np.asarray(self.policy.predict_chunk(batch), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - reported as an unhealthy worker, like live
            self._fail(f"inference error: {exc}")
            return
        ms = (time.perf_counter_ns() - t0) * 1e-6
        if chunk.ndim != 3 or chunk.shape[0] != 1:
            self._fail(f"inference error: chunk shape {chunk.shape}, batch of 1")
            return
        with self._cv:
            self.calls += 1
            self.batch_sizes.append(1)
            self._latency_ms.append(ms)
            self._results[bay_id] = (int(tick), chunk[0])


def _make_renderer(model: Any, image_hw: tuple[int, int]) -> CameraRenderer:
    """The eval's policy-camera renderer; a module-level name so tests can replace it."""
    return MirrorRenderer(model, image_hw)


# --------------------------------------------------------------------------- the gate


def fit_dmp_for(fault: str, episodes: Sequence[EvalEpisode]) -> SkillPrimitive | None:
    """A DMP fitted to the first successful operator rescue of ``fault`` (glove, else scripted)."""
    from teleop.unoq.dmp import fit_rbf_dmp

    for source in ("glove", "scripted"):
        for ep in episodes:
            if ep.fault != fault or not ep.success or ep.source != source:
                continue
            proto = StallSignature.from_vector(ep.signature) if ep.signature else None
            return fit_rbf_dmp(
                ep.t_s,
                ep.wrist_latch,
                ep.hand,
                ep.tau_hand_inf,
                prototype=proto,
                signature_id=f"eval_{fault}",
                skill_id=f"eval/{fault}",
            )
    return None


def evaluate(
    checkpoint: Path,
    episodes: Sequence[EvalEpisode],
    *,
    faults: Sequence[str],
    rollouts: int,
    bar: int,
    device: str,
    seed: int,
    realism: float,
    log: Callable[[str], None] = print,
    policy: PolicyBackend | None = None,
) -> GateManifest:
    """Run the gate and return the manifest (the caller saves it)."""
    from teleop.atc.coordinator import CoordinatorConfig
    from teleop.atc.recovery import DmpRecovery, RecoveryController
    from teleop.atc.validator import BayRig, SkillValidator
    from teleop.sim.hands import load_hand

    _, image_hw = read_policy_config(checkpoint)
    if policy is None:
        policy = load_act_policy(checkpoint, device)
    validator = SkillValidator(
        load_hand("inspire"),
        n_rollouts=rollouts,
        pass_min=min(bar, rollouts),
        seed=seed,
        realism=realism,
    )
    worker = SyncWorker(policy)
    renderer = _make_renderer(validator.model, image_hw)
    stub = GateManifest(SCHEMA_VERSION, "", (bar, rollouts), {}, {}, None)
    factory = ActRecoveryFactory(stub, worker, renderer, validator.data)
    cfg = CoordinatorConfig()
    results: dict[str, FaultResult] = {}
    prototypes: dict[str, list[Proto]] = {}
    try:
        for fault in faults:
            good = [e for e in episodes if e.fault == fault and e.success]
            if not good:
                results[fault] = FaultResult(fault, 0, 0, None, 0.0, None, note="no data")
                log(f"  {fault}: no successful episodes in the dataset, not evaluated")
                continue
            kind = cast("Literal['flap', 'pin']", fault)
            conds = validator.draw_conditions(f"eval/{fault}")
            act = validator.validate_controller(
                lambda rig: factory.create(rig, StallSignature(0.0, 0.0, 0.0, 0.0)),
                fault=kind,
                conditions=conds,
            )
            skill = fit_dmp_for(fault, episodes)
            dmp_passed: int | None = None
            if skill is not None:

                def make_dmp(rig: BayRig, skill: SkillPrimitive = skill) -> RecoveryController:
                    return DmpRecovery(
                        skill,
                        rig,
                        backoff_m=cfg.backoff_m,
                        backoff_s=cfg.backoff_s,
                        blend_s=cfg.blend_s,
                        hold_s=cfg.skill_hold_s,
                        tau_margin=cfg.tau_margin,
                        tau_hard_max=cfg.tau_hard_max,
                        timeout_s=cfg.skill_timeout_s,
                    )

                dmp_passed = validator.validate_controller(
                    make_dmp, fault=kind, conditions=conds
                ).n_passed
            clear = [r.time_to_clear_s for r in act.results if r.time_to_clear_s is not None]
            results[fault] = FaultResult(
                fault=fault,
                rollouts=act.total,
                passed_rollouts=act.n_passed,
                mean_time_to_clear_s=float(np.mean(clear)) if clear else None,
                max_tau_hand_inf=max((r.max_tau_ext_inf_nm for r in act.results), default=0.0),
                dmp_passed_rollouts=dmp_passed,
            )
            if act.n_passed >= bar:
                prototypes[fault] = [e.signature for e in good if e.signature]
        return GateManifest(
            schema_version=SCHEMA_VERSION,
            checkpoint_sha256=checkpoint_sha256(checkpoint),
            bar=(bar, rollouts),
            results=results,
            prototypes=prototypes,
            inference_p95_ms=worker.p95_ms(),
        )
    finally:
        factory.close()


def format_table(manifest: GateManifest) -> str:
    need, total = manifest.bar
    rows = [f"{'fault':<6} {'ACT':>7} {'DMP':>7} {'clear s':>8} {'max Nm':>7}  verdict"]
    passed = set(manifest.passed_faults())
    for fault, r in manifest.results.items():
        dmp = "-" if r.dmp_passed_rollouts is None else f"{r.dmp_passed_rollouts}/{r.rollouts}"
        t = "-" if r.mean_time_to_clear_s is None else f"{r.mean_time_to_clear_s:.2f}"
        verdict = "PASS" if fault in passed else (r.note or f"FAIL (need {need}/{total})")
        rows.append(
            f"{fault:<6} {f'{r.passed_rollouts}/{r.rollouts}':>7} {dmp:>7} {t:>8} "
            f"{r.max_tau_hand_inf:>7.2f}  {verdict}"
        )
    p95 = manifest.inference_p95_ms
    rows.append("inference p95: " + ("n/a" if p95 is None else f"{p95:.1f} ms"))
    return "\n".join(rows)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="teleop eval", description="Gate an ACT checkpoint offline and write gate.json."
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--faults", default=",".join(FAULTS), help="comma-separated: flap,pin")
    p.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS)
    p.add_argument("--bar", type=int, default=DEFAULT_BAR, help="rollouts that must pass")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--realism", type=float, default=None, help="gate randomization scale (default: live)"
    )
    return p


def main(argv: list[str]) -> int:
    ns = _parser().parse_args(argv)
    faults = [f.strip() for f in str(ns.faults).split(",") if f.strip()]
    unknown = sorted(set(faults) - set(FAULTS))
    if unknown or not faults:
        print(
            f"teleop eval: unknown --faults {', '.join(unknown) or '(empty)'}; "
            f"expected a subset of {','.join(FAULTS)}",
            file=sys.stderr,
        )
        return 2
    if not 0 < ns.bar <= ns.rollouts:
        print(f"teleop eval: --bar must be in 1..--rollouts ({ns.rollouts})", file=sys.stderr)
        return 2
    realism = ns.realism
    if realism is None:
        from teleop.atc.coordinator import DEFAULT_REALISM

        realism = DEFAULT_REALISM
    try:
        episodes = read_eval_episodes(ns.dataset)
        manifest = evaluate(
            ns.checkpoint,
            episodes,
            faults=faults,
            rollouts=ns.rollouts,
            bar=ns.bar,
            device=ns.device,
            seed=ns.seed,
            realism=float(realism),
        )
    except ManifestError as exc:
        print(f"teleop eval: {exc}", file=sys.stderr)
        return 2
    except (ImportError, RuntimeError) as exc:
        print(f"teleop eval: {exc}", file=sys.stderr)
        return 2
    out = Path(ns.checkpoint) / GATE_FILE
    manifest.save(out)
    print(format_table(manifest))
    print(f"wrote {out}")
    return 0 if manifest.passed_faults() else 1


__all__ = [
    "EvalEpisode",
    "MirrorRenderer",
    "SyncWorker",
    "checkpoint_sha256",
    "evaluate",
    "fit_dmp_for",
    "format_table",
    "main",
    "mirror_bay_state",
    "read_eval_episodes",
]
