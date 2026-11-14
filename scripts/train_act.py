"""Train an ACT recovery policy on a recorded ATC dataset by wrapping ``lerobot-train``.

Usage (needs ``uv sync --group act``)::

    uv run python scripts/train_act.py --dataset data/episodes/gen-smoke-clean \\
        --out models/act/smoke --steps 200 --batch-size 4

After a successful run it writes ``<out>/run.json`` (schema version, dataset hash, git commit,
preset) next to the checkpoints so a model can be traced back to its data and code. ``<out>``
must not exist yet (``lerobot-train`` refuses to overwrite a run). ``--dry-run`` prints the
``lerobot-train`` argv and does nothing else.

``lerobot-train`` flags used (LeRobot 0.7, checked with ``uv run lerobot-train --help``):
``--policy.type``, ``--policy.device``, ``--policy.chunk_size``, ``--policy.n_action_steps``,
``--policy.optimizer_lr``, ``--policy.push_to_hub``, ``--dataset.repo_id``, ``--dataset.root``,
``--output_dir``, ``--steps``, ``--batch_size``, ``--seed``, ``--save_freq``, ``--wandb.enable``.

This script never imports torch or lerobot itself; ``lerobot-train`` runs as a subprocess.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from teleop.policy.schema import SCHEMA_VERSION

REPO = Path(__file__).resolve().parents[1]
HASHED_META = (Path("meta") / "info.json", Path("meta") / "teleop_episodes.jsonl")


@dataclass(frozen=True)
class ActPreset:
    chunk_size: int = 30
    batch_size: int = 32
    steps: int = 50_000
    lr: float = 1e-5
    seed: int = 1000
    device: str = "cuda"
    save_freq: int = 5_000


def _lerobot_train() -> str:
    """The ``lerobot-train`` console script next to the running interpreter, else on PATH."""
    exe = Path(sys.executable).with_name(
        "lerobot-train.exe" if sys.platform == "win32" else "lerobot-train"
    )
    return str(exe) if exe.is_file() else shutil.which("lerobot-train") or "lerobot-train"


def build_train_args(dataset: Path, out: Path, preset: ActPreset) -> list[str]:
    """The full ``lerobot-train`` argv for one ACT run."""
    return [
        _lerobot_train(),
        "--policy.type=act",
        f"--policy.device={preset.device}",
        f"--policy.chunk_size={preset.chunk_size}",
        f"--policy.n_action_steps={preset.chunk_size}",
        f"--policy.optimizer_lr={preset.lr}",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id=local/{dataset.name}",
        f"--dataset.root={dataset}",
        f"--output_dir={out}",
        f"--steps={preset.steps}",
        f"--batch_size={preset.batch_size}",
        f"--seed={preset.seed}",
        f"--save_freq={preset.save_freq}",
        "--wandb.enable=false",
    ]


def dataset_sha256(dataset: Path) -> str:
    """Hash of the dataset's ``info.json`` and episode sidecar (missing files hash as empty)."""
    h = hashlib.sha256()
    for rel in HASHED_META:
        h.update(rel.as_posix().encode())
        p = dataset / rel
        h.update(p.read_bytes() if p.is_file() else b"")
    return h.hexdigest()


def write_run_json(dataset: Path, out: Path, preset: ActPreset, git_commit: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "run.json"
    data = {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(dataset),
        "dataset_sha256": dataset_sha256(dataset),
        "git_commit": git_commit,
        "preset": asdict(preset),
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _git_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def main(argv: list[str] | None = None, run: Callable[[list[str]], int] = subprocess.call) -> int:
    p = argparse.ArgumentParser(description="Train ACT on an ATC LeRobot dataset.")
    p.add_argument("--dataset", type=Path, required=True, help="LeRobot v3 dataset root")
    p.add_argument("--out", type=Path, required=True, help="output dir, e.g. models/act/<run>")
    p.add_argument("--steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--device", help="cuda (default), cpu or mps")
    p.add_argument("--dry-run", action="store_true", help="print the lerobot-train argv only")
    a = p.parse_args(argv)

    preset = ActPreset()
    if a.steps is not None:
        preset = replace(preset, steps=a.steps)
    if a.batch_size is not None:
        preset = replace(preset, batch_size=a.batch_size)
    if a.device is not None:
        preset = replace(preset, device=a.device)

    dataset: Path = a.dataset.resolve()
    out: Path = a.out.resolve()
    if not (dataset / "meta" / "info.json").is_file():
        print(f"error: {dataset} is not a LeRobot dataset (no meta/info.json)", file=sys.stderr)
        return 2

    args = build_train_args(dataset, out, preset)
    if a.dry_run:
        print(subprocess.list2cmdline(args))
        return 0
    if out.exists():
        # lerobot-train refuses an existing output dir unless resuming; fail before it starts.
        print(f"error: {out} already exists; pick a new --out", file=sys.stderr)
        return 2
    commit = _git_commit()  # before training, so edits made during a long run don't count
    code = run(args)
    if code == 0:
        # After the run: lerobot-train creates the output dir itself.
        write_run_json(dataset, out, preset, commit)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
