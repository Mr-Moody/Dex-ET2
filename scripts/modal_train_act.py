"""Train the ACT recovery policy on a Modal GPU (``scripts/train_act.py`` in a container).

One-time setup: ``uvx modal setup``. Then, with a cleaned dataset from ``teleop process``::

    uvx modal volume create act-data
    uvx modal volume put act-data data/episodes/gen-1/clean /gen-1-clean
    uvx modal run scripts/modal_train_act.py --dataset gen-1-clean --run first --steps 12000
    uvx modal volume get act-data /models/first models/act/first

Billing is per second while the function runs, nothing idles afterwards. Checkpoints and
``run.json`` land in the ``act-data`` volume under ``/models/<run>``. ``--steps 1000`` makes a
cheap timing run first.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
VOLUME = modal.Volume.from_name("act-data", create_if_missing=True)
IGNORE = [".venv", ".git", "data", "outputs", "models", "third_party", "**/__pycache__"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("lerobot[training]>=0.7,<0.8")
    .add_local_dir(str(REPO), "/repo", copy=True, ignore=IGNORE)
    .run_commands("pip install -e /repo --no-deps")
)
app = modal.App("teleop-act-train", image=image)


@app.function(gpu="L4", volumes={"/vol": VOLUME}, timeout=4 * 3600)
def train(dataset: str, run: str, steps: int, batch_size: int) -> str:
    import subprocess

    out = f"/vol/models/{run}"
    cmd = [
        "python", "/repo/scripts/train_act.py",
        "--dataset", f"/vol/{dataset}", "--out", out,
        "--steps", str(steps), "--batch-size", str(batch_size), "--device", "cuda",
    ]  # fmt: skip
    subprocess.run(cmd, check=True, cwd="/repo")
    VOLUME.commit()
    return out


@app.local_entrypoint()
def main(dataset: str, run: str, steps: int = 12_000, batch_size: int = 32) -> None:
    print("trained ->", train.remote(dataset, run, steps, batch_size))
