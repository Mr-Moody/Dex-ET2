"""Generate scripted ATC episodes and train the ACT recovery policy on Modal.

Everything lives on one Modal Volume (``act-training``): ``episodes/<name>/`` holds the raw
per-seed datasets and the cleaned merge, ``models/<name>/`` the checkpoints. Run it with the
Modal CLI (``pip install modal`` / ``modal setup``); the repo's own environment is not needed
locally::

    modal run scripts/modal_act.py::generate --name gen-1 --total 80 --workers 8
    modal run scripts/modal_act.py::train --name gen-1 --steps 50000
    modal volume get act-training models/gen-1 models/act/gen-1     # download the checkpoint

``generate`` is ``scripts/generate_episodes.py`` fanned out over containers (one seed each,
offscreen OSMesa rendering, no GPU) followed by ``teleop process``. ``train`` runs
``scripts/train_act.py`` unchanged on one GPU. Raise ``--seed-base`` (or change ``--name``) so
teammates do not generate the same data twice. See ``docs/training/modal.md``.

Training is billed per second while the function runs; it stops by itself when it finishes.
Set ``MODAL_GPU`` (default ``L40S``) to pick another GPU type before ``modal run``.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
APP_DIR = "/root/repo"  # teleop finds third_party/ and configs/ relative to src/teleop
VOLUME_DIR = "/vol"
GPU = os.environ.get("MODAL_GPU", "L40S")

app = modal.App("junction-act")
volume = modal.Volume.from_name("act-training", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # OSMesa gives MuJoCo an offscreen GL context on CPU-only containers; ffmpeg for LeRobot video.
    .apt_install(
        "libosmesa6",
        "libegl1",
        "libegl-mesa0",
        "libgles2",
        "libgl1-mesa-dri",
        "libgl1",
        "libglib2.0-0",
        "ffmpeg",
        "git",
    )  # fmt: skip
    .uv_sync(str(REPO), groups=["act"])
    # third_party/ is gitignored, so fetch the pinned Panda and Inspire models at build time.
    .add_local_file(
        REPO / "scripts" / "fetch_models.py", f"{APP_DIR}/scripts/fetch_models.py", copy=True
    )
    .run_commands(f"python {APP_DIR}/scripts/fetch_models.py")
    .env(
        {
            "PYTHONPATH": f"{APP_DIR}/src",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    .add_local_dir(REPO / "src", f"{APP_DIR}/src", ignore=["**/__pycache__/**"])
    .add_local_dir(REPO / "configs", f"{APP_DIR}/configs")
    .add_local_dir(REPO / "scripts", f"{APP_DIR}/scripts", ignore=["**/__pycache__/**"])
)


# LeRobot 0.7 replaced ``--wandb.enable`` with ``--trackio.enable``, but ``train_act.py`` (owned by
# T-194) still passes the old flag. Translate it here so that file stays untouched.
_TRAIN_SHIM = """
import sys
sys.path.insert(0, "scripts")
import train_act

_build = train_act.build_train_args
train_act.build_train_args = lambda *a, **k: [
    x.replace("--wandb.enable=false", "--trackio.enable=false") for x in _build(*a, **k)
]
raise SystemExit(train_act.main(sys.argv[1:]))
"""


def _run(argv: list[str], gl: str | None = None) -> int:
    print("+", " ".join(argv), f"(MUJOCO_GL={gl})" if gl else "", flush=True)
    env = {**os.environ, "MUJOCO_GL": gl, "PYOPENGL_PLATFORM": gl} if gl else None
    return subprocess.call(argv, cwd=APP_DIR, env=env)


@app.function(image=image, volumes={VOLUME_DIR: volume}, cpu=4, memory=8192, timeout=4 * 3600)
def generate_worker(name: str, seed: int, episodes: int) -> tuple[int, int]:
    """One ``teleop atc --headless --generate`` worker. Returns ``(seed, exit code)``."""
    root = f"{VOLUME_DIR}/episodes/{name}/raw/seed{seed}"
    argv = [
        sys.executable, "-m", "teleop.cli", "atc", "--headless", "--no-audio",
        "--generate", str(episodes), "--seed", str(seed), "--record-episodes", root,
    ]  # fmt: skip
    code = 1
    for gl in ("egl", "osmesa"):  # software GL differs per host; fall back if the renderer dies
        code = _run(argv, gl)
        if code in (0, 1):  # exit 1 means fewer episodes than asked, not a crash
            break
        print(f"{gl} rendering crashed (exit {code}); trying the next backend", flush=True)
    volume.commit()
    return seed, code


@app.function(image=image, volumes={VOLUME_DIR: volume}, cpu=4, memory=8192, timeout=3600)
def clean(name: str) -> str:
    """Clean and merge every generated seed folder with ``teleop process``."""
    volume.reload()
    raw = Path(VOLUME_DIR) / "episodes" / name / "raw"
    found = {
        p.parent.parent
        for pat in ("seed*/meta", "seed*/*/meta")
        for p in raw.glob(pat + "/teleop_episodes.jsonl")
    }
    if not found:
        raise RuntimeError(f"no generated datasets under {raw}")
    out = Path(VOLUME_DIR) / "episodes" / name / "clean"
    argv = [sys.executable, "-m", "teleop.cli", "process", "--out", str(out)]
    for s in sorted(found):
        argv += ["--in", str(s)]
    if _run(argv) != 0:
        raise RuntimeError("teleop process failed")
    volume.commit()
    return str(out)


@app.function(image=image, volumes={VOLUME_DIR: volume}, gpu=GPU, cpu=8, timeout=24 * 3600)
def train_act(
    name: str, steps: int | None, batch_size: int | None, git_commit: str
) -> dict[str, object]:
    """Run ``scripts/train_act.py`` on ``episodes/<name>/clean``, writing ``models/<name>``."""
    volume.reload()
    dataset = Path(VOLUME_DIR) / "episodes" / name / "clean"
    out = Path(VOLUME_DIR) / "models" / name
    argv = [sys.executable, "-c", _TRAIN_SHIM, "--dataset", str(dataset), "--out", str(out)]
    if steps is not None:
        argv += ["--steps", str(steps)]
    if batch_size is not None:
        argv += ["--batch-size", str(batch_size)]
    code = _run(argv)
    run_json = out / "run.json"
    if code == 0 and run_json.is_file():
        # The container has no git checkout, so stamp the commit the job was launched from.
        meta = json.loads(run_json.read_text(encoding="utf-8"))
        meta["git_commit"] = git_commit
        run_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    volume.commit()
    if code != 0:
        raise RuntimeError(f"train_act.py exited with {code}")
    return {"model": str(out), "run": json.loads(run_json.read_text(encoding="utf-8"))}


def _local_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


@app.local_entrypoint()
def generate(name: str, total: int = 60, workers: int = 6, seed_base: int = 1000) -> None:
    """Make ``total`` scripted episodes over ``workers`` containers, then clean and merge them."""
    if total < 1 or workers < 1:
        raise SystemExit("--total and --workers must be at least 1")
    per_worker = math.ceil(total / workers)
    jobs = [(name, seed_base + i, per_worker) for i in range(workers)]
    print(
        f"{workers} workers x {per_worker} episodes, seeds {seed_base}..{seed_base + workers - 1}"
    )
    results = list(generate_worker.starmap(jobs))
    for seed, code in results:
        print(f"worker seed {seed} exited with {code}")
    if all(code != 0 for _, code in results):
        raise SystemExit("every worker failed; see the logs above")
    print(f"cleaned dataset: {clean.remote(name)}")
    print(f"next: modal run scripts/modal_act.py::train --name {name}")


@app.local_entrypoint()
def train(name: str, steps: int = 0, batch_size: int = 0) -> None:
    """Train ACT on ``episodes/<name>/clean``. ``--steps`` / ``--batch-size`` 0 keeps the preset."""
    result = train_act.remote(name, steps or None, batch_size or None, _local_commit())
    print(json.dumps(result, indent=2))
    print(f"download: modal volume get act-training models/{name} models/act/{name}")
