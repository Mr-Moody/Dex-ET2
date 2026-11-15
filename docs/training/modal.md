# Generating data and training ACT on Modal

An alternative to [the Verda runbook](verda.md). `scripts/modal_act.py` runs the episode
generator on CPU containers and `scripts/train_act.py` on one GPU, with a Modal Volume
(`act-training`) holding the data and checkpoints. Nothing stays running afterwards, so there is
no instance to delete; you pay per second while a function runs.

## One-off setup

```bash
pip install modal        # or: uv tool install modal
modal setup              # logs in; creates ~/.modal.toml (never commit it)
```

## 1. Generate scripted episodes

```bash
modal run scripts/modal_act.py::generate --name gen-1 --total 80 --workers 8 --seed-base 1000
```

One container per seed (4 CPU each) runs `teleop atc --headless --generate`, rendering offscreen
with OSMesa. The seed folders land in `episodes/gen-1/raw/`, then `teleop process` merges them
into `episodes/gen-1/clean/`. Use a different `--seed-base` per teammate.

## 2. Train

```bash
modal run scripts/modal_act.py::train --name gen-1                 # preset: 50 000 steps, batch 32
modal run scripts/modal_act.py::train --name gen-1 --steps 200 --batch-size 4    # smoke test
MODAL_GPU=A100 modal run scripts/modal_act.py::train --name gen-1  # another GPU (default L40S)
```

Add `--detach` after `modal run` to let a long run continue if your laptop disconnects
(`modal run --detach scripts/modal_act.py::train ...`). `--name` must not have a model yet:
`lerobot-train` refuses to overwrite `models/<name>`.

## 3. Download the checkpoint

```bash
modal volume get act-training models/gen-1 models/act/gen-1
```

Check `models/act/gen-1/run.json` and `models/act/gen-1/checkpoints/last/` arrived.

## Using a dataset recorded locally

```bash
modal volume put act-training data/episodes/<clean-dataset> episodes/<name>/clean
modal run scripts/modal_act.py::train --name <name>
```

## Housekeeping

- `modal volume ls act-training episodes` / `modal volume rm -r act-training episodes/<name>`
- `modal app list` shows running apps; `modal app stop <id>` ends one.
- The image installs the lockfile's `act` group (torch from the CUDA 13.2 index), so the GPU
  host driver must support CUDA 13. If `torch.cuda.is_available()` is False in the logs, pick a
  newer GPU type with `MODAL_GPU`.
- Record the GPU type, wall time and cost in `docs/training/results.md`.
