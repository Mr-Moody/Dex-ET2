# Training ACT on Verda

> **Cost warning: a Verda GPU instance bills for every minute it exists, idle or not.**
> Hibernating does not save you money (Verda says it "has the same effect as delete").
> When training finishes, copy the checkpoint back and **delete the instance (step 10)**, then
> check `verda vm list` shows nothing and the dashboard shows no leftover volumes. Note the
> instance type, wall time and cost in `docs/training/results.md`.

The runbook turns a processed ATC dataset (`teleop process` output, LeRobot v3 with the
`meta/teleop_episodes.jsonl` sidecar) into `models/act/<run>/` on the laptop.
`scripts/train_act.py` wraps `lerobot-train` with the ACT preset for this schema and writes
`run.json` (schema version, dataset hash, git commit, preset) next to the checkpoints.

Commands marked *(unconfirmed)* were not in the Verda CLI docs when this was written; check
`verda <command> --help` before relying on them. CLI reference: <https://docs.verda.com/cli/>.

## 0. Before you start (laptop)

Check the run works locally first. It takes seconds and catches dataset problems for free:

```bash
uv sync --group act
uv run python scripts/train_act.py --dataset data/episodes/<dataset> --out models/act/dry --dry-run
uv run python scripts/train_act.py --dataset data/episodes/<dataset> --out models/act/smoke \
    --steps 200 --batch-size 4
```

## 1. Install the Verda CLI

```bash
# Linux / macOS
curl -sSL https://raw.githubusercontent.com/verda-cloud/verda-cli/main/scripts/install.sh | sh
# Windows (Scoop)
scoop bucket add verda https://github.com/verda-cloud/homebrew-tap
scoop install verda-cli

verda --version
verda doctor          # checks credentials, connectivity and configuration
```

## 2. API credentials

Create a client ID and secret in the Verda dashboard, then run `verda auth login`, which saves
them to `~/.verda/credentials`. You can also set `VERDA_CLIENT_ID` / `VERDA_CLIENT_SECRET` in
your shell. **Enter them yourself. Never put them in this repo, `.env` or a commit.**
Run `verda doctor` again and it should pass.

## 3. Register your SSH key

```bash
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519
verda ssh-key add       # prompts for a name and the public key: paste ~/.ssh/id_ed25519.pub
verda ssh-key list
```

## 4. Pick an instance type

```bash
verda instance-types --gpu
```

ACT on 128 × 128 images trains well on one GPU. A single 24 GB+ card is plenty for batch 32.
Pick the cheapest available single-GPU type. Write down its hourly price.

## 5. Create the instance

```bash
verda vm create --kind gpu --instance-type <TYPE> --location FIN-01 \
    --os ubuntu-24.04-cuda-13.0-open-docker --os-volume-size 100 \
    --hostname act-train --wait
verda vm list
```

`verda vm create` with no flags runs a wizard instead. Pick the same SSH key from step 3.
The `act` group uses CUDA 13.2 wheels, so pick an image whose driver supports CUDA 13
(any `cuda-13.*` image).

## 6. Set up the instance

```bash
verda ssh act-train
# on the instance:
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone https://github.com/Mr-Moody/Dex-ET2.git && cd Dex-ET2
git checkout <commit you are training>
uv sync --group act
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

## 7. Upload the dataset

From the laptop, with rsync (Git Bash or WSL on Windows; `verda vm describe act-train` shows
the IP *(unconfirmed field name)*):

```bash
rsync -avz --progress data/episodes/<dataset>/ ubuntu@<IP>:Dex-ET2/data/episodes/<dataset>/
```

Or push it to a **private** Hugging Face Hub dataset and download it on the instance with
`hf download --repo-type dataset <you>/<dataset> --local-dir data/episodes/<dataset>`.

## 8. Train (inside tmux)

```bash
tmux new -s act
uv run python scripts/train_act.py --dataset data/episodes/<dataset> --out models/act/<run>
# detach: Ctrl+B then D; reattach: tmux attach -t act
```

Defaults: 50 000 steps, batch 32, chunk 30, lr 1e-5, seed 1000, a checkpoint every 5 000
steps. `--steps`, `--batch-size` and `--device` override them. `--out` must not exist yet.
When it finishes, `models/act/<run>/run.json` is written.

## 9. Copy the checkpoint back

From the laptop:

```bash
rsync -avz --progress ubuntu@<IP>:Dex-ET2/models/act/<run>/ models/act/<run>/
```

Check `models/act/<run>/run.json` and `models/act/<run>/checkpoints/last/` arrived before you
go on.

## 10. Delete the instance

```bash
verda vm delete <instance-id>      # permanent; asks for confirmation
verda vm list                      # must not show act-train
```

Then open the dashboard and check no volumes are left over. Whether the OS volume is
deleted with the instance is *(unconfirmed)*, so delete it there if it is still listed.
Record the instance type, wall time and cost in `docs/training/results.md`.
