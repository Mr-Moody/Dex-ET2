# Air Traffic Control for Robot Hands

A fleet supervisor for EV-battery disassembly. Four MuJoCo bays run Franka Panda arms with LEAP
hands that unlatch a sprung slide-bolt latch. When a bay stalls, an operator rescues it with
Rokoko gloves. An Arduino UNO Q (or a local mock) learns the rescue as a skill, a headless
simulation gate checks it, and the fleet reuses it. The pipeline also has a compliance chain
and an LLM reasoning co-pilot (Kimi K3.1 Pro on Verda).

See `PROJECT_BRIEF.md` for the pitch and
`docs/superpowers/specs/2026-11-14-atc-fleet-rescue-design.md` for the design.

## Setup

You need [uv](https://docs.astral.sh/uv/). It installs Python 3.12 for you.

```bash
uv sync --all-extras --all-groups      # dependencies, dev tools and optional groups
uv run python scripts/fetch_models.py  # Panda/LEAP models into third_party/ (needed for the sim)
cp .env.example .env                   # then fill in what you need (see below)
```

Do this in every new git worktree. Worktrees do not share `third_party/` or `.env`.

### `.env`

`.env` is gitignored. Never commit it. Every service falls back to a local or offline mode when
its keys are unset, so the demo runs with an empty `.env`.

| Variable | Purpose | If unset |
|---|---|---|
| `VERDA_API_KEY` | Verda **Inference API** key (not the Cloud API client ID/secret) | Built-in deterministic reasoner text |
| `VERDA_BASE_URL` | `https://inference.verda.com/v1` | That default |
| `NEMOTRON_MODEL` | `kimi-k3-1-pro` (the variable name is historical) | That default |
| `SOLANA_RPC_URL`, `SOLANA_KEYPAIR_PATH` | Compliance chain | Local hash-chained audit log |
| `UNOQ_HOST`, `UNOQ_PORT`, `UNOQ_SERIAL_PORT` | Real UNO Q edge node | In-process loopback mock |
| `HOTSPOT_PASSWORD` | Laptop hotspot for the gloves (Windows) | Only needed with `--hotspot` |

Kimi must be enabled for your project in the Verda Console under Inference. Otherwise requests
return `subscription_required` and the reasoner quietly uses its offline text.

## Run the demo

Run everything from the repo root.

| What | Command |
|---|---|
| Scripted demo in the browser ops console (default) | `uv run teleop atc --source synthetic --script` |
| Scripted demo in the MuJoCo window | `uv run teleop atc --source synthetic --script --display viewer` |
| Drive it yourself with the keyboard | `uv run teleop atc --source synthetic` |
| Headless check (exits 0 only if all 6 beats pass) | `uv run teleop atc --source synthetic --headless --script --no-audio` |
| With the real gloves (start Rokoko Studio streaming first) | `uv run teleop atc --source studio --script` |

The console is served at <http://127.0.0.1:8765/> and opens in your browser.

The `--script` run plays six stage beats: routine autonomy, a fault and contact stall, the
operator rescue, the edge skill fit and 3-rollout sim gate, Bay 2 pose generalization, and the
negative control (refused).

### Keys

These work in the console page or the MuJoCo window.

| Key | Action |
|---|---|
| `1`–`4` | Select a bay |
| `SPACE` | Toggle the clutch (rescue) |
| `F` | Arm the Bay 1 flap jam (causes a contact stall) |
| `G` | Arm the Bay 2 flap jam |
| `P` | Arm the Bay 2 pinned bolt (out-of-distribution, refused) |
| `R` | Reset |

A typical manual run:

1. Press `F` and wait for Bay 1 to stall and queue.
2. Press `1`, then `SPACE`, to rescue it.
3. Watch the skill fit, the 3/3 gate and Bay 2 clearing itself.
4. Press `P` to see the refusal.

### Useful flags

| Flag | Effect |
|---|---|
| `--display console\|viewer` | Browser ops console (default) or the MuJoCo passive viewer |
| `--no-browser` | Print the console URL instead of opening it |
| `--console-port N`, `--console-fps N` | Console server port and feed frame rate |
| `--no-audio` | No spoken or audio cues |
| `--no-background` | Leave out background Bays 3 and 4 (lighter) |
| `--unoq-host IP` / `--serial-port PORT` | Use the real UNO Q instead of the mock |
| `--hotspot` | Keep the laptop hotspot up for the gloves |
| `--env-file PATH` | Read a different `.env` |
| `--config PATH` | Stage config (`configs/atc.yaml`) |

## Other tools

| Command | What it does |
|---|---|
| `uv run python -m teleop.reasoning.nemotron status` | Show reasoner mode, endpoint and model |
| `uv run python -m teleop.reasoning.nemotron demo` | Send three sample events to Kimi |
| `uv run python scripts/console_preview.py` | Ops console with fake telemetry, no MuJoCo |
| `uv run teleop unoq --serve` | Run the UNO Q edge service on its own |
| `uv run teleop sim` | Original two-hand lift-cube teleop scene |
| `uv run teleop calibrate` / `record` / `replay` / `process` | Glove calibration and LeRobot dataset pipeline |
| `uv run teleop hotspot {up,status,down,watch}` | Laptop hotspot control (Windows) |

Every command takes `--help`.

The reasoner `demo` is working when each result shows `(kimi-k3-1-pro, ~2000 ms)`. If it shows
`placeholder:...`, the key is missing or Kimi is not enabled.

## Troubleshooting

- **`... panda_nohand.xml missing`**: run `uv run python scripts/fetch_models.py`.
- **MuJoCo window misbehaves on Wayland**: prefix the command with `GLFW_PLATFORM=x11`.
- **Reasoner stays `offline`**: `VERDA_API_KEY` is not being read. Run from the repo root or
  pass `--env-file`.
- **`pyright` reports missing `lerobot`**: run `uv sync --all-extras --all-groups`.

## Contributing

Read `CLAUDE.md` and `CONTRIBUTING.md` first. In short:

1. Claim a task: `uv run task list --ready`, then `uv run task claim T-NNN --agent <you>`.
2. Use one worktree per task, on a `<type>/<kebab-summary>` branch.
3. Run the checks below before every PR.
4. Use Conventional Commits.
5. Never push to `main`.

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
```
