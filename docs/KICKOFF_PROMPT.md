# Kickoff prompt: Rokoko Smartgloves II → MuJoCo teleoperation & data pipeline

> Paste everything below the line into Claude Code in an empty repo directory.
> Before you start, install Superpowers: `/plugin install superpowers@claude-plugins-official`
> (or `/plugin marketplace add obra/superpowers-marketplace` then `/plugin install superpowers@superpowers-marketplace`).

---

You are the lead agent bootstrapping a new repository. Read this whole brief before acting.

## 1. Goal

Build a teleoperation and training pipeline. A human wearing **Rokoko Smartgloves II** drives a robot hand (optionally mounted on an arm) in a **MuJoCo** simulation in real time. Each session is recorded as a **LeRobot** dataset, which is then used to train and evaluate imitation-learning policies in the same sim.

End-to-end flow:

```
Smartgloves II ──(Rokoko Device SDK: RGMP stream)──► input adapter ──► HandFrame (canonical, timestamped)
                                                                         │
                                                                         ▼
                                                       retargeting (human hand → robot joint targets)
                                                                         │
                                                                         ▼
                              MuJoCo sim (fixed-rate control loop, position actuators, offscreen cameras)
                                                                         │
                                                                         ▼
                              LeRobot recorder (observation.state, observation.images.*, action, task)
                                                                         │
                                                                         ▼
                                      lerobot train (ACT / Diffusion) ──► sim rollout eval
```

## 2. Tech stack (use this unless brainstorming finds a concrete reason not to; record any deviation in an ADR)

| Layer | Choice | Notes |
|---|---|---|
| Language | Python **3.12** (pin in `.python-version`) | The Rokoko SDK Python examples need ≥3.12 |
| Packaging | **uv**: `uv init --package`, `src/` layout, `uv.lock` committed, dependency groups `dev`, `train`, `viz` | Only `uv add`, `uv run`, `uv sync`. Never pip. |
| Glove input (primary) | **Rokoko Device SDK**: `rokoko-sdk` driver; **RGMP** binary motion stream over TCP (default `127.0.0.1:12276`); **RCSP** for commands (list devices, `SubscribeToData`). USB streaming works on Smartgloves II; start with `rokoko-sdk --auto-stream-usb` | Needs an SDK licence. Port the logic of the SDK's `examples/python/rgmp_stream.py` into a typed client. Read `$ROKOKO_SDK_HOME` and the RGMP spec at sdk.rokoko.com before writing the parser. |
| Glove input (fallback) | **Rokoko Studio → Custom Streaming**: UDP, JSON v3 (optionally LZ4-compressed → `lz4` package), default port `14043` | For when there is no SDK licence or the driver isn't available |
| Glove input (dev/CI) | `ReplaySource` (plays back recorded raw streams from `data/raw/`) and `SyntheticSource` (scripted finger curls) | Every test and CI job must run without hardware |
| Wrist / arm pose | Open question for brainstorming. Options: glove IMU orientation only, plus position from keyboard/SpaceMouse; an external tracker; a fixed-base floating hand | Keep it behind a `WristSource` interface |
| Simulation | **mujoco** (latest 3.x), `mujoco.viewer.launch_passive` for interactive use, `mujoco.Renderer` for offscreen camera frames | |
| Robot models | **MuJoCo Menagerie** (vendored as a git submodule or fetched by script, never copied by hand). Default target: one dexterous hand (LEAP or Shadow), with Allegro selectable via config; optional arm (e.g. Franka) | Target hand is set in config, never hard-coded |
| Arm IK | **mink** (MuJoCo differential IK) | Only if an arm is used |
| Hand retargeting | **dex-retargeting** (vector / position / DexPilot optimizers, pinocchio-based; URDFs from dex-urdf) behind a `Retargeter` interface, with a simple MuJoCo-native baseline (direct joint-angle mapping and calibrated scaling) as the first implementation | Map joints **by name**, never by index |
| Datasets | **lerobot** (LeRobotDataset **v3**): `LeRobotDataset.create(...)`, `add_frame()`, `save_episode()`, `finalize()`, optional `push_to_hub()` | Feature schema is defined in one place, `schema.py` |
| Training / eval | `lerobot-train` with ACT, then Diffusion Policy; eval script rolls policies out in the MuJoCo env and reports success rate | `train` dependency group; GPU optional |
| Config | dataclasses + **draccus** (matches LeRobot's CLI style), YAML presets in `configs/` | |
| Visualisation / debug | `rerun-sdk` for live streams of glove frames, joint targets and camera images; `lerobot-dataset-viz` to inspect datasets | `viz` group |
| Quality | **ruff** (lint and format), **pyright** (strict on `src/`), **pytest** (+ `pytest-cov`, `hypothesis` for parser fuzzing), **pre-commit** | |
| Commit linting | `conventional-pre-commit` hook (commit-msg stage) | |
| CI | GitHub Actions with `astral-sh/setup-uv`: lint → typecheck → test (headless; `MUJOCO_GL=egl` or `osmesa`) | Required to pass before merge |

Proposed layout:

```
src/teleop/
  input/        # rokoko_sdk.py (RGMP/RCSP), rokoko_studio.py (UDP JSON), replay.py, synthetic.py, base.py
  types.py      # HandFrame, JointTargets, timestamps (the shared contract)
  retarget/     # base.py, direct.py, dex.py, calibration.py
  sim/          # env.py, scenes/, cameras.py, loop.py (fixed-rate, monotonic clock)
  record/       # schema.py, recorder.py (LeRobot writer)
  train/        # configs, eval_rollout.py
  cli.py        # `uv run teleop {stream,calibrate,sim,record,replay,eval}`
configs/  tests/  scripts/  docs/{adr,superpowers/specs,superpowers/plans}  tasks/
```

Real-time rules: the control loop runs at a fixed rate (start at 60 Hz control and 500 Hz+ physics with substeps), uses `time.monotonic_ns()`, and handles non-monotonic or late device timestamps. The Rokoko docs say epoch timestamps can step. Input I/O runs on its own thread with a latest-value buffer, so a slow consumer never stalls the stream.

Arduino Uno Q sensor kit: out of scope for v1, but leave an `AuxSensorSource` hook so tactile or other sensors can be added later as extra `observation.*` features.

## 3. Mandatory workflow (Superpowers)

All non-trivial work follows this sequence. Don't skip steps.

1. **brainstorming** → design spec in `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`. Ask me questions before freezing a spec.
2. **writing-plans** → implementation plan in `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`, broken into small tasks with exact file paths and verification steps.
3. Turn plan tasks into checklist entries (§5).
4. **using-git-worktrees** → each claimed task runs in its own worktree and branch.
5. **subagent-driven-development** for execution, or **dispatching-parallel-agents** when tasks are independent. Use **test-driven-development** inside each task.
6. **requesting-code-review** → a fresh reviewer subagent checks spec compliance first, then code quality. Fix findings with **receiving-code-review**.
7. **verification-before-completion** → run the full test, lint and typecheck suite and show the output before claiming anything is done.
8. **finishing-a-development-branch** → open a PR into `main`.

**Parallelism:** maximise it. Once the shared contracts are merged (`types.py`, `Retargeter`/`InputSource`/`WristSource` protocols, the dataset `schema.py`), run these as parallel subagents in separate worktrees:

- **A.** Rokoko SDK RGMP/RCSP client, Studio UDP fallback, replay/synthetic sources
- **B.** MuJoCo scene: Menagerie hand loading, actuators, cameras, fixed-rate loop
- **C.** Retargeting: direct baseline, calibration, dex-retargeting adapter
- **D.** LeRobot recorder and dataset round-trip tests
- **E.** Tooling: CI, pre-commit, docs

Only serialise work that has a real dependency.

## 4. Git and contribution rules (write these into `CONTRIBUTING.md` and `CLAUDE.md`)

- `main` is protected. No direct pushes. Every change lands via a PR into `main`, squash-merged, with CI green and a review approved.
- Branch naming: `<type>/<task-id>-<kebab-summary>`, e.g. `feat/T-007-rgmp-parser`.
- Commit messages and **PR titles** follow **Conventional Commits 1.0**: `type(scope): imperative summary`.
  - Types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`, `perf`, `build`, `ci`, `style`, `revert`.
  - Scopes: `input`, `sim`, `retarget`, `record`, `train`, `cli`, `infra`, `docs`, `tasks`.
  - Breaking changes: `!` after type/scope and a `BREAKING CHANGE:` footer.
- Rebase on `origin/main` before opening a PR. No merge commits on feature branches. Keep PRs small: one checklist task per PR.
- The PR template (`.github/pull_request_template.md`) must include the task ID, links to the spec and plan, a summary, test evidence (command plus output), a checklist (tests added, docs updated, no unrelated changes) and a "Closes T-xxx" line.
- Never commit secrets, licence keys, raw recordings or datasets. Gitignore `data/`, `outputs/` and `.env`.
- Add an ADR to `docs/adr/` for any stack change or cross-cutting decision.

## 5. Task checklist and claiming system (prevents agents from working over each other)

Implement this as part of the first PR, with a small CLI at `scripts/task.py`, run as `uv run task ...`:

- **One file per task** in `tasks/T-<NNN>-<slug>.md`, with YAML frontmatter: `id`, `title`, `status` (`todo | claimed | in_review | done | blocked`), `owner`, `branch`, `depends_on`, `plan` (path to the plan section), `claimed_at`. The body holds acceptance criteria as a `- [ ]` checklist. Use one file per task, not a single shared file, so parallel edits don't conflict.
- **Atomic claim lock:** `uv run task claim T-007 --agent <name>` runs `git update-ref refs/claims/T-007 HEAD ""`. The empty old-value means the command fails if the ref already exists. Claim refs live in the shared `.git` common dir, so every worktree sees them. The claim is also pushed (`git push origin refs/claims/T-007`) so other machines see it; if the push is rejected, the claim has lost the race and must be released locally.
- After a successful claim, the agent sets `status: claimed`, `owner` and `branch` in the task file on its branch.
- Other commands: `task list [--status todo] [--ready]` (ready means all `depends_on` are `done`), `task release T-007`, `task status T-007 in_review`, `task done T-007` (run as part of the merged PR; deletes the claim ref locally and on the remote).
- **Rules for every agent:** never start work without a successful claim; one active claim per agent; never edit files owned by another claimed task without coordinating through the lead agent; a claim older than N hours with no branch activity can be reclaimed by the lead after `task release`.
- `tasks/README.md` documents the protocol. `CLAUDE.md` tells every agent to run `uv run task list --ready` first.

## 6. First milestones (each one goes through §3)

- **M0 – Scaffold (do first, sequentially):** uv project, tooling, CI, `CLAUDE.md`, `CONTRIBUTING.md`, PR template, task CLI with tests, empty module skeleton, shared contracts in `types.py`. Seed `tasks/` with the M1 tasks.
- **M1 – Glove to sim in real time (parallel A–D):** `uv run teleop sim --source synthetic` and `--source rokoko-sdk` drive a Menagerie hand in the viewer with less than 50 ms end-to-end latency, measured and logged.
- **M2 – Recording:** `uv run teleop record --task "pick up the cube" --episodes 5` writes a valid LeRobot v3 dataset that loads with `LeRobotDataset(...)` and plays back in `lerobot-dataset-viz`. Include a replay-from-dataset check that re-simulates the actions.
- **M3 – Training and eval:** an ACT policy trained on a small dataset is rolled out in sim, with the success rate reported.
- **M4 (stretch):** arm plus wrist tracking via mink, a second hand model, push to the Hub.

## 7. What to do right now

1. Verify the facts in this brief before depending on them. Check the current docs for the Rokoko SDK (RGMP stream group names, rates and the per-joint value layout for Smartgloves II), the LeRobot v3 API and the Menagerie hand options. Note any discrepancy in the spec.
2. Run **brainstorming** with me on the M0+M1 design. Open questions to raise: target hand and arm, how the wrist is tracked, control rate, calibration flow (Rokoko straight-pose calibration plus our own min/max range calibration), whether the SDK licence is available (if not, start with the Studio UDP fallback), and the simulated task(s) to record.
3. Then **writing-plans**, seed `tasks/`, execute M0, and fan out M1 to parallel subagents in worktrees.

Ask before making irreversible choices: public repo visibility, Hub pushes, licence.
