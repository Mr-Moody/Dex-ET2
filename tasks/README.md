# Tasks

One markdown file per task: `tasks/T-NNN-<slug>.md`, YAML frontmatter (`id, title, status, owner, branch, depends_on, plan, claimed_at`) plus a checklist body. Status is one of `todo | claimed | in_review | done | blocked`.

## Protocol

Run from any worktree (`uv run task ...`); the repo root is found with `git rev-parse --show-toplevel`.

1. `uv run task list --ready` shows tasks that are `todo` with every dependency `done`.
2. `uv run task claim T-0NN --agent <name>` claims atomically. It fetches `refs/claims/*`, creates a unique claim commit, creates `refs/claims/T-0NN` locally (fails with "already claimed by ..." if it exists), then pushes it to `origin`. If the push is rejected you lost the race: the local ref is removed and the command exits 1. Pick another task.
3. After a successful claim the CLI sets `status: claimed`, `owner`, `branch` and `claimed_at` in the task file. Commit that change on your task branch.
4. `uv run task status T-0NN in_review` when the PR is open.
5. `uv run task done T-0NN` after merge: sets `done` and deletes the claim ref (local and `origin`). Commit the status change.
6. `uv run task release T-0NN` abandons a claim: deletes the ref and resets the task to `todo`.

The claim ref on `origin` is the source of truth for who owns a task; the file status is the human-readable copy.

## Stale claims

The lead may reclaim (release) a claim older than 6 hours with no activity on its branch. `uv run task list` shows each claim's owner and age.

## Hardware

Tasks marked hardware-laptop-only (T-010, T-011 Step 1, T-050 Step 6) go to the hardware laptop; teammates take the others first.

## Active Track (2026-11-14 Pivot): Air Traffic Control (`T-100`–`T-160`)

See `PROJECT_BRIEF.md`, `docs/adr/0002-atc-fleet-rescue-pivot.md`, and `docs/superpowers/plans/2026-11-14-atc-implementation-plan.md`.

- **Foundation (claim first):** `T-100` (`depends_on: [T-050]`) — shared ATC contracts, Franka Panda model fetch, 16-DoF LEAP YAML update, CLI subcommands (`atc`, `unoq`), and Solana / reasoner placeholders. Also marks any unstarted legacy `T-060`/`T-070` booth tasks `blocked` once `main` settles.
- **Parallel P0/P1 tracks unlocked by `T-100`:**
  - `T-110` -> `T-120`: 3-box sprung slide-bolt MJCF, 4-bay Franka Panda + LEAP scene, DLS differential IK, `mj_inverse` torques, relative-clutch pose indexing, and split cameras.
  - `T-130`: Arduino UNO Q QRB2210 Python edge service + automatic localhost mock (`1-Euro` filter, Mahalanobis stall-signature/OOD gate, `<20 ms` 30-kernel RBF-DMP ridge regression).
  - `T-131`: Arduino UNO Q STM32U585 1 kHz console/Judge Fault Rig firmware sketch + 8x13 LED Matrix serial/mock bridge.
  - `T-140` -> `T-150`: ATC Fleet Coordinator (`C(t)`, single-active FIFO hold queue, Bay 2 `SE(3)` replay, additive residual torque tube, 3-rollout headless validation gate, `x=0` Negative Control) and `teleop atc` live HUD + 2.5-min stage controls.
- **Reserved Teammate Tasks (unlocked by `T-100`):**
  - `T-160`: Solana compliance & alert chain (`src/teleop/compliance/solana.py`).

## Active Track (2026-11-14): ACT Recovery Policy (`T-190`–`T-198`)

See `docs/superpowers/specs/2026-11-14-act-recovery-policy-design.md` and `docs/superpowers/plans/2026-11-14-act-recovery-plan.md` (Global Constraints and the File Ownership Map live there).

- **Foundation (claim first):** `T-190`: recovery backend seam (`dmp` default, unchanged behaviour), policy schema, gate manifest, `M` toggle and `teleop atc` hooks.
- **Parallel after `T-190`:** `T-191` (wrist camera + episode recording), `T-193` (ATC QC in `teleop process`), `T-194` (CUDA `act` group, training wrapper, Verda runbook).
- **Then:** `T-192` (scripted generator) and `T-195` (ACT runtime) after `T-191`; `T-196` (offline gate, `teleop eval`) after `T-195`; `T-197` (console/HUD + README) after `T-196`.
- **Release:** `T-198` is operational. It is run by whoever holds the Verda credits on the hardware laptop.
