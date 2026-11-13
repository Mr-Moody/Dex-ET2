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
