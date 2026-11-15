---
id: T-229
title: Run scripted episode generation and ACT training on Modal
status: todo
owner: null
branch: null
depends_on: []
plan: null
claimed_at: null
---

A Modal alternative to the Verda runbook: generate scripted rescue episodes on CPU containers,
clean them with `teleop process` and train ACT on a GPU, all against one Modal Volume.
New files only (`scripts/modal_act.py`, `docs/training/modal.md`); `scripts/train_act.py`
(T-194) and `scripts/generate_episodes.py` stay unchanged.

- [ ] `modal run scripts/modal_act.py::generate` runs one seed per container and merges the seeds with `teleop process`
- [ ] `modal run scripts/modal_act.py::train` wraps `scripts/train_act.py` on a GPU and stamps `run.json` with the local git commit
- [ ] Headless MuJoCo rendering works in the container (offscreen GL)
- [ ] Runbook `docs/training/modal.md` (upload, generate, train, download, cost)
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`
