---
id: T-194
title: CUDA PyTorch group, ACT training wrapper and Verda runbook
status: in_review
owner: claude-code
branch: feat/act-training-setup
depends_on: [T-190]
plan: docs/superpowers/plans/2026-11-14-act-recovery-plan.md
claimed_at: '2026-11-14T18:05:14+00:00'
---

Makes ACT trainable on a Verda GPU instance and runnable on the laptop GPU.

- [x] uv `act` group with CUDA torch (default `uv sync` and CI unchanged)
- [ ] `torch.cuda.is_available()` is True on the laptop after `uv sync --group act` (deferred: 2 GB wheel at 0.25 MB/s on the hotspot)
- [x] `scripts/train_act.py` (`ActPreset`, `build_train_args`, `run.json`, `--dry-run`)
- [x] `docs/training/verda.md` runbook, cost and teardown warning first
- [ ] 200-step local smoke train writes a checkpoint (deferred to T-198: no dataset until T-192)
- [ ] Full verify passes

Spec: `docs/superpowers/specs/2026-11-14-act-recovery-policy-design.md`. Follow the plan's Global
Constraints and touch only the files the plan lists for this task.
