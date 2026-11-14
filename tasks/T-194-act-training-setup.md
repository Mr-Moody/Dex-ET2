---
id: T-194
title: CUDA PyTorch group, ACT training wrapper and Verda runbook
status: todo
owner: null
branch: null
depends_on: [T-190]
plan: docs/superpowers/plans/2026-11-14-act-recovery-plan.md
claimed_at: null
---

Makes ACT trainable on a Verda GPU instance and runnable on the laptop GPU.

- [ ] uv `act` group with CUDA torch (default `uv sync` and CI unchanged)
- [ ] `torch.cuda.is_available()` is True on the laptop after `uv sync --group act`
- [ ] `scripts/train_act.py` (`ActPreset`, `build_train_args`, `run.json`, `--dry-run`)
- [ ] `docs/training/verda.md` runbook, cost and teardown warning first
- [ ] 200-step local smoke train writes a checkpoint
- [ ] Full verify passes

Spec: `docs/superpowers/specs/2026-11-14-act-recovery-policy-design.md`. Follow the plan's Global
Constraints and touch only the files the plan lists for this task.
