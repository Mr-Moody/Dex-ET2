---
id: T-198
title: First Verda ACT training run and release
status: todo
owner: null
branch: null
depends_on: [T-192, T-193, T-194, T-196, T-197]
plan: docs/superpowers/plans/2026-11-14-act-recovery-plan.md
claimed_at: null
---

Operational. Run by whoever holds the Verda credits, on the hardware laptop. Generate and clean
data, train on Verda via the runbook, gate with `teleop eval`, check live, and flip
`configs/atc.yaml` to `hybrid` only if the gate passes.

- [ ] >= 100 clean episodes per fault
- [ ] Verda run done and the instance deleted (type, wall time, cost noted)
- [ ] `flap` passes >= 9/10; live 30 fps and p95 <= 30 ms with two bays
- [ ] `docs/training/results.md` written; default flipped only on pass

Spec: `docs/superpowers/specs/2026-11-14-act-recovery-policy-design.md`. Follow the plan's Global
Constraints and touch only the files the plan lists for this task.
