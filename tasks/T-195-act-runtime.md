---
id: T-195
title: Batched ACT inference worker and live ACT recovery
status: in_review
owner: claude-code
branch: feat/act-runtime
depends_on: [T-191]
plan: docs/superpowers/plans/2026-11-14-act-recovery-plan.md
claimed_at: '2026-11-14T18:14:22+00:00'
---

Runs one ACT checkpoint for every bay in ACT recovery in one batched call on a worker thread,
and drives recoveries from its action chunks inside the hand-torque, no-progress and timeout
checks.

- [ ] `InferenceWorker` (batching, 30 ms budget, 250 ms timeout, health)
- [ ] `load_act_policy` refuses schema mismatches; `load_act_factory` reads `gate.json`
- [ ] `ActRecovery`: 3 cm retract, settle, chunk playback, abort reasons
- [ ] `ReplayPolicy` stub clears a flap jam end to end in hybrid
- [ ] Laptop p95 latency (B=2) reported in the PR
- [ ] Full verify passes

Spec: `docs/superpowers/specs/2026-11-14-act-recovery-policy-design.md`. Follow the plan's Global
Constraints and touch only the files the plan lists for this task.
