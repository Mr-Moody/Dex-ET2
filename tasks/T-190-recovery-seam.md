---
id: T-190
title: Recovery backend seam, policy schema and gate manifest (dmp default)
status: in_review
owner: claude-code
branch: feat/recovery-seam
depends_on: [T-140, T-150]
plan: docs/superpowers/plans/2026-11-14-act-recovery-plan.md
claimed_at: '2026-11-14T10:01:18+00:00'
---

**Foundation for the ACT recovery track. Claim first.** It makes the DMP path one of several
recovery backends behind a toggle (`recovery_backend: dmp | hybrid | act`, `M` key), with zero
behaviour change in `dmp`. It also defines the policy schema, the gate manifest, the episode
observer hook and the lazy hooks that every later task plugs into.

- [ ] DMP golden-run regression test captured on `main` first (beats + compliance event order)
- [ ] `teleop.policy.schema`: 29 state names, 12 latch-frame action names, cameras, `observe`, action <-> command, `EpisodeMeta`
- [ ] `teleop.policy.manifest`: `GateManifest` load/save, `covers()` by fingerprint prototypes (not session signature IDs)
- [ ] `teleop.atc.recovery`: `RecoveryBackend`, `choose_recovery`, `DmpRecovery` (moved, unchanged numbers), `RecoveryController`/`RecoveryFactory`/`CoordinatorObserver`
- [ ] Coordinator: backend, ACT factory, observer, auto-downgrade, hybrid ACT abort -> DMP fallback
- [ ] `M` key cycles the backend; `teleop atc` flags + lazy hooks (exit 3 when a hook module is missing)
- [ ] `dmp` mode never imports torch (subprocess test)
- [ ] Full verify passes

Spec: `docs/superpowers/specs/2026-11-14-act-recovery-policy-design.md`. Follow the plan's Global
Constraints and touch only the files the plan lists for this task.
