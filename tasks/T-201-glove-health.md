---
id: T-201
title: Studio glove liveness and glove health check
status: in_review
owner: claude-code
branch: feat/glove-health
depends_on: [T-200]
plan: docs/superpowers/plans/2026-11-14-one-command-booth-health.md
claimed_at: '2026-11-14T11:16:44+00:00'
---

Implements Task 2 of the plan.

- [ ] `StudioSource.liveness()` and `tolerate_bind_error`
- [ ] `GloveCheck` levels and reconnect
- [ ] Full verify passes

Spec: `docs/superpowers/specs/2026-11-14-one-command-booth-health-design.md`. Follow the plan's Global Constraints and touch only the files the plan lists for
this task.
