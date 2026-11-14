---
id: T-200
title: Health model and monitor with off-thread reconnect
status: claimed
owner: claude-code
branch: feat/health-monitor
depends_on: []
plan: docs/superpowers/plans/2026-11-14-one-command-booth-health.md
claimed_at: '2026-11-14T11:04:09+00:00'
---

Implements Task 1 of the plan.

- [ ] `Level`, `Health`, `HealthCheck`, `health_to_dict`
- [ ] `HealthMonitor` (2 Hz poll, reconnect worker, busy flag)
- [ ] Full verify passes

Spec: `docs/superpowers/specs/2026-11-14-one-command-booth-health-design.md`. Follow the plan's Global Constraints and touch only the files the plan lists for
this task.
