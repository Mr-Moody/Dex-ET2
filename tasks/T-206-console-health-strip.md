---
id: T-206
title: Console hardware health strip with reconnect buttons
status: done
owner: null
branch: null
depends_on: [T-200]
plan: docs/superpowers/plans/2026-11-14-one-command-booth-health.md
claimed_at: null
---

Implements Task 7 of the plan.

- [ ] Server `on_reconnect` + validation
- [ ] `HealthPublisher`, `run_console(health=)`
- [ ] `health.js` strip, viewer HUD line
- [ ] Full verify passes

Spec: `docs/superpowers/specs/2026-11-14-one-command-booth-health-design.md`. Follow the plan's Global Constraints and touch only the files the plan lists for
this task.
