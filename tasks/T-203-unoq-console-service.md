---
id: T-203
title: Edge service serves the MCU console and bay states; App Lab auto-run app
status: done
owner: null
branch: null
depends_on: []
plan: docs/superpowers/plans/2026-11-14-one-command-booth-health.md
claimed_at: null
---

Implements Task 4 of the plan.

- [ ] `attach_console`, `set_bay_state` op, heartbeat age
- [ ] `teleop unoq --serve` attaches the console
- [ ] `firmware/unoq_app/` + README (on-board check is hardware)
- [ ] Full verify passes

Spec: `docs/superpowers/specs/2026-11-14-one-command-booth-health-design.md`. Follow the plan's Global Constraints and touch only the files the plan lists for
this task.
