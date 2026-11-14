---
id: T-204
title: Hot-swappable UNO Q client and control panel health check
status: in_review
owner: claude-code
branch: feat/unoq-hot-swap
depends_on: [T-200, T-203]
plan: docs/superpowers/plans/2026-11-14-one-command-booth-health.md
claimed_at: '2026-11-14T11:27:29+00:00'
---

Implements Task 5 of the plan.

- [ ] `SwitchableUnoQ` (skill replay, failover)
- [ ] `PanelCheck` (proactive ping, auto retry, serial bridge)
- [ ] Poller forwards bay states; `StageController.replace_hardware`
- [ ] Full verify passes

Spec: `docs/superpowers/specs/2026-11-14-one-command-booth-health-design.md`. Follow the plan's Global Constraints and touch only the files the plan lists for
this task.
