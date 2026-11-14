---
id: T-197
title: Recovery backend on the console and HUD, README section
status: todo
owner: null
branch: null
depends_on: [T-196]
plan: docs/superpowers/plans/2026-11-14-act-recovery-plan.md
claimed_at: null
---

Makes the backend, ACT health and the `M` toggle visible to the operator, and documents the
collect -> train -> gate -> run loop and the fallback.

- [ ] Telemetry `recovery` block and per-bay recovery tag
- [ ] Console chip, `M` key-bar entry, downgrade toasts; viewer HUD line
- [ ] README "Recovery policy (ACT)" section
- [ ] Full verify passes

Spec: `docs/superpowers/specs/2026-11-14-act-recovery-policy-design.md`. Follow the plan's Global
Constraints and touch only the files the plan lists for this task.
