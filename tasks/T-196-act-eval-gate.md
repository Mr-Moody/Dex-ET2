---
id: T-196
title: Offline ACT gate and release manifest (teleop eval)
status: todo
owner: null
branch: null
depends_on: [T-195]
plan: docs/superpowers/plans/2026-11-14-act-recovery-plan.md
claimed_at: null
---

`teleop eval` runs ACT and the DMP through the same headless rollouts per fault type and
writes `gate.json` (passed faults + fingerprint prototypes) that the live runtime trusts.

- [ ] `SkillValidator.validate_controller` (DMP gate numbers unchanged)
- [ ] `teleop eval` table, 9/10 bar, `no data` faults, exit codes
- [ ] Full verify passes

Spec: `docs/superpowers/specs/2026-11-14-act-recovery-policy-design.md`. Follow the plan's Global
Constraints and touch only the files the plan lists for this task.
