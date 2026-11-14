---
id: T-193
title: ATC cleaning, trimming and labelling in teleop process
status: todo
owner: null
branch: null
depends_on: [T-190]
plan: docs/superpowers/plans/2026-11-14-act-recovery-plan.md
claimed_at: null
---

`teleop process` detects ATC datasets and drops/trims/labels recovery episodes, filters by
`--sources`, merges sessions and prints a fault x source summary.

- [ ] `process/atc_qc.py` with the six drop reasons and idle trim
- [ ] `write_clean` carries sidecar labels through; legacy QC tests stay green
- [ ] Mixing ATC and legacy datasets exits 2
- [ ] Full verify passes

Spec: `docs/superpowers/specs/2026-11-14-act-recovery-policy-design.md`. Follow the plan's Global
Constraints and touch only the files the plan lists for this task.
