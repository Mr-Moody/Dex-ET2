---
id: T-193
title: ATC cleaning, trimming and labelling in teleop process
status: in_review
owner: claude-code
branch: feat/atc-episode-qc
depends_on: [T-190]
plan: docs/superpowers/plans/2026-11-14-act-recovery-plan.md
claimed_at: '2026-11-14T17:29:42+00:00'
---

`teleop process` detects ATC datasets and drops/trims/labels recovery episodes, filters by
`--sources`, merges sessions and prints a fault x source summary.

- [x] `process/atc_qc.py` with the six drop reasons and idle trim
- [x] `write_clean` carries sidecar labels through; legacy QC tests stay green
- [x] Mixing ATC and legacy datasets exits 2
- [x] Full verify passes

Spec: `docs/superpowers/specs/2026-11-14-act-recovery-policy-design.md`. Follow the plan's Global
Constraints and touch only the files the plan lists for this task.
