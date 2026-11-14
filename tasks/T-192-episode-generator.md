---
id: T-192
title: Headless scripted rescue episode generator
status: todo
owner: null
branch: null
depends_on: [T-191]
plan: docs/superpowers/plans/2026-11-14-act-recovery-plan.md
claimed_at: null
---

`teleop atc --headless --generate N --record-episodes DIR`: random faults, latch-pose
perturbations, timing jitter and pose noise on the scripted glove, until N successful episodes.

- [ ] `GeneratorConfig`, `latch_perturbation`, `run_generation`
- [ ] Seeded and reproducible; gives up after 3N attempts
- [ ] 20-episode smoke run recorded in the PR description
- [ ] Full verify passes

Spec: `docs/superpowers/specs/2026-11-14-act-recovery-policy-design.md`. Follow the plan's Global
Constraints and touch only the files the plan lists for this task.
