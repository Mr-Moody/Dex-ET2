---
id: T-179
title: Realism upgrades (pose estimate, touch re-localization, randomized gate)
status: in_review
owner: claude-code
branch: feat/realism-upgrades
depends_on: [T-178]
plan: docs/superpowers/plans/2026-11-14-realism-upgrades.md
claimed_at: '2026-11-14T07:51:16+00:00'
---

**Implements `docs/superpowers/specs/2026-11-14-realism-upgrades-design.md` per the plan.**

Acceptance:

- [ ] Control runs on a seeded latch-pose estimate; no control path reads the true pose
- [ ] Touch (bolt encoder or wrist springs) corrects the estimate along the push axis
- [ ] Randomized 20-rollout gate, pass at >= 18, run off the control loop at a sim-time deadline
- [ ] Console/HUD show pose sigma, touch corrections and the gate pass count; `--seed` CLI flag
- [ ] Full verify passes (ruff, pyright, pytest)
