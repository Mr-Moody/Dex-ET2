---
id: T-171
title: Canned routine cycle for background Bays 3-4
status: in_review
owner: lead-laptop
branch: feat/background-cycle
depends_on: []
plan: docs/superpowers/plans/2026-11-14-fleet-ops-console.md#task-t-171-canned-background-cycle-for-bays-3-4
claimed_at: '2026-11-14T05:35:17+00:00'
---

**Fleet ops console, background bays. Edit only `src/teleop/console/{__init__,background}.py`, `src/teleop/console/data/background_cycle.npz`, `scripts/record_background_cycle.py`, `tests/console/test_background.py`.**

Acceptance:

- [ ] `record_cycle` captures one Bay 1 routine cycle (qpos by joint name + C(t)); committed `.npz` < 1 MB, C(t) min > 0.85
- [ ] `BackgroundPlayer` animates Bays 3-4 phase-offset, seamless arm wrap, skips absent bays
- [ ] Display only: coordinator untouched
