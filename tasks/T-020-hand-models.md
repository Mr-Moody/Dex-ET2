---
id: T-020
title: Hand models and per-hand config
status: todo
owner: null
branch: null
depends_on: [T-003]
plan: docs/superpowers/plans/2026-11-13-m0-m1-glove-to-sim.md#task-t-020-hand-models
claimed_at: null
---

Acceptance (from the plan):

- [ ] `uv run python scripts/fetch_models.py` creates files under third_party/
- [ ] test_hand_loads_in_mujoco[inspire|leap]
- [ ] test_flexmap_actuators_exist
- [ ] test_flexmap_joints_are_openxr
- [ ] CI step `uv run python scripts/fetch_models.py` appended to ci.yml
