---
id: T-120
title: Relative-clutch pose indexing, force bleed-off on release & split cameras
status: done
owner: null
branch: null
depends_on: [T-100, T-110]
plan: docs/superpowers/plans/2026-11-14-atc-implementation-plan.md#task-t-120-relative-clutch-pose-indexing-force-bleed-off-on-release--split-cameras
claimed_at: null
---

**P0 Ergonomic Teleop Rescue track. Edit only `src/teleop/retarget/clutch.py` and `tests/retarget/test_clutch.py`.**

Acceptance (from the plan):

- [ ] `RelativeClutchIndexer`: zero pose jump on clutch press; incremental desk-frame translation and orientation while held
- [ ] `bleed_off_targets`: clutch release immediately sets commanded joint velocity to zero and `q_target = q_actual` across 7 Panda + 16 LEAP actuators
- [ ] `RescueTrajectoryBuffer`: records latch-local `SE(3)` wrist + 16-DoF LEAP joints + `tau_ext_inf`; discards incomplete buffer if bolt is not cleared on release
- [ ] Split perspective + orthographic top/side rescue camera helper configured
- [ ] All tests in `tests/retarget/test_clutch.py` pass
