---
id: T-021
title: Bimanual scene, lift-cube task, SimSink, fixed-rate loop
status: claimed
owner: lead-laptop
branch: feat/T-021-scene-sink-loop
depends_on: [T-020]
plan: docs/superpowers/plans/2026-11-13-m0-m1-glove-to-sim.md#task-t-021-scene-sink-loop
claimed_at: '2026-11-13T22:36:50+00:00'
---

Acceptance (from the plan):

- [ ] test_scene_has_both_hands_prefixed
- [ ] test_sink_maps_by_name_and_clips
- [ ] test_sink_ignores_unknown_actuator
- [ ] test_wrist_mocap_quat_converted
- [ ] test_substeps
- [ ] test_loop_absolute_deadlines_no_drift
- [ ] test_loop_skips_when_late
- [ ] test_lift_cube_success_when_lifted
