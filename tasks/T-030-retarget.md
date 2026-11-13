---
id: T-030
title: Flexion, calibration, DirectRetargeter
status: claimed
owner: lead-laptop
branch: feat/T-030-retarget
depends_on: [T-003, T-020]
plan: docs/superpowers/plans/2026-11-13-m0-m1-glove-to-sim.md#task-t-030-retarget
claimed_at: '2026-11-13T22:37:22+00:00'
---

Acceptance (from the plan):

- [ ] test_twist_angle_pure_axis
- [ ] test_twist_ignores_swing
- [ ] test_twist_sign_and_wrap
- [ ] test_calibrate_median
- [ ] test_calibration_json_roundtrip
- [ ] test_retarget_open_maps_lo_closed_maps_hi
- [ ] test_retarget_clips_beyond_range
- [ ] test_degenerate_range_falls_back
- [ ] test_missing_joint_holds_previous
