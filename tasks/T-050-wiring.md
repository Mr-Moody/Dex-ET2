---
id: T-050
title: teleop sim / calibrate / record wiring, latency, live check
status: claimed
owner: lead-laptop
branch: feat/T-050-teleop-sim
depends_on: [T-011, T-012, T-021, T-030, T-031, T-040]
plan: docs/superpowers/plans/2026-11-13-m0-m1-glove-to-sim.md#task-t-050-wiring
claimed_at: '2026-11-13T22:54:30+00:00'
---

**Step 6 is hardware laptop only.**

Acceptance (from the plan):

- [ ] test_sim_synthetic_moves_actuators
- [ ] test_sim_survives_source_dropout
- [ ] test_record_two_short_episodes (lerobot)
- [ ] Step 5: `uv run teleop sim --source synthetic` opens the viewer and both hands open and close
- [ ] Step 6 (hardware): both hands follow the fingers with Studio, wrist follows Coil Pro or IMU, logged p95 latency < 50 ms (else open a perf task)
