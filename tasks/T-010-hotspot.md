---
id: T-010
title: Hotspot manager + keep-alive watchdog (Windows)
status: todo
owner: null
branch: null
depends_on: [T-003]
plan: docs/superpowers/plans/2026-11-13-m0-m1-glove-to-sim.md#task-t-010-hotspot
claimed_at: null
---

**Hardware laptop only.**

Acceptance (from the plan):

- [ ] test_password_validation
- [ ] test_ensure_up_configures_and_starts
- [ ] test_ensure_up_noop_when_correct
- [ ] test_band_unsupported_raises
- [ ] test_watchdog_restarts_when_off
- [ ] test_reconfigures_on_drift
- [ ] test_disable_idle_timeout_called_once
- [ ] Step 5 (hardware, manual): SSID matches DEFAULT_SSID char by char (apostrophe); `teleop hotspot up` reconnects gloves; watchdog restores hotspot within 10 s
