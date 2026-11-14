---
id: T-216
title: UNO Q console controls (joystick calibration, per-robot E-stop, clutch-gated glove)
status: claimed
owner: cursor-agent
branch: feat/unoq-console-controls
depends_on: []
plan: docs/hardware/console/README.md
claimed_at: '2026-11-14T15:15:31+00:00'
---

Make the physical console (`docs/hardware/console/wiring.html`) drive the web fleet console.

- [x] Joystick bay map is runtime-configurable (`CMD:JOY_MAP`) and calibrated on the console
      itself (`CMD:JOY_CAL` or a 2 s stick press); the host persists it and restores it on MCU boot
- [x] `teleop unoq --calibrate-joystick` guides the calibration over the edge service
- [x] E-stop latches per robot: a press cuts power to the selected bay (brakes on, belt stopped,
      glove ignored), a second press on that bay restores it; LED solid red while cut
- [x] Power state reaches the coordinator, telemetry and the web console bay cards
- [x] Buttons auto-detect their idle level at boot (wiring rule 3)
- [x] Clutch must be held on the selected, powered bay for the glove to drive it
- [x] Firmware compiles for `arduino:zephyr:unoq` and both sketch copies stay in sync
- [x] Full verify passes
