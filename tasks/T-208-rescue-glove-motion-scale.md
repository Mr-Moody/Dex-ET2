---
id: T-208
title: Scale down glove motion during an operator rescue
status: todo
owner: null
branch: null
depends_on: [T-205]
plan: null
claimed_at: null
---

Live glove rescues lock out after about a second: the relative clutch maps hand motion 1:1, a
natural 12-25 cm reach drives the Inspire hand into the latch, the arm torque guard trips
(ratio >= 0.8) and `breach_lock` holds until the clutch is released. Replaying
`data/raw/coil_on_2.jsonl` through a forced Bay 1 rescue locks at t = 1.5 s.

- [ ] `RelativeClutchIndexer` scales the rotation delta as well as the translation, per update
- [ ] `OperatorInput` carries the motion scale; the scripted rescue keeps 1.0
- [ ] `teleop atc` glove input scaled by `rescue_pos_scale` / `rescue_rot_scale`
      (`configs/atc.yaml`, `--rescue-pos-scale`, `--rescue-rot-scale`)
- [ ] Tests; README flags; full verify passes

Files: `src/teleop/retarget/clutch.py`, `src/teleop/atc/coordinator.py`,
`src/teleop/commands/atc.py`, `configs/atc.yaml`, `README.md`, tests for these.
