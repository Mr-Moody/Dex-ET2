---
id: T-132
title: Operator console hardware doc (sensor suite, pin map, wiring diagram)
status: done
owner: null
branch: null
depends_on: [T-100]
plan: null
claimed_at: null
---

**Docs only. Create `docs/hardware/console/` and nothing else outside this task file.**

Records the hardware that the single-plate prototype console actually uses, so `T-131`
(firmware + bridge) and `T-150` (HUD) build against the real pins.

- [ ] Sensor suite: which joy-it SEN-Kit02 modules, how many, what each one does
- [ ] Pin map from UNO Q header pins to modules, including power rails
- [ ] Wiring diagram that renders on GitHub, plus a standalone schematic page
- [ ] Wiring rules (3V3 only on A0/A1, LED series resistors, button idle-level detection)
- [ ] Differences from the `T-131` acceptance list (joystick instead of four bay buttons)
