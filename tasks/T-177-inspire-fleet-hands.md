---
id: T-177
title: Swap the fleet LEAP hands for five-finger Inspire hands
status: claimed
owner: claude-code
branch: feat/inspire-fleet-hands
depends_on: []
plan: null
claimed_at: '2026-11-14T06:45:37+00:00'
---

**Fleet hands. Replaces the right LEAP hand (3 fingers + thumb) on every bay's Panda with the
Inspire RH56 (thumb + 4 fingers) and ports everything keyed on LEAP actuator names.**

Acceptance:

- [ ] Every bay mounts a right Inspire hand; actuator names come from one shared tuple
- [ ] Routine and scripted rescue finger poses re-tuned so the latch cycle and rescue still pass
- [ ] Coordinator, validator, edge DMP, clutch retargeter and HUD use the Inspire actuators
- [ ] `background_cycle.npz` re-recorded for the new joints
- [ ] Full verify passes (ruff, pyright, pytest)
