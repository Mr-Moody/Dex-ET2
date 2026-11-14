---
id: T-150
title: teleop atc command, 4-bay HUD, audio prompts & 2.5-min stage rehearsal controls
status: done
owner: lead-laptop
branch: feat/atc-stage-hud
depends_on: [T-140, T-131]
plan: docs/superpowers/plans/2026-11-14-atc-implementation-plan.md#task-t-150-teleop-atc-command-4-bay-hud-audio-prompts--25-min-stage-rehearsal-controls
claimed_at: '2026-11-14T03:28:07+00:00'
---

**P1 Stage Demo App & HUD. Edit only `configs/atc.yaml`, `src/teleop/atc/{hud,stage}.py`, `src/teleop/commands/atc.py`, and `tests/atc/test_stage_hud.py`.**

Acceptance (from the plan):

- [ ] `uv run teleop atc` launches the 4-bay supervisor (`--source studio|synthetic|replay`, `--unoq-host`, `--headless --steps`, `--hotspot`)
- [ ] Large-font HUD overlay displays 4-bay status, live `C(t)`, FIFO queue order, active clutch/rescue timer, UNO Q fit + 3-rollout gate stats, and latest Solana/Nemotron status lines
- [ ] Keyboard stage controls (`1..4` bay select, `SPACE` software clutch, `F` Bay 1 fault latch, `G` Bay 2 35deg/12cm flap fault, `P` Negative Control `x=0` pinned bolt, `R` reset) mirror hardware console for full rehearsal
- [ ] Non-blocking audio status cues (disabled automatically in `--headless`)
- [ ] `tests/atc/test_stage_hud.py` (including scripted 6-beat headless run) passes
