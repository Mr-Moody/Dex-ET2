---
id: T-131
title: UNO Q STM32U585 1 kHz console, fault rig & 8x13 LED matrix bridge
status: done
owner: null
branch: null
depends_on: [T-100]
plan: docs/superpowers/plans/2026-11-14-atc-implementation-plan.md#task-t-131-uno-q-stm32u585-1-khz-console-fault-rig--813-led-matrix-bridge
claimed_at: null
---

**P1 Physical Console & Judge Fault Rig track. Edit only `firmware/unoq_console/unoq_console.ino`, `src/teleop/unoq/mcu_bridge.py`, and `tests/unoq/test_mcu_bridge.py`.**

Acceptance (from the plan):

- [ ] `firmware/unoq_console/unoq_console.ino`: 1 kHz `INPUT_PULLUP` polling for Bay 1-4 select buttons, Hold-to-Control Dead-Man's Clutch, E-Stop, and one-way edge-triggered Judge Fault Rig + 8x13 LED Matrix 4-bay status animations
- [ ] Physical Judge Fault Rig rising edge latches once (`FAULT_RIG:LATCHED`) and ignores subsequent toggles until `CMD:RESET_RIG`
- [ ] `McuConsoleBridge` parses MCU events and formats per-bay LED state updates over an injectable transport
- [ ] All unit tests in `tests/unoq/test_mcu_bridge.py` pass without hardware
