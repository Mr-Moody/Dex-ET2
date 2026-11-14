---
id: T-175
title: Run teleop atc through the fleet ops console by default
status: done
owner: null
branch: null
depends_on: [T-173, T-174]
plan: docs/superpowers/plans/2026-11-14-fleet-ops-console.md#task-t-175-wire-the-console-into-teleop-atc
claimed_at: null
---

**Fleet ops console, wiring. Edit only `src/teleop/console/run.py`, `src/teleop/commands/atc.py`, `configs/atc.yaml`, `tests/console/test_run.py`.**

Acceptance:

- [ ] `teleop atc` opens the console by default; `--display viewer` keeps the passive viewer; `--console-port`, `--console-fps`, `--no-browser`
- [ ] `--headless`/`--script` headless unchanged and never import the renderer
- [ ] Busy port exits 1 with a clear message
- [ ] End-to-end `teleop atc --script` plays all 6 beats in the console at 1080p without a hitch
