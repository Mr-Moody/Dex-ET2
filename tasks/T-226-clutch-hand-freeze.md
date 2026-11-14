---
id: T-226
title: Keep the clutch-held hand moving instead of freezing under fast or pressing glove motion
status: claimed
owner: clutch-hand-freeze
branch: fix/clutch-hand-freeze
depends_on: []
plan: null
claimed_at: '2026-11-14T22:15:05+00:00'
---

Reported: with the clutch held the hand follows the glove for under a second, then freezes.
The relative-clutch target was fed to the IK unbounded, so fast glove motion or pressing down
onto the latch saturated a Panda joint, tripped the arm torque guard and locked the clutch out
until release.

- [ ] Headless regression tests: a fast free-space glove sweep and a press into the latch keep
  the clutch engaged and the arm torque ratio under the guard
- [ ] Clutch target leads the measured wrist by a bounded amount, less under load, and yields
  before the guard; the guard itself and the scripted rescue are unchanged
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`
