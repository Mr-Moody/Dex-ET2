---
id: T-176
title: Fix teleop atc viewer freeze (set_texts called under viewer.lock)
status: done
owner: null
branch: null
depends_on: [T-150]
plan: docs/superpowers/plans/2026-11-14-atc-implementation-plan.md
claimed_at: null
---

`teleop atc` (viewer mode) freezes at 100% CPU seconds after start: `_run_viewer` calls
`viewer.set_texts()` while holding `viewer.lock()`. MuJoCo's `set_texts` spins until the render
thread consumes the previous text request, and the render thread needs that lock.

Acceptance:

- [ ] `set_texts` is called outside `viewer.lock()`
- [ ] HUD overlay updates are throttled so the 60 Hz loop does not block on every render frame
- [ ] Regression test with a fake viewer asserting the lock is not held during `set_texts`
