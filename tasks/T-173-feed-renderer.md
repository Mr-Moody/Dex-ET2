---
id: T-173
title: Offscreen CCTV feed renderer and render worker thread
status: todo
owner: null
branch: null
depends_on: [T-170, T-171, T-172]
plan: docs/superpowers/plans/2026-11-14-fleet-ops-console.md#task-t-173-offscreen-feed-renderer-and-render-worker
claimed_at: null
---

**Fleet ops console, rendering. Edit only `src/teleop/console/{snapshot,render}.py`, `tests/console/test_render.py`, and `pyproject.toml`/`uv.lock` (`uv add pillow` only).**

Acceptance:

- [ ] Render thread owns its own MjData + GL context; physics never waits on rendering
- [ ] Grid and rescue feeds render to JPEG; Bays 3-4 animated by the background player
- [ ] Worker survives render errors and keeps telemetry flowing
- [ ] Measured mean frame time <= 33 ms in grid and rescue layouts on the laptop (numbers in PR)
