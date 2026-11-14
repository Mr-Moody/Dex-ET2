---
id: T-174
title: Fleet ops console web front end and preview server
status: todo
owner: null
branch: null
depends_on: [T-172]
plan: docs/superpowers/plans/2026-11-14-fleet-ops-console.md#task-t-174-console-front-end
claimed_at: null
---

**Fleet ops console, front end. Edit only `src/teleop/console/web/*`, `scripts/console_preview.py`, `tests/console/test_web_assets.py`.**

Acceptance:

- [ ] CCTV 2x2 grid, side panel (queue, C(t) sparklines, gate, event log, Solana/reasoner), rescue mode with two angles, clutch and timer
- [ ] Stale-feed SIGNAL LOST, LINK DOWN reconnect, layout_id frame filtering, key forwarding
- [ ] No external URLs; works at 1920x1080, 1280x720, 1024x768 (screenshots in PR)
