---
id: T-172
title: Console feed layout, telemetry adapter and WebSocket/HTTP server
status: claimed
owner: lead-laptop
branch: feat/console-protocol
depends_on: []
plan: docs/superpowers/plans/2026-11-14-fleet-ops-console.md#task-t-172-console-protocol-feed-layout-telemetry-adapter-and-server
claimed_at: '2026-11-14T05:39:01+00:00'
---

**Fleet ops console, protocol. Edit only `src/teleop/console/{layout,telemetry,server}.py` and their tests under `tests/console/`.**

Acceptance:

- [ ] Grid and rescue feed layouts per the plan
- [ ] Telemetry JSON per the plan's wire protocol, with C(t) history, event log, STANDBY/replayed bays, layout_id
- [ ] Server: static files with traversal rejection, latest-state on connect, bounded per-client buffering, key callback, OSError on busy port
