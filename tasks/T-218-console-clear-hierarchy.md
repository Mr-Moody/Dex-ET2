---
id: T-218
title: Fleet ops console UI/UX pass (clear information hierarchy, projector legibility)
status: claimed
owner: claude-code
branch: feat/console-clear-hierarchy
depends_on: []
plan: docs/superpowers/specs/2026-11-14-fleet-ops-console-design.md
claimed_at: '2026-11-14T17:21:50+00:00'
---

UI/UX pass only on `src/teleop/console/web/{index.html, console.css, console.js, health.js}`.
No change to ATC logic, the WebSocket protocol or telemetry fields.

- [x] Three-tier hierarchy: banner, feeds, rescue, link-down, belt/E-stop and queue dominant; gate, audit trail, C(t) sparklines, key bar and health quieter
- [x] Dummy glove panel moves to a collapsible dev drawer; redundant header text, activity log, rail footer and scanlines demoted or removed
- [x] One status colour system, minimum text size for projector legibility
- [x] Verified at 1920x1080, 1280x720 and 1024x768 in every preview state
