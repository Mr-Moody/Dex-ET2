---
id: T-200
title: Interactive console camera pinning, zoom, wrist cam and temporary dummy glove streamer
status: in_review
owner: tara-kas
branch: feat/console-pinning-zoom-dummy-glove
depends_on: [T-174, T-175]
plan: docs/superpowers/plans/2026-11-14-fleet-ops-console.md
claimed_at: '2026-11-14T13:00:00+00:00'
---

Makes the fleet ops console interactive and non-hardcoded while physical Rokoko gloves are
disconnected:

- [x] Scrollable Activity history in the right rail with scroll-offset preservation
- [x] Google Meet-style focus pinning, 4-up equal grid toggle, drag-to-focus/reorder panes
- [x] Eye-in-hand `bay{k}/wrist_cam` on all bays, full-headroom CCTV framing, 3D scroll-zoom/pan
- [x] Standalone temporary Rokoko Studio JSON v3 UDP streamer + manual UI teleop override (`TODO(rokoko-gloves-swap)`)
