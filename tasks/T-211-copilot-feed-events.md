---
id: T-211
title: Copilot console feed client, frame store and event detector
status: done
owner: null
branch: null
depends_on: [T-210]
plan: docs/superpowers/plans/2026-11-14-operator-copilot-plan.md#t-211-console-feed-client-frame-store-and-event-detector
claimed_at: null
---

WebSocket client (`FeedClient`) for the console telemetry + binary JPEG stream (plus `"copilot_audit"` status publishing for standalone `python -m teleop.copilot run`, never `"key"` control messages), per-bay `FrameStore` (`<= 512 px` JPEG downscaling and rescue keyframe ring buffer), and telemetry-diff `EventDetector` with recorded stage fixtures.

Owned files: `src/teleop/copilot/{feed,frames,events}.py`, `tests/copilot/{test_feed,test_events}.py`, `tests/copilot/fixtures/*`.
See the plan section for the full checklist. Never sends `"key"` or `.on_key` robot control commands.
