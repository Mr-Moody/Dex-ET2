---
id: T-220
title: Fix the console wrist feed clipping the hand and latch (near plane)
status: in_review
owner: claude-code
branch: fix/console-wrist-near-plane
depends_on: []
plan: null
claimed_at: '2026-11-14T18:44:28+00:00'
---

The console's `bay{k}/wrist_cam` feed renders with MuJoCo's extent-relative near plane. The
warehouse scene is ~37 m across, so the near plane sits ~0.37 m out and clips the hand and
latch from the wrist view (found while landing T-191, #72). `FeedRenderer` should use the same
metric clip planes as `teleop.policy.cameras.policy_clip` for body-mounted cameras.

- [x] Wrist feeds render with a near plane of a few millimetres; world-fixed feeds are unchanged
- [x] The renderer's model copy is left with its original clip planes after each frame
- [x] Test covers both
- [x] Full verify passes
