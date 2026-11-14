---
id: T-199
title: Show the Panda arm in the console rescue close-ups
status: in_review
owner: ethan
branch: fix/rescue-views-show-arm
depends_on: [T-173]
plan: docs/superpowers/plans/2026-11-14-atc-implementation-plan.md
claimed_at: '2026-11-14T09:52:11+00:00'
---

The console's rescue close-ups (top and side ortho views) render with the Panda link meshes
hidden (`rescue_view_option`), so the hand appears to float. Render them with the full arm.

- [ ] `FeedRenderer` renders every feed with the default scene option (arm visible)
- [ ] Test that rescue feeds no longer hide the Panda visual group
