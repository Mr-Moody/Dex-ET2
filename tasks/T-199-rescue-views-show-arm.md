---
id: T-199
title: Show the Panda arm in the console rescue close-ups
status: done
owner: null
branch: null
depends_on: [T-173]
plan: docs/superpowers/plans/2026-11-14-atc-implementation-plan.md
claimed_at: null
---

The console's rescue close-ups (top and side ortho views) render with the Panda link meshes
hidden (`rescue_view_option`), so the hand appears to float. Render them with the full arm.

- [ ] `FeedRenderer` renders every feed with the default scene option (arm visible)
- [ ] Test that rescue feeds no longer hide the Panda visual group
