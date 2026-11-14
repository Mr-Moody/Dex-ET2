---
id: T-191
title: Wrist camera and LeRobot episode recording of ATC recoveries
status: todo
owner: null
branch: null
depends_on: [T-190]
plan: docs/superpowers/plans/2026-11-14-act-recovery-plan.md
claimed_at: null
---

Adds `bay{k}/wrist_cam` and records every rescue and autonomous recovery as a LeRobot v3
episode (29-value state, 12-value latch-frame action, `wrist` + `top` cameras, sidecar labels),
without ever blocking the 60 Hz loop.

- [ ] Wrist camera in four-bay and single-bay scenes; `BayCameraRenderer`
- [ ] `EpisodeSink` (observer): per-bay buffers, 30 fps, one writer thread, `close()` keeps finished episodes
- [ ] `teleop atc --record-episodes DIR` works with gloves and synthetic source
- [ ] Tests: rescue, discarded rescue, two bays, close mid-episode, non-blocking tick
- [ ] Full verify passes

Spec: `docs/superpowers/specs/2026-11-14-act-recovery-policy-design.md`. Follow the plan's Global
Constraints and touch only the files the plan lists for this task.
