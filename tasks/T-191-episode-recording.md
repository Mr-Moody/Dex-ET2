---
id: T-191
title: Wrist camera and LeRobot episode recording of ATC recoveries
status: claimed
owner: claude-code
branch: feat/episode-recording
depends_on: [T-190]
plan: docs/superpowers/plans/2026-11-14-act-recovery-plan.md
claimed_at: '2026-11-14T17:05:56+00:00'
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

**Per-episode source label (booth health, T-207).** `teleop atc` now defaults to
`source: studio`, and its operator is a `SelectingGlove` (`teleop.atc.glove_select`) that picks
the real glove or the scripted rescue once per rescue, at the clutch engage edge (gloves not
healthy then -> scripted). A startup-wide `"glove"` label would therefore save scripted fallback
rescues as human glove episodes, which must never reach ACT training. Requirement
(spec `docs/superpowers/specs/2026-11-14-one-command-booth-health-design.md`, "Episode
provenance"): `open_episode_sink(root, controller, source)` must also accept a callable
`source: str | Callable[[], str]`, evaluated when each episode starts, and record that label on the
episode. `teleop atc` should then pass `lambda: selecting.rescue_input or "glove"` when the
operator is a `SelectingGlove` (see `TODO(T-191)` in `src/teleop/commands/atc.py`).
