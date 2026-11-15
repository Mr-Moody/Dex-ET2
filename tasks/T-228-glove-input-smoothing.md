---
id: T-228
title: Smooth the rescue glove wrist and finger targets with 1-Euro filters
status: todo
owner: null
branch: null
depends_on: []
plan: null
claimed_at: null
---

Reported: the clutch-held hand jitters. `GloveOperator.latest()` handed raw glove wrist poses
and raw finger targets to the relative clutch; the 1-Euro filters in `teleop.unoq.filter` were
only used by the UNO Q `filter_step` op, which the ATC loop never calls.

- [ ] `GloveOperator` filters the wrist pose (`PoseOneEuroFilter`) and finger targets
  (`JointsOneEuroFilter`) once per new glove frame, timed by the device clock
- [ ] Filters restart after a stream gap and on recalibration; `glove_filter: false` turns it off
- [ ] Tests on synthetic frames: jitter is attenuated, repeated frames do not re-filter, gaps reset
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`
