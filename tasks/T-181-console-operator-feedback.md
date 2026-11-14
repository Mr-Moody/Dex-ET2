---
id: T-181
title: Fleet ops console operator feedback (key acks, next-step banner, key bar)
status: in_review
owner: ethan
branch: feat/console-operator-feedback
depends_on: [T-174, T-175]
plan: docs/superpowers/plans/2026-11-14-atc-implementation-plan.md
claimed_at: '2026-11-14T07:17:54+00:00'
---

Browser key presses reach the stage but the page never says what happened. Make the console
explain itself.

- [ ] `StageController` records the outcome of every operator action (accepted or ignored, why)
- [ ] Telemetry carries action acks, armed faults, bolt travel and the stage beat
- [ ] Page: "what is happening / what to do next" banner, action toasts, live key bar with
      context-aware hints, per-bay armed-fault and bolt-progress badges
- [ ] Fix the refused-stall log text (it said "unseen jam")
- [ ] Tests for action outcomes and the new telemetry fields
