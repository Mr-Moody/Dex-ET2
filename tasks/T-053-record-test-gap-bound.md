---
id: T-053
title: Fix flaky max_gap_s bound in the record integration test
status: claimed
owner: lead-laptop
branch: fix/record-test-gap-bound
depends_on: [T-051]
plan: null
claimed_at: '2026-11-14T00:23:28+00:00'
---

`test_record_two_short_episodes` asserts `max_gap_s < 1.0` and fails intermittently on CI
(1.11 s, 1.16 s): a single slow tick stalls the loop past the 1 s cap.

- [ ] Find what stalls the tick
- [ ] Make the test (or the recorder) robust to it
