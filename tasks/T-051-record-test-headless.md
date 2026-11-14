---
id: T-051
title: Run the record integration test headless
status: in_review
owner: lead-laptop
branch: fix/T-051-record-test-headless
depends_on: [T-050]
plan: null
claimed_at: '2026-11-13T23:49:30+00:00'
---

CI crashes in `test_record_two_short_episodes`: it calls `record.main` without `--headless`,
so `TeleopApp.session()` opens the passive viewer, GLFW fails on the display-less runner and
MuJoCo `exit(1)`s the pytest process.

- [ ] Pass `--headless` in the test argv
- [ ] CI green on the PR
