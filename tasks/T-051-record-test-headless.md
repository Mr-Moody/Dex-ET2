---
id: T-051
title: Run the record integration test headless
status: done
owner: null
branch: null
depends_on: [T-050]
plan: null
claimed_at: null
---

CI crashes in `test_record_two_short_episodes`: it calls `record.main` without `--headless`,
so `TeleopApp.session()` opens the passive viewer, GLFW fails on the display-less runner and
MuJoCo `exit(1)`s the pytest process.

- [ ] Pass `--headless` in the test argv
- [ ] CI green on the PR
