---
id: T-221
title: Stop the RGMP fuzz test flaking under load and the keypair test failing on Windows
status: done
owner: null
branch: null
depends_on: []
plan: null
claimed_at: null
---

Both were reported as pre-existing failures in #72.

- [x] `test_parse_frames_fuzz_never_raises` does not fail Hypothesis's too-slow health check or
  deadline when the machine is busy
- [x] `test_keypair_file_roundtrip_is_owner_only_and_not_overwritten` checks owner-only mode
  bits only where they mean something (not on Windows)
- [x] Full verify passes
