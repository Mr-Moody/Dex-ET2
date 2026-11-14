---
id: T-221
title: Stop the RGMP fuzz test flaking under load and the keypair test failing on Windows
status: in_review
owner: claude-code
branch: test/stabilise-fuzz-and-keypair
depends_on: []
plan: null
claimed_at: '2026-11-14T18:58:40+00:00'
---

Both were reported as pre-existing failures in #72.

- [x] `test_parse_frames_fuzz_never_raises` does not fail Hypothesis's too-slow health check or
  deadline when the machine is busy
- [x] `test_keypair_file_roundtrip_is_owner_only_and_not_overwritten` checks owner-only mode
  bits only where they mean something (not on Windows)
- [x] Full verify passes
