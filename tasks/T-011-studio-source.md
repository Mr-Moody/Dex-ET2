---
id: T-011
title: Studio raw capture + StudioSource parser
status: claimed
owner: lead-laptop
branch: feat/T-011-studio-parser
depends_on: [T-003]
plan: docs/superpowers/plans/2026-11-13-m0-m1-glove-to-sim.md#task-t-011-studio-source
claimed_at: '2026-11-13T22:39:13+00:00'
---

**Hardware laptop for Step 1 only; teammates' agents take the rest after fixtures exist.**

Acceptance (from the plan):

- [ ] Step 1 (hardware): captures coil_on, coil_off, lz4 (30 s each) trimmed to <=200 KB under tests/fixtures/studio/, docs/studio-json-v3.md written
- [ ] test_parses_both_hands_from_coil_on
- [ ] test_coil_off_has_no_wrist_position
- [ ] test_lz4_payload_parses_same_as_plain
- [ ] test_parser_tolerates_timestamp_jump
- [ ] test_garbage_payload_returns_empty
- [ ] test_studio_source_receives_udp
