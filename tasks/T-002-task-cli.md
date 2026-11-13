---
id: T-002
title: Task checklist CLI with atomic claims, seed tasks
status: done
owner: null
branch: null
depends_on: [T-001]
plan: docs/superpowers/plans/2026-11-13-m0-m1-glove-to-sim.md#task-t-002-task-cli
claimed_at: null
---

Acceptance (from the plan):

- [x] test_frontmatter_roundtrip_preserves_body
- [x] test_ready_requires_all_deps_done
- [x] test_claim_creates_ref_and_updates_file
- [x] test_second_claim_same_clone_fails
- [x] test_claim_race_two_clones
- [x] test_release_removes_local_and_remote_ref
- [x] test_done_sets_status_and_releases
- [x] `uv run task list --ready` lists only T-003
