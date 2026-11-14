---
id: T-219
title: Latches riding the conveyor wear the printed-rig skin, not grey placeholder boxes
status: claimed
owner: claude-code
branch: fix/conveyor-latch-skins
depends_on: []
plan: docs/superpowers/specs/2026-11-14-fleet-ops-console-design.md
claimed_at: '2026-11-14T18:35:58+00:00'
---

Packs on the belt carry a grey 3-box placeholder latch; only the latch under the arm has the
printed-rig meshes (`assets/latch`), so the skin appears to spawn when a pack reaches the arm.

- [x] Belt packs carry the same visual meshes and colours as the bay latch (visual only, no physics change)
- [x] Falls back to the boxes when the mesh files are missing
- [x] Test in `tests/sim/test_cells.py`
