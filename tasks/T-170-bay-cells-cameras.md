---
id: T-170
title: Enclosed bay cells, lighting, per-bay CCTV and live rescue cameras
status: claimed
owner: lead-laptop
branch: feat/bay-cells
depends_on: []
plan: docs/superpowers/plans/2026-11-14-fleet-ops-console.md#task-t-170-bay-cells-lighting-and-cameras
claimed_at: '2026-11-14T05:19:40+00:00'
---

**Fleet ops console, scene. Edit only `src/teleop/sim/cells.py` (new), `src/teleop/sim/bays.py`, `tests/sim/test_cells.py`.**

Acceptance:

- [ ] Bays at 4 m spacing, each in a visual-only walled cell; no CCTV camera can see another bay (mj_ray test)
- [ ] `bay{k}/cctv` on every bay; `bay{k}/top_ortho`/`side_ortho` in the live scene for Bays 1-2
- [ ] Per-cell shadowed key light, skybox, offscreen MSAA; checked by eye with rendered PNGs
- [ ] Existing `tests/sim`, `tests/atc`, `tests/retarget` still pass
