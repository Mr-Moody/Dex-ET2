# EV-module latch meshes

Visual meshes that make the sim latch look like the 3D-printed EV-module retention rig: a graphite
pallet with a bolt housing, a blue battery module with terminals, a yellow T-handle slide bolt and an
orange hazard-striped safety flap. Binary STL, **metres**, +X is the bolt's unlatching direction,
+Z up.

`teleop.sim.latch.add_latch_to_spec` attaches them as visual-only geoms (`contype=conaffinity=0`,
zero mass) and makes the physics boxes transparent. All contact still happens on the boxes, so the
latch's physics, stall position (about 3.5 cm) and clear threshold (`BOLT_CLEARED_M`) do not change.

These meshes are the **sim-scale** twin (Fusion document `latch_sim`). The surfaces the hand touches
match the boxes exactly: the bolt bar is 80 x 24 x 16 mm, the hook is 16 x 28 x 35 mm, the paddle is
40 x 40 x 8 mm and the hinge is at (56.5, 0, -8) mm. The physical rig that judges flip (Fusion
document `latch`) is a smaller print-friendly version of the same design.

## Files and frames

| File | Mesh frame | Body |
|---|---|---|
| `pallet_base.stl` | latch frame (top-centre of the 140 x 80 mm track) | `latch_base` |
| `battery_module.stl` | latch frame | `latch_base` |
| `terminal_pos.stl`, `terminal_neg.stl` | latch frame | `latch_base` |
| `slide_bolt.stl` | latch frame, bolt at x = 0 | `slide_bolt` (slide joint +X) |
| `safety_flap.stl` | hinge frame, origin at `FLAP_HINGE_POS` | `stopper_flap` (hinge +Y, positive angle lifts the free end) |

The pallet reaches past the track at -X, out to -125 mm, to carry the module that the bolt's back
end locks into. Those surfaces are visual only.

## Regenerating

Export each body of `latch_sim` from Fusion as binary STL (mm). The CAD frame already is the latch
frame, so scale everything by 0.001 and also shift the flap by (-56.5, 0, +8) mm into the hinge
frame. `tests/sim/test_latch_visuals.py` checks that the bolt and flap meshes still line up with
their boxes.
