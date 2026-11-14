---
id: T-110
title: 3-box sprung slide-bolt MJCF, 4-bay Panda+LEAP scene, DLS diff-IK & mj_inverse
status: in_review
owner: lead-laptop
branch: feat/latch-panda-sim
depends_on: [T-100]
plan: docs/superpowers/plans/2026-11-14-atc-implementation-plan.md#task-t-110-3-box-sprung-slide-bolt-mjcf-4-bay-pandaleap-scene-dls-diff-ik--mj_inverse
claimed_at: '2026-11-14T01:00:34+00:00'
---

**P0 Simulation & Kinematics track. Edit only `src/teleop/sim/{latch,ik,torques,bays}.py` and `tests/sim/test_latch_panda.py`.**

Acceptance (from the plan):

- [ ] `src/teleop/sim/latch.py`: 3 oversized `<geom type="box">` collision bodies (`Base Track` 14x8x2 cm, `Slide Bolt` with hook 7 cm from flap, `Sprung Stopper Flap` 4x4 cm with `springref=+35 deg`), one-way `arm_flap_fault()`, and `pinned_bolt` lock at `x = 0`
- [ ] `src/teleop/sim/ik.py`: `PandaDiffIK` solving 6-DoF wrist pose onto 7 Franka Panda joints with posture nullspace regularization and joint limits
- [ ] `src/teleop/sim/torques.py`: `ExternalTorqueEstimator` using scratch `MjData` + `mj_inverse` (`tau_ext = tau_act - tau_id`)
- [ ] `src/teleop/sim/bays.py`: `build_four_bay_scene` (Bays 1-2 active Panda+LEAP physics, Bays 3-4 lightweight kinematic background) + `build_single_bay_scene`
- [ ] All 6 unit tests in `tests/sim/test_latch_panda.py` pass headless
