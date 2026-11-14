---
id: T-140
title: ATC Fleet Coordinator — C(t), FIFO queue, Bay 2 SE(3) replay, torque tube & 3-rollout gate
status: claimed
owner: claude
branch: feat/atc-fleet-coordinator
depends_on: [T-110, T-120, T-130]
plan: docs/superpowers/plans/2026-11-14-atc-implementation-plan.md#task-t-140-atc-fleet-coordinator--ct-fifo-queue-bay-2-se3-replay-additive-torque-tube-3-rollout-gate--negative-control
claimed_at: '2026-11-14T02:24:05+00:00'
---

**P0/P1 Fleet Coordinator & Validation Gate integration. Zero observation of `flap_hinge` `qpos` permitted. Edit only `src/teleop/atc/{coordinator,validator}.py` and `tests/atc/test_coordinator.py`.**

Acceptance (from the plan):

- [ ] Explicit `compute_confidence` formula `C(t)` triggering stall when `C(t) < 0.3` continuously for 300 ms (or 3.0 s watchdog), with zero pre-contact flap state observation
- [ ] Single-Active-Rescue FIFO hold queue: concurrent stalls retract 3 cm to zero-velocity safe hover and wait in `QUEUED`
- [ ] `SkillValidator`: 3 headless MuJoCo rollouts across pose perturbations (`+-30 deg`, `+-10 cm`) completing in `< 0.4 s` total on laptop CPU
- [ ] Bay 2 contact-consistent autonomy: hits `Jam_Sig_1` at `x ~ 3.5 cm`, matches Fleet Library via UNO Q within 300 ms, backs off 1 cm, warps `SE(3)` wrist DMP to Bay 2 (`35 deg`, `12 cm`), and clears autonomously inside additive torque guard `||tau_ext(s) - tau_demo(s)||_inf <= tau_margin` and `||tau_ext(s)||_inf < tau_hard_max`
- [ ] Negative Control (`x = 0 cm` pinned bolt) rejected as OOD by UNO Q classifier and escalated to human; non-blocking events emitted to `ComplianceSink` and `FleetReasoner`
- [ ] All tests in `tests/atc/test_coordinator.py` pass
