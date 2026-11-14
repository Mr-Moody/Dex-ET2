---
id: T-130
title: UNO Q QRB2210 Python edge service + local mock (1-Euro, Mahalanobis, 30-RBF DMP)
status: in_review
owner: lead-laptop
branch: feat/unoq-edge-dmp
depends_on: [T-100]
plan: docs/superpowers/plans/2026-11-14-atc-implementation-plan.md#task-t-130-uno-q-qrb2210-python-edge-service--local-mock-1-filter-mahalanobis-gate-30-rbf-dmp
claimed_at: '2026-11-14T01:41:30+00:00'
---

**P0 Edge Compute track. Pure NumPy + stdlib only (no `mujoco` or `torch` imports inside `src/teleop/unoq/`). Edit only `src/teleop/unoq/` (excluding `mcu_bridge.py`), `src/teleop/commands/unoq.py`, and `tests/unoq/test_edge_dmp.py`.**

Acceptance (from the plan):

- [ ] `OneEuroFilter` + `PoseOneEuroFilter` for 83.3 Hz wrist/joint streams
- [ ] `MahalanobisStallClassifier`: matches `Jam_Sig_1` (`x_bolt ~ 3.5 cm`) and rejects Negative Control pinned bolt (`x_bolt = 0.0 cm`) as OOD
- [ ] `fit_rbf_dmp` & `evaluate_rbf_dmp`: closed-form 30x30 Gaussian RBF ridge regression (`< 20 ms`), encoding latch-frame `SE(3)` wrist + 16-DoF LEAP joints + `tau_demo_profile`
- [ ] `UnoQServer` / `UnoQClient.connect_or_mock` + `uv run teleop unoq --serve` with automatic localhost fallback when UNO Q is unplugged
- [ ] All tests in `tests/unoq/test_edge_dmp.py` pass without hardware
