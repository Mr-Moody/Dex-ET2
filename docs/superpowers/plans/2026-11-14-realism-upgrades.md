# Realism Upgrades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fleet control runs on a noisy, seeded latch-pose estimate that touch corrects. New
skills must pass a randomized 18/20 headless gate that runs off the control loop. The console
shows all of it.

**Architecture:**
- A new pure module, `teleop.atc.estimate`, samples and touch-corrects per-bay latch-pose
  estimates.
- `BayRig.world_latch()` / `world_from_latch()` now return the estimate. The true pose moves to
  `true_world_latch()`, so every existing control call site switches to the estimate without
  being edited.
- The coordinator detects touches (bolt encoder or wrist springs), corrects the estimate, and
  re-anchors routine and replay targets.
- `SkillValidator` runs N seeded randomized rollouts in a worker thread. The coordinator applies
  the result at a fixed sim-time deadline.

**Tech Stack:** Python 3.12, MuJoCo, numpy, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-14-realism-upgrades-design.md`. Executors read it
alongside this plan.

## Global Constraints

- Python 3.12, uv only, xyzw quaternions, joints and actuators by name, no hardware in tests
  (plan `2026-11-13-m0-m1-glove-to-sim.md` Global Constraints).
- **No ground truth in control:** no control or decision path may read the true latch pose,
  `flap_hinge`, or `data.contact`. Ground truth may appear only in labelled display or debug
  values and in tests.
- **Safety semantics are unchanged:**
  - arm torque margin 0.8;
  - a jam trips the 300 ms confidence stall, not the arm-torque guard;
  - torque tube `TAU_MARGIN_NM` 3.5 and `TAU_HARD_MAX_NM` 12.
- **Default pose sigma:** 3 mm in x and y, 1.5° in yaw. Touch sigma about 1 mm. Gate: N = 20,
  pass at ≥ 18 (minimum N 10 if timing forces it).
- **Gate randomization:**
  - pose offsets: the spec-4.4 set `PERTURBATIONS_DEG`, cycled, plus jitter;
  - flap stiffness ±20%;
  - flap `springref` ±3°;
  - hand and latch friction ±25%;
  - estimate error at replay: 1 mm along the push axis, full G1 error in y and yaw.
- **Seeding:** every random draw is seeded from the run seed (default 0). The gate RNG is seeded
  from the run seed plus the skill id.
- **Determinism:** the gate result is applied at a fixed sim-time deadline, so a seeded run is
  tick-for-tick reproducible.
- **Commits:** Conventional Commits with scope `atc` (allowed scopes are in
  `.pre-commit-config.yaml`). No `Co-Authored-By` or "Generated with" lines.
- **Before any PR:**
  `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`

## Review Focus

1. **sigma = 0 reproduces today:** with the estimator's sigma at 0 and no touch correction
   larger than 0.5 mm, every existing coordinator and stage test outcome is unchanged
   (Task 2 test).
2. **No false touches:** approach, lift and free sliding with the bolt resting at 0 never
   register a touch. A bolt left at 0.06 from the previous cycle is reset before the routine
   arms the detector (Task 3 test).
3. **Pinned bolt (negative control):** the encoder never moves. The wrist-spring touch fires,
   the correction uses `x_bolt` = 0, and the stall still escalates as OOD (Task 3 test).
4. **Gate worker failure:** an exception in the worker yields a FAIL report at the deadline. The
   bay leaves `VALIDATING` and never hangs (Task 5 test).
5. **Outlier touch:** a correction larger than 4 sigma (> 12 mm) is rejected and logged, and the
   estimate is kept, so a mis-detected touch cannot throw the arm (Task 1 test).

---

### Task 1: Latch pose estimator module

**Files:**
- Create: `src/teleop/atc/estimate.py`
- Test: `tests/atc/test_estimate.py`

**Interfaces:**
- Produces:
  - `POSE_SIGMA_XY_M = 0.003`, `POSE_SIGMA_YAW_RAD = math.radians(1.5)`,
    `TOUCH_SIGMA_M = 0.001`, `MAX_TOUCH_SIGMAS = 4.0`, `HOOK_BACK_X_M` (the hook's back face in
    the latch frame at `x_bolt` = 0: `HOOK_POS[0] - HOOK_HALF[0]`, i.e. -0.0215, imported from
    `teleop.sim.latch`), `INDEX_PAD_RADIUS_M` (index pad centre to surface, start 0.006, tuned
    in Task 3).
  - `@dataclass(frozen=True, slots=True) class LatchEstimate`:
    - `pose: LatchPose`
    - `sigma_xy_m: float`
    - `sigma_x_m: float` (push axis; equals `sigma_xy_m` until touched)
    - `sigma_yaw_rad: float`
    - `touch_corrected: bool = False`
    - `last_correction_m: float | None = None`
    - `rejected_correction_m: float | None = None`
  - `class LatchEstimator`:
    - `__init__(self, seed: int = 0, sigma_xy_m: float = POSE_SIGMA_XY_M, sigma_yaw_rad: float = POSE_SIGMA_YAW_RAD)`
    - `sample(self, true_pose: LatchPose, bay_id: int, cycle: int) -> LatchEstimate`: one
      draw per (seed, bay, cycle) from `np.random.default_rng((seed, bay_id, cycle))`, so the
      result does not depend on call order.
  - `touch_correct(est: LatchEstimate, index_tip_world: ArrayLike, x_bolt: float) -> LatchEstimate`:
    - It is a pure function.
    - The new latch origin is moved along the estimate's own +X axis so that
      `HOOK_BACK_X_M + x_bolt` lands at `index_tip + INDEX_PAD_RADIUS_M` (projected on that
      axis). y and yaw are unchanged, and `sigma_x_m` becomes `TOUCH_SIGMA_M`.
    - If `abs(shift) > MAX_TOUCH_SIGMAS * est.sigma_x_m`, it returns `est` with
      `rejected_correction_m=shift` instead.

- [ ] **Step 1: Write the failing tests**

```python
def test_sample_is_seeded_per_bay_and_cycle():
    e = LatchEstimator(seed=7)
    a, b = e.sample(CANONICAL_LATCH, 1, 3), LatchEstimator(seed=7).sample(CANONICAL_LATCH, 1, 3)
    assert a == b
    assert e.sample(CANONICAL_LATCH, 1, 4) != a and e.sample(CANONICAL_LATCH, 2, 3) != a


def test_sample_error_statistics_match_sigma():
    e = LatchEstimator(seed=0)
    errs = np.array(
        [
            np.subtract(
                e.sample(CANONICAL_LATCH, 1, k).pose.position[:2], CANONICAL_LATCH.position[:2]
            )
            for k in range(2000)
        ]
    )
    np.testing.assert_allclose(errs.std(axis=0), [0.003, 0.003], rtol=0.1)


def test_zero_sigma_returns_truth():
    est = LatchEstimator(sigma_xy_m=0.0, sigma_yaw_rad=0.0).sample(BAY2_LATCH, 2, 0)
    assert est.pose == BAY2_LATCH


def test_touch_moves_estimate_along_push_axis_only():
    est = LatchEstimate(LatchPose((0.5, 0.01, 0.02), 0.3), 0.003, 0.003, 0.026)
    # A tip exactly where the estimate expects the hook (x_bolt = 0.01) gives no shift.
    expect = local_point(est.pose, HOOK_BACK_X_M + 0.01 - INDEX_PAD_RADIUS_M, 0.0, 0.05)
    assert touch_correct(est, expect, 0.01).last_correction_m == pytest.approx(0.0, abs=1e-9)
    moved = touch_correct(est, expect + 0.004 * axis_x(est.pose), 0.01)
    assert moved.last_correction_m == pytest.approx(0.004)
    assert moved.pose.yaw_rad == est.pose.yaw_rad and moved.sigma_x_m == TOUCH_SIGMA_M
    assert moved.touch_corrected


def test_outlier_touch_is_rejected():
    est = LatchEstimate(CANONICAL_LATCH, 0.003, 0.003, 0.026)
    tip = local_point(est.pose, HOOK_BACK_X_M - INDEX_PAD_RADIUS_M + 0.013, 0.0, 0.05)
    out = touch_correct(est, tip, 0.0)
    assert out.pose == est.pose and out.rejected_correction_m == pytest.approx(0.013)
```

`local_point` and `axis_x` are small helpers in the test file: they turn latch-frame
coordinates into world coordinates.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/atc/test_estimate.py -v`. Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `estimate.py`** with the interfaces above.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/atc/test_estimate.py -v`. Expected: 5 passed.

- [ ] **Step 5: Commit**

Commit message: `feat(atc): seeded latch pose estimator with touch correction`.

---

### Task 2: Route all control through the estimate

**Files:**
- Modify: `src/teleop/atc/validator.py`. `BayRig` (about lines 186-310): the true-pose
  accessors and the estimate.
- Modify: `src/teleop/atc/coordinator.py`. `CoordinatorConfig` (about lines 140-160);
  `start_cycle` and `_control_approach`, where the estimate is resampled when the latch is
  reset.
- Modify: `src/teleop/atc/stage.py:617`. The scripted rescue anchors on the estimate (no code
  change; check it uses `world_from_latch`).
- Test: `tests/atc/test_coordinator.py` (new tests). Existing tests run with the defaults.

**Interfaces:**
- Consumes (Task 1): `LatchEstimator`, `LatchEstimate`.
- Produces:
  - `BayRig.true_world_latch() -> LatchPose` (ground truth, for display and tests only).
  - `BayRig.estimate: LatchEstimate`.
  - `BayRig.set_estimate(est: LatchEstimate) -> None`.
  - `BayRig.world_latch()` / `world_from_latch()` now return `estimate.pose` / its `Pose`.
  - `BayRig.set_world_latch(pose)` still moves the true latch body, and also sets the estimate
    to that pose (sigma 0).
  - `BayRig.index_tip_world() -> FloatArray`: forward kinematics of the
    `{prefix}right_index_tip` site, i.e. proprioception, which is allowed.
  - `CoordinatorConfig` gains:
    - `seed: int = 0`
    - `pose_sigma_xy_m: float = POSE_SIGMA_XY_M`
    - `pose_sigma_yaw_rad: float = POSE_SIGMA_YAW_RAD`
  - `AtcCoordinator.estimate(bay_id) -> LatchEstimate`.
  - `AtcCoordinator.cycle_count(bay_id) -> int`: increments in `start_cycle`.

- [ ] **Step 1: Write the failing tests**

```python
def test_control_uses_estimate_not_truth(unoq, hand_spec, validator):
    rig = Rig(unoq, hand_spec, validator)
    rig.co.start_cycle(1)
    for _ in range(300):
        rig.co.tick()
        if rig.co.in_routine(1):
            break
    est, truth = rig.co.estimate(1).pose, rig.co.rig(1).true_world_latch()
    assert est != truth  # seeded error is applied
    assert rig.co.rig(1).world_latch() == est


def test_zero_sigma_reproduces_true_frame_cycles(unoq, hand_spec, validator):
    cfg = CoordinatorConfig(pose_sigma_xy_m=0.0, pose_sigma_yaw_rad=0.0)
    co = AtcCoordinator.build(
        unoq,
        RecordingSink(),
        RecordingReasoner(),
        hand_spec=hand_spec,
        validator=validator,
        config=cfg,
    )
    for bay in (1, 2):
        co.start_cycle(bay)
    for _ in range(600):
        co.tick()
    assert co.outcome(1) == co.outcome(2) == "cleared"
```

The `Rig` helper in this file gains an optional `config` parameter.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/atc/test_coordinator.py -k "estimate_not_truth or zero_sigma" -v`.
Expected: FAIL (`AttributeError: estimate`).

- [ ] **Step 3: Implement**
  - Build a `LatchEstimator(config.seed, sigma…)` in `AtcCoordinator`.
  - In `_control_approach`, at the existing "lifted clear: reset the latch" point, call
    `rig.set_estimate(estimator.sample(rig.true_world_latch(), bay_id, cycle))`.
  - At build time, each active bay gets the cycle-0 sample.
  - The validator's own rig keeps truth = estimate until Task 4.

- [ ] **Step 4: Run the full coordinator and stage suites with the default sigma**

Run: `uv run pytest tests/atc -v`. Expected: all pass.

If a seeded error breaks an existing test before Task 3's touch correction exists, mark that
test `@pytest.mark.xfail(strict=True, reason="needs touch correction (Task 3)")`. Task 3 must
remove every such mark.

- [ ] **Step 5: Commit**

Commit message: `feat(atc): run fleet control on the estimated latch pose`.

---

### Task 3: Touch detection and re-localization

**Files:**
- Modify: `src/teleop/atc/coordinator.py`. A new `TouchDetector` next to `StallDetector`;
  `_sense` (about line 700) for the routine and the stall; `_control_auto` for replay
  re-anchoring.
- Modify: `src/teleop/atc/validator.py`.
  - `SkillReplay` gains `rebase(world_from_latch: Pose, t_s: float, blend_s: float = 0.2) -> None`,
    which blends the replay frame from old to new over `blend_s`.
  - `BayRig.wrist_push_torque() -> float`: the compliant-wrist spring torque
    (`-stiffness * q` of `wrist_comp_x/y`, read from the rig's own joints) projected on the
    estimated push axis, expressed as a torque about the flange.
- Test: `tests/atc/test_coordinator.py`

**Interfaces:**
- Consumes (Task 1): `touch_correct`. Consumes (Task 2): `BayRig.index_tip_world`,
  `BayRig.set_estimate`, `AtcCoordinator.estimate`.
- Produces:
  - `class TouchDetector`:
    - `__init__(self, encoder_m: float = 0.0005, wrist_nm: float = <tuned>, ticks: int = 2)`
    - `reset() -> None`
    - `update(x_bolt: float, wrist_push_nm: float) -> bool`: True once per arming, on the
      first tick where `x_bolt > encoder_m`, or after `ticks` consecutive ticks with
      `wrist_push_nm > wrist_nm`.
  - Coordinator events: on each accepted correction, log
    `"bay %d touched latch: pose corrected %.1f mm"` and emit a `TOUCH_CORRECTED` entry to the
    HUD and log feed only (not the compliance sink).
  - `AtcCoordinator.last_touch(bay_id) -> LatchEstimate | None`.

**Behaviour:**
- **Detector lifecycle:** armed when the routine becomes active (after the latch reset), and
  disarmed after it fires.
- **Correction:** on firing, compute `touch_correct(estimate, index_tip_world(), x_bolt)`, then
  set the new estimate.
- **Routine re-anchoring:** the routine target is recomputed from the corrected frame each tick,
  so the shift lands through the existing push-lead clamp. No extra blend is needed, but assert
  that the target jump is no more than `push_lead_limit_m` + the correction.
- **At a stall:** if the routine's touch has not fired, apply it from the stall tick's state
  before the retract starts. The detector is armed again for `AUTONOMOUS_SKILL`; on firing,
  `auto.replay.rebase(new_wfl, auto.t)`.
- **Standoff check:** the index-tip-to-hook-back-face gap at the routine start, minus the pad
  radius, must be ≥ 12 mm. Today it is about 17 mm. Add a test, and change `ROUTINE.start_x`
  only if it fails.
- **Tuning:** tune `wrist_nm` and `INDEX_PAD_RADIUS_M` so that, with sigma at 0, the
  correction stays under 1 mm on both bays.

- [ ] **Step 1: Write the failing tests**

```python
def test_touch_detector_encoder_and_wrist_paths():
    d = TouchDetector(wrist_nm=0.5)
    assert not d.update(0.0, 0.1) and not d.update(0.0004, 0.2)
    assert d.update(0.0006, 0.0)
    assert not d.update(0.01, 0.0)  # fires once per arming
    d.reset()
    assert not d.update(0.0, 0.6) and d.update(0.0, 0.6)  # two consecutive wrist ticks


def test_no_false_touch_in_approach_or_lift(rig):  # Review Focus 2
    co = rig.co
    co.start_cycle(1)
    for _ in range(80):  # still approaching, the bolt was just reset
        co.tick()
        assert co.last_touch(1) is None or co.in_routine(1)


def test_routine_touch_shrinks_estimate_error_to_mm(rig):
    co = rig.co
    co.start_cycle(1)
    err0 = _x_err(co, 1)  # |estimate - truth| along the latch axis; test-only truth read
    for _ in range(600):
        co.tick()
        if co.outcome(1):
            break
    assert co.outcome(1) == "cleared" and co.last_touch(1) is not None
    assert _x_err(co, 1) < 0.0015 and _x_err(co, 1) < err0


def test_pinned_bolt_touch_via_wrist_still_escalates(rig):  # Review Focus 3
    rig.stall_bay(1, bolt_pinned=True)
    assert rig.co.state(1) == BayState.OOD_ESCALATED
    assert rig.co.last_touch(1) is not None


def test_routine_standoff_clears_hook_at_3_sigma():
    gap = _routine_index_gap_m()  # kinematic, from tip offsets at ROUTINE.fingers
    assert gap - 3 * POSE_SIGMA_XY_M >= 0.003
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/atc/test_coordinator.py -k "touch or standoff" -v`. Expected: FAIL.

- [ ] **Step 3: Implement** `TouchDetector`, `BayRig.wrist_push_torque`, `SkillReplay.rebase`
  and the coordinator wiring above. Tune `wrist_nm` and `INDEX_PAD_RADIUS_M`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/atc -v`. Expected: all pass, and every Task 2 `xfail` mark removed.

- [ ] **Step 5: Commit**

Commit message: `feat(atc): touch re-localization from the bolt encoder and wrist springs`.

---

### Task 4: Randomized validation gate

**Files:**
- Modify: `src/teleop/atc/validator.py`. `RolloutResult`, `ValidationReport`,
  `SkillValidator.__init__`, `validate` and `_rollout`.
- Test: `tests/atc/test_coordinator.py`. Update `test_end_to_end_…` for `total == 20`.

**Interfaces:**
- Consumes (Tasks 1-3): `LatchEstimator`, `touch_correct`, `TouchDetector`,
  `SkillReplay.rebase`.
- Produces:
  - `GATE_ROLLOUTS = 20`, `GATE_PASS_MIN = 18`.
  - `@dataclass(frozen=True, slots=True) class RolloutConditions`:
    - `perturbation: tuple[float, float, float]`
    - `est_error: tuple[float, float, float]` (dx m, dy m, dyaw rad)
    - `flap_stiffness_scale: float`
    - `flap_springref_deg: float`
    - `friction_scale: float`
  - `RolloutResult` gains `conditions: RolloutConditions`.
  - `ValidationReport` gains `threshold: int` and `pass_rate: float` (property). `passed` now
    means `sum(r.passed) >= threshold`.
  - `SkillValidator.__init__(…, n_rollouts: int = GATE_ROLLOUTS, pass_min: int = GATE_PASS_MIN, seed: int = 0)`.
  - `SkillValidator.draw_conditions(skill_id: str) -> tuple[RolloutConditions, ...]`:
    deterministic, seeded with `(seed, zlib.crc32(skill_id))`. Perturbations cycle through
    `PERTURBATIONS_DEG`, plus jitter of ±1 cm and ±3°.

**Behaviour:**
- **Before each rollout:**
  - set the true latch;
  - scale the flap joint's `jnt_stiffness` and `qpos_spring` (the springref), and the
    hand/latch `geom_friction[:, 0]`, on the validator's own model;
  - restore all of them after the rollout.
- **Replay frame:** truth + `est_error`, with x drawn at 1 mm sigma and y and yaw at the G1
  sigma.
- **Mid-replay touch:** run the `TouchDetector` during replay; on firing, `rebase` the replay
  as live.
- **Pass checks:** the per-rollout checks are unchanged.

- [ ] **Step 1: Write the failing tests**

```python
def test_gate_conditions_are_seeded_and_cover_spec_offsets(validator):
    a, b = validator.draw_conditions("s/1"), validator.draw_conditions("s/1")
    assert a == b and len(a) == 20
    assert {round(c.perturbation[2] / 30) for c in a} == {-1, 0, 1}  # all three spec yaws
    assert all(0.8 <= c.flap_stiffness_scale <= 1.2 for c in a)
    assert all(abs(c.flap_springref_deg - 35.0) <= 3.0 for c in a)
    assert all(0.75 <= c.friction_scale <= 1.25 for c in a)


def test_rollout_restores_model_parameters(validator, rescued_skill):
    before = validator.model.geom_friction.copy(), validator.model.jnt_stiffness.copy()
    validator.validate(rescued_skill)
    np.testing.assert_array_equal(validator.model.geom_friction, before[0])
    np.testing.assert_array_equal(validator.model.jnt_stiffness, before[1])


def test_threshold_rule():
    r = ValidationReport(
        results=(_res(True),) * 18 + (_res(False),) * 2, elapsed_s=1.0, threshold=18
    )
    assert r.passed and r.pass_rate == pytest.approx(0.9)
    assert not dataclasses.replace(r, threshold=19).passed
```

In the e2e test, change the assertions to `report.total == 20 and report.passed` and remove the
3-perturbation list assertion. Add a module fixture `rescued_skill` that runs `_rescued_rig`
once.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/atc/test_coordinator.py -k "gate or threshold or restores or end_to_end" -v`.
Expected: FAIL.

- [ ] **Step 3: Implement** the interfaces and behaviour above.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/atc -v`. Expected: all pass.

The e2e rescue skill must reach ≥ 18/20 at the default seed. If not, tune the rescue waypoints
or the gate sigma within the spec's ranges before touching thresholds.

- [ ] **Step 5: Commit**

Commit message: `feat(atc): 20-rollout randomized skill gate (pose error, spring, friction)`.

---

### Task 5: Gate off the control loop at a fixed sim-time deadline

**Files:**
- Modify: `src/teleop/atc/coordinator.py`. `_finalize_rescue` (about line 944) splits into
  submit and apply; `tick`; `close`.
- Test: `tests/atc/test_coordinator.py`

**Interfaces:**
- Produces:
  - `CoordinatorConfig.gate_deadline_s: float = 5.0` (sim seconds from submission).
  - `AtcCoordinator.gate_pending(bay_id) -> bool`.
  - `AtcCoordinator.close() -> None`: shuts down the single-worker `ThreadPoolExecutor`.

**Behaviour:**
- **Submit:** `_finalize_rescue` fits the DMP, emits `RESCUE_LOGGED`, submits
  `validator.validate(skill)` to the executor, and records `deadline = time_s + gate_deadline_s`.
  The bay stays `VALIDATING` and holds hover.
- **Apply:** on the first tick with `time_s >= deadline`, take `future.result()` (blocking if
  not done). Then run today's post-validation code: `VALIDATION_GATE`, deploy or reject, and
  the state change.
- **Worker exception:** convert it to a failing `ValidationReport` (empty results,
  `threshold = pass_min`) and log it.

- [ ] **Step 1: Write the failing tests**

```python
def test_gate_result_applies_exactly_at_deadline(rig):
    rig.stall_bay(1, flap_jammed=True)
    run_rescue(rig, 1, max_extra_ticks=0)  # release only, don't wait for validation
    co = rig.co
    assert co.state(1) == BayState.VALIDATING and co.gate_pending(1)
    t_submit = co.time_s
    while co.state(1) == BayState.VALIDATING:
        co.tick()
    assert co.time_s - t_submit == pytest.approx(co.config.gate_deadline_s, abs=2 * CONTROL_DT)


def test_gate_worker_exception_fails_cleanly(rig, monkeypatch):  # Review Focus 4
    monkeypatch.setattr(rig.co.validator, "validate", lambda s: 1 / 0)
    rig.stall_bay(1, flap_jammed=True)
    run_rescue(rig, 1)
    assert rig.co.state(1) != BayState.VALIDATING and len(rig.co.library) == 0
```

`run_rescue` gains `max_extra_ticks` semantics, so it can return as soon as the bay is
`VALIDATING`. All existing waits on validation use `gate_deadline_s / CONTROL_DT` + margin
ticks.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/atc/test_coordinator.py -k "deadline or worker_exception" -v`.
Expected: FAIL.

- [ ] **Step 3: Implement** the submit/apply split, the executor and `close()`. Call `close()`
  from `StageController.close` and from the `teleop atc` command's shutdown path.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/atc -v`. Expected: all pass. The stage 6-beat test still passes;
widen the director's beat waits by `gate_deadline_s` where needed.

- [ ] **Step 5: Commit**

Commit message: `feat(atc): run the skill gate in a worker with a sim-time deadline`.

---

### Task 6: Console, HUD and CLI seed

**Files:**
- Modify: `src/teleop/atc/types.py`. `BaySnapshot` gains `pose_sigma_mm: float | None = None`
  and `touch_correction_mm: float | None = None`.
- Modify: `src/teleop/atc/coordinator.py`. `snapshots()` fills them.
- Modify: `src/teleop/atc/hud.py`.
  - `GateStatus` gains `conditions: str | None`.
  - `format_gate` renders `"Skill gate 19/20 PASS (randomized pose, friction, spring, jam)"`.
  - The bay line gains `"pose ±3 mm"` and, after a touch, `"touch 4.2 mm"`.
- Modify: `src/teleop/atc/stage.py`. Fill `GateStatus` from the report; add the seed parameter
  to `StageController.build`.
- Modify: `src/teleop/console/telemetry.py`.
  - `_bay_entry` adds `"pose_sigma_mm"` and `"touch_correction_mm"`.
  - The event log gains the touch line and the gate line, with the worst failing condition.
- Modify: `src/teleop/console/web/console.js`, `console.css`. Render the two bay fields and the
  gate line.
- Modify: `src/teleop/commands/atc.py`, `configs/atc.yaml`. Add `--seed` (int; config key
  `seed: 0`), passed to `CoordinatorConfig.seed` and `SkillValidator(seed=…)`.
- Modify: `PROJECT_BRIEF.md`. Add a "Presenter notes: realism" section with one justification
  line each for the sensor model, tactile localization and the randomized gate.
- Test: `tests/atc/test_stage_hud.py`, `tests/console/test_telemetry.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_format_gate_shows_randomized_pass_count():
    g = GateStatus(fit_ms=12.0, passed=19, total=20, elapsed_s=3.1, ok=True,
                   conditions="randomized pose, friction, spring, jam")
    assert "19/20 PASS (randomized pose, friction, spring, jam)" in format_gate(g)

def test_telemetry_bay_carries_pose_estimate_fields():
    snap = BaySnapshot(1, BayState.NOMINAL, 0.9, 0.0, 0.0, "ev_latch_v1",
                       pose_sigma_mm=3.0, touch_correction_mm=4.2)
    entry = TelemetryAdapter(...)._bay_entry(1, {1: snap}, {})
    assert entry["pose_sigma_mm"] == 3.0 and entry["touch_correction_mm"] == 4.2

def test_cli_seed_reaches_coordinator(monkeypatch):
    # parse `teleop atc --headless --steps 1 --seed 5` and assert the built coordinator's
    # config.seed == 5 (patch the run loop the way test_atc_command_headless_smoke does)
```

`...` is the adapter's existing constructor arguments, as used elsewhere in
`test_telemetry.py`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/atc/test_stage_hud.py tests/console/test_telemetry.py -v`.
Expected: FAIL.

- [ ] **Step 3: Implement** the fields, formatting, console rendering, the CLI flag and the
  presenter notes.

- [ ] **Step 4: Verify end to end**

- Run: `uv run pytest`. Expected: all pass, except the pre-existing Windows-only Solana
  permission test.
- Run: `uv run teleop atc --headless --script`. Expected: all 6 beats complete; the log shows a
  touch-correction line per cycle and `Skill gate N/20 PASS` with N ≥ 18.

- [ ] **Step 5: Commit**

Commit message: `feat(atc): show pose estimate, touch corrections and randomized gate on the console`.
