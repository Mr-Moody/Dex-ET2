# ACT Recovery Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save rescues as LeRobot episodes, train ACT on Verda, gate it offline, and run it live
as the "release" recovery behind the DMP "hotfix", with a one-switch fallback to today's DMP-only
behaviour.

**Architecture:** T-190 lands a recovery seam in the coordinator (`RecoveryController` /
`RecoveryFactory`, a `recovery_backend` toggle, an episode observer hook) plus the shared
policy schema and gate manifest. The coordinator and `teleop atc` call later tasks only through
those interfaces and lazy imports. After T-190, the recording, cleaning, training, runtime,
gate and console tracks run in parallel worktrees with no overlapping files.

**Tech Stack:** Python 3.12, uv, MuJoCo 4, LeRobot 0.7 (ACT policy, v3 datasets), PyTorch
(CUDA build for the `act` group), Verda CLI for remote training.

**Spec:** `docs/superpowers/specs/2026-11-14-act-recovery-policy-design.md`

## Global Constraints

- Python 3.12. uv only (`uv add`, `uv run`, `uv sync`, never `pip`). Quaternions xyzw. Joints and actuators by name. `time.monotonic_ns()` clock.
- Default `pytest` needs no hardware, GPU, display, network or Verda account. Tests that need `lerobot` use `pytest.importorskip("lerobot")`. Tests never train a real model.
- Do not edit `src/teleop/cli.py`, `scripts/fetch_models.py`, `.env.example`, `.pre-commit-config.yaml`, `src/teleop/compliance/protocol.py`, `src/teleop/atc/types.py` (T-100 shared contracts).
- Each task edits only the files listed under its **Files**. If you need a file another task owns, stop and ask in the team channel.
- `recovery_backend: dmp` (the default) must behave exactly like `main` today and must not import `torch` or `lerobot`.
- The coordinator, validator and policy inputs never read `flap_hinge`, latch fault flags or `bolt_x` to decide anything. `bolt_x` is allowed only for success labels and gate pass rules (as today). The `fault` label comes from the stage and is used only for labels and eval.
- The policy observation is the 29-value `STATE_NAMES` vector plus the `wrist` and `top` cameras. The action is the 12-value `ACTION_NAMES` vector in the latch frame. Both are defined once, in `teleop.policy.schema`.
- Dataset rate is 30 fps (every second 60 Hz tick). Images are 128 × 128 RGB unless the schema constant changes.
- Live budget: p95 inference ≤ 30 ms for a batch of 2. No action chunk within 250 ms counts as a failure. 10 consecutive over-budget calls make ACT unhealthy.
- Gate bar: at least 9 of 10 rollouts per fault type.
- Verify before every PR: `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`
- Conventional Commits with an allowed scope (`atc`, `record`, `train`, `sim`, `docs`, `tasks`, …). No `Co-Authored-By` or "Generated with" lines.

## Review Focus

1. **Backend switched with `M` while a bay is mid-recovery.** The running recovery finishes on the backend it started with, and the switch applies from the next stall. Test in Task 1.
2. **A checkpoint trained on a different schema** (other state or action size, missing camera, old `schema_version`). Loading refuses with a clear message, and `teleop atc` downgrades to `dmp` instead of crashing. Test in Task 6.
3. **Recording interrupted (Ctrl+C or window closed) mid-episode.** Finished episodes stay loadable, the partial one is discarded, and the dataset is finalised. Test in Task 2.
4. **Two bays need ACT at once.** One batched forward pass serves both. One bay aborting does not stop the other. Test in Task 6.
5. **Eval asked for a fault with no training episodes** (`--faults pin` on a flap-only dataset). The fault is reported as "no data", is left out of the manifest, and nothing crashes. Test in Task 7.

## File Ownership Map

```text
configs/atc.yaml                               # T-190 (keys), T-198 (flip default after release)
pyproject.toml  uv.lock                        # T-194
scripts/train_act.py                           # T-194
docs/training/verda.md                         # T-194
docs/training/results.md                       # T-198
README.md (new "Recovery policy" section only) # T-197
src/teleop/
  policy/__init__.py  schema.py  manifest.py   # T-190
  policy/cameras.py                            # T-191
  policy/act_runtime.py                        # T-195
  atc/recovery.py                              # T-190
  atc/coordinator.py                           # T-190
  atc/stage.py                                 # T-190 (M key); T-197 (hud_view() recovery line only)
  atc/episodes.py                              # T-191
  atc/generate.py                              # T-192
  atc/act_recovery.py                          # T-195
  atc/validator.py                             # T-196
  atc/hud.py                                   # T-197
  sim/bays.py                                  # T-191 (wrist camera)
  process/atc_qc.py                            # T-193
  process/qc.py  commands/process.py           # T-193
  commands/atc.py                              # T-190
  commands/eval.py                             # T-196
  console/telemetry.py  console/web/*          # T-197
tests/
  policy/test_schema.py  policy/test_manifest.py  atc/test_recovery.py   # T-190
  policy/test_cameras.py  atc/test_episodes.py                           # T-191
  atc/test_generate.py                                                   # T-192
  process/test_atc_qc.py                                                 # T-193
  test_train_act.py                                                      # T-194
  policy/test_act_runtime.py  atc/test_act_recovery.py                   # T-195
  atc/test_eval_gate.py                                                  # T-196
  console/test_recovery_telemetry.py                                     # T-197
```

Dependency graph: `T-190 → {T-191, T-193, T-194}`; `T-191 → {T-192, T-195}`; `T-195 → T-196`;
`T-196 → T-197`; `{T-192, T-193, T-194, T-196, T-197} → T-198`.

---

### Task 1 (T-190): Recovery seam, policy schema, gate manifest, backend toggle

**Files:**
- Create: `src/teleop/policy/__init__.py`, `src/teleop/policy/schema.py`, `src/teleop/policy/manifest.py`, `src/teleop/atc/recovery.py`
- Modify: `src/teleop/atc/coordinator.py` (move `_Move`, `_start_autonomous`, `_control_auto`, `_sense_auto` behind `DmpRecovery`; add backend, ACT factory and observer), `src/teleop/atc/stage.py` (M key), `src/teleop/commands/atc.py`, `configs/atc.yaml`
- Test: `tests/policy/test_schema.py`, `tests/policy/test_manifest.py`, `tests/atc/test_recovery.py`

**Interfaces:**
- Consumes: `BayRig` (`teleop.atc.validator`), `SkillReplay`, `torque_tube_ok`, `MahalanobisStallClassifier` (`teleop.unoq.classifier`), `compose_pose`/`relative_pose`/`so3_log`/`so3_exp` (`teleop.unoq.dmp`), `HAND_ACTUATORS`, `ROUTINE`, `StallSignature`, `SkillPrimitive`.
- Produces:
  - `teleop.policy.schema`:
    - `SCHEMA_VERSION = 1`, `FPS = 30`, `IMAGE_HW = (128, 128)`, `CAMERAS = ("wrist", "top")`
    - `STATE_NAMES: tuple[str, ...]`, 29 entries in this order: `arm/joint1..7`, `hand/<a>` for `a` in `HAND_ACTUATORS`, `wrist/x,y,z,qx,qy,qz,qw`, `tau_hand/<a>`, `tau_wrist/joint5..7`
    - `ACTION_NAMES: tuple[str, ...]`, 12 entries: `wrist/x,y,z,rx,ry,rz`, `hand/<a>`
    - `camera_name(bay_id: int, cam: str) -> str`: `wrist` → `f"bay{bay_id}/wrist_cam"`; `top` → first of `rescue_camera_names(bay_prefix(bay_id))`
    - `observe(rig: BayRig) -> FloatArray` with shape `(29,)`
    - `action_from_command(world_from_latch: Pose, wrist_world: Pose, fingers: Mapping[str, float]) -> FloatArray` with shape `(12,)`
    - `command_from_action(world_from_latch: Pose, action: ArrayLike) -> tuple[Pose, dict[str, float]]`
    - `lerobot_features(image_hw: tuple[int, int] = IMAGE_HW) -> dict[str, dict[str, Any]]`
    - `EpisodeSource = Literal["glove", "scripted", "dmp", "act"]`
    - `@dataclass(frozen=True) EpisodeMeta(source, fault: str | None, bay: int, signature_id: str | None, signature: tuple[float, float, float, float] | None, latch_pose: dict[str, float], success: bool, bolt_x_final: float, max_tau_hand_inf: float, max_arm_ratio: float, aborted_reason: str | None)` with `to_json() -> dict[str, Any]` and `from_json(d) -> EpisodeMeta`
  - `teleop.policy.manifest`:
    - `class ManifestError(ValueError)`
    - `@dataclass(frozen=True) FaultResult(fault: str, rollouts: int, passed_rollouts: int, mean_time_to_clear_s: float | None, max_tau_hand_inf: float, dmp_passed_rollouts: int | None, note: str = "")`
    - `@dataclass(frozen=True) GateManifest(schema_version: int, checkpoint_sha256: str, bar: tuple[int, int], results: dict[str, FaultResult], prototypes: dict[str, list[tuple[float, float, float, float]]], inference_p95_ms: float | None)`, with `passed_faults() -> tuple[str, ...]`, `save(path: Path) -> None` and `@classmethod load(path: Path) -> GateManifest` (raises `ManifestError` if the file is missing, unreadable or has the wrong `schema_version`)
    - `covers(manifest: GateManifest, z: StallSignature, ood_threshold: float = 3.0) -> str | None`: the fault label of the nearest passed-fault prototype within the threshold, else `None`
  - `teleop.atc.recovery`:
    - `class RecoveryBackend(StrEnum)`: `DMP = "dmp"`, `HYBRID = "hybrid"`, `ACT = "act"`
    - `next_backend(b: RecoveryBackend) -> RecoveryBackend` (dmp → hybrid → act → dmp)
    - `SmoothMove(start: Pose, end: Pose, duration_s: float)`: the moved `_Move`, unchanged
    - `@dataclass(frozen=True) BaySignals(bay_id: int, t_s: float, tau_hand_inf: float, confidence: float, fingers: Mapping[str, float])`
    - `@dataclass(frozen=True) RecoveryCommand(wrist: Pose, fingers: Mapping[str, float], v_cmd: float | None = None)`
    - `class RecoveryController(Protocol)`: `name: str`; `command(rig: BayRig, s: BaySignals) -> RecoveryCommand | None` (`None` = hold pose); `check(rig: BayRig, s: BaySignals) -> str | None` (abort reason); `close() -> None`
    - `class RecoveryFactory(Protocol)`: `name: str`; `covers(z: StallSignature) -> bool`; `healthy() -> bool`; `unhealthy_reason: str`; `create(rig: BayRig, z: StallSignature) -> RecoveryController`; `close() -> None`
    - `DmpRecovery(skill: SkillPrimitive, rig: BayRig, *, backoff_m: float, backoff_s: float, blend_s: float, hold_s: float, tau_margin: float, tau_hard_max: float, timeout_s: float)`
    - `choose_recovery(backend: RecoveryBackend, *, act_covers: bool, act_healthy: bool, dmp_available: bool, act_failed: bool = False) -> Literal["act", "dmp"] | None`
    - `class CoordinatorObserver(Protocol)`: `on_episode_start(bay_id: int, source: Literal["operator", "dmp", "act"], signature_id: str | None, signature: StallSignature | None) -> None`; `on_tick(bay_id: int, rig: BayRig, wrist_world: Pose, fingers: Mapping[str, float]) -> None`; `on_episode_end(bay_id: int, success: bool, aborted_reason: str | None) -> None`
  - `AtcCoordinator` additions: constructor and `build()` keywords `recovery_backend: RecoveryBackend = RecoveryBackend.DMP`, `act: RecoveryFactory | None = None`, `observer: CoordinatorObserver | None = None`; property `recovery_backend`; `set_recovery_backend(b: RecoveryBackend) -> None`; `set_act(f: RecoveryFactory | None) -> None`; `set_observer(o: CoordinatorObserver | None) -> None`; `recovery_name(bay_id: int) -> str | None` (`"dmp"`/`"act"` while autonomous); `backend_note: str` (last switch or downgrade message, empty at start)
  - `StageAction.CYCLE_BACKEND` bound to `ord("M")`. It calls `set_recovery_backend(next_backend(...))` and records an accepted action note `"Recovery backend: <name> (next stall)"`. `CONTROL_KEYS_HINT = "Use 1-4, SPACE, F, G, P, R or M"`
  - `teleop atc` flags and `AtcConfig` fields: `--recovery-backend {dmp,hybrid,act}` (`recovery_backend: str = "dmp"`), `--act-checkpoint PATH`, `--act-device` (default `cuda`), `--record-episodes DIR`, `--generate N`, `--seed` (default 0). Lazy hooks, imported only when their flag is set:
    - `teleop.atc.act_recovery.load_act_factory(checkpoint: Path, device: str, model: Any, data: Any) -> RecoveryFactory`
    - `teleop.atc.episodes.open_episode_sink(root: Path, controller: StageController, operator_source: Literal["glove", "scripted"]) -> EpisodeSinkHandle`, where `EpisodeSinkHandle` is a `CoordinatorObserver` with `close() -> None`
    - `teleop.atc.generate.run_generation(controller: StageController, sink: EpisodeSinkHandle, n: int, seed: int) -> int` (successful episodes)
  - If a hook module is missing, `teleop atc` prints `teleop atc: <flag> is not implemented yet (<module>)` and returns 3. If `load_act_factory` raises `ManifestError` or `OSError`, it logs a warning, sets backend `dmp` and continues.

`choose_recovery` truth table (spec section 1):

| backend | act_covers ∧ act_healthy ∧ ¬act_failed | dmp_available | result |
|---|---|---|---|
| dmp | any | True / False | `"dmp"` / `None` |
| hybrid | True | any | `"act"` |
| hybrid | False | True / False | `"dmp"` / `None` |
| act | True / False | any | `"act"` / `None` |

Coordinator flow on a stall:
1. Classify as today. `dmp_available` = not OOD and a deployed skill exists. `act_covers` = `act is not None and act.covers(sig)`.
2. If backend ≠ dmp and (`act is None` or not `act.healthy()`), downgrade: set backend dmp, set `backend_note` to `"ACT unavailable (<reason>): back to dmp"`, log a warning.
3. `"act"` → create the controller and enter `AUTONOMOUS_SKILL`. `"dmp"` → `DmpRecovery` (today's path). `None` → `_escalate` as today.
4. An ACT abort in hybrid with `dmp_available` starts `DmpRecovery` immediately (`act_failed=True`). Otherwise it goes to the queue as today, emitting `STALL_ALERT` with payload `"backend": "act"`.

The observer is called on:
- the first engaged clutch tick of a rescue (`"operator"`);
- autonomous start (`"dmp"` or `"act"`);
- every control tick of those phases, with the commanded world wrist pose and fingers;
- clutch release (`success = bolt cleared`), autonomous complete (`True`) and abort (`False`, reason).

- [ ] **Step 1: Capture the DMP golden run.** On `main`, before changing anything, write `tests/atc/test_recovery.py::test_dmp_mode_script_unchanged`. It runs `StageDirector` headless (as `teleop atc --script --headless` does) and asserts every beat `ok`. It also asserts the list of compliance event kinds in emission order equals a literal `GOLDEN_EVENT_KINDS` copied from that run. Run it on `main`: PASS. This is the regression guard for the refactor.
- [ ] **Step 2: Write failing schema and manifest tests.**
  - `test_names_lengths`: `len(STATE_NAMES) == 29`, `len(ACTION_NAMES) == 12`, no duplicates.
  - `test_action_roundtrip`: for a random latch pose and wrist pose, `command_from_action(wfl, action_from_command(wfl, w, f))` returns `w` within 1e-9 (position) and 1e-9 (quaternion up to sign), and `f`.
  - `test_observe_by_name`: in a four-bay scene, `observe(rig)[STATE_NAMES.index("arm/joint4")]` equals the `bay1/joint4` qpos.
  - `test_manifest_roundtrip`, `test_manifest_missing_raises`, `test_manifest_wrong_schema_raises`.
  - `test_covers_near_and_far`: a prototype mean → returns the fault; mean + 10σ → `None`; a failed fault's prototypes are never matched.
- [ ] **Step 3: Write failing seam tests.**
  - `test_choose_recovery_table`: parametrised over every row of the table above.
  - `test_next_backend_cycles`.
  - `test_backend_switch_mid_recovery_finishes_on_old`: fake ACT factory covering everything; hybrid; arm the Bay 2 flap; once `recovery_name(2) == "act"`, call `set_recovery_backend(DMP)`; `recovery_name(2)` stays `"act"` until the bay leaves `AUTONOMOUS_SKILL`.
  - `test_downgrade_when_act_unhealthy`: factory `healthy()` → False; on the next stall `recovery_backend == DMP` and `"back to dmp"` in `backend_note`.
  - `test_hybrid_act_abort_falls_back_to_dmp`: fake controller whose `check` returns `"no progress"` on the first tick; with a deployed DMP the bay continues in `AUTONOMOUS_SKILL` with `recovery_name == "dmp"`; without one it is `QUEUED`.
  - `test_observer_sees_rescue_and_auto_episodes`: a recording fake observer gets start/tick/end in that order for one operator rescue and one DMP auto recovery, with `success=True`.
  - `test_dmp_mode_does_not_import_torch`: subprocess runs `teleop.commands.atc.main(["--headless", "--steps", "30", "--no-audio"])` and then asserts `"torch" not in sys.modules`.
  - `test_unknown_backend_exits_2` and `test_missing_hook_exits_3` (monkeypatch `importlib.import_module` to raise `ModuleNotFoundError` for `teleop.atc.episodes`).
  - `test_m_key_cycles_backend`: `controller.on_key(ord("M"))` then `tick()` → `coordinator.recovery_backend == HYBRID`.
- [ ] **Step 4: Run the tests to confirm they fail.** Run: `uv run pytest tests/policy tests/atc/test_recovery.py -v`. Expected: the new tests FAIL with import errors. `test_dmp_mode_script_unchanged` still PASSES.
- [ ] **Step 5: Implement `teleop.policy.schema` and `teleop.policy.manifest`** to the interfaces above. `covers` builds a `MahalanobisStallClassifier` with one prototype per passed fault (SKU `"release"`). The rotation part of the action is `so3_log(R_ref^T R_latch)` with `R_ref = ROUTINE.rotation()`, imported inside the function to avoid an import cycle.
- [ ] **Step 6: Implement `teleop.atc.recovery`.** Move `_Move` to `SmoothMove` and the DMP back-off/replay/tube/timeout logic to `DmpRecovery` with no numeric changes.
- [ ] **Step 7: Rewire the coordinator** to the flow above. Then add the `M` key to `stage.py`, and the flags, config fields, lazy hooks and three `configs/atc.yaml` keys (`recovery_backend: dmp`, `act_checkpoint: null`, `record_episodes: null`) to `commands/atc.py`.
- [ ] **Step 8: Run everything.** Run: `uv run pytest -v`. Expected: all PASS, including `test_dmp_mode_script_unchanged` with the unchanged golden list and the existing `tests/atc/*`.
- [ ] **Step 9: Verify and commit.** Run the full verify command. Then: `git commit -m "feat(atc): recovery backend seam with dmp default, policy schema and gate manifest"`

### Task 2 (T-191): Wrist camera and episode recording

**Files:**
- Create: `src/teleop/policy/cameras.py`, `src/teleop/atc/episodes.py`
- Modify: `src/teleop/sim/bays.py` (add `bay{k}/wrist_cam` on each active bay's Inspire hand base, in both the four-bay and single-bay scenes)
- Test: `tests/policy/test_cameras.py`, `tests/atc/test_episodes.py`

**Interfaces:**
- Consumes: Task 1's `schema.*`, `CoordinatorObserver`, `EpisodeMeta`, `camera_name`; `EpisodeRecorder` (`teleop.record.recorder`); `StageController.hud_view().faults`.
- Produces:
  - `teleop.policy.cameras.BayCameraRenderer(model: Any, image_hw: tuple[int, int] = IMAGE_HW)` with `render(data: Any, bay_id: int) -> dict[str, NDArray[np.uint8]]` (keys `CAMERAS`, HWC uint8) and `close() -> None`. One `mujoco.Renderer` reused for all cameras.
  - `teleop.atc.episodes.EpisodeSink(root: Path, model: Any, data: Any, fault_of: Callable[[int], str | None], operator_source: Literal["glove", "scripted"] = "glove", renderer: BayCameraRenderer | None = None)`: implements `CoordinatorObserver` plus `close() -> None` and a read-only `written: int` (episodes saved).
  - `open_episode_sink(root, controller, operator_source) -> EpisodeSink` (the Task 1 hook). `fault_of(bay)` returns the kind of the `"active"` fault for that bay from `controller.hud_view().faults`, or `None`.

Behaviour:
- Keep frames per bay in memory. Sample on every second `on_tick` (30 fps).
- On `on_episode_end`, hand the finished episode to one writer thread. The thread owns the `EpisodeRecorder` and writes one episode at a time with `task = f"recover latch: {fault or 'unknown'}"` and `EpisodeMeta` as the sidecar line.
- Map `"operator"` → `operator_source`.
- `close()` drops open (unfinished) episodes, drains the queue and calls `recorder.close()` (finalise).

- [ ] **Step 1: Write the failing tests.**
  - `test_wrist_camera_exists_in_both_scenes`: `mj_name2id(..., "bay1/wrist_cam") >= 0` in the four-bay scene and the validator's single-bay scene.
  - `test_render_shapes`: `BayCameraRenderer(model).render(data, 1)` returns `{"wrist", "top"}` with arrays of shape `(128, 128, 3)` and dtype `uint8`, and the wrist image is not uniform. Skip if there's no GL context (`pytest.skip` on `mujoco.FatalError`).
  - `test_records_operator_rescue` (importorskip lerobot): drive Bay 1 flap → `ScriptedGlove` rescue to clear with the sink attached → after `close()` the dataset has 1 episode; sidecar `source == "glove"`, `fault == "flap"`, `success is True`; `observation.state` has 29 names and `action` 12; frame count is within ±2 of rescue duration × 30.
  - `test_discarded_rescue_is_failure`: release the clutch before the bolt clears → one episode with `success is False`.
  - `test_two_bays_two_episodes`: a Bay 2 DMP auto recovery and a Bay 1 rescue in the same run → 2 episodes, `bay` 1 and 2.
  - `test_close_mid_episode_keeps_finished` (Review Focus 3): close while Bay 1's rescue is open, after one finished episode → the dataset loads with `LeRobotDataset` and holds exactly 1 episode.
  - `test_on_tick_never_blocks`: a writer stub that sleeps 0.5 s per episode; `on_tick` p99 < 5 ms over 300 ticks.
  - Inject a fake renderer returning zeros wherever GL is not needed.
- [ ] **Step 2: Run the tests to confirm they fail.** Run: `uv run pytest tests/policy/test_cameras.py tests/atc/test_episodes.py -v`. Expected: FAIL (modules missing).
- [ ] **Step 3: Add the wrist camera** in `bays.py`. Mount it on the hand base body, looking along the fingers, with `fovy` about 70°. The exact pose is your call: the index fingertip and the bolt must both be in view at the routine start pose.
- [ ] **Step 4: Implement `cameras.py` and `episodes.py`** to the interfaces above.
- [ ] **Step 5: Run the tests to confirm they pass.** Run: `uv run pytest tests/policy tests/atc -v`. Expected: PASS, including the existing scene and coordinator tests.
- [ ] **Step 6: Check by hand.** Run `uv run teleop atc --record-episodes data/episodes/smoke` with the synthetic source, do one rescue, quit, and open the dataset (`LeRobotDataset("local/smoke", root=...)`). Expect 1 episode, and the console still at 30 fps.
- [ ] **Step 7: Verify and commit.** `git commit -m "feat(record): save ATC recoveries as LeRobot episodes with wrist camera"`

### Task 3 (T-192): Scripted episode generator

**Files:**
- Create: `src/teleop/atc/generate.py`
- Test: `tests/atc/test_generate.py`

**Interfaces:**
- Consumes: `StageController` (`arm`, `on_key`, `tick`, `rescue_bay`), `ScriptedGlove`/`scripted_rescue` (`teleop.atc.stage`), `BayRig.set_world_latch`, Task 2's `EpisodeSink`.
- Produces:
  - `@dataclass(frozen=True) GeneratorConfig(perturb_xy_m: float = 0.10, perturb_yaw_deg: float = 40.0, timing_jitter: float = 0.2, pose_noise_m: float = 0.002, faults: tuple[str, ...] = ("bay1_flap", "bay2_flap", "bay2_pin"), max_sim_s_per_episode: float = 30.0)`
  - `run_generation(controller, sink, n, seed, config: GeneratorConfig = GeneratorConfig()) -> int`: the Task 1 hook signature plus an optional config.
  - `latch_perturbation(rng: np.random.Generator, config: GeneratorConfig) -> tuple[float, float, float]` (dx m, dy m, dyaw deg).

Behaviour:
- Each cycle: pick a fault with the RNG and arm it. Move the bay's latch by a perturbation, only while the bay is between cycles (NOMINAL with an outcome set). Hold the clutch on the queued bay. Feed a `ScriptedGlove` whose samples get time-scaling (1 ± `timing_jitter`) and Gaussian position noise (`pose_noise_m`).
- Stop after `n` successful episodes (`sink.written`), or after 3n attempts, whichever comes first.
- The DMP backend stays on, so DMP auto recoveries are recorded too (`source: "dmp"`).
- Sink `operator_source` is `"scripted"`.

- [ ] **Step 1: Write the failing tests.**
  - `test_perturbation_bounds_and_seed`: 1000 draws stay within ±0.10 m and ±40°, and the same seed gives the same sequence.
  - `test_generates_n_successes` (importorskip lerobot, n=3, seed=0): returns ≥ 3; the sidecar has ≥ 3 `success: true` lines with `source` in `{"scripted", "dmp"}` and `fault` in `{"flap", "pin"}`.
  - `test_gives_up_after_3n_attempts`: with a sink whose episodes always fail (a fake that never counts `written`), it returns 0 and stops.
- [ ] **Step 2: Run the tests to confirm they fail.** Run: `uv run pytest tests/atc/test_generate.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `generate.py`.**
- [ ] **Step 4: Run the tests to confirm they pass.** Then do a 20-episode run: `uv run teleop atc --headless --no-audio --generate 20 --record-episodes data/episodes/gen-smoke`. Expect `kept` ≥ 20 when processed by Task 4 (or the sidecar count if Task 4 hasn't landed).
- [ ] **Step 5: Verify and commit.** `git commit -m "feat(atc): headless scripted rescue episode generator"`

### Task 4 (T-193): ATC cleaning and labelling in `teleop process`

**Files:**
- Create: `src/teleop/process/atc_qc.py`
- Modify: `src/teleop/process/qc.py` (`write_clean` carries every input sidecar field into the output sidecar), `src/teleop/commands/process.py`
- Test: `tests/process/test_atc_qc.py`

**Interfaces:**
- Consumes: `EpisodeMeta`, `STATE_NAMES`, `ACTION_NAMES` (Task 1); `QCResult`, `write_clean`, `compute_trim` (existing); `TAU_HARD_MAX_NM`, `ARM_TORQUE_MARGIN` (`teleop.atc.validator`); Inspire actuator ranges via `teleop.sim.hands.actuator_ranges`.
- Produces:
  - `@dataclass AtcQCThresholds(min_len_s: float = 0.5, max_gap_s: float = 0.2, idle_eps: float = 1e-3)`
  - `is_atc_dataset(root: Path) -> bool` (sidecar lines carry `source`)
  - `run_atc_qc(root: Path, thresholds: AtcQCThresholds, sources: frozenset[str]) -> list[QCResult]`
  - `summarize(results: Sequence[QCResult], sidecars: Mapping[Path, Mapping[int, Mapping[str, Any]]]) -> str`: a fault × source table of kept/total
  - `teleop process` flag `--sources` (default `glove,scripted,dmp`). ATC mode is auto-detected. Mixing ATC and legacy datasets in one call exits 2 with `teleop process: cannot mix ATC and legacy datasets`.

Drop reasons (exact strings, used in reports):
- `"not successful"`
- `"source excluded"`
- `"too short"` (< 0.5 s after trim)
- `"timestamp gap"` (> 0.2 s)
- `"torque breach"` (`max_tau_hand_inf >= TAU_HARD_MAX_NM` or `max_arm_ratio >= ARM_TORQUE_MARGIN`)
- `"action out of range"` (any `hand/<a>` action outside its ctrlrange by > 1e-6)

Trim: drop leading frames until the first frame whose max abs action delta exceeds `idle_eps`, and trailing frames after the last such frame.

- [ ] **Step 1: Write the failing tests.** Build tiny datasets with `EpisodeRecorder` and hand-written sidecars (importorskip lerobot).
  - `test_drops_each_reason`: one episode per drop reason plus one good one; exactly the good one is kept and each reason string appears.
  - `test_trims_idle_head_and_tail`: 10 idle frames, 30 moving, 10 idle → trim `(10, 40)`.
  - `test_sources_filter`: `--sources glove` drops a `scripted` episode with `"source excluded"`.
  - `test_clean_sidecar_carries_labels`: the output sidecar keeps `fault`, `source`, `signature`.
  - `test_mixed_datasets_exit_2`.
  - `test_summary_counts`.
- [ ] **Step 2: Run the tests to confirm they fail.** Run: `uv run pytest tests/process/test_atc_qc.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `atc_qc.py`, the `write_clean` sidecar carry-through and the `process.py` wiring.** Legacy `tests/process/test_qc.py` must stay green.
- [ ] **Step 4: Run the tests to confirm they pass.** Run: `uv run pytest tests/process -v`. Expected: PASS.
- [ ] **Step 5: Verify and commit.** `git commit -m "feat(record): QC, trim and label ATC recovery datasets"`

### Task 5 (T-194): CUDA PyTorch group, training script, Verda runbook

**Files:**
- Modify: `pyproject.toml`, `uv.lock`
- Create: `scripts/train_act.py`, `docs/training/verda.md`
- Test: `tests/test_train_act.py`

**Interfaces:**
- Consumes: `SCHEMA_VERSION`, `FPS` (Task 1). The `lerobot-train` CLI of the installed LeRobot 0.7. Check flag names with `uv run lerobot-train --help` and write them down in the script's docstring.
- Produces:
  - uv dependency group `act`: LeRobot training dependencies plus `torch`/`torchvision` from a PyTorch CUDA wheel index, declared as an explicit `[[tool.uv.index]]` and selected in `[tool.uv.sources]` only for the `act` group. A plain `uv sync` must stay CPU and unchanged for CI. Pick the newest CUDA index with wheels for the torch version uv resolves. The laptop driver is 610.78.
  - `@dataclass(frozen=True) ActPreset(chunk_size: int = 30, batch_size: int = 32, steps: int = 50_000, lr: float = 1e-5, seed: int = 1000, device: str = "cuda", save_freq: int = 5_000)`
  - `build_train_args(dataset: Path, out: Path, preset: ActPreset) -> list[str]`: the full `lerobot-train` argv (`--policy.type=act`, `--dataset.repo_id=local/<dataset.name>`, `--dataset.root=<dataset>`, `--output_dir=<out>`, plus preset fields)
  - `write_run_json(dataset: Path, out: Path, preset: ActPreset, git_commit: str) -> Path`: writes `{schema_version, dataset_sha256 (of meta/info.json + meta/teleop_episodes.jsonl), git_commit, preset}`
  - `main(argv: list[str] | None = None, run: Callable[[list[str]], int] = subprocess.call) -> int`. Flags: `--dataset`, `--out`, `--steps`, `--batch-size`, `--device`, `--dry-run` (print argv only).

`docs/training/verda.md` is the runbook in spec section 7, in this order:
1. **Cost and teardown warning** at the top.
2. Install the Verda CLI. Check with `verda doctor`.
3. Create API credentials in the dashboard. They are entered by the person, never committed.
4. Register `~/.ssh/id_ed25519.pub`.
5. `verda instance-types`, then `verda vm create`.
6. `verda ssh`, then: clone, `uv sync --group act`, `nvidia-smi`.
7. Upload the dataset: rsync, or a private HF Hub dataset.
8. `uv run python scripts/train_act.py --dataset ... --out ...` inside `tmux`.
9. Copy `models/act/<run>/` back.
10. **Delete the instance**, and confirm it with `verda vm list`.

Mark any CLI subcommand you could not confirm against the docs.

- [ ] **Step 1: Write the failing tests.**
  - `test_build_train_args`: contains `--policy.type=act`, the dataset root and the output dir, and `--steps=50000` by default.
  - `test_dry_run_does_not_call_run`.
  - `test_run_json_contents`: keys present; `schema_version == SCHEMA_VERSION`; the sha changes when the sidecar changes.
  - `test_main_returns_run_exit_code`.
- [ ] **Step 2: Run the tests to confirm they fail.** Run: `uv run pytest tests/test_train_act.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement the script and the dependency group.** Then run `uv lock`.
- [ ] **Step 4: Check the group on the laptop.** Run `uv sync --group act` then `uv run python -c "import torch; print(torch.cuda.is_available())"`. Expected: `True`. Then run `uv sync` and the full verify. Expected: unchanged, all green.
- [ ] **Step 5: Smoke-train locally.** Run `uv run python scripts/train_act.py --dataset data/episodes/gen-smoke-clean --out models/act/smoke --steps 200 --batch-size 4`. Expected: it finishes and writes a checkpoint plus `run.json`.
- [ ] **Step 6: Verify and commit.** `git commit -m "build(train): CUDA act group, ACT training wrapper and Verda runbook"`

### Task 6 (T-195): ACT runtime and live ACT recovery

**Files:**
- Create: `src/teleop/policy/act_runtime.py`, `src/teleop/atc/act_recovery.py`
- Test: `tests/policy/test_act_runtime.py`, `tests/atc/test_act_recovery.py`

**Interfaces:**
- Consumes: Task 1's `RecoveryFactory`, `RecoveryController`, `RecoveryCommand`, `BaySignals`, `SmoothMove`, `GateManifest`, `covers`, `ManifestError`, `observe`, `command_from_action`, `STATE_NAMES`, `ACTION_NAMES`, `CAMERAS`. Task 2's `BayCameraRenderer`.
- Produces:
  - `@dataclass(frozen=True) ObsBatch(state: NDArray[np.float32], images: dict[str, NDArray[np.uint8]])`, shapes `(B, 29)` and `(B, H, W, 3)`
  - `class PolicyBackend(Protocol)`: `predict_chunk(batch: ObsBatch) -> NDArray[np.float32]` with shape `(B, chunk, 12)`
  - `load_act_policy(checkpoint: Path, device: str) -> PolicyBackend`. It imports torch and lerobot lazily, and raises `ManifestError` if the checkpoint's input or output features don't match `STATE_NAMES`/`ACTION_NAMES` sizes and `observation.images.{wrist,top}`.
  - `InferenceWorker(policy: PolicyBackend, budget_ms: float = 30.0, timeout_ms: float = 250.0, over_budget_limit: int = 10)` with:
    - `submit(bay_id: int, state: NDArray, images: dict[str, NDArray], tick: int) -> None` (non-blocking, latest wins per bay)
    - `latest(bay_id: int) -> tuple[int, NDArray[np.float32]] | None` (`(obs tick, chunk)`)
    - `healthy() -> bool`, `unhealthy_reason: str`, `p95_ms() -> float | None`, `close() -> None`
    - one daemon thread that batches every pending bay into one `predict_chunk` call
  - `ActRecoveryFactory(manifest: GateManifest, worker: InferenceWorker, renderer: BayCameraRenderer, data: Any, *, query_every: int = 5, retract_m: float = 0.03, retract_s: float = 0.5, settle_s: float = 0.7, timeout_s: float = 8.0, tau_hard_max: float = TAU_HARD_MAX_NM, stall_confidence: float = 0.3, stall_ticks: int = 18)`. Implements `RecoveryFactory` with `name = "act"`.
  - `load_act_factory(checkpoint: Path, device: str, model: Any, data: Any) -> ActRecoveryFactory` (the Task 1 hook). It reads `checkpoint/gate.json` via `GateManifest.load`.
  - `ReplayPolicy(actions_latch: NDArray[np.float32], chunk: int = 30)`: a test and eval stub whose `predict_chunk` returns the next `chunk` rows of a recorded latch-frame action sequence, chosen by nearest wrist position. Make it public, because Task 7 uses it.

`ActRecovery` phases:
1. Retract 3 cm along latch −X (`SmoothMove`).
2. Hold for `settle_s`.
3. Policy phase: every `query_every` ticks, `submit(observe(rig), renderer.render(data, bay), tick)`. Each tick, play row `(tick − obs_tick) // 2` of the latest chunk through `command_from_action`, clamped to the last row. Return `None` (hold) until the first chunk arrives.

`check()` returns, in this priority:
- `"inference unavailable"` if the worker is unhealthy;
- `"inference timeout"` if there's no chunk 250 ms after the first submit;
- `"hand torque limit"` if `tau_hand_inf >= tau_hard_max`;
- `"no progress"` if confidence < 0.3 for 18 consecutive policy-phase ticks;
- `"act timeout"` after `timeout_s`.

- [ ] **Step 1: Write the failing tests.**
  - `test_worker_batches_two_bays` (Review Focus 4): a stub counting calls; submit for bays 1 and 2 before the worker wakes → one `predict_chunk` call with B=2, and both `latest()` are filled.
  - `test_worker_over_budget_turns_unhealthy`: a stub sleeping 40 ms → unhealthy after 10 calls, reason mentions `"budget"`.
  - `test_worker_exception_turns_unhealthy`.
  - `test_load_rejects_schema_mismatch` (Review Focus 2): write a fake checkpoint `config.json` with a state dim of 20 → `ManifestError`. With a `gate.json` of the wrong `schema_version` → `load_act_factory` raises `ManifestError`.
  - `test_act_recovery_clears_flap_with_replay_stub`: record one scripted Bay 1 flap rescue's latch-frame actions (via `action_from_command` on the `ScriptedGlove` commands), build a manifest whose `flap` prototype is that stall's signature, run the coordinator in hybrid with a `ReplayPolicy` worker → the bay reaches `cleared autonomously` with `recovery_name == "act"`.
  - `test_one_bay_abort_does_not_stop_other` (Review Focus 4): Bay 1 hits the hand torque limit and aborts. Bay 2's ACT recovery continues and clears.
  - `test_atc_downgrades_on_bad_checkpoint`: `teleop atc --headless --steps 30 --recovery-backend hybrid --act-checkpoint <dir without gate.json>` → exit 0, a warning is logged and the backend is `dmp`.
- [ ] **Step 2: Run the tests to confirm they fail.** Run: `uv run pytest tests/policy/test_act_runtime.py tests/atc/test_act_recovery.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `act_runtime.py`, then `act_recovery.py`.** `load_act_policy` uses LeRobot's ACT policy class `from_pretrained` and its pre/post-processors as saved with the checkpoint. Images are converted HWC uint8 → CHW float the way the processor expects. Run under `torch.inference_mode()` with autocast fp16 on CUDA.
- [ ] **Step 4: Run the tests to confirm they pass.** Run: `uv run pytest tests/policy tests/atc -v`. Expected: PASS.
- [ ] **Step 5: Check latency on the laptop** with the Task 5 smoke checkpoint: `uv run --group act python -c "..."`, or a tiny bench in the test file marked `@pytest.mark.hardware`. Record the p95 for B=2 in the PR description. Expected: ≤ 30 ms.
- [ ] **Step 6: Verify and commit.** `git commit -m "feat(atc): batched ACT inference worker and live ACT recovery"`

### Task 7 (T-196): Offline gate (`teleop eval`) and release manifest

**Files:**
- Create: `src/teleop/commands/eval.py`
- Modify: `src/teleop/atc/validator.py`
- Test: `tests/atc/test_eval_gate.py`

**Interfaces:**
- Consumes: `RecoveryController`, `DmpRecovery`, `GateManifest`, `FaultResult`, `EpisodeMeta` (Task 1); `ActRecoveryFactory`, `InferenceWorker`, `load_act_policy`, `ReplayPolicy` (Task 6); `BayCameraRenderer` (Task 2); `fit_rbf_dmp` (`teleop.unoq.dmp`).
- Produces:
  - `SkillValidator.validate_controller(make: Callable[[BayRig], RecoveryController], *, fault: Literal["flap", "pin"] = "flap", perturbations: Sequence[tuple[float, float, float]] | None = None) -> ValidationReport`. Every rollout starts in today's stall state. Each controller does its own pre-move: the DMP 1 cm back-off, or the ACT 3 cm retract and settle. `validate(skill)` keeps its exact behaviour and becomes `validate_controller(DmpRecovery…)`. Existing validator tests stay green unchanged.
  - `random_perturbations(n: int, rng: np.random.Generator, xy_m: float = 0.08, yaw_deg: float = 30.0) -> list[tuple[float, float, float]]`
  - `teleop eval --checkpoint DIR --dataset DIR [--faults flap,pin] [--rollouts 10] [--bar 9] [--device cuda] [--seed 0]`. It writes `DIR/gate.json` and prints one row per fault (rollouts passed for ACT and for the DMP, mean time-to-clear, max hand torque, p95 ms). It returns 0 if at least one fault passed, else 1.
  - Module-level `_load_policy = load_act_policy`, so tests can monkeypatch it.

Rollouts per fault: the 3 `PERTURBATIONS_DEG`, then `rollouts − 3` random ones (seeded).

The DMP comparison fits `fit_rbf_dmp` to the first successful `glove` (else `scripted`) episode of that fault in `--dataset`, using its state's wrist pose and hand values, and runs the same rollouts.

Prototypes for a passed fault are all `signature` vectors from that fault's successful dataset episodes. A fault with no episodes gets a result with `note = "no data"`, is not passed and has no prototypes.

- [ ] **Step 1: Write the failing tests.**
  - `test_validate_unchanged_for_dmp`: the existing DMP gate on the canonical skill gives the same `n_passed` and `max_tau_ext_inf_nm` (to 1e-9) as before the refactor. Capture the values on `main` first, as in Task 1 Step 1.
  - `test_eval_writes_manifest_with_passed_fault` (`ReplayPolicy` from a scripted flap episode, `--rollouts 4 --bar 3`): `gate.json` passes `flap` and `prototypes["flap"]` is non-empty.
  - `test_eval_fault_below_bar_not_passed`: a `ReplayPolicy` of an idle sequence → `flap` not in `passed_faults()`, exit 1.
  - `test_eval_no_data_fault` (Review Focus 5): `--faults flap,pin` on a flap-only dataset → `results["pin"].note == "no data"`, no exception.
  - `test_pin_fault_uses_bolt_pinned`: in a `fault="pin"` rollout, the latch reset is called with `bolt_pinned=True`.
- [ ] **Step 2: Run the tests to confirm they fail.** Run: `uv run pytest tests/atc/test_eval_gate.py -v`. Expected: FAIL.
- [ ] **Step 3: Generalise the validator, then implement `eval.py`.**
- [ ] **Step 4: Run the tests to confirm they pass.** Run: `uv run pytest tests/atc -v`. Expected: PASS, including all pre-existing validator tests.
- [ ] **Step 5: Verify and commit.** `git commit -m "feat(atc): offline ACT gate and release manifest via teleop eval"`

### Task 8 (T-197): Console and HUD for the recovery backend, README

**Files:**
- Modify: `src/teleop/console/telemetry.py`, `src/teleop/console/web/console.js`, `src/teleop/console/web/console.css`, `src/teleop/console/web/index.html`, `src/teleop/atc/hud.py`, `src/teleop/atc/stage.py` (`hud_view()` only), `README.md` (new section)
- Test: `tests/console/test_recovery_telemetry.py`

**Interfaces:**
- Consumes: `AtcCoordinator.recovery_backend`, `backend_note`, `recovery_name(bay)`, the `act` factory's `healthy()`/`unhealthy_reason` and, when it's an `ActRecoveryFactory`, `worker.p95_ms()` and `manifest.passed_faults()`.
- Produces:
  - `HudView.recovery: RecoveryStatus`, where `@dataclass(frozen=True) RecoveryStatus(backend: str = "dmp", note: str = "", act_loaded: bool = False, act_healthy: bool = False, p95_ms: float | None = None, faults: tuple[str, ...] = ())`. It lives in `hud.py`, and `StageController.hud_view()` fills it.
  - Telemetry JSON gains `"recovery": {"backend", "note", "act": {"loaded", "healthy", "p95_ms", "faults"}}`, plus `"recovery": "dmp" | "act" | null` per bay entry.
  - Console:
    - a backend chip in the telemetry strip (`DMP`, `HYBRID`, `ACT`);
    - an `M` key in the key bar with hint `"Switch recovery: <next>"`;
    - a toast whenever `note` changes;
    - an `ACT` or `DMP` tag on a bay in `AUTONOMOUS_SKILL`.
  - Viewer HUD: one line `RECOVERY hybrid | ACT ok p95 12 ms | flap`.
  - README section **Recovery policy (ACT)**: collect → process → train (link the Verda runbook) → eval → run with `--recovery-backend hybrid`, and how to fall back (`M` or `--recovery-backend dmp`).

- [ ] **Step 1: Write the failing tests.**
  - `test_telemetry_has_recovery_block`: the default coordinator gives `backend == "dmp"` and `act.loaded is False`.
  - `test_note_change_reaches_telemetry`: a forced downgrade note appears in the next frame.
  - `test_bay_recovery_tag`: a fake ACT recovery on Bay 2 → `bays[1]["recovery"] == "act"`.
  - `test_hud_recovery_line`: `hud_lines` contains `RECOVERY dmp`.
  - Web assets: extend `tests/console/test_web_assets.py`-style checks so `console.js` references `recovery` and the `M` key.
- [ ] **Step 2: Run the tests to confirm they fail.** Run: `uv run pytest tests/console/test_recovery_telemetry.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement it.**
- [ ] **Step 4: Run the tests to confirm they pass.** Then check in the browser: `uv run teleop atc`, press `M` three times, and see the chip cycle and toasts. Start with `--recovery-backend hybrid --act-checkpoint models/act/smoke` and confirm the chip shows `ACT ok` with a p95 value.
- [ ] **Step 5: Verify and commit.** `git commit -m "feat(atc): show recovery backend, ACT health and M toggle on console and HUD"`

### Task 9 (T-198): First Verda training run and release (operational)

**Files:**
- Create: `docs/training/results.md`
- Modify: `configs/atc.yaml` (`recovery_backend: hybrid` and `act_checkpoint`, **only if** the gate passes), `.gitignore` (`data/`, `models/` if missing)

This task is run by the person holding the Verda credits, on the hardware laptop. No new code.
Model checkpoints and datasets stay out of git (`data/` and `models/` are gitignored; add them
in this task if they aren't).

- [ ] **Step 1: Generate data.** `uv run teleop atc --headless --no-audio --generate 300 --seed 1 --record-episodes data/episodes/gen-001`. Add any glove sessions recorded with `--record-episodes`.
- [ ] **Step 2: Clean and merge.** `uv run teleop process --in data/episodes/gen-001 [--in data/episodes/<glove sessions>] --out data/clean/release-001`. Expected: the summary shows ≥ 100 kept per fault.
- [ ] **Step 3: Train.** Follow `docs/training/verda.md` end to end: train `models/act/release-001`, copy it back, **delete the instance**, and note the instance type, wall time and cost.
- [ ] **Step 4: Gate.** `uv run --group act teleop eval --checkpoint models/act/release-001 --dataset data/clean/release-001 --faults flap,pin`. Expected: `flap` passes (≥ 9/10).
- [ ] **Step 5: Run live.** `uv run --group act teleop atc --recovery-backend hybrid --act-checkpoint models/act/release-001`. Check, for two bays recovering at once:
  - the console holds 30 fps;
  - p95 ≤ 30 ms on the chip;
  - `M` back to `dmp` restores DMP recoveries;
  - `--recovery-backend dmp --script --headless` passes every beat.
- [ ] **Step 6: Record and release.** Write the eval table, latency, data counts and cost into `docs/training/results.md`. If the gate passed, set `recovery_backend: hybrid` and `act_checkpoint: models/act/release-001` in `configs/atc.yaml`. Commit: `git commit -m "docs(train): first ACT release results; enable hybrid recovery"`
