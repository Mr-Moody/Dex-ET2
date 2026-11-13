# M2 + M3: Booth App, Processing, Training and Attract Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A walk-up booth (`uv run teleop booth`) where a stranger calibrates in about 15 s and plays a scored 60 s lift-cube challenge with both hands. Every attempt is auto-recorded as a LeRobot v3 episode. Episodes are cleaned (rule-based QC, with a VLM annotator hook) and an ACT policy is trained on them. Attract mode replays the policy, or the best episode, when nobody is wearing the gloves.

**Architecture:** The booth is a small state machine (`teleop.booth.machine`) driving M1's `TeleopApp` (`teleop.app`). The HUD is drawn with MuJoCo viewer overlays. Processing (`teleop.process`) reads a recorded dataset and writes a cleaned copy plus a QC report. Training is `lerobot-train` driven by a config. Eval and attract roll the policy out through the same `SimSink`.

**Tech Stack:** As M1, plus `lerobot-train` (ACT) on CUDA, and an optional Gemma or other VLM client for annotation (behind `Annotator`, so it's off by default).

**Spec:** `docs/superpowers/specs/2026-11-13-glove-teleop-booth-design.md` (§5.5, §5.7–5.10, §7). M1 plan: `docs/superpowers/plans/2026-11-13-m0-m1-glove-to-sim.md`.

## Global Constraints

- Everything in the M1 plan's Global Constraints still applies. **No AI co-author trailers or "Generated with" lines in commits or PRs** (user preference).
- **Hard freeze Sun 2026-11-15 10:00**: the code submitted at 10:00 is the demo build. M3 training must start by Sat ~22:00 so it can run overnight.
- Visitors never touch the terminal. Between visitors the booth must recover by itself: no glove frames → "put on the gloves" screen. Studio restart → reconnect.
- Calibration UX ≤ 15 s: flat 2 s → fist 2 s, with a skip key that uses the last good calibration or the default.
- Each booth session writes to a fresh dataset root `data/datasets/booth-<YYYYmmdd-HHMMSS>` (the recorder refuses non-empty roots).
- HUD text must be readable from 1–2 m: large font, high contrast.

## Review Focus

1. **A visitor walks away mid-attempt** (gloves go stale). The attempt ends as "abandoned", the episode is discarded (not saved), and the booth returns to idle within 3 s. Pinned in T-060 `test_stale_gloves_abandons_and_discards`.
2. **Recording dies** (disk full, encoder error). The booth keeps running without recording, the HUD shows "REC OFF", and the error is logged. Pinned in T-060 `test_recorder_failure_keeps_booth_running`.
3. **A degenerate calibration** (the visitor didn't move). The flagged actuators fall back to the last good calibration or the default, and the HUD says "calibration partial". Pinned in T-060 `test_degenerate_calibration_falls_back`.
4. **An empty or tiny dataset handed to processing or training.** Processing writes an empty clean set and a report saying why. The train script refuses with a clear message if there are fewer than 10 kept episodes. Pinned in T-061 `test_empty_dataset_report` and T-070 `test_train_refuses_tiny_dataset`.
5. **The policy checkpoint is missing or fails to load at attract time.** Attract falls back to replaying the best recorded episode, or to the synthetic open/close. Pinned in T-070 `test_attract_falls_back_without_checkpoint`.

---

### Task T-060: Booth state machine, HUD, auto-record

**Files:** Create `src/teleop/booth/__init__.py`, `src/teleop/booth/machine.py`, `src/teleop/booth/hud.py`, `src/teleop/commands/booth.py`. Modify `src/teleop/cli.py` (add `booth`). Test `tests/booth/test_machine.py`.
**Interfaces:**
- Consumes: M1 `TeleopApp` (T-050), `calibrate`, `EpisodeRecorder`, `AutoWrist.coil.recenter`, `LiftCube.score`.
- Produces:
  - `class BoothState(StrEnum)`: `IDLE, ATTRACT, CALIB_FLAT, CALIB_FIST, READY, PLAYING, RESULT`.
  - `class BoothMachine(clock, cfg: BoothConfig)` with `tick(now_ns, status: SourceStatus, frame: BimanualFrame, score: TaskScore | None) -> list[BoothEvent]` (pure logic, no MuJoCo) and the property `state`.
  - Events tell the app what to do: `StartCalibration, CalibrationDone(calib, partial: bool), StartEpisode, EndEpisode(save: bool, reason), ResetTask, Recenter, EnterAttract`.
  - `@dataclass BoothConfig`: `idle_to_attract_s=20`, `calib_hold_s=2.0`, `play_s=60`, `result_s=6`, `stale_abandon_s=3`.
- Flow:
  - IDLE becomes CALIB_FLAT once gloves are connected and moving.
  - CALIB_FLAT → CALIB_FIST → READY (recenter happens here), then 3-2-1 → PLAYING (auto-record starts).
  - PLAYING ends on success, timeout, or stale gloves (abandon). Success or timeout goes to RESULT (save=True). Abandon goes to IDLE (save=False).
  - RESULT → IDLE. IDLE goes to ATTRACT after `idle_to_attract_s` with no gloves.
- HUD (`hud.py`): `render_overlay(viewer, state, info)` draws the state title, instructions, countdown, score, "REC ●", gloves/hotspot status and wrist mode. It uses `viewer.set_texts` or `mjr_overlay` and the large font.
- [ ] Tests (pure machine, fake clock): `test_happy_path_events`, `test_stale_gloves_abandons_and_discards`, `test_recorder_failure_keeps_booth_running` (app-level, with a fake recorder that raises), `test_degenerate_calibration_falls_back`, `test_idle_to_attract_after_timeout`, `test_skip_calibration_key_uses_last_good`.
- [ ] Manual: `uv run teleop booth --source replay --replay-file data/raw/coil_on.jsonl` runs through calibration and play. Then, on the hardware laptop, `--source studio --hotspot`.
- [ ] Commit `feat(booth): add booth state machine, HUD and auto-record`.

### Task T-061: Processing / QC + annotator hook

**Files:** Create `src/teleop/process/__init__.py`, `qc.py`, `annotate.py`, `src/teleop/commands/process.py`. Test `tests/process/test_qc.py`.
**Interfaces:**
- Produces:
  - `@dataclass QCResult(episode_index: int, keep: bool, reasons: list[str], trim: tuple[int, int])`.
  - `run_qc(ds_root: Path, thresholds: QCThresholds) -> list[QCResult]`. Rules:
    - `too_short` (< 2 s)
    - `input_dropout` (sidecar status shows frames gap > 0.5 s, or a disconnect event)
    - `hotspot_event` (any hotspot watchdog restart during the episode)
    - `out_of_range` (any action outside actuator range > 5 % of frames)
    - `no_progress` (cube max height < 3 cm)
  - Idle head and tail are trimmed by action velocity < ε.
  - `class Annotator(Protocol)` with `annotate(episode_frames: Sequence[np.ndarray], meta: Mapping) -> Annotation(label: str, success: bool | None, notes: str)`.
  - `RuleAnnotator` labels by score and success. `VlmAnnotator(client)` is the stretch task T-081.
  - `write_clean(ds_root, results, out_root)` writes a new LeRobot dataset with only the kept, trimmed episodes, plus `qc_report.json` and `qc_report.md`.
  - CLI: `teleop process --in data/datasets/booth-... [--in ...] --out data/datasets/clean-<ts>` merges several sessions.
- [ ] Tests: `test_too_short_rejected`, `test_dropout_rejected`, `test_trim_idle_ends`, `test_merge_two_sessions`, `test_empty_dataset_report`, `test_clean_dataset_loads`.
- [ ] Commit `feat(record): add episode QC, trimming and annotator hook`.

### Task T-070: Train, eval, attract

**Files:** Create `configs/train/act_booth.yaml`, `scripts/train_act.py` (or `src/teleop/train/train.py`), `src/teleop/train/eval_rollout.py`, `src/teleop/commands/eval.py`, `src/teleop/booth/attract.py`. Test `tests/train/test_eval.py`.
**Interfaces:**
- `train_act.py --dataset data/datasets/clean-... --out outputs/act-<ts>` wraps `lerobot-train --policy.type=act --dataset.root=... --policy.device=cuda`. Pin the CUDA torch index for Windows in pyproject if needed (ADR).
- It refuses to start if there are fewer than 10 episodes.
- `PolicyRunner(checkpoint: Path)` has `act(obs: dict) -> np.ndarray` (action in schema order). `rollout(app, runner, episodes: int, max_s: float) -> EvalReport(success_rate, mean_score)`.
- `teleop eval --checkpoint ... --episodes 20` prints and writes the report.
- `AttractPlayer` uses a policy if one is available, else the best recorded episode (highest score) replayed through SimSink, else synthetic. The booth's ATTRACT state uses it.
- [ ] Tests: `test_train_refuses_tiny_dataset`, `test_rollout_with_dummy_policy_reports`, `test_attract_falls_back_without_checkpoint`, `test_attract_replays_best_episode`.
- [ ] Commit `feat(train): add ACT training wrapper, sim eval and attract mode`.

### Task T-080 (stretch): Real Inspire sink
Decided on Saturday once the hardware is seen. `InspireRealSink(RobotSink)` maps the same actuator targets to the Inspire command range over the Aalto integration or Unitree DDS `rt/inspire/cmd`. Rate-limited. Its own plan section gets written when it's picked up.

### Task T-081 (stretch): VLM annotator
`VlmAnnotator` samples 4 frames per episode and asks a local or hosted VLM (e.g. Gemma) "did the hand lift the cube? describe the grasp". It stores the label and notes in the QC report. It's off by default and enabled with `--annotator vlm`. Needs the user's choice of model and endpoint before it starts.
