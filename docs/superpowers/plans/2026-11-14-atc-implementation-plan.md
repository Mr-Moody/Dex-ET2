# Air Traffic Control for Robot Hands (`T-100`–`T-161`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `uv run teleop atc` — a 1-to-4 EV battery disassembly fleet supervisor where Bays 1–2 run active 7-DoF Franka Panda + 16-DoF LEAP Hand MuJoCo physics against a 3-box sprung slide-bolt latch, an operator rescues contact-discovered jams via Rokoko Smartglove II relative-clutch teleop, an Arduino UNO Q (or automatic localhost mock) runs a $1\text{€}$ filter + Mahalanobis stall classifier + $<20\text{ ms}$ 30-kernel RBF-DMP fit, a 3-rollout headless laptop gate validates the skill, Bay 2 autonomously clears rotated/shifted jams ($35^\circ, 12\text{ cm}$) while Negative Control ($x = 0\text{ cm}$) is safely refused, and decoupled teammate hooks emit events to **Solana** (`T-160`) and **Verda Nemotron** (`T-161`).

**Architecture:** `T-100` lands first and owns **all** edits to shared files (`cli.py`, `scripts/fetch_models.py`, `configs/hands/leap.yaml`, `.env.example`, `.pre-commit-config.yaml`) plus shared contracts (`teleop.atc.types`, `teleop.compliance`, `teleop.reasoning`). Once `T-100` merges, five tracks (`T-110`, `T-130`, `T-131`, `T-160`, `T-161`) run in parallel git worktrees with **zero overlapping file edits**. `T-120` follows `T-110`; `T-140` integrates `T-110 + T-120 + T-130`; `T-150` wires the live HUD and 2.5-minute stage script.

**Tech Stack:** Python 3.12, `uv`, `mujoco` (`MjSpec`, `mj_jacSite`, `mj_inverse`, `launch_passive`, `Renderer`), `numpy` (closed-form RBF-DMP ridge regression & Mahalanobis classifier — **no PyTorch MLP**), `draccus`, `pyyaml`, `python-dotenv`, `pytest` + `hypothesis`, `ruff`, `pyright` (strict on `src/teleop`).

**Brief:** `PROJECT_BRIEF.md` · **Spec:** `docs/superpowers/specs/2026-11-14-atc-fleet-rescue-design.md` · **ADR:** `docs/adr/0002-atc-fleet-rescue-pivot.md`

---

## Global Constraints

1. **M1 Constraints Apply:** Python 3.12, `uv` only (`uv add`, `uv run`, `uv sync` — never `pip`), quaternions **xyzw** (`(x, y, z, w)` everywhere in Python; convert to MuJoCo `wxyz` only at `MjData` boundary), joints/actuators mapped **by name, never by index**, `time.monotonic_ns()` clock, and zero hardware/GPU/display/network required in default `pytest` (`-m "not hardware"`).
2. **No AI Attribution Trailers:** Do **not** include `Co-Authored-By` or `Generated with` lines in commits or PRs.
3. **Preserve Existing M1 Tests & Commands:** Do **not** break `teleop sim`, `capture`, `calibrate`, `record`, `replay`, or `process`. Existing `inspire` and `lift_cube` tests must stay green.
4. **Strict File Ownership (Zero Merge Conflicts):**
   - **Only `T-100`** may edit `src/teleop/cli.py`, `scripts/fetch_models.py`, `third_party/README.md`, `configs/hands/leap.yaml`, `.env.example`, and `.pre-commit-config.yaml`.
   - **Only `T-160`** (Solana teammate) edits `src/teleop/compliance/solana.py` after `T-100`.
   - **Only `T-161`** (Verda Nemotron teammate) edits `src/teleop/reasoning/nemotron.py` after `T-100`.
   - Every other task (`T-110`–`T-150`) creates and edits **only** the files listed in its own section below.
5. **Causal Contact-Only Jam Discovery:** Neither `coordinator.py`, `validator.py`, nor `unoq/` may ever read `flap_hinge` `qpos` to decide whether a jam exists or which skill to run. Jam detection uses $C(t)$ and $x_{\text{bolt}}$ / $\tau_{\text{ext}}$ strictly during physical contact.
6. **Verification Before Every PR:**
   ```bash
   uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
   ```

---

## File Ownership Map

```text
PROJECT_BRIEF.md
docs/adr/0002-atc-fleet-rescue-pivot.md
docs/superpowers/specs/2026-11-14-atc-fleet-rescue-design.md
docs/superpowers/plans/2026-11-14-atc-implementation-plan.md
configs/
  atc.yaml                                  # T-150
  hands/leap.yaml                           # T-100 (extend from 12 -> 16 actuators)
scripts/
  fetch_models.py                           # T-100 (add franka_emika_panda to menagerie sparse_dirs)
firmware/
  unoq_console/unoq_console.ino             # T-131 (STM32U585 1 kHz buttons/clutch/fault-rig + 8x13 LEDs)
src/teleop/
  cli.py                                    # T-100 (register "atc" and "unoq" in COMMANDS)
  atc/
    __init__.py  types.py                   # T-100
    coordinator.py  validator.py            # T-140
    hud.py  stage.py                        # T-150
  sim/
    latch.py  ik.py  torques.py  bays.py    # T-110
  retarget/
    clutch.py                               # T-120
  unoq/
    __init__.py  filter.py  classifier.py   # T-130
    dmp.py  protocol.py  server.py          # T-130
    client.py                               # T-130
    mcu_bridge.py                           # T-131
  compliance/
    __init__.py  protocol.py                # T-100
    solana.py                               # T-100 (placeholder) -> T-160 (Solana implementation)
  reasoning/
    __init__.py  protocol.py                # T-100
    nemotron.py                             # T-100 (placeholder) -> T-161 (Verda Nemotron implementation)
  commands/
    unoq.py                                 # T-130
    atc.py                                  # T-150
tests/
  atc/test_types_and_placeholders.py        # T-100
  sim/test_latch_panda.py                   # T-110
  retarget/test_clutch.py                   # T-120
  unoq/test_edge_dmp.py                     # T-130
  unoq/test_mcu_bridge.py                   # T-131
  atc/test_coordinator.py                   # T-140
  atc/test_stage_hud.py                     # T-150
  compliance/test_solana.py                 # T-160
  reasoning/test_nemotron.py                # T-161
```

---

## Tasks

### Task T-100: ATC shared contracts, CLI registration, Panda fetch, 16-DoF LEAP & placeholders

**Sequential prerequisite (`depends_on: [T-050]`). Run first after pulling latest `origin/main`.**

**Files:**
- Modify: `scripts/fetch_models.py`, `third_party/README.md`, `configs/hands/leap.yaml`, `src/teleop/cli.py`, `.env.example`, `.pre-commit-config.yaml` (add `atc,unoq` scopes if needed), and retire any unstarted legacy M2/M3 tasks (`T-013`, `T-060`, `T-070`, `T-080`, `T-081` $\rightarrow$ `status: blocked` if still `todo`).
- Create: `src/teleop/atc/__init__.py`, `src/teleop/atc/types.py`, `src/teleop/compliance/__init__.py`, `src/teleop/compliance/protocol.py`, `src/teleop/compliance/solana.py`, `src/teleop/reasoning/__init__.py`, `src/teleop/reasoning/protocol.py`, `src/teleop/reasoning/nemotron.py`, `tests/atc/__init__.py`, `tests/atc/test_types_and_placeholders.py`.

**Step-by-step instructions:**
- [ ] **Step 1 (Retire unstarted legacy tasks safely):** After rebasing onto latest `origin/main`, for each of `T-013`, `T-060`, `T-070`, `T-080`, `T-081` whose `status` is still `todo`, run `uv run task status <ID> blocked` so `uv run task list --ready` only lists active `T-100+` tasks. Leave any task already marked `done` untouched.
- [ ] **Step 2 (Fetch Franka Emika Panda):** In `scripts/fetch_models.py`, update `SOURCES[0]` (`menagerie`, commit `370c43965323c0944b433b202aac5058b5f907a9`) to `sparse_dirs=("leap_hand", "franka_emika_panda")` and add `"franka_emika_panda/panda_nohand.xml"` to `expect`. Document `franka_emika_panda` (Apache-2.0 / Menagerie licence) in `third_party/README.md`.
- [ ] **Step 3 (16-DoF LEAP Hand YAML):** Extend `configs/hands/leap.yaml` with `th_axl_act`, `if_rot_act`, `mf_rot_act`, and `rf_rot_act` so `len(load_hand("leap").flex) == 16`. Verify `tests/sim/test_hands.py` still passes.
- [ ] **Step 4 (CLI & `.env.example`):** Add `"atc": "teleop.commands.atc"` and `"unoq": "teleop.commands.unoq"` to `COMMANDS` in `src/teleop/cli.py`. Append commented placeholder env keys (`UNOQ_HOST`, `UNOQ_PORT`, `UNOQ_SERIAL_PORT`, `SOLANA_RPC_URL`, `SOLANA_KEYPAIR_PATH`, `VERDA_BASE_URL`, `VERDA_API_KEY`, `NEMOTRON_MODEL`) to `.env.example`.
- [ ] **Step 5 (Contracts & Placeholders with TDD):** Write `tests/atc/test_types_and_placeholders.py` first, then implement:
  - `src/teleop/atc/types.py`: frozen dataclasses `LatchPose`, `StallSignature`, `StallMatch`, `SkillPrimitive`, `BayState` (`NOMINAL | STALLED | QUEUED | RESCUE | VALIDATING | AUTONOMOUS_SKILL | OOD_ESCALATED`), `BaySnapshot`, `ConsoleInputState`.
  - `src/teleop/compliance/protocol.py` & `solana.py`: `ComplianceEvent`, `ComplianceSink`, and thread-safe `SolanaCompliancePlaceholder` (non-blocking ring buffer + optional JSONL file writer).
  - `src/teleop/reasoning/protocol.py` & `nemotron.py`: `ReasoningContext`, `ReasoningVerdict`, `FleetReasoner`, and `VerdaNemotronPlaceholder` (deterministic offline triage generator reading `VERDA_*` env vars).
- [ ] **Step 6 (Verify & Commit):** Run `uv run python scripts/fetch_models.py`, then `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`. Commit `feat(infra): add ATC contracts, Panda model fetch, 16-DoF LEAP and placeholders`.

---

### Task T-110: 3-box sprung slide-bolt MJCF, 4-bay Panda+LEAP scene, DLS diff-IK & `mj_inverse`

**Parallel Track A (`depends_on: [T-100]`). Priority: P0.**

**Files:**
- Create: `src/teleop/sim/latch.py`, `src/teleop/sim/ik.py`, `src/teleop/sim/torques.py`, `src/teleop/sim/bays.py`, `tests/sim/test_latch_panda.py`.

**Interfaces & Requirements:**
- `latch.py`:
  - `add_latch_to_spec(spec, prefix: str, pose: LatchPose) -> None` building strictly 3 `<geom type="box">` collision bodies (`Base Track` $14 \times 8 \times 2\text{ cm}$, `Slide Bolt` range $[0, 6\text{ cm}]$ with hook $7\text{ cm}$ from flap, `Sprung Stopper Flap` $4 \times 4\text{ cm}$ with `stiffness=0.45, springref=+35 deg`), plus virtual `flap_catch` and `bolt_pin` constraints.
  - `LatchController(model, data, prefix)`: `reset(flap_jammed=False, bolt_pinned=False)`, `arm_flap_fault()` (one-way latched until next `reset()`), `bolt_x() -> float`, `is_cleared() -> bool` ($x_{\text{bolt}} \ge 0.052\text{ m}$).
- `ik.py`:
  - `PandaDiffIK(model, bay_prefix: str, damping: float = 0.05, nullspace_gain: float = 0.5)`: `solve(data, target_pose: Pose, dt: float) -> np.ndarray` returning 7 joint position targets for `bay{k}/actuator1..7` using `mj_jacSite` on `bay{k}/wrist_site`.
- `torques.py`:
  - `ExternalTorqueEstimator(model, bay_prefix: str)`: allocates one scratch `MjData` in `__init__` and computes `estimate(data) -> np.ndarray` ($\tau_{\text{ext}} = \tau_{\text{act}} - \tau_{\text{id}}$ via `mj_inverse`) without mutating contact forces in live `data`.
- `bays.py`:
  - `build_four_bay_scene(bay1_pose: LatchPose, bay2_pose: LatchPose, include_background_bays: bool = True) -> MjModel` attaching `panda_nohand.xml` + `leap_hand/right_hand.xml` for active Bays 1–2 and lightweight `contype=0, conaffinity=0` kinematic rigs for Bays 3–4. Also provide `build_single_bay_scene(pose: LatchPose) -> MjModel` for fast $<0.4\text{ s}$ headless validation rollouts in `T-140`.
- **Tests (`tests/sim/test_latch_panda.py`):**
  - [ ] `test_routine_flap_flat_allows_one_finger_bolt_slide`
  - [ ] `test_sprung_flap_blocks_bolt_around_35mm_unless_depressed`
  - [ ] `test_arm_flap_fault_is_one_way_latched_until_reset`
  - [ ] `test_pinned_bolt_fault_locks_bolt_at_zero`
  - [ ] `test_panda_dls_ik_tracks_rotated_and_shifted_latch_pose` ($\pm 35^\circ$ yaw, $\pm 12\text{ cm}$ shift)
  - [ ] `test_external_torque_spike_detected_on_flap_collision`
- [ ] Commit `feat(sim): add 3-box sprung latch, 4-bay Panda+LEAP scene, diff-IK and torque estimator`.

---

### Task T-120: Relative-clutch pose indexing, force bleed-off on release & split cameras

**Follows `T-110` (`depends_on: [T-100, T-110]`). Priority: P0.**

**Files:**
- Create: `src/teleop/retarget/clutch.py`, `tests/retarget/test_clutch.py`.

**Interfaces & Requirements:**
- `RelativeClutchIndexer`:
  - `update(clutch_pressed: bool, glove_wrist: Pose, current_sim_wrist: Pose) -> Pose | None`
  - On rising edge `False -> True`: anchors `(current_sim_wrist, glove_wrist)` and returns `current_sim_wrist` (zero discontinuity).
  - While `True`: applies scaled relative translation + relative quaternion delta in ergonomic desk frame.
- `bleed_off_targets(model, data, bay_prefix: str) -> tuple[np.ndarray, dict[str, float]]`:
  - Returns `q_target = q_actual` for all 7 Panda joints and 16 LEAP joints and zeroes commanded velocity immediately on clutch release.
- `RescueTrajectoryBuffer`:
  - Records `(t_s, wrist_in_latch_se3, leap_q16, tau_ext_inf)` while `clutch_pressed == True`; `finish(bolt_cleared: bool)` returns the clean payload if cleared or discards and returns `None` if released mid-jam.
- Split rescue camera helper configuring `top_ortho` (`orthographic=True` over active latch) and `side_ortho` cameras per active bay.
- **Tests (`tests/retarget/test_clutch.py`):**
  - [ ] `test_clutch_engage_has_zero_pose_jump`
  - [ ] `test_relative_translation_and_rotation_track_after_re_clutch`
  - [ ] `test_clutch_release_sets_qtarget_to_qactual_and_discards_incomplete_buffer`
  - [ ] `test_successful_clutch_rescue_transforms_wrist_into_local_latch_frame`
- [ ] Commit `feat(retarget): add relative-clutch pose indexing and force bleed-off`.

---

### Task T-130: UNO Q QRB2210 Python edge service + local mock ($1\text{€}$ filter, Mahalanobis gate, 30-RBF DMP)

**Parallel Track B (`depends_on: [T-100]`). Priority: P0. Pure NumPy + stdlib (zero MuJoCo/PyTorch imports so it deploys directly onto UNO Q Cortex-A53!).**

**Files:**
- Create: `src/teleop/unoq/__init__.py`, `src/teleop/unoq/filter.py`, `src/teleop/unoq/classifier.py`, `src/teleop/unoq/dmp.py`, `src/teleop/unoq/protocol.py`, `src/teleop/unoq/server.py`, `src/teleop/unoq/client.py`, `src/teleop/commands/unoq.py`, `tests/unoq/__init__.py`, `tests/unoq/test_edge_dmp.py`.

**Interfaces & Requirements:**
- **Dependency rule:** Nothing under `src/teleop/unoq/` may import `mujoco` or `torch` so the folder runs unmodified on Arduino UNO Q Linux (`QRB2210`, Python 3.12 + `numpy`).
- `filter.py`: `OneEuroFilter` (scalar/vector) + `PoseOneEuroFilter` (3D position + hemisphere-consistent quaternion smoothing) tuned for $83.3\text{ Hz}$ streams.
- `classifier.py`: `MahalanobisStallClassifier(ood_threshold=3.0, sigma_floor=(0.005, 0.4, 0.4, 0.4))`:
  - `register_prototype(part_sku_id: str, signature_id: str, samples: np.ndarray) -> None` (works from a **single** $1 \times 4$ rescue stall vector via `sigma_floor` shrinkage, and updates covariance as more samples arrive).
  - `classify(part_sku_id: str, z: np.ndarray) -> StallMatch`.
- `dmp.py`:
  - `fit_rbf_dmp(timestamps_s, wrist_latch_se3, fingers_q16, tau_ext_inf, n_kernels=30, ridge=1e-4) -> SkillPrimitive`: closed-form $30 \times 30$ ridge regression over canonical phase $s(t) = \exp(-4 t / T)$.
  - `evaluate_rbf_dmp(skill: SkillPrimitive, s: float, world_from_latch: Pose) -> tuple[Pose, dict[str, float], float]`: returns world wrist `Pose` (warped by `world_from_latch`), 16 named LEAP actuator targets, and nominal $\bar{\tau}_{\text{demo}}(s)$.
- `server.py` & `client.py` & `commands/unoq.py`:
  - `uv run teleop unoq --serve --host 0.0.0.0 --udp-port 15051 --tcp-port 15052` runs on UNO Q Linux.
  - `UnoQClient.connect_or_mock(host, udp_port, tcp_port)` connects to a live UNO Q if reachable within $150\text{ ms}$, else automatically starts an in-process localhost `UnoQServer` thread and logs `"unoq: using local loopback node"`.
- **Tests (`tests/unoq/test_edge_dmp.py`):**
  - [ ] `test_one_euro_filter_reduces_jitter_with_low_step_lag`
  - [ ] `test_rbf_dmp_fit_completes_under_20ms_and_reconstructs_trajectory`
  - [ ] `test_rbf_dmp_se3_warp_rotates_35deg_and_shifts_12cm_accurately`
  - [ ] `test_mahalanobis_matches_flap_jam_at_35mm_and_rejects_pinned_bolt_at_0mm_as_ood`
  - [ ] `test_unoq_client_roundtrips_over_local_loopback_server`
- [ ] Commit `feat(net): add UNO Q edge 1-Euro filter, Mahalanobis stall gate and 30-RBF DMP service`.

---

### Task T-131: UNO Q STM32U585 1 kHz console, fault rig & 8×13 LED matrix bridge

**Parallel Track C (`depends_on: [T-100]`). Priority: P1.**

**Files:**
- Create: `firmware/unoq_console/unoq_console.ino`, `src/teleop/unoq/mcu_bridge.py`, `tests/unoq/test_mcu_bridge.py`.

**Interfaces & Requirements:**
- `firmware/unoq_console/unoq_console.ino`:
  - $1\text{ kHz}$ debounced polling (`INPUT_PULLUP`) for Bay 1–4 select buttons, Dead-Man's Clutch, E-Stop, and Judge 3D-Printed Fault Rig switch.
  - **Strict One-Way Fault Rig:** Rising edge on Fault Rig emits `FAULT_RIG:LATCHED\n` once and ignores all subsequent edges until serial receives `CMD:RESET_RIG\n`.
  - Drives the Arduino UNO Q `8x13` LED Matrix divided into four $4 \times 3$ bay tiles (+ status column): breathing pulse in `NOMINAL`, fast blink in `STALLED`/`QUEUED`, solid indicator in `RESCUE`, horizontal sweep during `VALIDATING`.
- `src/teleop/unoq/mcu_bridge.py`:
  - `McuConsoleBridge` parsing serial lines (`BTN:BAY:1..4`, `CLUTCH:0|1`, `ESTOP:0|1`, `FAULT_RIG:LATCHED`) over `SerialLike` injectable transport (no `pyserial` required in unit tests; optional runtime serial port or software keyboard state).
- **Tests (`tests/unoq/test_mcu_bridge.py`):**
  - [ ] `test_fault_rig_latches_once_and_ignores_toggles_until_reset_cmd`
  - [ ] `test_clutch_and_bay_select_transitions_parsed`
  - [ ] `test_led_state_commands_formatted_per_bay`
- [ ] Commit `feat(net): add UNO Q STM32U585 console sketch and serial/mock bridge`.

---

### Task T-140: ATC Fleet Coordinator — $C(t)$, FIFO queue, Bay 2 `SE(3)` replay, additive torque tube, 3-rollout gate & Negative Control

**Integration Task (`depends_on: [T-110, T-120, T-130]`). Priority: P0/P1.**

**Files:**
- Create: `src/teleop/atc/coordinator.py`, `src/teleop/atc/validator.py`, `tests/atc/test_coordinator.py`.

**Interfaces & Requirements:**
- `compute_confidence(tau_ext_inf, v_cmd, v_act, alpha=0.8, beta=0.9, tau_limit=6.0, v_nom=0.05) -> float`: exact formula from Spec §4.1.
- `SkillValidator(hand_spec)` (`validator.py`):
  - Pre-builds one reusable headless single-bay `MjModel` + `MjData` so validating a new `SkillPrimitive` across 3 pose perturbations (`(-8 cm, -6 cm, -30°)`, `(0, 0, 0°)`, `(+8 cm, +6 cm, +30°)`) executes in $< 0.4\text{ s}$ total without recompiling XML!
  - Checks `bolt_x >= 0.052`, `max_tau_ext_inf < tau_hard_max`, and Panda joint limits.
- `AtcCoordinator` (`coordinator.py`):
  - Runs routine 1-finger unlatching on active Bays 1–2 ($C(t) > 0.85$ when unjammed).
  - Detects contact stall when $C(t) < 0.3$ for $300\text{ ms}$ (or $3.0\text{ s}$ step watchdog), with **zero access to `flap_hinge` `qpos`**.
  - Queries `UnoQClient.classify_stall(part_sku_id, z)`:
    - **Matched known signature + skill in Fleet Library (e.g. Bay 2 hitting `Jam_Sig_1` at $x \approx 3.5\text{ cm}$):** backs off $1\text{ cm}$ along local $-X$, switches to `AUTONOMOUS_SKILL`, warps $T_{\text{wrist}}^{\text{latch}}(s)$ to Bay 2's rotated/shifted latch frame ($35^\circ, 12\text{ cm}$), and executes DMP wrapped in the **Additive Residual Torque Tube**:
      $$\|\tau_{\text{ext}}(s) - \bar{\tau}_{\text{demo}}(s)\|_\infty \le \tau_{\text{margin}} \quad \text{and} \quad \|\tau_{\text{ext}}(s)\|_\infty < \tau_{\text{hard\_max}}$$
    - **Unseen / OOD signature (Bay 1 first jam, or Negative Control pinned bolt at $x = 0\text{ cm}$):** retracts $3\text{ cm}$ to zero-velocity safe hover, appends bay to **Single-Active-Rescue FIFO Hold Queue**, emits `STALL_ALERT` or `OOD_REFUSAL` to `ComplianceSink`, and submits `ReasoningContext` to `FleetReasoner`.
  - Enforces that **at most one bay** is in `RESCUE` at any instant; concurrent stalls wait safely in `QUEUED`.
  - On successful operator rescue + clutch release: calls `UnoQClient.fit_dmp`, runs `SkillValidator` (3 rollouts), emits `RESCUE_LOGGED` + `VALIDATION_GATE` to `ComplianceSink`, promotes approved skill + prototype to Fleet Library, and resets bay.
- **Tests (`tests/atc/test_coordinator.py`):**
  - [ ] `test_confidence_high_in_nominal_slide_and_below_0_3_on_300ms_flap_stall`
  - [ ] `test_additive_torque_tube_allows_free_space_zero_demo_torque_and_aborts_on_excess`
  - [ ] `test_single_active_fifo_queue_serializes_concurrent_bay_stalls`
  - [ ] `test_end_to_end_bay1_rescue_fits_dmp_passes_3_rollouts_and_bay2_clears_35deg_12cm`
  - [ ] `test_negative_control_pinned_bolt_at_x0_rejected_as_ood_and_escalates`
- [ ] Commit `feat(sim): add ATC fleet coordinator, 3-rollout validation gate and Bay 2 SE(3) replay`.

---

### Task T-150: `teleop atc` command, 4-bay HUD, audio prompts & 2.5-min stage rehearsal controls

**Final Demo Wiring (`depends_on: [T-140, T-131]`). Priority: P1.**

**Files:**
- Create: `configs/atc.yaml`, `src/teleop/atc/hud.py`, `src/teleop/atc/stage.py`, `src/teleop/commands/atc.py`, `tests/atc/test_stage_hud.py`.

**Interfaces & Requirements:**
- `uv run teleop atc [--source studio|synthetic|replay] [--unoq-host HOST] [--headless --steps N] [--hotspot]`
- `hud.py`: MuJoCo passive-viewer large-font overlay rendering:
  - 4-Bay status table (`BAY 1..4` state, $C(t)$ live bar/value, FIFO queue order)
  - Active Rescue HUD (selected bay, clutch state `HOLD CLUTCH` / `ENGAGED`, elapsed rescue timer)
  - Edge & Gate banner (`UNO Q DMP fit: 11.4 ms | Sim Gate: 3/3 PASS (0.28 s)`)
  - Latest `ComplianceSink` event hash & `FleetReasoner` triage headline
- `stage.py`: Keyboard shortcuts for rehearsing all 6 beats of the 2.5-minute stage script even without physical switches:
  - `1..4`: select bay from FIFO queue
  - `SPACE`: toggle/hold software clutch (mirrors hardware clutch)
  - `F`: arm Bay 1 Judge Fault Rig (one-way sprung flap trigger — Beat 2)
  - `G`: arm Bay 2 sprung flap ($35^\circ, 12\text{ cm}$ — Beat 5)
  - `P`: arm Negative Control pinned bolt ($x = 0\text{ cm}$ — Beat 6)
  - `R`: reset selected/all bays
- Optional non-blocking pre-recorded/synthesized audio status cues (`"Bay 1 stalled — queued"`, `"Skill validated — broadcasting to fleet"`, `"Out-of-distribution fault — refusing skill transfer"`), silent in `--headless`.
- **Tests (`tests/atc/test_stage_hud.py`):**
  - [ ] `test_atc_headless_scripted_6_beat_stage_run_completes`
  - [ ] `test_hud_formats_bay_states_confidence_and_fifo_queue`
- [ ] Commit `feat(cli): add teleop atc command, 4-bay HUD and 2.5-min stage controls`.

---

### Task T-160: Solana compliance & alert chain integration *(Teammate Task)*

**Parallel Track D (`depends_on: [T-100]`). Reserved for Solana teammate.**

**Files:**
- Modify: `src/teleop/compliance/solana.py` (and add `pyproject.toml` dependency only if needed, noted in PR).
- Create: `tests/compliance/__init__.py`, `tests/compliance/test_solana.py`.

**Interfaces & Requirements:**
- Implement `SolanaComplianceSink(ComplianceSink)` inside `src/teleop/compliance/solana.py` without changing `ComplianceEvent` or `ComplianceSink` signatures in `protocol.py`.
- Anchor SHA-256 digests + structured metadata of `STALL_ALERT`, `RESCUE_LOGGED`, `VALIDATION_GATE`, `SKILL_DEPLOYED`, and `OOD_REFUSAL` asynchronously on a background worker thread using `SOLANA_RPC_URL` and `SOLANA_KEYPAIR_PATH`.
- Must never block `emit()` ($< 0.5\text{ ms}$ enqueue) and must fall back cleanly to local signed/JSONL audit logging if RPC is unreachable during live stage demo or CI.

---

### Task T-161: Verda Nemotron fleet reasoning co-pilot *(Teammate Task)*

**Parallel Track E (`depends_on: [T-100]`). Reserved for Verda/Nemotron teammate.**

**Files:**
- Modify: `src/teleop/reasoning/nemotron.py`.
- Create: `tests/reasoning/__init__.py`, `tests/reasoning/test_nemotron.py`.

**Interfaces & Requirements:**
- Implement `VerdaNemotronReasoner(FleetReasoner)` inside `src/teleop/reasoning/nemotron.py` without changing `ReasoningContext`, `ReasoningVerdict`, or `FleetReasoner` signatures in `protocol.py`.
- Call the Verda-hosted Nemotron endpoint (`VERDA_BASE_URL`, `VERDA_API_KEY`, `NEMOTRON_MODEL`) on a background worker thread (`urllib.request` or `httpx` if added) to produce concise root-cause stall analysis, operator rescue guidance, OOD refusal explanations, and compliance rationales.
- Must never block `submit()` on the $60\text{ Hz}$ loop and must retain `VerdaNemotronPlaceholder` deterministic fallback when `VERDA_API_KEY` is absent or requests time out.
