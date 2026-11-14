# ACT Recovery Policy: Hotfix-then-Release Fleet Learning

Date: 2026-11-14. Builds on the ATC pivot (`docs/superpowers/specs/2026-11-14-atc-fleet-rescue-design.md`,
ADR 0002) and the fleet ops console (`docs/superpowers/specs/2026-11-14-fleet-ops-console-design.md`).

## Intent

Today a human glove rescue teaches the fleet one thing: a DMP (a fitted replay of that one
motion, relative to the latch). It reaches every bay within minutes, but it only replays. It
cannot see or react, it doesn't get better with more rescues, and two different jams that
produce similar force readings get the same replay.

We want the robots running **trained policies** learned from the collected rescue data, in a way
that would carry over to real hardware. The chosen model is **ACT** (Action Chunking
Transformer, the policy from the ALOHA work, available in the installed LeRobot 0.7). It is
trained on a remote GPU cluster (Verda) and runs on the laptop.

The DMP stays. The fleet learns on two timescales:

- **Hotfix (minutes, unchanged):** the first rescue of a new failure becomes a DMP, gated by the
  3 rollouts and shared with the whole fleet.
- **Release (hours):** every rescue is also saved as an ACT training episode. Periodically ACT
  is retrained on the cluster on everything collected. Once a checkpoint passes the gate for a
  failure type, it takes over from that failure's DMP.

The stall fingerprint (Mahalanobis classifier) stays the gatekeeper for both. It decides whether
a bay attempts recovery itself or calls the operator.

**Fallback is a first-class requirement.** If ACT doesn't work, one config value (or one key
press on the console) puts the system back on today's DMP behaviour, unchanged.

## Goals

1. **Collect:** every successful recovery (glove rescue, scripted rescue, autonomous DMP) can be
   saved as a LeRobot v3 episode with cameras, robot state, forces and latch-frame actions,
   plus a sidecar with signature and outcome labels.
2. **Generate:** a headless generator produces hundreds of scripted rescue episodes with
   randomised latch poses, fault types and operator noise, to bootstrap training before many
   glove demos exist.
3. **Clean and label:** `teleop process` QCs, trims and labels these datasets and merges
   sessions into one training set.
4. **Train remotely:** one documented command trains ACT on a Verda GPU instance from the merged
   dataset and brings back a checkpoint.
5. **Gate offline:** `teleop eval` runs the checkpoint through headless rollouts per failure
   signature (the same perturbations as the DMP gate, more rollouts) and writes a release
   manifest listing the signatures it passed.
6. **Run live:** `teleop atc --recovery-backend hybrid` uses ACT for gated signatures and the DMP
   for the rest, inside the existing safety checks, without stuttering the 60 Hz loop or the
   30 fps console.
7. **Fall back:** `--recovery-backend dmp` (the default) reproduces today's behaviour exactly.
   Any ACT failure downgrades to the DMP, then the operator.

## Non-goals

- SmolVLA or other VLAs (deferred until there are several tasks to pick between by instruction).
- ACT running the routine unlatch work (option B). It only does recovery after a stall.
- Turning Bays 3-4 into simulated bays. They stay console-only replays.
- Retraining during the live demo, online learning or RL.
- Running inference on Verda. Verda is for training only.
- Real-hardware deployment in this round. The design must not block it (see Constraints).
- Changing the DMP fit, the classifier, C(t), the stall detector or the FIFO queue.

## Anti-goals (failure even if it works)

- The `dmp` backend behaves differently from today in any way, or needs `torch` installed.
- ACT runs on a failure signature its checkpoint hasn't passed the gate for.
- The live loop drops below 60 Hz or the console feeds hitch because of inference, rendering
  for the policy, or episode encoding.
- The policy reads simulator-only state (`bolt_x`, `flap_hinge`, the latch controller's fault
  flags) as an input. Then it could never run on a real robot.
- Actions in world coordinates. Then one policy can't serve every bay or a real latch.

## Constraints

- Global Constraints of `docs/superpowers/plans/2026-11-13-m0-m1-glove-to-sim.md` apply:
  Python 3.12, uv only, xyzw quaternions, joints and actuators by name, no hardware in tests.
- **Do not edit `src/teleop/cli.py`.** Use the existing `atc`, `process` and `eval` command
  slots. Training is a script (`scripts/train_act.py`) that runs on the cluster.
- `teleop atc` in `dmp` mode must not import `torch` or `lerobot`. ACT code loads lazily.
- The laptop is an RTX 4060 Laptop GPU (8 GB), i7-12650H, 16 GB RAM. The current venv has a
  **CPU-only** PyTorch build (`torch.cuda.is_available() is False`), so a CUDA build must be
  configured through uv for the ACT groups.
- The 60 Hz control loop owns the main thread. Inference and video encoding must not block it.
- Tests run without a GPU, without lerobot training and without network. ACT runtime tests use
  a stub policy with the same interface.
- A team demo is scheduled for 2026-11-15 10:00. Every new path is off by default, so this
  work can land at any point without risking it.

## Approaches considered

| Approach | Trade-off | Decision |
|---|---|---|
| **DMP only** (today) | One demo, minutes to deploy, deterministic. It can't see or adapt and doesn't improve with more data. | Kept as the hotfix and the fallback |
| **ACT only** | Reacts to cameras and forces and combines many demos. It needs about 20-50 demos per failure and hours to retrain, and has no built-in "out of my depth" check. | Rejected: new failures would wait hours with no automated recovery |
| **Hotfix-then-release hybrid** | DMP covers new failures in minutes, ACT replaces it once trained and gated. The classifier gates both. More moving parts. | **Chosen** |
| SmolVLA | Language conditioning adds little with one task. Pretrained on other arms and grippers, so it mostly learns from our demos anyway. Can't easily take force readings. | Deferred |
| Inference on Verda | Frees the laptop GPU but adds network latency to a contact task, and LeRobot 0.7 here has no async inference server. | Rejected: one ACT on the RTX 4060 is fast enough |

## Design

### 1. Recovery backends and the toggle

A new seam, `src/teleop/atc/recovery.py`, defines how a bay recovers after a matched stall:

```python
class RecoveryController(Protocol):
    name: str  # "dmp" | "act"

    def start(self, bay: BayContext) -> None: ...
    def command(self, bay: BayContext, t_s: float) -> RecoveryCommand | None: ...
    def check(self, bay: BayContext, t_s: float) -> str | None: ...  # abort reason or None
```

- `DmpRecovery` wraps today's path **without changing it**: 1 cm back-off, blend, `SkillReplay`,
  torque tube, skill timeout. The coordinator's current `_start_autonomous`, `_control_auto`
  and `_sense_auto` logic moves behind it unchanged.
- `ActRecovery` runs the trained policy (section 5).

Selection, in `configs/atc.yaml` with matching CLI flags:

```yaml
recovery_backend: dmp        # dmp (default) | hybrid | act
act_checkpoint: null         # models/act/<run>/ (contains the checkpoint and gate.json)
record_episodes: null        # directory: save recoveries as LeRobot episodes (off when null)
```

| Backend | On a matched stall | On ACT abort |
|---|---|---|
| `dmp` | Today's DMP replay, byte-for-byte | n/a |
| `hybrid` | ACT if the signature is in `gate.json`, else the DMP if one is deployed | DMP if deployed, else operator queue |
| `act` | ACT if the signature is in `gate.json`, else operator queue | Operator queue |

Unmatched or out-of-distribution stalls always go to the operator queue, in every mode. DMPs
are still fitted, gated and deployed after every rescue in every mode, so switching back to
`dmp` never loses the hotfix library.

**Live toggle:** the `M` key (console page and MuJoCo viewer) cycles `dmp → hybrid → act`.
It takes effect at the next recovery start. A running recovery finishes on its backend. The
console shows the current backend in the telemetry strip and the key bar, and every switch
emits a `BACKEND_CHANGED` compliance event.

**Automatic downgrade to `dmp`** happens, with a console toast and a compliance event, if:

- the checkpoint or `gate.json` is missing, unreadable or doesn't match the observation and
  action schema;
- the inference worker crashes or returns no action chunk within 250 ms;
- measured inference latency stays above budget (section 5) for 10 consecutive calls.

### 2. What the policy sees and does

Everything is relative to the bay's latch frame, so one policy serves every bay and maps onto a
real latch whose pose comes from perception.

**Observation** (`observation.state`, by name, float32):

| Group | Values | Real-world source |
|---|---|---|
| Arm joint positions | 7 (`joint1..7`) | Panda encoders |
| Hand actuator positions | 6 (`HAND_ACTUATORS` order) | Inspire position feedback |
| Wrist pose in latch frame | 7 (x, y, z, qx, qy, qz, qw) | Forward kinematics + latch perception |
| Hand external torque per actuated joint | 6 | Inspire per-finger force readings |
| Wrist external torque | 3 (`joint5..7` of `tau_ext`) | Panda torque sensing |

Not included, on purpose: `bolt_x`, `flap_hinge`, latch fault flags, C(t). They are sim-only or
derived. `bolt_x` is still recorded in the sidecar for labels and QC.

**Cameras** (`observation.images.*`):

- `wrist`: a new camera mounted on the Inspire hand base, looking along the fingers. This is
  the camera that carries over to real hardware.
- `top`: the existing latch-relative orthographic rescue camera (`rescue_camera_names()`).

Resolution is the builder's call within 224 px on the long side. Start small (for example
128 × 128) and raise it only if eval shows it helps.

**Action** (`action`, 12 values per step): target wrist position in the latch frame (3), target
wrist rotation as a rotation vector relative to the routine's start orientation in the latch
frame (3), and 6 hand actuator targets. The coordinator composes these into a world wrist pose
with `compose_pose(world_from_latch, ·)` and sends them through the existing IK and position
servos, exactly as a glove command would.

### 3. Episodes: what gets recorded

One episode is one recovery attempt, from the moment the bay hands over (clutch first engaged
for a human, first commanded tick for a scripted or DMP recovery) to bolt cleared or abort.

Episodes start where the human demos start: at the hover pose after the 3 cm retract. So in
`hybrid` and `act` modes, a matched stall routed to ACT first does the same 3 cm retract and
settle as an escalation, then starts the policy. The training and live start states match.

Recorded at 30 fps (every second 60 Hz tick):

- observation and action as above (the action is what was actually commanded: glove after
  clutch indexing, scripted target, or DMP output);
- `task` string `"recover latch: <signature_id>"` (required by LeRobot, and it keeps a later VLA
  option open).

Sidecar (`meta/teleop_episodes.jsonl`, the same mechanism `teleop process` already reads),
one line per episode:

```json
{"episode_index": 12, "source": "glove|scripted|dmp|act", "signature_id": "Jam_Sig_1",
 "signature": [0.035, 1.2, 2.4, 0.3], "fault": "flap|pin", "bay": 1,
 "latch_pose": {...}, "success": true, "bolt_x_final": 0.054,
 "max_tau_hand_inf": 2.1, "max_arm_ratio": 0.42, "aborted_reason": null}
```

Failed and aborted attempts are recorded too (`success: false`). They are excluded from
training by default and kept for analysis and later failure-aware training.

**Where recording hooks in:** the coordinator already records the clutch hold into
`RescueTrajectoryBuffer`. A new `EpisodeSink` observer (`src/teleop/atc/episodes.py`) gets a
per-tick callback with the bay, its phase and the commanded action, and receives
start/finish events. The coordinator calls it only when `record_episodes` is set. Camera
rendering for episodes uses its own offscreen renderer at 30 fps. Encoding uses LeRobot's
parallel encoding off the control thread, so recording during a live session doesn't block the
loop. Each live session writes a fresh dataset directory: `data/episodes/<YYYYmmdd-HHMMSS>/`.

### 4. Scripted episode generator

`teleop atc --headless --generate N --record-episodes DIR` runs the coordinator headless:

- it arms a random fault (Bay 1 flap, Bay 2 flap, Bay 2 pin) each cycle;
- it perturbs each bay's latch pose within the DMP gate's perturbation ranges and beyond;
- it plays `ScriptedGlove` with randomised timing, approach offsets and small pose noise as the
  operator, holding the clutch on whichever bay is queued;
- it stops after N successful episodes.

The DMP path stays active (`dmp` backend), so autonomous DMP recoveries of known signatures are
also recorded with `source: dmp`. Generated data is labelled `source: scripted`, and training
can weight or exclude sources.

### 5. ACT runtime on the laptop

- **One model, batched.** One ACT checkpoint is loaded once (lazily, when the backend first
  needs it). All bays currently in ACT recovery are stacked into one forward pass.
- **Off the control thread.** An inference worker thread holds the model on CUDA. The control
  loop posts observations and reads back action chunks. It never waits. Until a bay's first
  chunk arrives it holds its pose.
- **Chunked execution.** The policy is queried every k ticks (the builder picks k, about 4-6, so
  10-15 Hz), and the bay plays the latest chunk in between. Temporal ensembling is the
  builder's call if eval shows it helps.
- **Budget:** p95 inference ≤ 30 ms for a batch of 2 on the RTX 4060. Above this for 10
  consecutive calls, the backend downgrades (section 1). CPU inference is allowed for tests and
  debugging but warns at load.
- **Safety, same layers as today:**
  - the IK speed limit and arm torque margin;
  - the hand hard torque limit `tau_hard_max`;
  - a recovery timeout;
  - a no-progress watchdog: C(t) below the stall threshold for the same 300 ms as the routine
    stall detector.

  The DMP torque tube needs a demo torque profile, which ACT doesn't have. ACT uses the hard
  limit plus the watchdog instead. Any of these aborts the recovery (section 1 decides what
  happens next).

### 6. Cleaning and labelling

`teleop process` gains an ATC mode (detected from the sidecar's `source` field):

- **Drop:** not successful, shorter than 0.5 s, gaps over 0.2 s, any torque or arm-ratio
  breach, actions outside actuator ranges.
- **Trim:** idle frames at the start (clutch engaged but no motion) and the hold after bolt
  clear.
- **Label:** carry `signature_id`, `fault` and `source` into the clean dataset's sidecar.
  `--sources glove,scripted,dmp` selects which sources go in. A summary prints counts per
  signature and source.
- **Merge:** several session directories into one training dataset (as today).

### 7. Training on Verda

- `scripts/train_act.py --dataset DIR --out models/act/<run>` wraps `lerobot-train` with an ACT
  preset for this schema (chunk size, image size, batch size, steps, seed) and writes a
  `run.json` with the dataset hash, schema and git commit.
- A uv dependency group `act` adds the LeRobot training extras and a CUDA PyTorch build for
  Linux (the cluster) and Windows (the laptop) through a uv index. The default sync stays as is.
- `docs/training/verda.md`: the end-to-end runbook:
  1. Install the Verda CLI (`verda doctor`).
  2. Create API credentials in the Verda dashboard. The person running it enters them; they are
     never committed.
  3. Register `~/.ssh/id_ed25519.pub`.
  4. Pick an instance type (`verda instance-types`).
  5. `verda vm create`.
  6. `verda ssh`: clone, `uv sync --group act`.
  7. Upload the dataset (rsync, or a private HF Hub dataset).
  8. Train.
  9. Copy the checkpoint back.
  10. **Delete the instance.**

  Cost and the delete step are called out first, because idle GPU instances burn credits.

### 8. Offline gate and release manifest

`teleop eval --checkpoint models/act/<run> [--signatures ...] [--rollouts 10]`:

- Generalises `SkillValidator` to take any `RecoveryController`, not just a `SkillPrimitive`.
- Per signature, runs headless single-bay rollouts from the post-retract hover, across the DMP
  gate's latch perturbations plus random ones. Each rollout uses the same pass rules as today:
  bolt cleared, hand torque below `tau_hard_max`, arm ratio below the margin.
- Default pass bar: at least 9 of 10 rollouts per signature. This is stricter than the DMP's 3
  of 3 because the policy covers more variation.
- Reports success rate, time-to-clear and peak torques next to the DMP for the same signature,
  and the inference latency measured on this machine.
- Writes `models/act/<run>/gate.json`: passed signatures, the bar used, per-signature results,
  the checkpoint hash and schema version. The live runtime trusts only signatures listed there.

The gate runs offline, never inside the live loop: rendering cameras and running ACT for
dozens of rollouts would stall the demo.

## Success criteria

1. `teleop atc --script --headless` passes every beat with `recovery_backend: dmp`, and its
   compliance event sequence matches today's.
2. In `dmp` mode, `teleop atc` runs in a venv without `torch`.
3. The generator produces 300 or more successful scripted episodes headless. `teleop process`
   cleans and merges them with glove sessions.
4. One Verda training run, following only `docs/training/verda.md`, produces a checkpoint.
5. `teleop eval` passes that checkpoint for at least the flap-jam signature. It reports success
   rate and time-to-clear next to the DMP.
6. Live `hybrid` with ACT active in two bays at once holds 60 Hz and 30 fps console feeds. p95
   inference ≤ 30 ms.
7. Pressing `M` back to `dmp`, or deleting the checkpoint, returns to today's behaviour with no
   restart.

## Left to the builder

- ACT hyperparameters (chunk size, k, temporal ensembling, image size, steps), within the
  latency budget.
- Exact wrist-camera mount pose on the Inspire base.
- Instance type on Verda and whether the dataset moves by rsync or a private HF Hub repo.
- Task split and ordering for the plan, following the file-ownership style of the ATC plan.

## Calls made without explicit sign-off (check these)

- Default `recovery_backend: dmp` until a checkpoint passes `teleop eval`. Then flip
  `configs/atc.yaml` to `hybrid`.
- In `hybrid`, an ACT abort falls back to the DMP before the operator.
- ACT recoveries start after the same 3 cm retract as human rescues (not the DMP's 1 cm
  back-off), so live start states match the training data.
- Policy inputs exclude `bolt_x` and other sim-only state.
- Gate bar of 9 of 10 rollouts per signature.
- Episode recording off by default, so the demo laptop isn't encoding video unless asked.
- `M` key for the live backend toggle.
