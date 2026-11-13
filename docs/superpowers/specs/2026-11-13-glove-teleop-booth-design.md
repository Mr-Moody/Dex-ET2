# Glove teleop booth + data pipeline — design (M0–M3)

Date: 2026-11-13 (Fri) · Deadline: **Sun 2026-11-15 10:00** — final code submission, booth demos start immediately after (hard freeze) · Status: draft for approval
Source brief: `docs/KICKOFF_PROMPT.md` (this spec overrides it where they differ; see §9).

## 1. Intent

Entry for Rokoko's Junction challenge ("use mocap gloves to control robot hands… surprise us").
Judging is booth-style peer review: people walk up to the table, **put on the Smartgloves II and
try it**. The physical, hands-on experience is the priority.

Underneath the demo sits a **robot-agnostic teleop data pipeline** — capture → calibrate →
record → clean/process → train → evaluate — that the team will **specialise later** (ideas
floated, *not* to be built now: learning skills across countries via teleop; pressure
sensing via an Arduino Uno Q). Automated data cleaning with a VLM (e.g. Gemma) is a bonus.

## 2. Goals, non-goals, anti-goals

**Goals**
1. Walk-up booth: a stranger puts on **both gloves (left + right)**, does a ~15 s guided calibration, and drives
   **two** simulated dexterous hands in MuJoCo through a short scored mini-challenge (pick up / stack).
2. Every booth attempt is auto-recorded as a LeRobot v3 episode (crowd-sourced demos).
3. Cleaning/processing stage with rule-based QC and a pluggable annotator hook (VLM stretch).
4. Train ACT on the collected episodes (overnight Sat) and show an autonomous "attract mode"
   rollout when nobody is wearing the gloves.
5. Laptop-managed Wi-Fi hotspot that the gloves connect to, kept alive by our code.
6. Stretch: the same retargeted Inspire-hand commands drive the **real** Inspire hand (Aalto
   integration / G1) if that hardware is free on Saturday.

**Non-goals (v1):** Arduino Uno Q / pressure sensing (hook only), cross-country teleop, arm IK
with mink, full-body Smartsuit, Hub pushes, Diffusion Policy (ACT only).

**Anti-goals (failure even if it "works")**
- Demo needs a laptop restart / terminal fiddling between visitors.
- Gloves silently drop because the hotspot timed out.
- Hard-coding the robot, task, or input source so later specialisation means a rewrite.
- Any test/CI path requiring the gloves or the hotspot.

## 3. Constraints and verified facts

- **Platform:** Windows 11 laptop, RTX 4060 Laptop (8 GB). uv 0.9.7. Python **3.12** pinned.
  MuJoCo viewer runs natively on Windows (`launch_passive`). CI (GitHub Actions, Linux) uses
  `MUJOCO_GL=egl`.
- **Hardware on hand:** Smartgloves II **left and right**; Coil Pro present but **untested** — assume it
  works, but everything must work without it (contingency in §5.7). **Rokoko Studio is set up
  and gloves verified** over the laptop hotspot. SDK licence: obtainable via the event form but
  not confirmed — so Studio is primary.
- **Rokoko Studio Custom Streaming:** UDP, JSON v3, default port `14043`, optional LZ4.
  Exact JSON field names for glove joints are **not verified** — the first M1 task captures real
  packets to `data/raw/` and builds the parser + fixtures from them.
- **Rokoko Device SDK (secondary adapter):** driver RGMP on `127.0.0.1:12276` streams
  **per-sensor orientations, not joints** (5 finger sensors + hub + forearm, + world forearm
  pose with Coil Pro). Joints come from the `rkk-hand-solver` sidecar (`127.0.0.1:12277`,
  26-joint, OpenXR `XrHandJointEXT` order) or the pure-Python solver example. RGMP: TCP,
  little-endian, `[u32 prefix][u32 len][payload]`, prefix 1=definition JSON, 2=data
  (`u32 device_id, u32 group_id, u64 timestamp_us` + packed values), 3=disconnect;
  quaternions xyzw; protocol v2. RCSP commands on TCP `45451` (`ListDevices`,
  `SubscribeToData`). Gloves rate ≤ 83 Hz. Studio and the SDK driver are assumed mutually
  exclusive on the gloves.
- **LeRobot:** `lerobot>=0.4` (dataset v3). `LeRobotDataset.create(repo_id, fps, features, root, …)`
  → `add_frame(frame)` with `"task"` key in the frame → `save_episode()` → **`finalize()` is
  mandatory**. Verify signatures against the installed source before coding.
- **Robot models:** Menagerie has `leap_hand`, `wonik_allegro`, `shadow_hand`, … — **no Inspire**.
  Inspire RH56DFX MJCF: open-sourced alongside arXiv 2603.08988 (sim2real-validated); 6
  actuators / 12 joints. Real hand control (Unitree): DDS topic `rt/inspire/cmd`.
- **Time:** ~35 h total including sleep. Everything is scoped against that. No visitors before Sunday —
  Saturday data comes from the team and other hackers.
- **Team:** 3 people on 3 laptops. **This laptop is the hardware host** (gloves, Studio, hotspot).
  Teammates develop against Replay/Synthetic sources, and can also join the hotspot and receive
  live data: Studio Custom Streaming can target another laptop's IP. Claim refs are pushed
  to `origin` (`github.com/Mr-Moody/Dex-ET2`) so all three machines see them; GitHub Actions is
  assumed enabled.
- **Shared captures:** small, trimmed raw Studio captures are committed under `tests/fixtures/`
  (the exception to "no raw recordings in git"); full captures/datasets are shared out of band.

## 4. Approaches considered

- **A. Sim-first walk-up booth.** All in MuJoCo; fewest external dependencies.
- **B. Real robot first** (Inspire/G1). Most physical wow, but shared hardware, availability
  and safety risk with strangers; pipeline starves.
- **C. A + real-Inspire output as stretch.** ✅ **Chosen.** Same core as A; retargeting outputs
  Inspire-shaped commands consumed by a `RobotSink` that can be the sim *or* the real hand.

## 5. Decided design

### 5.1 Architecture

```
gloves ─Wi-Fi(hotspot)─► Rokoko Studio ─UDP JSON─► StudioSource ┐
                         rokoko-sdk/solver ─RGMP─► SdkSource    ├► InputSource → HandFrame
                          data/raw replay ───────► ReplaySource │   (latest-value buffer,
                          scripted ──────────────► SyntheticSource┘    own thread)
                                                     │
            WristSource (Coil Pro pose | keyboard | fixed) ──┤
            AuxSensorSource (hook, none in v1) ──────────────┤
                                                     ▼
                         Calibration (per-user min/max) → Retargeter → JointTargets (by name)
                                                     ▼
                         RobotSink: SimSink (MuJoCo) | InspireRealSink (stretch)
                                                     ▼
                         Booth app: task/scene, score, HUD, auto-record → LeRobot episode
                                                     ▼
                         Process: QC rules → Annotator hook (rule | VLM) → clean dataset
                                                     ▼
                         lerobot-train ACT → eval rollout / attract mode
```

### 5.2 Shared contracts (`src/teleop/types.py` + protocols; merged first, everything depends on them)
- `HandFrame`: device timestamp, host `monotonic_ns` receive time, side (L/R) — one per hand; a
  `BimanualFrame` pairs the latest L and R, per-joint
  rotations keyed by **canonical joint name** (OpenXR hand joint names), optional wrist pose,
  source id, sequence number.
- `JointTargets`: `dict[str, float]` keyed by robot joint/actuator name + timestamp.
- Protocols: `InputSource`, `WristSource`, `AuxSensorSource`, `Retargeter`, `RobotSink`,
  `Task` (scene + reset + success/score), `Annotator`.
- `record/schema.py`: the single definition of LeRobot features.
- Everything is selected by config (draccus dataclasses + YAML in `configs/`): source, hand,
  task, sink. **No robot/task/source hard-coded.** New tasks/hands register via a small registry
  — this is the specialisation seam.

### 5.3 Network / hotspot (`src/teleop/net/hotspot.py`, Windows only)
- Manages Windows Mobile Hotspot via WinRT `NetworkOperatorTetheringManager`.
- SSID **`Router? I barely know 'er`**, passphrase from `.env` `HOTSPOT_PASSWORD` (gitignored;
  `.env.example` committed), band **2.4 GHz** (check `IsBandSupported`; fail loudly).
- Disables the "turn off when no devices connected" timeout.
- `uv run teleop hotspot {up,status,down}`; a keep-alive watchdog thread (poll state +
  client count, restart if off) runs standalone or inside `sim`/`booth`.
- Drops and reconnects are logged into session metadata for QC.
- Mocked in tests/CI.

### 5.4 Input
- **StudioSource (primary):** UDP listener, JSON v3 (+LZ4), parses glove joints into HandFrame.
  Raw packets optionally teed to `data/raw/` for replay.
- **SdkSource (secondary):** typed RGMP/RCSP client per §3; consumes the hand-solver output.
- **ReplaySource / SyntheticSource:** for dev and CI.
- Input I/O on its own thread with a latest-value buffer; tolerate late/non-monotonic
  timestamps; auto-reconnect.

### 5.5 Calibration (booth-friendly)
- ~15 s on-screen guided flow: "flat hand" (hold 2 s) → "fist" (hold 2 s) → done. Stores per-joint
  min/max per user/session on top of Studio's own straight-pose calibration. Skippable with
  defaults. Saved with the episode metadata.

### 5.6 Retargeting
- `DirectRetargeter` first: calibrated finger flexion → robot joints **by name**, per-hand YAML
  mapping. Inspire's 6 actuators (thumb rotation, thumb bend, 4 finger curls) make this direct.
- `dex-retargeting` adapter is optional later, behind the same interface.

### 5.7 Sim + booth app
- Default hand **Inspire** (MJCF from the arXiv 2603.08988 release); LEAP (Menagerie, submodule or
  fetch script) selectable via config.
- **Two** floating hands (left + right models, mirrored where the hand has no left variant).
- Wrist from Coil Pro world pose (primary). **Contingency without Coil Pro:** wrist *orientation* from
  the glove hub/forearm IMU, wrist *position* held at a per-hand home pose above the table with
  keyboard nudge (and optional height-from-forearm-pitch heuristic). Selected by config/auto-detect
  (no positions in the stream → contingency), shown on the HUD.
- Fixed-rate loop: 60 Hz control, ≥500 Hz physics via substeps, `time.monotonic_ns()`.
- Mini-challenge task(s): v1 "pick up the cube and lift it above a line" (either hand) with timer and score;
  table, cube, offscreen camera(s) for recording.
- HUD/status: gloves connected, hotspot up, calibration state, score, recording indicator.
- Booth loop: idle/attract → calibrate → play (auto-record) → score screen → idle. No terminal
  interaction between visitors.
- End-to-end latency (glove packet receive → actuator command) measured and logged; target
  < 50 ms.

### 5.8 Recording
- Each attempt = one LeRobot v3 episode. Schema is **always bimanual** (fixed, so it never changes
  mid-event): `observation.state` = L+R hand joint positions **plus L+R wrist pose**,
  `observation.images.<cam>`, `action` = L+R joint targets **plus L+R wrist pose targets** (a policy
  must be able to move the hands to lift anything), `task` string; episode metadata
  includes calibration, source, hotspot/glove dropout events, score/success.
- Local datasets only (`data/`, gitignored). `finalize()` on shutdown, including on Ctrl-C.

### 5.9 Processing
- `uv run teleop process`: rule-based QC (input dropouts, timestamp gaps, out-of-range joints,
  too short, no task progress) → trim idle head/tail → write a cleaned dataset + QC report.
- `Annotator` hook: v1 ships a rule-based annotator; a VLM annotator (e.g. Gemma) that
  labels success/failure or rejects episodes from camera frames is a stretch task.

### 5.10 Train + eval
- `lerobot-train --policy.type=act` on the cleaned dataset (`train` group, CUDA).
- `uv run teleop eval`: rolls out the policy in the same task, reports success rate; the
  booth's attract mode uses this rollout. **Fallback (builder's call, approved path):** if the
  policy isn't convincing by Sunday, attract mode replays the best recorded episode.

### 5.11 Real Inspire (stretch)
- `InspireRealSink` consuming the same `JointTargets` (actuator names → Inspire command
  range). Transport (Aalto integration vs Unitree DDS `rt/inspire/cmd`) decided on Saturday
  once we see what's available.

## 6. Process (lightened for 35 h)
- Three builders: the lead agent on this laptop owns hardware-touching tasks (hotspot, Studio
  capture, live latency check); teammates' agents take hardware-free tasks first.
- Superpowers flow kept: spec → plan → tasks → worktrees → subagents with TDD → verification.
- Task files + `uv run task` CLI + atomic claim refs (`git update-ref refs/claims/T-NNN HEAD ""`)
  as in brief §5 — kept, because parallel agents need it.
- Conventional Commits, branch naming, PR template as in brief §4.
- **Dropped:** separate reviewer-agent pass per PR. A PR merges (squash) once CI is green;
  the lead agent does a quick check. Branch protection / required review on GitHub not
  enforced for the weekend.
- Repo visibility, licence, and Hub pushes: ask the user before any change.

## 7. Schedule (targets)
- **M0 — Fri night:** uv scaffold, tooling, CI, CLAUDE.md, CONTRIBUTING.md, PR template, task CLI
  with tests, contracts, hotspot module, raw Studio packet capture. Seed M1 tasks.
- **M1 — Sat midday:** gloves → Studio → sim hand live with calibration, latency logged.
  Parallel: input, sim/scene, retarget+calibration, recorder, tooling.
- **M2 — Sat afternoon/evening:** booth loop, mini-challenge, auto-record, processing/QC.
  Collect episodes from visitors/teammates.
- **M3 — Sat night (train overnight):** ACT trained, eval + attract mode.
- **Sun morning:** freeze, rehearse demo, submit by 10:00 — demos start immediately, so the
  submitted code *is* the demo build. Stretch items only if M2 is solid.

## 8. Left to the builder
- Exact Studio JSON parsing (from captured packets), joint name canonicalisation details.
- Scene composition, camera placement, image resolution (keep ACT-trainable on 8 GB).
- HUD tech (MuJoCo viewer overlays vs a small window) — must be readable from 1–2 m.
- How to vendor the Inspire MJCF (submodule vs fetch script) and its licence check; if the model
  can't be obtained and loaded within ~1 h, ship LEAP as default and note it in an ADR.
- QC thresholds, ACT hyperparameters.

## 9. Deviations from the kickoff brief
- Studio UDP is **primary**, SDK secondary (brief had the opposite).
- SDK driver streams sensor orientations; joints need the hand-solver (brief implied joints).
- Default hand Inspire (not in Menagerie), LEAP fallback (brief: LEAP/Shadow).
- Wrist: Coil Pro (brief: open question).
- Added: hotspot management, booth app, processing/QC stage + annotator hook.
- Removed for v1: Diffusion Policy, mink/arm, per-PR reviewer agent.
