# Project Brief — Air Traffic Control for Robot Hands

> **Active Architecture Docs:**
> - Engineering Spec: `docs/superpowers/specs/2026-11-14-atc-fleet-rescue-design.md`
> - Multi-Agent Implementation Plan: `docs/superpowers/plans/2026-11-14-atc-implementation-plan.md`
> - Architecture Decision Record: `docs/adr/0002-atc-fleet-rescue-pivot.md`
> - Active Task Checklist: `tasks/T-100-*.md` through `tasks/T-161-*.md` (`uv run task list --ready`)

---

## Proposal — Air Traffic Control for Robot Hands

### 1. Problem and Concept

Disassembling EV battery modules and e-waste requires multi-fingered dexterity: depressing a sprung safety flap with a thumb while sliding a retention bolt with an index finger. Standard two-finger grippers fail, while autonomous dexterous hands stall whenever a retainer arrives bent or latched unexpectedly.

**Solution:** Building on interactive fleet-learning paradigms (**HG-DAgger**, **Fleet-DAgger**), one technician at a central desk supervises **four simulated dexterous robot arms/hands** across disassembly bays (one active rescue at a time via a strict FIFO hold queue). Routine unlatching runs autonomously. When a robot unexpectedly stalls against a raised stopper flap, its confidence drops below threshold, it retracts to a safe hover pose, and queues for a 10-second human rescue via a **Rokoko Smartglove II**. The **Arduino UNO Q** encodes the stall signature and fits an object-centric **Dynamic Movement Primitive (DMP / Radial Basis Function model)** from that single rescue in milliseconds. After a fast headless simulation check on the host workstation, the skill is shared across the fleet—allowing other bays to recognize the same stall signature and clear the jam across pose variations ($\pm 30^\circ\text{–}40^\circ$ yaw, $\pm 10\text{–}12\text{ cm}$ shift) for that latch SKU, while refusing transfer to incompatible jams.

### 2. Physical Fault Twin and Simulated Mechanism (3-Box Sprung Slide-Bolt)

To prevent MuJoCo convex-hull mesh lockups and retargeting sensitivity, the MuJoCo MJCF latch is built strictly from **three oversized `<geom type="box">` primitives** (with optional non-colliding visual meshes `contype="0" conaffinity="0"`):

- **Base Track (Static Grey Box):** Rectangular tray ($14\text{ cm} \times 8\text{ cm}$) with a central X-axis channel retaining a battery cell block.
- **Slide Bolt (Moving Blue Box):** Features a tall vertical finger hook/tab spaced $7\text{ cm}$ from the stopper flap. Slides right ($+X$) to release the cell.
- **Sprung Stopper Flap (Pivoting Orange Box):** Wide $4\text{ cm} \times 4\text{ cm}$ thumb paddle in the channel.
  - *Routine State:* Flap is latched flat ($0^\circ$), allowing a routine 1-finger bolt slide.
  - *Jammed Fault State:* A MuJoCo hinge spring (`stiffness`, `springref = +35 deg`) actively pushes the flap upward into the bolt path. Clearing it requires real physics: the robot thumb must actively press and hold the sprung flap down against spring torque while the index finger pulls the bolt across it.
- **Edge-Triggered Physical Fault Rig (Judge-Operated):** A matching 3D-printed latch sits at the front of the desk (outside the Coil Pro's EM workspace) wired to UNO Q `INPUT_PULLUP` pins. Flipping the physical orange flap up acts strictly as a **one-way latched fault trigger**: it releases Bay 1's virtual flap catch in MuJoCo so the hinge spring pops the virtual flap up for the next pallet cycle. Subsequent physical switch changes are ignored until the bay resets, ensuring the operator's rescue relies 100% on simulated thumb-to-flap contact physics rather than closing a wire.
- **Ergonomic Relative-Clutch Rescue Workspace:** Rather than co-locating the operator's hand onto rigid plastic (where $5\text{ mm}$ of EM drift or human-to-LEAP hand link mismatch would physically block the human thumb before the virtual flap closes), the operator performs rescues from a forearm-supported workspace using **Relative Pose Indexing** upon clutch press and a split perspective/orthographic camera view.

### 3. System Architecture, Compute Split, and DMP Math

- **Host Workstation (Laptop — Simulation & Differential IK):**
  - Captures Rokoko Studio UDP packets ($83.3\text{ Hz}$) and forwards joint/wrist streams to the UNO Q (or local mock service when unplugged).
  - Runs 4-bay MuJoCo simulation (**Bays 1–2** active physics with **7-DoF Franka Emika Panda + 16-DoF LEAP Hand**; **Bays 3–4** lightweight kinematic background loops), damped-least-squares differential IK, `mj_inverse` external torque estimation, and **3 headless validation rollouts** (~$0.3\text{ s}$ total on laptop CPU).
- **Arduino UNO Q Linux MPU (QRB2210 — Edge Filtering & Skill Learning):**
  - Communicates with the Host Laptop over UDP/TCP (USB-C network gadget or Wi-Fi), with an automatic localhost Python fallback node when hardware is unplugged.
  - Runs a low-lag adaptive **One-Euro Filter ($1\text{€}$)** on incoming Rokoko joint/wrist streams.
  - Runs a **Mahalanobis Stall-Signature Classifier** over the $300\text{ ms}$ stall window $z = [x_{\text{bolt,stall}}, \Delta \tau_{\text{ext}}]$ to classify known vs. unseen (out-of-distribution) jams.
  - Compiles the **Object-Centric RBF Movement Primitive (DMP)** via closed-form ridge regression (30 Gaussian basis kernels, $< 20\text{ ms}$ compute on Cortex-A53) upon successful rescue.
- **Arduino UNO Q Real-Time MCU (STM32U585) & Physical Console:**
  - Polls physical Bay-Select buttons (`1–4`), the Hold-to-Control Dead-Man's Clutch, the E-Stop, and the Judge Fault Rig at $1\text{ kHz}$.
  - Drives the **8×13 LED Matrix** as a live 4-Bay Status & Queue Dashboard (green nominal pulse, flashing red on stall/queue, sweep animation during skill validation).
- **Decoupled Task-Space Wrist + Retargeted Robot Finger DMP Math:**
  1. *Wrist Trajectory:* Recorded as the 6-DoF robot wrist pose in the latch's local frame, $T_{\text{wrist}}^{\text{latch}}(s) \in SE(3)$, parameterized by the canonical DMP phase variable $s(t) \in (0, 1]$.
  2. *Robot Finger Trajectory:* Recorded after human-to-robot retargeting as the robot hand's joint trajectory $q_{\text{robot}}(s) \in \mathbb{R}^{16}$ (LEAP Hand), phase-locked to $s(t)$.
  3. *Bounded Pose Generalization:* When Bay 2 encounters an identical latch SKU shifted within $\pm 10\text{–}12\text{ cm}$ and rotated within $\pm 30^\circ\text{–}40^\circ$, laptop differential IK transforms $T_{\text{wrist}}^{\text{latch}}(s)$ into Bay 2's world frame, rotating the 7-DoF Panda arm + wrist to preserve approach geometry while replaying $q_{\text{robot}}(s)$.

### 4. Explicit Metrics and Focused Safeguards

- **Mathematical Confidence Definition & Stall Trigger:**
  Instantaneous confidence $C(t) \in (0, 1]$ combines normalized external joint torque residual ($\tau_{\text{ext}} = \tau_{\text{act}} - \tau_{\text{id}}$ via `mj_inverse`) and velocity stall error:
  $$C(t) = \exp\!\left(-\alpha \frac{\|\tau_{\text{ext}}(t)\|_\infty}{\tau_{\text{limit}}} - \beta \frac{\max(0, v_{\text{cmd}}(t) - v_{\text{act}}(t))}{v_{\text{nom}}}\right)$$
  No pre-contact flap state is observed. A robot only discovers a jam during contact when $C(t) < 0.3$ continuously for $300\text{ ms}$ (or a $3.0\text{ s}$ step watchdog expires).
- **Single-Task FIFO Queue & Hardware Dead-Man's Clutch:**
  Only one bay can be teleoperated at a time; concurrent stalls retract $3\text{ cm}$ to a zero-velocity safe hover and wait in the FIFO queue. Releasing the physical clutch mid-rescue immediately sets $\dot{q}_{\text{cmd}} = 0$ and $q_{\text{target}} = q_{\text{actual}}$ to bleed off contact force, discarding incomplete trajectory buffers.
- **Additive Residual Tube, SKU Gate, and Negative Control:**
  Skills are indexed by `(Part_SKU_ID, Stall_Signature_ID)`. Autonomous replay is wrapped in an additive torque guard plus a hard torque ceiling (avoiding percentage-of-zero false triggers in free space):
  $$\|\tau_{\text{ext}}(s) - \bar{\tau}_{\text{demo}}(s)\|_\infty \le \tau_{\text{margin}} \quad \text{and} \quad \|\tau_{\text{ext}}(s)\|_\infty < \tau_{\text{hard\_max}}$$
  If a bay hits an incompatible jam (e.g., bolt pinned solid at $x = 0\text{ cm}$, producing an out-of-distribution stall signature $z$), the UNO Q classifier rejects skill transfer and escalates directly to the human.
- **Pre-Deployment Simulation Gate (Risk-Reduction Check):**
  Before fleet broadcast, the host runs **3 fast headless MuJoCo rollouts** across $\pm 30^\circ$ yaw and $\pm 10\text{ cm}$ shifts (~$0.3\text{ s}$) verifying latch clearance without exceeding $\tau_{\text{hard\_max}}$ or IK joint limits.

### 5. Minimum Viable Demonstration (MVD) and Build Priority

- **P0 Build Priority (Must work 10/10 times before anything else):**
  1. MuJoCo 3-box latch with sprung virtual flap + Panda 7-DoF diff-IK + 16-DoF LEAP Hand + Rokoko $1\text{€}$ retargeting + relative-clutch takeover in Bay 1 (`T-100`, `T-110`, `T-120`).
  2. UNO Q closed-form DMP fit + $SE(3)$ wrist-warped autonomous replay clearing the sprung flap in rotated Bay 2 ($35^\circ$ yaw, $12\text{ cm}$ shift) with zero switch input (`T-130`, `T-140`).
- **P1 Build Priority (Add once P0 is locked):**
  1. Edge-triggered physical fault rig wiring + explicit $C(t)$ stall detector & live UI plot + UNO Q 8×13 LED 4-bay status grid (`T-131`, `T-150`).
  2. 3-rollout laptop validation gate + Negative Control jam (pinned bolt at $x = 0$ rejected by UNO Q stall-signature classifier) + kinematic background Bays 3–4 + short pre-recorded audio status prompts (`T-140`, `T-150`).
- **Live Stage Script (2.5 Minutes):**
  1. *Routine Autonomy:* Bays cycle unjammed bolts with 1 finger ($C(t) > 0.85$, UNO Q LEDs pulsing green).
  2. *Physical Fault & Contact Stall:* Judge flips up the physical orange Stopper Flap on the table rig, arming Bay 1's virtual flap spring. On the next cycle, Bay 1's finger collides with the raised flap, $C(t)$ drops below $0.3$, the arm retracts $3\text{ cm}$, and Bay 1 flashes red on the UNO Q grid.
  3. *10-Second Rescue:* Operator presses Bay 1, holds the clutch, holds the virtual sprung flap flat with the thumb while sliding the bolt open with the index finger, and releases the clutch.
  4. *Edge Fit & Sanity Check:* UNO Q fits the RBF-DMP weights and logs the stall signature ($< 50\text{ ms}$) while the laptop completes 3 pose-perturbed check rollouts (~$0.4\text{ s}$).
  5. *Bounded Pose Generalization:* Bay 2 hits the same sprung flap rotated $35^\circ$ and shifted $12\text{ cm}$, recognizes `Stall_Signature_1` on initial contact, retracts $1\text{ cm}$, warps the learned 2-finger DMP to the rotated latch frame, and clears it autonomously.
  6. *Negative Control (15 s):* A pinned-bolt fault ($x = 0\text{ cm}$) triggers an unseen stall signature; the UNO Q refuses unsafe skill reuse and cleanly pages the operator.

---

## Compliance & Reasoning Extensions (Teammate Modules)

To keep the real-time $60\text{ Hz}$ control loop completely non-blocking and allow parallel teammate development without merge conflicts, two decoupled extension layers are defined via protocols in `T-100` with dedicated implementation tasks:

1. **Solana Compliance & Alert Chain (`T-160` — Placeholder in `T-100`):**
   - **Interface:** `ComplianceSink` (`src/teleop/compliance/`) receiving structured, SHA-256-digested `ComplianceEvent` records asynchronously for `STALL_ALERT`, `RESCUE_LOGGED`, `VALIDATION_GATE`, and `OOD_REFUSAL`.
   - **Status:** `T-100` provides `SolanaCompliancePlaceholder` (no-op / local JSONL log); a teammate owns `T-160` to wire on-chain Solana compliance and alert anchoring.

2. **Verda Nemotron Reasoning Co-Pilot (`T-161` — Placeholder in `T-100`):**
   - **Interface:** `FleetReasoner` (`src/teleop/reasoning/`) invoked asynchronously off the real-time loop on stall events, validation gates, and OOD refusals to synthesize root-cause triage, operator rescue hints, and human-readable compliance rationales.
   - **Hosting:** Served via **Verda** (`VERDA_BASE_URL`, `VERDA_API_KEY`, `NEMOTRON_MODEL` in `.env`).
   - **Status:** `T-100` provides `VerdaNemotronPlaceholder` (deterministic offline status text); a teammate owns `T-161` to connect the Verda Nemotron endpoint.

---

## Presenter notes: realism

For the experts in the audience. One line each:

- **Sensor model:** the latch pose is not read from the simulator. The estimator is a simulated
  fiducial-tag sensor (about 3 mm in x and y, 1.5 degrees in yaw), the error a tag-based setup
  would realistically give, so the controller never sees ground truth.
- **Tactile localization:** a bolt encoder plus the compliant wrist spring detect first contact
  with the latch and correct the pose by touch (about 1 mm). Replay uses a touch offset
  calibrated from the demonstration.
- **Randomized gate:** a learned skill must pass at least 18 of 20 rollouts with randomized
  latch pose, estimate error, flap stiffness and rest angle, friction (plus or minus 25%) and
  jam, before it is shared with the fleet.

Honesty note: the default-seed test gate passes 18/20. The scripted show runs at the fixed
`STAGE_SEED` 5. The scripted demo skill's pooled per-rollout pass rate is about 70% across
seeds 0-9, so the robustness claim is "every deployed skill passed at least 18/20 randomized
trials", not "every demo passes".
