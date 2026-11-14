# ADR 0002: Pivot Application Layer to 4-Bay Air Traffic Control Fleet Rescue

Status: accepted (2026-11-14). Supersedes the application/booth defaults in ADR 0001 while preserving M1 core infrastructure.

## Context

Following M0 and M1 completion (`StudioSource`, `ReplaySource`, `DirectRetargeter`, `Calibration`, `CoilWrist`/`ImuWrist`, `SimSink`, `EpisodeRecorder`, `task` CLI) and two design critiques (documented in `PROJECT_BRIEF.md`), the team refined the Junction entry from a walk-up `lift_cube` bimanual booth (`T-060`/`T-070`) into **"Air Traffic Control for Robot Hands"** — a 1-to-4 EV battery disassembly fleet supervisor combining Rokoko Smartglove II rescue, Arduino UNO Q edge skill learning, host-side MuJoCo validation, a Solana compliance/alert chain, and Verda-hosted Nemotron reasoning.

## Decisions

1. **Preserve M1 Core Unchanged:**
   Keep `teleop.input`, `teleop.retarget.flexion/calibration/direct`, `teleop.wrist`, `teleop.record`, `teleop.sim.hands/scene/sink/loop`, and all M1 unit tests intact so existing CI and commands (`sim`, `capture`, `calibrate`, `record`, `replay`, `process`) remain green.
2. **Active Bay Robot = 7-DoF Franka Emika Panda + 16-DoF Right LEAP Hand (`T-100`, `T-110`):**
   - Fetch `franka_emika_panda` alongside `leap_hand` from the already-pinned MuJoCo Menagerie commit (`370c43965323c0944b433b202aac5058b5f907a9`) in `scripts/fetch_models.py`.
   - Extend `configs/hands/leap.yaml` so all **16 actuators** (12 flexion + thumb opposition `th_axl_act` + 3 abduction `*_rot_act`) are mapped for retargeting and DMP recording ($q_{\text{robot}}(s) \in \mathbb{R}^{16}$).
   - Solve 6-DoF wrist pose targets $T_{\text{wrist}}^{\text{world}}(t) \in SE(3)$ onto the 7 Panda arm joints via damped-least-squares differential IK with posture nullspace regularization.
3. **4-Bay Disassembly Scene & 3-Box Sprung Slide-Bolt (`T-110`, `T-120`, `T-140`):**
   - Bays 1–2 run full active contact physics with the 3-box sprung slide-bolt latch (oversized box geoms, `springref = +35 deg`, one-way virtual catch, and `pinned_bolt` fault at $x = 0\text{ cm}$ for Negative Control). Bays 3–4 run lightweight kinematic background loops.
   - Operator rescues use **Relative-Clutch Pose Indexing** in an ergonomic armrest workspace (keeping the Judge's 3D-printed fault rig outside the Coil Pro EM field).
4. **Split Compute on Arduino UNO Q with Local Mock Fallback (`T-130`, `T-131`):**
   - **QRB2210 Linux MPU** (`src/teleop/unoq/`): runs the $1\text{€}$ filter, Mahalanobis Stall-Signature Classifier / OOD Gate, and closed-form $30 \times 30$ RBF-DMP ridge regression ($< 20\text{ ms}$, pure NumPy, zero PyTorch/GPU) over UDP/TCP. When UNO Q hardware is unplugged, the host automatically spawns the exact same service locally on loopback so all tests and rehearsals run hardware-free.
   - **STM32U585 MCU** (`firmware/unoq_console/`): polls Bay 1–4 buttons, Dead-Man's Clutch, E-Stop, and the one-way Judge Fault Rig at $1\text{ kHz}$ and drives the 8×13 LED Matrix.
5. **Isolated Teammate Placeholders for Solana (`T-160`) and Verda Nemotron (`T-161`):**
   - `T-100` defines non-blocking `ComplianceSink` (`src/teleop/compliance/`) and `FleetReasoner` (`src/teleop/reasoning/`) protocols plus zero-dependency placeholder classes (`SolanaCompliancePlaceholder`, `VerdaNemotronPlaceholder`).
   - Teammates implement `T-160` (Solana compliance/alert anchoring) and `T-161` (Verda-hosted Nemotron reasoning) strictly inside those isolated subpackages without colliding with simulation, UNO Q, or coordinator code.

## Consequences

- `T-100`–`T-161` become the active task series (`tasks/README.md`, `CLAUDE.md`); legacy `lift_cube` booth/ACT tasks (`T-060`, `T-070`, `T-080`, `T-081`, `T-013`) are superseded once any in-flight `main` PRs land.
- No PyTorch MLP is used anywhere on the control or skill-learning path; DMP fitting requires only `numpy` on Python 3.12+.
