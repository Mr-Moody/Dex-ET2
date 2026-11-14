# Air Traffic Control for Robot Hands — Engineering Design Spec (`T-100`–`T-161`)

Date: 2026-11-14 (Sat) · Deadline: **Sun 2026-11-15 10:00** (hard code freeze & live demo build)
Source brief: `PROJECT_BRIEF.md` · ADR: `docs/adr/0002-atc-fleet-rescue-pivot.md`
M1 foundation spec: `docs/superpowers/specs/2026-11-13-glove-teleop-booth-design.md` (§3–§5.6 remain valid for M1 modules).

---

## 1. System Overview & Compute Split

One human operator wearing a **Rokoko Smartglove II** (right hand primary, Coil Pro or IMU wrist) supervises **four simulated EV-battery disassembly bays** via an interactive fleet-learning loop (HG-DAgger / Fleet-DAgger):

1. **Host Workstation (Laptop — Python 3.12 + MuJoCo):**
   - Receives Rokoko Studio UDP JSON v3 packets at $83.3\text{ Hz}$ (`teleop.input.studio.StudioSource`, or `ReplaySource` / `SyntheticSource` in dev/CI).
   - Runs the $60\text{ Hz}$ control / $600\text{ Hz}$ physics 4-bay MuJoCo supervisor (`teleop.atc`):
     - **Bays 1 & 2 (Active Physics):** 7-DoF **Franka Emika Panda** (`franka_emika_panda/panda_nohand.xml`) + 16-DoF right **LEAP Hand** (`leap_hand/right_hand.xml`) + **3-Box Sprung Slide-Bolt Latch**.
     - **Bays 3 & 4 (Lightweight Kinematic Background):** Visual-only (`contype=0, conaffinity=0`) Panda + LEAP + Latch cycling nominal 1-finger unlatching so rendering 4 bays adds negligible contact cost.
   - Computes damped-least-squares (DLS) differential IK, `mj_inverse` external joint torque residuals $\tau_{\text{ext}}$, instantaneous confidence $C(t)$, the strict **Single-Active-Rescue FIFO Queue**, and the **3-Rollout Headless Simulation Gate** ($< 0.4\text{ s}$ total on CPU).
   - Emits non-blocking events to `ComplianceSink` (**Solana** placeholder, `T-160`) and `FleetReasoner` (**Verda Nemotron** placeholder, `T-161`).

2. **Arduino UNO Q Linux MPU (QRB2210 Cortex-A53 — `teleop.unoq`):**
   - Runs a zero-GPU, pure-NumPy + stdlib Python UDP/TCP service (`python -m teleop.unoq.server`) over USB-C network gadget or Wi-Fi (`UNOQ_HOST`, `UNOQ_PORT`).
   - **Automatic Hardware-Free Fallback:** When `UNOQ_HOST` is unset or unreachable, the host automatically starts the exact same `UnoQEdgeService` in-process on `127.0.0.1` so all unit tests, CI, and rehearsals execute identically without hardware.
   - Executes three lightweight edge models:
     1. Adaptive **One-Euro ($1\text{€}$) Filter** on incoming $83.3\text{ Hz}$ wrist poses and retargeted joint targets.
     2. **Mahalanobis Stall-Signature Classifier / OOD Gate** over the $300\text{ ms}$ stall window $z$.
     3. **Object-Centric RBF-DMP Fitter**: Closed-form $30 \times 30$ ridge regression ($< 20\text{ ms}$ on Cortex-A53) fitting $T_{\text{wrist}}^{\text{latch}}(s) \in SE(3)$ and $q_{\text{robot}}(s) \in \mathbb{R}^{16}$.

3. **Arduino UNO Q Real-Time MCU (STM32U585 — `firmware/unoq_console/`):**
   - Polls physical Bay-Select buttons (`1–4`), Hold-to-Control Dead-Man's Clutch, E-Stop, and Judge 3D-Printed Fault Rig (`INPUT_PULLUP`) at $1\text{ kHz}$.
   - Drives the **8×13 LED Matrix** showing 4-bay status (pulsing green nominal, flashing red on stall/queued, horizontal sweep during validation).
   - Mirrored by keyboard bindings in `teleop atc` so stage rehearsals and tests work without physical switches.

---

## 2. Simulated Mechanism & Robot Kinematics (`T-100`, `T-110`)

### 2.1 3-Box Sprung Slide-Bolt MJCF (`src/teleop/sim/latch.py`)

Constructed strictly from three `<geom type="box">` collision primitives per bay (optional visual-only meshes must set `contype="0" conaffinity="0"`). Local latch frame origin is at the Base Track center on the bay table surface ($+X$ slide direction, $+Z$ up):

| Component | MJCF Body / Joint | Box Size (full L×W×H) | Physics / Fault Behavior |
|---|---|---|---|
| **Base Track** | `bay{k}/latch_base` (static relative to bay frame, placed at `LatchPose(pos, yaw)`) | $0.14 \times 0.08 \times 0.02\text{ m}$ (grey) + channel guide walls | Retains cell block; defines local latch frame $T_{\text{world}}^{\text{latch}}$. |
| **Slide Bolt** | `bay{k}/slide_bolt`, slide joint `bay{k}/bolt_slide` along local $+X$, range $[0.0, 0.06]\text{ m}$, damping $2.0\text{ Ns/m}$ | Main bar $0.08 \times 0.024 \times 0.016\text{ m}$ + vertical finger hook tab ($0.016 \times 0.028 \times 0.035\text{ m}$) spaced **$0.07\text{ m}$** from stopper flap | Cleared when $x_{\text{bolt}} \ge 0.052\text{ m}$. **Routine free slide** when flap $\theta_{\text{flap}} \le 8^\circ$. Collides cleanly with raised flap at $x_{\text{bolt}} \approx 0.035\text{ m}$ when $\theta_{\text{flap}} > 12^\circ$. |
| **Sprung Stopper Flap** | `bay{k}/stopper_flap`, hinge joint `bay{k}/flap_hinge` about local $-Y$, range $[0^\circ, 38^\circ]$, `stiffness = 0.45`, `springref = +35 deg` ($0.6109\text{ rad}$), damping $0.03$ | **$0.04 \times 0.04 \times 0.008\text{ m}$** orange thumb paddle | **Routine State:** Held at $0^\circ$ by virtual equality/limit catch (`bay{k}/flap_catch`). **Jammed State:** Releasing `flap_catch` lets hinge spring drive paddle to $+35^\circ$ into bolt path; robot thumb must actively hold $\theta_{\text{flap}} \le 8^\circ$ against spring torque while index slides bolt past $0.052\text{ m}$. |
| **Pinned-Bolt Fault (Negative Control)** | Equality weld/joint lock `bay{k}/bolt_pin` | Locks `bolt_slide` at $x_{\text{bolt}} = 0.0\text{ m}$ | Activated for Beat 6 Negative Control; produces OOD stall at $x_{\text{bolt}} = 0.0\text{ m}$. |

**One-Way Edge-Triggered Fault Latch Rule:**
Calling `arm_flap_fault(bay_id=1)` (from physical Judge Fault Rig rising edge or keyboard trigger) disables `flap_catch` **once** for that pallet cycle and ignores any further switch toggles until `reset_bay(bay_id=1)` runs. Closing the physical switch never lowers the virtual flap.

### 2.2 7-DoF Franka Panda + 16-DoF LEAP Hand (`scripts/fetch_models.py`, `configs/hands/leap.yaml`, `src/teleop/sim/bays.py`)

1. **Model Fetching (`T-100`):**
   Update `SOURCES[0]` (`menagerie`, commit `370c43965323c0944b433b202aac5058b5f907a9`) in `scripts/fetch_models.py`:
   - `sparse_dirs = ("leap_hand", "franka_emika_panda")`
   - `expect = ("leap_hand/left_hand.xml", "leap_hand/right_hand.xml", "franka_emika_panda/panda_nohand.xml")`
2. **Full 16-DoF LEAP Hand Mapping (`T-100`):**
   Update `configs/hands/leap.yaml` so all **16 position actuators** are mapped by name:
   - Existing 12 flexion actuators: `if_mcp_act`, `if_pip_act`, `if_dip_act`, `mf_mcp_act`, `mf_pip_act`, `mf_dip_act`, `rf_mcp_act`, `rf_pip_act`, `rf_dip_act`, `th_cmc_act`, `th_mcp_act`, `th_ipl_act`.
   - Added 4 opposition/abduction actuators:
     - `th_axl_act`: `joints: [thumb_metacarpal]`, `axis: y` (thumb opposition over the $4 \times 4\text{ cm}$ paddle)
     - `if_rot_act`: `joints: [index_proximal]`, `axis: z`
     - `mf_rot_act`: `joints: [middle_proximal]`, `axis: z`
     - `rf_rot_act`: `joints: [ring_proximal]`, `axis: z`
3. **Attachment & Damped-Least-Squares Differential IK (`T-110`, `src/teleop/sim/ik.py`):**
   - Attach `leap_hand/right_hand.xml` (`palm`) to Panda's `link7` / `attachment_site` with prefix `bay{k}/` (`bay1/joint1..joint7`, `bay1/if_mcp_act..`, etc.) and define site `bay{k}/wrist_site` at the LEAP palm frame.
   - Given desired world wrist pose $T_{\text{des}} = (p_{\text{des}}, R_{\text{des}}) \in SE(3)$ and current site pose $(p_{\text{cur}}, R_{\text{cur}})$:
     - Position error $e_p = p_{\text{des}} - p_{\text{cur}} \in \mathbb{R}^3$.
     - Orientation error $e_R = R_{\text{cur}} \operatorname{Log}(R_{\text{cur}}^T R_{\text{des}}) \in \mathbb{R}^3$, stacked as $e = [e_p; e_R] \in \mathbb{R}^6$.
     - Compute $6 \times 7$ arm columns of `mj_jacSite` ($J$) and solve:
       $$\Delta q = J^T (J J^T + \lambda^2 I_6)^{-1} e + (I_7 - J^+ J) k_{\text{null}} (q_{\text{home}} - q)$$
       with $\lambda = 0.05$, $k_{\text{null}} = 0.5$, step clamp $\|\Delta q\|_\infty \le \dot{q}_{\max} \Delta t$, and joint-limit clipping to Panda `ctrlrange`.

### 2.3 4-Bay Layout (`src/teleop/sim/bays.py`)

- **Bay 1 (Active Physics):** Base at $(-0.45, 0.0, 0.0)$, canonical latch offset $(0.50, 0.0, 0.02)$, $\text{yaw} = 0^\circ$, `part_sku_id = "ev_latch_v1"`.
- **Bay 2 (Active Physics):** Base at $(+0.45, 0.0, 0.0)$, perturbed latch offset shifted by $\Delta p = (+0.08, +0.09, 0.0)$ ($\|\Delta p_{xy}\| \approx 0.12\text{ m}$) and rotated $\text{yaw} = +35^\circ$ (configurable across $\pm 12\text{ cm}$, $\pm 40^\circ$), `part_sku_id = "ev_latch_v1"`.
- **Bays 3 & 4 (Kinematic Background):** Placed in back row $(-0.45, +0.85, 0.0)$ and $(+0.45, +0.85, 0.0)$ with collision disabled, driven by a lightweight phase oscillator cycling routine 1-finger bolt slides.

---

## 3. Ergonomic Relative-Clutch Teleoperation (`T-120`)

Implemented in `src/teleop/retarget/clutch.py`:

1. **Clutch Engage (`clutch: False -> True`):**
   - Captures current simulated wrist pose $T_{\text{wrist}}^{\text{anchor}} = (p_w^0, R_w^0)$ of the active bay and current filtered glove wrist pose $T_{\text{glove}}^{\text{anchor}} = (p_g^0, R_g^0)$.
   - Zero motion jump at engagement ($T_{\text{cmd}}(t_0) = T_{\text{wrist}}^{\text{anchor}}$).
2. **While Clutch Held (`clutch == True`):**
   - Position: $p_{\text{cmd}}(t) = p_w^0 + s_{\text{pos}} R_{\text{align}} (p_g(t) - p_g^0)$.
   - Orientation: $R_{\text{cmd}}(t) = R_{\Delta}(t) R_w^0$, where $R_{\Delta}(t) = R_{\text{align}} R_g(t) (R_g^0)^T R_{\text{align}}^T$.
   - Appends `(t, T_wrist_in_latch_frame, q_robot_16, tau_ext_inf)` to the active rescue trajectory buffer.
3. **Clutch Release (`clutch: True -> False`):**
   - **Immediate Force Bleed-Off:** Sets $\dot{q}_{\text{cmd}} = 0$ and $q_{\text{target}} = q_{\text{actual}}$ for all 7 arm + 16 hand actuators of the active bay.
   - If the bay's bolt cleared ($x_{\text{bolt}} \ge 0.052\text{ m}$), forwards the recorded rescue trajectory to UNO Q `fit_dmp`; otherwise discards the incomplete buffer cleanly and keeps the bay in safe hover for retry.
4. **Split Rescue Camera View:**
   - Main perspective camera framing all 4 bays + active-bay orthographic top/side picture-in-picture views rendered via `mujoco.Renderer` so thumb-to-flap depth and index-to-bolt alignment are unmistakable without physical plastic stops.

---

## 4. Mathematical Formulations & UNO Q Edge Models (`T-130`, `T-140`)

### 4.1 External Torque Residual & Confidence $C(t)$

On each $60\text{ Hz}$ control tick for active Bays 1 & 2 (`src/teleop/sim/torques.py`):
- Copy `qpos, qvel, qacc` into a dedicated scratch `MjData` (so contact state in live `data` is undisturbed) with `qfrc_applied = 0, xfrc_applied = 0`, call `mj_inverse(model, inv_data)`, and compute over active bay DoFs:
  $$\tau_{\text{ext}} = \tau_{\text{act}} - \tau_{\text{id}} = \text{data.qfrc\_actuator}[\text{dofs}] - \text{inv\_data.qfrc\_inverse}[\text{dofs}]$$
- Instantaneous confidence $C(t) \in (0, 1]$:
  $$C(t) = \exp\!\left(-\alpha \frac{\|\tau_{\text{ext}}(t)\|_\infty}{\tau_{\text{limit}}} - \beta \frac{\max(0, v_{\text{cmd}}(t) - v_{\text{act}}(t))}{v_{\text{nom}}}\right)$$
  Default constants (configurable in YAML): $\alpha = 0.8$, $\beta = 0.9$, $\tau_{\text{limit}} = 6.0\text{ Nm}$, $v_{\text{nom}} = 0.05\text{ m/s}$.
- **Contact-Only Stall Trigger:**
  No flap joint angle (`flap_hinge` `qpos`) may ever be read by the coordinator, classifier, or policy! A stall triggers strictly when:
  1. $C(t) < 0.3$ continuously for $300\text{ ms}$ ($18$ ticks at $60\text{ Hz}$), **or**
  2. Active routine step elapsed time exceeds the $3.0\text{ s}$ watchdog without reaching $x_{\text{bolt}} \ge 0.052\text{ m}$.

### 4.2 Mahalanobis Stall-Signature Classifier (`src/teleop/unoq/classifier.py`)

- Over the $300\text{ ms}$ stall window ($N = 18$ samples), extract feature vector $z \in \mathbb{R}^4$:
  $$z = \begin{bmatrix} x_{\text{bolt,stall}} & \mu_{\|\tau_{\text{ext}}\|_\infty} & \max \|\tau_{\text{ext}}\|_\infty & \|\Delta \tau_{\text{ext,wrist}}\|_2 \end{bmatrix}^T$$
- For each registered `(part_sku_id, signature_id)` prototype $(\mu_k, \Sigma_k)$ (with diagonal shrinkage $\Sigma_k = \hat{\Sigma}_k + \operatorname{diag}(\sigma_{\text{floor}}^2)$ so single-shot prototypes after 1 rescue are well-conditioned!):
  $$d_M(z, k) = \sqrt{(z - \mu_k)^T \Sigma_k^{-1} (z - \mu_k)}$$
- **Decision Rule:**
  - If $\min_k d_M(z, k) \le \gamma_{\text{ood}}$ (default $\gamma_{\text{ood}} = 3.0$): return `Match(signature_id=k*, distance=d_M, is_ood=False)`.
    - Raised stopper flap (`Jam_Sig_1`): stalls reproducibly at $x_{\text{bolt,stall}} \approx 0.035\text{ m} \implies d_M < 1.5$ in Bay 2!
  - Else (`d_M > \gamma_{\text{ood}}` or no skills for SKU): return `Match(signature_id=None, distance=d_M, is_ood=True)`.
    - Pinned bolt Negative Control ($x_{\text{bolt,stall}} = 0.0\text{ m}$): $0.035\text{ m}$ separation on $x_{\text{bolt}}$ with $\sigma_{x,\text{floor}} = 0.005\text{ m}$ alone gives $d_M \ge 7.0 > 3.0 \implies$ **guaranteed OOD rejection**!

### 4.3 Closed-Form 30-Kernel RBF-DMP (`src/teleop/unoq/dmp.py`)

- **Canonical Phase:** $s(t) = \exp(-\alpha_s t / T) \in (0, 1]$, with $\alpha_s = 4.0$ and rescue duration $T$.
- **Basis Functions:** $K = 30$ Gaussian kernels centered log-spaced in phase $c_i = \exp(-\alpha_s \frac{i - 1}{K - 1})$, widths $h_i = \frac{1}{(0.65 (c_{i+1} - c_i))^2}$:
  $$\psi_i(s) = \exp(-h_i (s - c_i)^2), \quad \phi_i(s) = \frac{\psi_i(s) \, s}{\sum_{j=1}^K \psi_j(s) + 10^{-10}}$$
- **Decoupled Latch-Frame Encoding:**
  - Transform recorded world wrist poses $T_{\text{world}}^{\text{wrist}}(t_n)$ into Bay 1's local latch frame:
    $$T_{\text{latch}}^{\text{wrist}}(t_n) = \left(T_{\text{world}}^{\text{latch}_1}\right)^{-1} T_{\text{world}}^{\text{wrist}}(t_n)$$
    represented as 6D vector $y_{\text{wrist}}(t_n) = [p_{\text{local}}(t_n); \operatorname{Log}(R_{\text{goal}}^T R_{\text{local}}(t_n))] \in \mathbb{R}^6$.
  - Concatenate with retargeted LEAP joint positions $q_{\text{robot}}(t_n) \in \mathbb{R}^{16} \implies Y \in \mathbb{R}^{N \times 22}$.
- **Closed-Form $30 \times 30$ Ridge Regression ($< 20\text{ ms}$ on Cortex-A53):**
  Given design matrix $\Phi \in \mathbb{R}^{N \times 30}$ and target residual $F = Y - \big((1 - s) y_{\text{goal}} + s \, y_0\big)$:
  $$W = (\Phi^T \Phi + \lambda_{\text{ridge}} I_{30})^{-1} \Phi^T F \in \mathbb{R}^{30 \times 22}, \quad \lambda_{\text{ridge}} = 10^{-4}$$
  Also bin-averages $\|\tau_{\text{ext}}(t_n)\|_\infty$ into a 30-point nominal torque profile $\bar{\tau}_{\text{demo}}(s)$.
- **$SE(3)$ Replay in Target Bay $b$:**
  Evaluate $y(s) = (1 - s) y_{\text{goal}} + s \, y_0 + \phi(s)^T W$, reconstruct $T_{\text{latch}}^{\text{wrist}}(s) \in SE(3)$, warp to Bay $b$'s world frame $T_{\text{world}}^{\text{wrist}}(s) = T_{\text{world}}^{\text{latch}_b} \cdot T_{\text{latch}}^{\text{wrist}}(s)$, feed to Panda DLS differential IK, and apply $q_{\text{robot}}(s) \in \mathbb{R}^{16}$ to LEAP actuators.

### 4.4 Additive Residual Torque Guard & 3-Rollout Validation Gate (`src/teleop/atc/coordinator.py`, `validator.py`)

- **Runtime Replay Guard (every tick during autonomous DMP execution):**
  $$\|\tau_{\text{ext}}(s) - \bar{\tau}_{\text{demo}}(s)\|_\infty \le \tau_{\text{margin}} \quad \text{and} \quad \|\tau_{\text{ext}}(s)\|_\infty < \tau_{\text{hard\_max}}$$
  Defaults: $\tau_{\text{margin}} = 3.5\text{ Nm}$, $\tau_{\text{hard\_max}} = 12.0\text{ Nm}$. Note that when $\bar{\tau}_{\text{demo}}(s) = 0$ in free space, $|\tau_{\text{ext}}(s)| \le 3.5\text{ Nm}$ holds cleanly without false percentage-of-zero aborts. Violation immediately triggers zero-velocity retract ($3\text{ cm}$) and escalates to the human FIFO queue.
- **Pre-Deployment 3-Rollout Headless Gate:**
  Before adding a newly fitted `SkillPrimitive` to the Fleet Library, `SkillValidator` runs 3 headless single-bay rollouts across pose perturbations:
  1. $(\Delta x = -0.08\text{ m}, \Delta y = -0.06\text{ m}, \Delta \psi = -30^\circ)$
  2. $(\Delta x = 0.0\text{ m}, \Delta y = 0.0\text{ m}, \Delta \psi = 0^\circ)$
  3. $(\Delta x = +0.08\text{ m}, \Delta y = +0.06\text{ m}, \Delta \psi = +30^\circ)$
  Each rollout must clear the bolt ($x_{\text{bolt}} \ge 0.052\text{ m}$), stay below $\tau_{\text{hard\_max}}$, and respect Panda joint limits. Total wall time target: $< 0.4\text{ s}$.

---

## 5. Wire Protocols (`T-130`, `T-131`)

### 5.1 Host Laptop $\leftrightarrow$ UNO Q Linux MPU (`teleop.unoq`)

Newline-delimited JSON over UDP (fast `filter_step` stream, default port `15051`) and TCP request/response (`classify_stall`, `fit_dmp`, `register_skill`, `console_state`, default port `15052`):

- `{"op": "filter_step", "t_ns": int, "wrist": [x,y,z,qx,qy,qz,qw], "joints": dict[str,float]}` $\rightarrow$ `{"ok": true, "wrist": [...], "joints": {...}}`
- `{"op": "classify_stall", "part_sku_id": str, "z": [float, float, float, float]}` $\rightarrow$ `{"ok": true, "matched_signature_id": str | null, "distance": float, "is_ood": bool}`
- `{"op": "fit_dmp", "part_sku_id": str, "z": [...], "timestamps_s": [...], "wrist_latch_se3": [[...], ...], "fingers_q": [[...], ...], "tau_ext_inf": [...]}` $\rightarrow$ `{"ok": true, "fit_ms": float, "skill": {...}}`

### 5.2 UNO Q STM32U585 $\leftrightarrow$ Linux MPU Serial Protocol (`115200 8N1`)

- **MCU $\rightarrow$ MPU ($1\text{ kHz}$ edge/change events + $20\text{ Hz}$ heartbeat):**
  - `BTN:BAY:<1|2|3|4>`
  - `CLUTCH:<0|1>`
  - `ESTOP:<0|1>`
  - `FAULT_RIG:LATCHED` (emitted **once** on physical flap rising edge; re-armed only after `CMD:RESET_RIG`)
- **MPU $\rightarrow$ MCU:**
  - `CMD:BAY_STATE:<1..4>:<NOMINAL|STALLED|QUEUED|RESCUE|VALIDATING|OOD>`
  - `CMD:RESET_RIG`

---

## 6. Teammate Extension Contracts (`T-100`, `T-160`, `T-161`)

Both extensions run **strictly off the $60\text{ Hz}$ control thread** via bounded background queues (`queue.SimpleQueue` / daemon worker thread) so network latency or RPC failures can never stall MuJoCo or teleop.

### 6.1 Solana Compliance & Alert Chain (`src/teleop/compliance/`)

```python
ComplianceKind = Literal[
    "STALL_ALERT",
    "RESCUE_LOGGED",
    "VALIDATION_GATE",
    "SKILL_DEPLOYED",
    "OOD_REFUSAL",
]

@dataclass(frozen=True, slots=True)
class ComplianceEvent:
    kind: ComplianceKind
    bay_id: int
    part_sku_id: str
    timestamp_ns: int
    payload_sha256: str
    summary: str
    metadata: Mapping[str, Any]

@runtime_checkable
class ComplianceSink(Protocol):
    def emit(self, event: ComplianceEvent) -> None: ...
    def close(self) -> None: ...
```

- **`T-100` Placeholder (`src/teleop/compliance/solana.py`):** `SolanaCompliancePlaceholder` appends events to an in-memory ring buffer (for HUD display & unit tests) and optional local `outputs/compliance_audit.jsonl`, reading `SOLANA_RPC_URL` and `SOLANA_KEYPAIR_PATH` from `.env`.
- **`T-160` Teammate Implementation:** Replaces the stub inside `src/teleop/compliance/solana.py` to submit signed Solana transactions / alerts while keeping the local fallback if offline.

### 6.2 Verda Nemotron Reasoning Co-Pilot (`src/teleop/reasoning/`)

```python
@dataclass(frozen=True, slots=True)
class ReasoningContext:
    event_kind: ComplianceKind
    bay_id: int
    part_sku_id: str
    confidence: float
    x_bolt_m: float
    tau_ext_inf_nm: float
    matched_signature_id: str | None
    mahalanobis_distance: float | None
    is_ood: bool
    rollout_passed: int | None = None
    rollout_total: int | None = None

@dataclass(frozen=True, slots=True)
class ReasoningVerdict:
    headline: str
    operator_guidance: str
    compliance_rationale: str
    model: str
    latency_ms: float

@runtime_checkable
class FleetReasoner(Protocol):
    def submit(self, ctx: ReasoningContext) -> None: ...
    def latest_verdict(self, bay_id: int) -> ReasoningVerdict | None: ...
    def close(self) -> None: ...
```

- **`T-100` Placeholder (`src/teleop/reasoning/nemotron.py`):** `VerdaNemotronPlaceholder` reads `VERDA_BASE_URL`, `VERDA_API_KEY`, and `NEMOTRON_MODEL` from `.env` and immediately populates deterministic structured `ReasoningVerdict` messages (e.g., explaining why `x_bolt = 0.0 m` was rejected as OOD or summarizing 3/3 rollout validation).
- **`T-161` Teammate Implementation:** Connects `src/teleop/reasoning/nemotron.py` to the Verda-hosted Nemotron endpoint asynchronously, falling back to the deterministic template on timeout/offline.
