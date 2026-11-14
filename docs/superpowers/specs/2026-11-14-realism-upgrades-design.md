# Realism Upgrades: Estimated Latch Pose, Touch Re-localization, Randomized Skill Gate

Date: 2026-11-14. Task: T-178. Builds on T-177 (Inspire hands with compliant wrist, PR #52).

## Intent

Today the fleet-rescue loop works only under best-case assumptions:

- **Perfect pose:** every bay reads the latch's exact pose from the simulator
  (`BayRig.world_latch()` reads `model.body_pos`).
- **Open-loop replay:** a learned skill is replayed open loop in that perfect frame.
- **Easy validation:** "validated" means 3 fixed pose offsets with identical physics, all of
  which must pass.

Anyone who knows robotics will ask how the robot knows where the latch is, and what happens when
the next case differs a little from the demonstration.

The audience is hackathon judges. Most are not roboticists, and they judge realism by what they
can see. A few are, and they will probe the design; the presenter has to justify each decision
in a sentence. So every upgrade must be:

1. **Visible:** shown on the fleet ops console in plain words or numbers.
2. **Defensible:** a standard technique the presenter can name and justify.
3. **Stage-safe:** the scripted 6-beat demo must still pass every time.

## Goals

- **G1 Estimated latch pose:** control never uses the true latch pose. Each bay works from an
  estimate that carries realistic sensor error.
- **G2 Touch re-localization:** contact with the bolt hook corrects the estimate along the push
  axis, measured by the compliant wrist, not by reading simulator contacts. This happens both at
  a routine stall and during autonomous skill replay.
- **G3 Randomized validation gate:** a new skill must pass at least 90% of about 20 randomized
  headless rollouts. Randomization covers pose-estimate error, flap spring stiffness, contact
  friction and jam angle. The console reports the result as "N/20 passed".
- **G4 Console visibility:** each active bay shows its pose-estimate uncertainty and, after a
  touch, how far the touch corrected it. The gate shows the pass count and the randomized
  conditions it covered.

## Non-goals

- Multi-demonstration skill learning and trained policies (ACT, diffusion, VLA). The setup has
  no hardware or data for them. They stay on the roadmap the presenter talks to.
- Real computer vision on the rendered feeds.
- Continuous force-feedback (admittance) control during replay.
- Changes to the background Bays 3-4 (visual-only replay).

## Anti-goals

- **A flaky stage demo:** any randomness must be seeded so a rehearsal equals the show, and the
  scripted run must pass every time.
- **Hidden ground truth:** no control or decision path may read the true latch pose,
  `flap_hinge`, or simulator contact lists. Ground truth may appear only in clearly labelled
  display or debug values.
- **Blocking the 60 Hz loop:** the larger gate must not freeze the control loop or the console
  feeds.

## Constraints

- Follow the plan's Global Constraints (Python 3.12, uv, xyzw quaternions, joints by name, no
  hardware in tests).
- Existing safety semantics stay:
  - the arm torque margin is 0.8, and a jam must trip the 300 ms confidence stall, not the guard;
  - the torque tube still applies;
  - the coordinator never reads `flap_hinge`.
- The current coordinator, validator and console tests keep passing, adjusted only where they
  assert the old 3-rollout gate.
- The spec-4.4 rollout perturbations (±8 cm / ±6 cm / ±30°) remain part of the randomized set,
  as the pose range a skill must generalise to.

## Approaches considered

**Pose estimate source.**

- **A. Simulated sensor (chosen):** the true pose plus seeded Gaussian error sized like a
  fiducial-tag camera. It is cheap and reliable, and experts accept it if we name the sensor it
  stands in for.
- **B. Real vision:** tag pose from the rendered CCTV feed with OpenCV. It is more convincing,
  but brittle under the cell lighting, adds a vision stack, and risks the demo.

Chosen: A. The presenter's line is "a fiducial tag camera gives a few-mm estimate; we model
its error, and touch fixes the rest".

**Coping with error during replay.**

- **A. Touch re-localization (chosen):** move until the finger touches the hook, correct the
  frame from where the touch happened, and continue. It is easy to show and explain, robust to
  mm-level error, and a named technique (tactile localization).
- **B. Continuous force feedback:** more general, but harder to tune, explain and keep
  stage-safe.

Chosen: A.

**Gate size.** One sensible way: about 20 rollouts, a 90% threshold, run off the control loop.

## Design

### 1. Latch pose estimator (G1)

- A new `LatchEstimate` per active bay holds:
  - the estimated latch pose (x, y, yaw in the bay's frame; z is assumed known from the
    fixture);
  - the per-axis 1-sigma uncertainty;
  - a `touch_corrected` flag and the last correction in mm.
- **When it updates:** at the start of every pallet cycle, at the same moment as today's
  `reset_latch` after the lift.
- **New sample:** truth plus seeded Gaussian error. Default sigma: 3 mm in x and y, 1.5° in yaw.
- **Seeding:** the RNG is seeded per run (`--seed`, default fixed). The stage script uses a
  fixed seed, so the show reproduces the rehearsal.
- **Who uses it:** every control-path use of the latch frame goes through the estimate:
  - the routine target (`compose_pose(wfl, routine.pose_at(t))`);
  - approach and lift poses;
  - skill replay;
  - the rescue recording frame (`RescueTrajectoryBuffer`);
  - backoff and retract axes.
  `BayRig.world_latch()` stays as the ground-truth accessor, but control code may not call it.
  Rename it to make that obvious, e.g. `true_world_latch()`.
- **Clutch rescue:** the operator works from the camera views, so their demonstration is
  naturally correct in the true frame. It is recorded relative to the bay's touch-corrected
  estimate.

### 2. Touch re-localization (G2)

- **Contact sensors:** a touch is declared on whichever comes first:
  - **the wrist:** the two compliant-wrist springs (`wrist_comp_x/y`) act as a wrist
    force/torque sensor, and the spring torque projected on the push axis exceeds a threshold
    (to be tuned; free motion stays well below it) for at least 2 consecutive ticks;
  - **the bolt encoder:** `x_bolt` (the spec-2.1 encoder the coordinator already reads)
    starts to move.

  A freely sliding bolt resists with only about 0.1 N, too little for the wrist springs to
  detect, so the encoder is what catches a touch in a clean cycle. The wrist catches a touch
  against a blocked bolt. Both are real sensors. *(Builder-check call.)*
- **Standoff:** the routine starts with the index far enough behind the estimated hook that a
  3-sigma x error cannot put the finger on top of or in front of the hook. Today the start is
  about 2 mm from the hook; at 3 mm sigma it needs to be at least about 12 mm. This costs a
  fraction of a second per cycle. *(Builder-check call.)*
- **What a touch means:** the index is against the hook's back face. Given the bolt's
  commanded/assumed position at that moment, the known hand geometry (index pad offset from
  the wrist site) and the hook geometry, the measured wrist pose gives the hook's x in the world.
  The estimate's x is replaced by that measurement, and its x sigma shrinks to the touch
  accuracy (about 1 mm). Only the push axis is corrected; y and yaw keep their estimate.
- **Bolt position at touch:**
  - In the routine, the bolt is at 0 when the cycle starts, so the first touch localizes the
    latch.
  - At a stall, the bolt has moved by an unknown amount. Use the stall touch only if the bolt
    position is observable: `x_bolt` is already used by the coordinator as a sensor (the bolt
    encoder of spec 2.1). In that case hook x = estimated latch x + `x_bolt` + hook offset.
- **When it applies:**
  - **Routine:** the first touch of each cycle corrects the estimate. The remaining push is
    re-anchored smoothly; no jump larger than the IK lead is allowed.
  - **Autonomous skill replay:** the replay runs in the current estimate. When the skill's index
    reaches the hook, the touch correction updates the frame and the rest of the replay continues
    in the corrected frame, blended over about 0.2 s.
  - **Stall:** the stall itself is a touch. The estimate is corrected before the retract, so the
    operator's rescue and its recording use the corrected frame.
- **What the touch cannot fix:** y and yaw error. These are tolerated by geometry (the hook is
  28 mm wide, the thumb presses a 40 mm flap) and by wrist compliance. The randomized gate
  measures whether that tolerance is enough. This is the honest answer to "what if it's off
  sideways".

### 3. Randomized validation gate (G3)

- `SkillValidator.validate` runs N = 20 rollouts (configurable).
- **Each rollout draws, from a seeded RNG:**
  - **Latch pose:** one of the spec-4.4 offsets, cycling so each appears, plus small extra
    jitter.
  - **Pose-estimate error, as a live bay faces it at replay time:** a skill only replays after
    a stall, and the stall touch has already corrected the push axis. So the error is about
    1 mm sigma along the push axis, plus the full G1 error in y and yaw. *(Builder-check call.)*
  - **Flap spring stiffness:** ±20%.
  - **Flap rest angle (`springref`):** ±3°. This changes how the jam actually behaves; moving
    only the bolt's start position would not. *(Builder-check call.)*
  - **Contact friction on the hand and latch:** ±25%.
- **Pass rule:** at least 18/20, and the existing per-rollout checks (cleared, torque, joint
  limits) are unchanged.
- **Report:** `ValidationReport` gains `pass_rate`, the threshold, and a list of the randomized
  conditions per rollout, so failures can be explained ("failed at +25% friction, 2.8 mm yaw
  error").
- **Off the control loop:** validation moves to a worker thread with its own `MjModel`/`MjData`
  (the validator already owns a separate single-bay model). The bay holds its safe hover in
  `VALIDATING` while the coordinator keeps ticking. The result is applied on the control thread
  when ready. Expected wall time: about 3-4 s. The stage script's beat timing must allow for it.
- **Determinism:** the gate RNG is seeded from the run seed plus the skill id, so the stage run
  always produces the same N/20.
- **Reporting at a fixed sim time:** the worker's result is applied at a fixed sim-time
  deadline after the gate starts. The deadline covers the expected wall time. If the result is
  late, the control thread waits for it at that tick. A seeded rehearsal therefore plays out
  tick-for-tick like the show. *(Builder-check call.)*

### 4. Console and HUD (G4)

- **Per active bay:** "Pose est. ±3 mm / ±1.5°". After a touch: "Touch-corrected 4.2 mm"
  (the correction size, which needs no ground truth).
- **Debug value (optional, off by default):** the true remaining error, labelled "sim truth".
- **Gate line:** "Skill gate 19/20 PASS (randomized pose, friction, spring, jam)" instead of
  "3/3".
- **Event log:**
  - "Bay 2 touched latch: pose corrected 3.1 mm";
  - one line per gate result, with the worst failing condition if any.
- **Telemetry:** the console's telemetry adapter carries the new fields. The HUD text mirror
  shows the same.

### 5. Stage demo

- The 6-beat script runs with the fixed seed. Beat timings are re-checked with the threaded
  gate.
- The stage story gains one line per upgrade. The presenter notes (in the brief) list the
  expert-facing justification for each: sensor model, tactile localization, randomized gate.

## Decided with the lead

- Audience: mostly non-robotics judges, plus a few experts the presenter must convince.
- Build: the simulated-sensor pose estimate, touch re-localization, and the randomized gate.
- Multi-demo learning and trained policies are out of scope (no hardware or data).
- **Robustness target:**
  - The stage run and the coordinator/stage tests use fixed seeds that are confirmed to pass
    with the realistic error switched on.
  - The demo does not have to pass for an arbitrary seed.
  - The robustness claim comes from the randomized gate ("passes 19/20 randomized trials
    before deployment").

## Left to the builder

- **Exact values:** the sigma values, touch threshold and blend times. Starting values are
  above; tune for a reliable demo.
- **Code shape:** where `LatchEstimate` lives and how it is threaded through `BayRig` and the
  coordinator, as long as control code cannot reach the true pose.
- **Rollout count:** N if 20 does not fit the stage timing (minimum 10).
- **Console layout** of the new fields within the existing design.
