# Console Pages and Always-On Data Collection

Date: 2026-11-14. Builds on the fleet ops console
(`docs/superpowers/specs/2026-11-14-fleet-ops-console-design.md`), the ACT recovery policy
(`docs/superpowers/specs/2026-11-14-act-recovery-policy-design.md`, T-190 to T-198) and the console
glove calibration panel (`docs/superpowers/specs/2026-11-14-console-glove-calibration-design.md`).

## Intent

Every time a human takes over a stalled robot, that rescue is the most valuable data the plant
produces. Today it is only kept if someone remembered to start `teleop atc --record-episodes DIR`,
and the operator can't see from the dashboard whether anything is being saved. The DMP skills the
fleet learns from those rescues are lost when `teleop atc` exits.

We want:

1. **Rescues saved automatically** as LeRobot episodes whenever a human intervenes, building up
   across sessions into one pile large enough for a big ACT training run.
2. **DMP skills saved to disk** and reloaded next run, so the fleet keeps what it learned. The
   DMP is the latch-frame, model-agnostic policy that carries over to other robots, so the saved
   library is itself a portable artifact.
3. **Collection controlled and visible from the dashboard**, not the command line.
4. **The dashboard split into three pages**, because one screen is now too cluttered.

## Goals

- Recording is **on by default** in live (non-headless) `teleop atc` runs. Episodes go to a new
  session folder under `data/episodes/` without any flag.
- The operator can pause and resume recording from the dashboard, separately for **human
  rescues** (glove or scripted) and **autonomous recoveries** (DMP or ACT). Both default on.
- A **Data & training** page shows the session and all-time totals, so the operator can see the
  training pile growing.
- Deployed DMP skills are written to `data/skills/library.json` as they deploy and reloaded at
  startup (re-registered with the UNO Q classifier). The operator can start fresh from the
  dashboard.
- Three pages: **Operate**, **Data & training**, **Setup & health**.
- A rescue alert is visible on every page, so switching pages never hides a robot that needs a
  human.

## Non-goals

- Running `teleop process`, ACT training or `teleop eval` from the dashboard. The Data page shows
  the commands to copy; they stay CLI jobs (training runs on Verda).
- Deleting or editing episodes from the UI. Bad episodes are dropped by `teleop process` QC as
  today.
- Building DMP skills from recorded episodes offline. Skills still come only from the live gate.
- A camera-only (CCTV) page. Cameras stay on Operate (see Approaches).
- Changing the episode schema, the DMP math or the skill gate.
- Booth mode (`teleop booth`).

## Anti-goals

- Stalling the 60 Hz control loop or the 30 fps console for disk writes. Skill saves and episode
  writes stay off the control thread.
- Changing `recovery_backend: dmp` behaviour on a fresh library. With no saved skills, a run must
  be byte-for-byte today's behaviour, and the headless DMP stage golden must not change.
- Losing a learned skill or a finished episode to a crash or Ctrl+C. Writes are atomic, and the
  dataset is still finalised on exit as today.
- A switch to another page hiding a rescue.

## Constraints

- Global Constraints from the M0/M1 plan (Python 3.12, uv only, xyzw, joints by name, no hardware
  in tests) and the ACT plan's file-ownership map.
- `lerobot` is in the optional `train` group. If it is missing, recording shows as **unavailable**
  on the Data page with the fix (`uv sync --group train`). `teleop atc` still starts.
- No new npm or JS deps. The console stays plain HTML/CSS/JS under `src/teleop/console/web/`, one
  WebSocket, one page load.
- Do not edit `src/teleop/cli.py`. New behaviour goes through flags on `teleop atc`
  (`src/teleop/commands/atc.py`).
- Tests are pure: fake recorder, fake board, no browser or hardware in CI.

## Approaches

**Page split** (chosen: A)

| | Approach | Trade-off |
|---|---|---|
| **A (chosen)** | Three pages: Operate (cameras + queue + activity), Data & training, Setup & health | Rescue needs cameras and queue together, so they stay on one page; data and setup are things you visit between rescues |
| B | Four pages, cameras on their own CCTV page | During a rescue the operator would flip between cameras and the queue |
| C | One page with collapsible rail sections | Least work, but doesn't fix the clutter |

**Navigation** (chosen: hash tabs in one document). The pages are tabs in the top bar
(`#operate`, `#data`, `#setup`). All three live in `index.html` and share one WebSocket, so a page
switch never drops the live link or reloads the camera stream. Separate HTML files were rejected:
each page load would reconnect and lose activity history.

**Skill persistence** (chosen: one JSON library file, written on deploy). The alternatives were
one file per skill, or saving only at exit. Saving only at exit loses skills on a crash. One file
per skill makes "start fresh" and the coordinator's library harder to keep consistent.
`SkillPrimitive` already has `to_dict`/`from_dict`.

## Pages

```
┌ Fleet operations ── [Operate] [Data & training ●REC 12] [Setup & health ⚠] ── Live 14:02 ┐
```

The top bar keeps the site name, live dot and clock. It gains the three tabs. The Data tab shows a
small REC indicator and the session episode count. The Setup tab shows a warning dot when any
health check is not OK. The camera-layout buttons (Overview, 4-Up, Auto Focus, pins) only show on
Operate.

On Data and Setup, a slim **rescue banner** under the top bar shows when any bay is waiting for or
under human rescue ("Bay 2 needs a rescue · Go to Operate"). Clicking it switches to Operate.

### 1. Operate (default)

What the operator uses during a shift: status brief, camera wall, keys strip, toasts, and the
rail with **Rescue queue**, **Confidence**, **Learned skill** and **Activity**. A small "REC" chip
on the brief shows when the current rescue is being recorded.

### 2. Data & training

- **Recording**: two switches, *Human rescues* and *Autonomous recoveries*. A line shows the
  session folder path, or "Recording unavailable: run `uv sync --group train`".
- **This session**: episodes saved and still writing, split by source (glove, scripted, DMP, ACT)
  and by fault type, with success vs aborted.
- **All sessions** (the training pile): totals across every session folder under
  `data/episodes/`, by fault type and source, read from each session's
  `meta/teleop_episodes.jsonl`. Each fault type gets a bar toward a target count (default 50,
  a constant) so the operator can see when there's enough for an ACT run.
- **Latest episodes**: the last 10 saved this session (time, bay, fault, source, frames, success).
- **DMP skill library**: each saved skill (id, fault/signature, when learned, source session), the
  library file path, and a **Start fresh** button behind a confirm. Start fresh moves the file to
  `data/skills/library-<timestamp>.json.bak`, clears the in-memory library and wipes the board's
  prototypes and skills (see Skill persistence behaviour), so the next rescue teaches the fleet from scratch (useful for demos).
- **Next steps**: copyable commands for `teleop process` (merge all sessions) and the ACT training
  runbook from T-194.
- **Audit trail** and **Fleet advisor** hooks move here from the Operate rail.

### 3. Setup & health

- The full hardware health view (today's `#health` section) with reconnect buttons.
- **Glove calibration** panel (from the calibration spec; it moves here from the right rail). That
  panel isn't built yet. This work leaves an empty, labelled slot for it on Setup, and the
  calibration work fills the slot. (Builder call: neither work waits on the other.)
- **Dummy glove** TEMP panel (moves here from the Operate rail).
- UNO Q board footer details.

## Recording behaviour

- Live `teleop atc` (not `--headless`) opens the episode sink on `data/episodes/` by default, and
  **always** writes to a new `data/episodes/<YYYYmmdd-HHMMSS>/` subfolder. Note that
  `session_root` today writes straight into an empty root, so the default path needs an
  always-subfolder variant. (Builder call: otherwise the first run turns `data/episodes/` itself
  into a dataset and later runs nest inside it.) The all-sessions scan reads every direct child of
  `data/episodes/` that has a `meta/teleop_episodes.jsonl`.
- `--record-episodes DIR` still overrides the root and keeps today's `session_root` behaviour, so
  the T-191 tests and `--generate` don't change.
- Episodes keep the wrist and top cameras (ACT needs them). Rendering stays on the writer thread,
  so a slow writer means a growing queue, not a slower control loop. The plan includes a manual
  check on this laptop: 30 fps console held with two bays in rescue at once while recording.
  If it fails, lower the render rate on the writer, not the camera set. (Builder call.) A new `--no-record` turns it off. Headless
  runs keep today's behaviour: no recording unless `--record-episodes` is passed (so `--generate`
  and the stage golden don't change).
- The pause switches are read in `EpisodeSink.on_episode_start`. A paused source does not open an
  episode. Changing a switch mid-rescue affects the next episode, not the one in progress.
- Human rescue = `EpisodeStartSource` from a human takeover (glove or scripted).
  Autonomous = DMP or ACT recovery. The mapping lives in `EpisodeSink._label`.
- Switch state is per run and resets to on at startup.

## Skill persistence behaviour

- New flags on `teleop atc`: `--skills PATH` (default `data/skills/library.json` for live runs,
  none for headless), `--fresh-skills` (ignore the file this run; it is overwritten as skills
  deploy).
- **Load at startup**: read the file, rebuild each `SkillPrimitive`, call
  `unoq.register_skill(skill)` and put it in `coordinator.library`. A skill whose
  `actuator_names` don't match the current hand, or that fails to parse, is skipped with a
  warning. Loaded skills are trusted: they passed the gate when saved and are not re-gated.
- **Save on deploy**: when the gate passes (`coordinator.py`, after `self.library[...] = skill`),
  hand the skill to a small writer that rewrites the whole library atomically (temp file + rename)
  off the control thread.
- **Clearing the board** (decided: a new board op). The UNO Q classifier outlives `teleop atc`,
  and today it only has `register_skill`. Add `OP_CLEAR_SKILLS` (`clear_skills(part_sku_id)`)
  across `unoq/protocol.py`, `unoq/server.py`, `unoq/client.py` and `unoq/switch.py`. It drops
  every stall prototype and registered skill for that SKU and resets signature numbering. The
  switch also empties its `_skills` replay list, so a node swap doesn't bring the skills back.
  **Start fresh** and `--fresh-skills` call it before anything else. **Load at startup** also calls
  it first, then re-registers the saved skills. That way a board left on from an earlier run
  doesn't double-count prototype samples, and the board's state always matches the library file.
  The board app has to be redeployed to the UNO Q after this lands (a manual hardware step in the
  plan). If the board doesn't know the op (old firmware), log a warning, clear the laptop side
  anyway, and show "board not cleared: redeploy the UNO Q app" on the Data page.
- File format: `{"version": 1, "skills": [{"skill": <SkillPrimitive.to_dict()>, "learned_at":
  ISO time, "session": <episode session folder or null>}]}`.

## Wire protocol

- Client to server, alongside today's `view`/`fault`/`dummy`:
  - `{"type": "record", "source": "human" | "autonomous", "on": bool}`
  - `{"type": "skills", "action": "reset"}`
- Server to client: telemetry gains a `data` object: `available`, `session_path`,
  `switches`, `session` counts, `latest` list, `skills` list. The all-sessions totals are
  recomputed on a background thread at most every 10 s and when an episode finishes, never in the
  telemetry build.

## Files touched

- `src/teleop/console/web/index.html`, `console.js`, `console.css`, `health.js`: tabs, pages,
  rescue banner, Data page rendering.
- `src/teleop/console/server.py`: accept `record` and `skills` messages.
- `src/teleop/console/run.py`, `telemetry.py`: route the new messages, add the `data` object.
- `src/teleop/atc/episodes.py`: pause switches, session counts, latest-episodes list.
- New `src/teleop/atc/skill_store.py`: load, save, reset and the atomic writer.
- New `src/teleop/record/pile.py` (or similar): all-sessions totals from sidecars.
- `src/teleop/atc/coordinator.py`: a save hook after a skill deploys, and a library reset.
- `src/teleop/unoq/protocol.py`, `server.py`, `client.py`, `switch.py`, `classifier.py`: the
  `clear_skills` op. Then redeploy the board app.
- `src/teleop/commands/atc.py`: the default recording root, `--no-record`, `--skills`,
  `--fresh-skills`, and wiring.
- Calibration spec: change "right rail" to "Setup & health page".

## Task ownership

This is a new `T-23x` series. It touches files owned by T-190 (`coordinator.py`, merged),
T-191 (`episodes.py`, in review) and T-197 (`console/telemetry.py`, `console/web/*`, todo), plus
console work in T-215. Episode-sink changes start after T-191 merges. The console page split goes
first among the console tasks. T-197 and T-215 then add their panels to the page that fits them:
the recovery backend goes on Operate, and copilot output goes on Operate (call-outs) and Data
(reviews). Record this in the ATC/ACT plan's file-ownership map. (Builder call.)

## Left to the builder

- Exact visual design of the tabs, banner and Data page, within the existing console style.
- Whether hidden pages pause their DOM updates.
- Whether the server skips sending camera frames to a client that isn't on Operate (saves
  bandwidth; needs a `{"type": "page"}` message).
- Skill id collisions: if a newly fitted skill reuses an id already in the loaded library,
  disambiguate. Check how the board assigns ids first.
- The per-fault target count constant and how the bars are drawn.
