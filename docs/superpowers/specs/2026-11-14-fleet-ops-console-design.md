# Fleet Ops Console: Remote-Supervision Visuals for the ATC Demo

Date: 2026-11-14. Demo: 2026-11-15 10:00. Builds on the ATC pivot
(`docs/superpowers/specs/2026-11-14-atc-fleet-rescue-design.md`, ADR 0002).

## Intent

Today the demo runs in the stock MuJoCo passive viewer: one flat top-down light, a checker
floor, the Menagerie textures and lights removed, all four bays visible in one shot, and an ASCII
text HUD in the corners. It looks flat, it stutters, the text looks dated, and it doesn't tell
the story.

The story is **remote supervision**. One technician watches a fleet they can't see directly,
through camera feeds, the way someone watches a CCTV room. They act as a safeguard who can take
over any bay from afar, clear the jam and hand control back. The screen should look like the
real product that technician would use. That console is what judges should remember.

## Goals

- A fullscreen **browser console** that replaces the MuJoCo viewer window. It shows one live
  "CCTV" feed per bay plus fleet telemetry, and judges should grasp it within about 5 seconds.
- **Rescue mode:** when the operator takes a bay, its feed expands to two angles with clutch
  state and a rescue timer. The other bays shrink but stay visible.
- **Each bay is its own enclosed cell,** far enough from the others that no camera ever shows a
  neighbouring bay.
- **MuJoCo scene polish:** proper lighting with shadows, the original robot materials, and
  styled cell surfaces.
- **Smooth video:** at least 30 fps per feed while physics runs, with no visible hitch when the
  3-rollout gate fires.
- Look as polished as possible. The team has until the demo to iterate.

## Non-goals

- Game-engine rendering (Unity, Unreal, Godot) or client-side three.js. Both were considered and
  rejected (see Approaches).
- Changing the ATC logic: coordinator state machine, C(t), queue, DMP, gate, classifier.
- Mobile layouts and multi-user access. It's one laptop driving one screen or projector.
- Recording or exporting video.

## Anti-goals (failure even if it works)

- **Adding latency or jitter to the glove rescue** that makes the 10-second rescue harder. The
  rescue feed must feel live.
- **A new way for the live demo to fail.** If the browser or stream dies mid-demo, physics,
  the UNO Q, audio and the hardware console must keep working, and reloading the page must
  bring the console back.
- **Breaking `--headless` and `--script` runs or the existing tests.** CI must stay green, with
  no GPU or browser required in tests.
- **A generic "AI dashboard" look.** It should read as an industrial operations console.

## Constraints

- Global Constraints in `docs/superpowers/plans/2026-11-13-m0-m1-glove-to-sim.md` apply:
  Python 3.12, uv only, joints by name, xyzw quaternions, no hardware in tests.
- Do not edit `src/teleop/cli.py`. The console is launched by `teleop atc` (`commands/atc.py`).
- Hardware: Windows 11 laptop with an RTX 4060 Laptop GPU. Offscreen GL rendering through
  `mujoco.Renderer` on Windows (GLFW hidden context) is available.
- Libraries: `websockets` 17.4 is already installed. New dependencies must be added with `uv add`.
  Front-end libraries come from pinned CDN versions or are vendored, never fetched at runtime
  from an unpinned URL. **The demo venue may have no internet, so vendoring is preferred.**
- Control loop: `FixedRateLoop(60 Hz)` calls `StageController.tick()`. The validation gate
  (`coordinator.py:957`, `validator.validate`) runs synchronously inside a tick (~0.3-0.4 s).
- Existing per-bay rescue cameras are orthographic top and side views
  (`retarget/clutch.py:add_rescue_cameras`, `rescue_view_option()` hides Panda links).
- Task process: claim a task (`uv run task claim`), one worktree per task, Conventional Commits,
  no AI attribution. `T-150` (`commands/atc.py`, `atc/hud.py`) has its code on `main` but is
  still marked `in_review`. Close it, or coordinate with its owner, before editing those files.

## Approaches considered

1. **Browser control room fed by MuJoCo offscreen renders (chosen).** The sim process renders
   each bay camera with `mujoco.Renderer` on a render thread, JPEG-encodes the frames and pushes
   them plus a telemetry JSON stream over a local WebSocket to a static HTML/CSS/JS page.
   Modern styling is easy and it runs fullscreen on a projector. The cost is encode time and
   about 10-30 ms of extra latency on the rescue view, which is acceptable for a 10-second
   rescue.
2. **Native Dear ImGui window.** Same layout with lower latency in one process, but it looks
   "pro tool", not "product", and polished styling is harder.
3. **Browser three.js scene driven by streamed joint poses.** Smoothest motion and best
   materials, but the Panda, LEAP and latch geometry would have to be ported. Too much work for
   the time left.
4. **Game engine bridge.** Same porting cost as 3, plus a second toolchain and another live
   process to fail on stage.

The user picked 1, with as much MuJoCo scene polish as possible.

## Design (as decided)

### Architecture

```
  physics thread (60 Hz, unchanged tick)        render thread (~30 fps)        browser
  StageController.tick() ──► snapshot qpos ──►  own MjData, mj_kinematics ──►  ws: frames
                         └─► hud_view() JSON ─────────────────────────────────►  ws: telemetry
                                                      ◄────────────── ws: keys/actions
```

- **Physics and rendering are separate.** After each tick, the physics thread publishes a
  snapshot (qpos, plus mocap pos/quat if any) and the latest telemetry into a latest-value slot.
  The render thread keeps **its own `MjData`**: it copies the snapshot in, runs kinematics and
  renders. It never holds the physics lock while rendering. During the ~0.4 s gate the feeds
  just freeze on the last pose (honest, since the robots are waiting) while the page keeps
  animating.
- **Streaming:** one local WebSocket server (`websockets`) on localhost. It sends binary frames
  tagged with a camera id and JSON telemetry messages, and receives browser key and action
  events, which go into the existing `StageController.apply()` / `on_key` path. If the page
  disconnects, nothing in the sim changes.
- **Static page** served from the package (vendored assets, no runtime internet). `teleop atc`
  serves it and prints the URL, and opening the browser automatically is a flag.
- `--headless` and `--script` keep working with no renderer. The console has its own flag or
  default for non-headless runs, and the passive viewer stays available behind a flag as a
  fallback.

### Cameras and feeds

- Each bay gets a **"CCTV" camera**: a high corner mount, slight downward angle, wide
  perspective, framing the robot, table and latch. The cell enclosure guarantees that only that
  bay is visible.
- **Rescue mode** adds the bay's existing top and side rescue cameras (with
  `rescue_view_option()`) as the large views.
- Only render feeds that are visible: four tiles normally. In rescue mode, two large views and
  three small tiles, with the small tiles at a lower resolution and rate if needed.

### Scene polish (MuJoCo)

- **Bay layout:** bases about 4 m apart (for example on a 2×2 grid, keeping `BAY_BASES` as the
  single source of truth). Latch poses remain bay-local, so the coordinator, the Bay 2 SE(3)
  replay and the single-bay validator scene are unaffected.
- **Cell per bay:** a work table under the latch, a dark matte floor patch, partition walls or
  an enclosure with an industrial look (for example safety-yellow edge strips), and the bay
  number on the wall. Walls are visual-only (`contype=conaffinity=0`).
- **Lighting:** per-cell key, fill and rim lights with shadows on the key light; higher shadow
  map size, offscreen MSAA and haze in `spec.visual`; a neutral dark skybox.
- **Materials:** stop stripping Menagerie materials and textures in `bays.py` (`_strip_globals`
  removes lights only, which is still needed). Rename collisions are handled the way the
  existing `white` → `leap_white` rename is. Give the latch boxes specular/reflectance
  materials that keep the blue bolt / orange flap colour coding.
- **Camera look:** the "CCTV" styling (timestamp, bay label, REC dot, slight vignette or scan
  overlay) is done **in the browser with CSS over the image**, not baked into the render.

### Console layout

```
┌ FLEET OPS ─ 4 bays ─ 1 operator ─────── ● LIVE 14:02:31 ┐
│ ┌BAY 1 ▲STALL C.21┐ ┌BAY 2 ●NOM C.92┐ │ QUEUE  [B1]       │
│ │ cctv feed  REC● │ │ cctv feed     │ │ C(t) sparkline x4 │
│ └─────────────────┘ └───────────────┘ │ GATE  3/3 ✓ 0.4s  │
│ ┌BAY 3 ●NOM ──────┐ ┌BAY 4 ●NOM ────┐ │ EVENT LOG         │
│ │ cctv feed       │ │ cctv feed     │ │ SOLANA tx 4f2…    │
│ └─────────────────┘ └───────────────┘ │ NEMOTRON advisory │
└─────────────────────────────────────────┴─────────────────┘
```

- **Bay tiles:** state badge (`NOMINAL`, `STALLED`, `QUEUED`, `RESCUE`, `VALIDATING`,
  `AUTO SKILL`, `OOD - HUMAN`), live C(t) value, and a tile border coloured by state. A stalled
  bay pulses.
- **Side panel:** FIFO queue, per-bay C(t) sparklines with the 0.3 threshold line, gate result
  (fit ms, rollouts passed out of total, time), a timestamped event log, and the
  Solana/Nemotron (Kimi) feeds.
- **Rescue mode:** the rescued bay takes the main area with top and side views, a clutch
  indicator (HELD/RELEASED), a rescue timer and a "LIVE CONTROL" banner. The other bays drop to
  a strip.
- **Data source:** the same `HudView` that `hud_view()` builds today, serialised to JSON. Any
  missing fields (for example the event log, or a C(t) history the HUD doesn't keep) are added
  in a console-side adapter, not in the coordinator.
- **Visual language:** dark industrial ops theme, monospace for numbers, a clear status colour
  system, no gradients-and-glow "AI" style. Use `frontend-design` guidance when building it.

## Decisions from the builder check (Claude's calls unless noted)

- **Background Bays 3-4 play a canned loop (user's decision).** Record one routine Bay 1
  unlatch cycle once (headless, as a joint trajectory for the arm, hand and latch, plus its
  C(t) trace) and save it as a small data file. The render side replays it on Bays 3 and 4,
  phase-offset so they don't move in sync, and the console shows them NOMINAL with that C(t)
  trace. This is display only: the coordinator still treats them as static, they never stall,
  and the code labels the trace as replayed. With `--no-background` the tiles show STANDBY.

- **Rescue cameras aren't in the live scene yet.** `add_rescue_cameras` is only called in tests.
  The scene work must add them for each active bay (and the CCTV camera for every bay).
- **Who drives what.** The console is the operator's main view on the laptop, mirrored to the
  projector. Real control goes through the glove plus the UNO Q hardware console (bay select,
  clutch, E-stop). Browser keyboard events reproduce the existing `KEYMAP` keys as a backup and
  for rehearsal, so the browser is not the only control path. Reloading the page must restore
  the full view from the next telemetry message; the page keeps no state of its own.
- **Display target.** Design for 1920×1080 at 16:9, with a layout that still works at 1280×720
  and 4:3 (no clipped panels).
- **Build order and cut line.** (1) Decouple rendering and stream the four feeds plus
  telemetry to a basic console. (2) Separate the bays into cells and add lighting and
  materials. (3) Rescue mode and the side panel. (4) Final styling. After step 1 the demo is
  already better than today and can be shown, and each later step stands on its own. The
  passive viewer stays behind a flag as a last-resort fallback.
- **Tasks and ownership.** New tasks in the `T-170+` range, each listing the files it owns:
  scene/cells (`sim/bays.py`, a new `sim/cells.py`), render+stream (a new `teleop/console/`
  server package), front end (a new `teleop/console/web/`), and wiring (`commands/atc.py`,
  `configs/atc.yaml`, dependency adds). `T-150`'s code is already merged on `main`; mark it
  done before the wiring task claims `commands/atc.py`. `atc/hud.py` stays as is (still used
  for headless logs); the console reads `HudView` directly.

## Left to the builder

- Exact resolutions and JPEG quality, as long as the frame-rate goal is met. Measure before and
  after.
- Whether frames go out as JPEG over the WebSocket or MJPEG over HTTP.
- Front-end tooling (plain JS or a small vendored framework) and chart library choice.
- Exact cell dimensions, wall heights, colours and light placement.
- How the work splits into tasks. Suggested split: (a) scene and cells, (b) render and stream
  server, (c) console front end, (d) `atc` wiring.
