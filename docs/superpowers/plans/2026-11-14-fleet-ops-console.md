# Fleet Ops Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the MuJoCo passive viewer in `teleop atc` with a browser-based remote-supervision console. It shows one CCTV feed per enclosed bay cell plus fleet telemetry, and a two-angle rescue view.

**Architecture:** Physics keeps its 60 Hz `StageController.tick()` loop on the main thread and publishes a copied `SimSnapshot` each tick. A render thread owns its own `MjData`, poses it from the latest snapshot (Bays 3-4 from a canned cycle), renders the cameras in the current feed layout offscreen, JPEG-encodes them, and pushes frames plus 10 Hz telemetry JSON through a `websockets` server that also serves the static page. The browser only displays; key events go back into the existing `StageController.on_key` queue.

**Tech Stack:** Python 3.12, MuJoCo 4.0.1 (`MjSpec`, `mujoco.Renderer`), `websockets` 17 (asyncio server, HTTP via `process_request`), Pillow (JPEG), NumPy, and plain HTML/CSS/JS with no framework or CDN.

**Spec:** `docs/superpowers/specs/2026-11-14-fleet-ops-console-design.md`

## Global Constraints

- The Global Constraints in `docs/superpowers/plans/2026-11-13-m0-m1-glove-to-sim.md` apply (Python 3.12, uv only, xyzw quaternions, joints by name, no hardware in tests).
- Do not edit `src/teleop/cli.py`. The console is reached through `teleop atc`.
- Tests need no GPU, browser or internet. Rendering tests request the `offscreen_render_ok` fixture (`tests/conftest.py`); model-dependent tests use a `needs_models` skip like `tests/atc/test_stage_hud.py`.
- Front-end assets are vendored under `src/teleop/console/web/`. No CDN and no Google Fonts: use Windows system fonts (`Bahnschrift` for labels, `Cascadia Mono`/`Consolas` for numbers), with generic fallbacks.
- `--headless` and `--script` behave exactly as today and never import the renderer or server.
- The passive viewer stays available with `--display viewer`.
- The browser is never the only control path. The UNO Q console and glove behave as today.
- Before any PR, verify with: `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`
- Conventional Commits with an allowed scope (`sim`, `atc`, `cli`, `infra`, `docs`, `tasks`). Use `atc` for `teleop/console`. No `Co-Authored-By` and no "Generated with" lines.
- Branch and worktree per task, named `<type>/<kebab>`, with no task IDs in the names.

### Wire protocol (shared by Tasks T-172, T-173, T-174, T-175)

- `GET /` serves `index.html`. `GET /<file>` serves a file directly inside `web/` with extension `.html .css .js .svg .png .woff2`. Anything else, including any path containing `/` after the first or containing `..`, returns 404.
- WebSocket path `/ws`.
- **Server → client, text:** one JSON object with `"type": "telemetry"`, sent at 10 Hz:

```json
{"type": "telemetry", "layout_id": 3, "sim_t": 41.2, "wall": "14:02:31", "threshold": 0.3,
 "mode": "grid" | "rescue",
 "feeds": [{"slot": 0, "camera": "bay1/cctv", "bay": 1, "role": "tile"|"main"|"strip"}],
 "bays": [{"id": 1, "state": "STALLED", "label": "STALLED", "confidence": 0.21,
           "history": [0.93, ...], "queue_position": null, "skill": null,
           "background": false, "replayed": false}],
 "queue": [1], "selected_bay": 1,
 "rescue": {"bay": 1, "clutch": true, "elapsed_s": 3.2} | null,
 "gate": {"fit_ms": 12.0, "passed": 3, "total": 3, "elapsed_s": 0.41, "ok": true} | null,
 "events": [{"sim_t": 40.9, "wall": "14:02:30", "bay": 1, "level": "alert"|"warn"|"info"|"ok",
             "text": "Bay 1 stalled · C(t) 0.21"}],
 "compliance": "...", "reasoning": "...", "beat": "...", "unoq": "..."}
```

- **Server → client, binary:** byte 0 is the feed `slot` (0-7), byte 1 is `layout_id % 256`, and the rest is one JPEG. The client draws a frame only when its `layout_id` matches the latest telemetry.
- **Client → server, text:** `{"type": "key", "code": <int GLFW key code>}`. The server accepts integers 32..348 and ignores anything else.
- Bay `state` values are the `BayState` values plus `"STANDBY"` for a bay absent from the model. `label` is the `teleop.atc.hud.STATE_LABELS` text (`"STANDBY"` for absent bays).

## Review Focus

1. **Page reload or second tab mid-demo:** the new client gets the latest telemetry and the latest frame for every slot at once, not after the next change (T-172 `test_new_client_gets_latest_state`).
2. **Browser closed or never opened:** the sim runs normally, publishing with zero clients is a no-op, and memory doesn't grow (T-172 `test_publish_without_clients_is_noop`).
3. **A slow or stalled client** (projector laptop sleeps) holds at most one pending frame per slot plus one telemetry message. Frames are dropped, not queued (T-172 `test_slow_client_keeps_only_latest`).
4. **Rescue starts or ends mid-stream:** the layout changes and slots change meaning. A frame from the old layout must never paint into a new role (T-172 `test_layout_id_bumps_on_layout_change`, T-174 JS drops mismatched `layout_id`).
5. **A render or GL failure on the render thread** (driver hiccup, camera missing): physics, UNO Q and audio keep running, the thread logs and keeps retrying once a second, and the page shows SIGNAL LOST for stale feeds (T-173 `test_worker_survives_render_error`, T-174 stale-feed overlay).

---

## File map

| File | Task | Responsibility |
|---|---|---|
| `src/teleop/sim/cells.py` (new) | T-170 | Cell geometry, lights, CCTV camera, global visual style |
| `src/teleop/sim/bays.py` (modify) | T-170 | New `BAY_BASES`, floor size, call cells + rescue cameras, overview camera |
| `tests/sim/test_cells.py` (new) | T-170 | Cell isolation and camera tests |
| `src/teleop/console/__init__.py`, `background.py`, `data/background_cycle.npz` (new) | T-171 | Canned Bay 3-4 cycle: record, save, load, play |
| `scripts/record_background_cycle.py` (new) | T-171 | Regenerates the `.npz` |
| `tests/console/test_background.py` (new) | T-171 | |
| `src/teleop/console/layout.py`, `telemetry.py`, `server.py` (new) | T-172 | Feed layout, HudView→JSON adapter with history and event log, WS/HTTP server |
| `tests/console/test_layout.py`, `test_telemetry.py`, `test_server.py` (new) | T-172 | |
| `src/teleop/console/snapshot.py`, `render.py` (new) | T-173 | Snapshot handoff, offscreen feed renderer, render worker thread |
| `tests/console/test_render.py` (new) | T-173 | |
| `src/teleop/console/web/index.html`, `console.css`, `console.js` (new) | T-174 | The console page |
| `scripts/console_preview.py` (new) | T-174 | Fake-telemetry server for front-end work without the sim |
| `tests/console/test_web_assets.py` (new) | T-174 | |
| `src/teleop/console/run.py` (new), `src/teleop/commands/atc.py`, `configs/atc.yaml`, `pyproject.toml`/`uv.lock` (modify) | T-175 | Wiring, flags, Pillow dependency |
| `tests/console/test_run.py` (new) | T-175 | |

Dependencies: T-170, T-171 and T-172 start in parallel. T-174 needs only the protocol above plus T-172's server, so it can start alongside T-172 against `scripts/console_preview.py` once `server.py` lands. T-173 needs T-170, T-171 and T-172. T-175 needs T-173 and T-174.

**Pillow:** T-173 is the first task to need it. T-173 runs `uv add pillow` and owns that `pyproject.toml`/`uv.lock` change. T-175 makes no further dependency edits.

---

### Task T-170: Bay cells, lighting and cameras

**Files:**
- Create: `src/teleop/sim/cells.py`, `tests/sim/test_cells.py`
- Modify: `src/teleop/sim/bays.py` (`BAY_BASES`, `_new_spec` floor and light, `build_four_bay_spec`, `_add_overview_camera`)

**Interfaces:**
- Consumes: `teleop.retarget.clutch.add_rescue_cameras(spec, bay_prefix, world_latch) -> tuple[str, str]` (import it inside `build_four_bay_spec` if a top-level import cycles), and `bays.latch_world_pose`.
- Produces:
  - `teleop.sim.bays.BAY_BASES = {1: (-2.0, -2.0, 0.0), 2: (2.0, -2.0, 0.0), 3: (-2.0, 2.0, 0.0), 4: (2.0, 2.0, 0.0)}`
  - `teleop.sim.cells.cctv_camera_name(bay_id: int) -> str`, which returns `"bay{k}/cctv"`
  - `teleop.sim.cells.cell_bounds(bay_id: int) -> tuple[float, float, float, float]`, which returns the inner wall faces `(x_min, x_max, y_min, y_max)`
  - `teleop.sim.cells.WALL_HEIGHT_M = 2.2`, `CELL_HALF_M = 1.5` (cell centred on base + (0.3, 0))
  - `teleop.sim.cells.add_cell(spec, bay_id: int) -> None` adds the floor tile, jig plate under the latch, four walls named `bay{k}/cell_wall_{n,s,e,w}`, safety-yellow edge strips, lights and the CCTV camera.
  - `teleop.sim.cells.apply_visual_style(spec) -> None` adds a dark gradient skybox, dims the headlight (ambient ≤ 0.15, diffuse ≤ 0.2), sets `quality.shadowsize = 4096`, `quality.offsamples = 8`, and haze.
  - After this task, `build_four_bay_spec` adds a cell and CCTV camera for every bay present, and rescue cameras (`bay{k}/top_ortho`, `bay{k}/side_ortho`) for Bays 1 and 2. `build_single_bay_scene` stays bare: no cells and no style, because the validator speed matters.

Rules the builder must respect:
- Every cell geom is visual-only (`contype = conaffinity = 0`) and in geom group 0, so the default `MjvOption` shows it.
- The floor plane grows to `size=[7.0, 7.0, 0.1]`.
- Nothing visual sits within 1 mm of the floor plane's z=0, to avoid z-fighting. The jig plate fills the 1 cm gap under the latch track (track bottom at z = 0.01).
- Each cell has one key spot light with `castshadow=True` and at most two fill lights without shadows. Remove the global top light from `_new_spec`.
- Keep the latch's blue bolt and orange flap colours, and give the latch geoms materials with `specular ≥ 0.3`.

- [ ] **Step 1: Write the failing tests** in `tests/sim/test_cells.py` (with the `needs_models` skip)

```python
def test_every_bay_has_cctv_inside_its_cell():
    model = build_four_bay_scene()
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    for k in (1, 2, 3, 4):
        x0, x1, y0, y1 = cell_bounds(k)
        x, y, z = data.cam_xpos[model.camera(cctv_camera_name(k)).id]
        assert x0 < x < x1 and y0 < y < y1
        assert z < WALL_HEIGHT_M


def test_cctv_cannot_see_other_bays():
    # A ray from bay k's CCTV camera to each other bay's robot base (+0.5 m) hits one of bay k's walls first.
    ...  # mj.mj_ray(model, data, pnt, vec, None, 1, -1, geomid); the hit geom name starts with f"bay{k}/cell_wall"


def test_rescue_cameras_in_live_scene():
    model = build_four_bay_scene()
    for k in (1, 2):
        for name in rescue_camera_names(f"bay{k}/"):
            assert model.camera(name).id >= 0


def test_cell_geoms_do_not_collide():
    model = build_four_bay_scene()
    for i in range(model.ngeom):
        if "/cell_" in str(model.geom(i).name):
            assert model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0


def test_cells_cast_shadows():
    model = build_four_bay_scene()
    assert sum(bool(model.light_castshadow[i]) for i in range(model.nlight)) >= 4
```

- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/sim/test_cells.py -v`. Expected: ImportError for `teleop.sim.cells`.
- [ ] **Step 3: Implement `cells.py` and the `bays.py` changes** as specified above. Aim the CCTV camera from a high inner corner of the cell at about base + (0.35, 0, 0.25) (`fovy` around 55) so the robot, jig and latch fill the frame. Put the overview camera high above the origin, looking down into all four open-top cells.
- [ ] **Step 4: Run the new tests plus the existing sim and ATC suites.** Run `uv run pytest tests/sim tests/atc tests/retarget -v`. Expected: all pass. The coordinator and the 6-beat `--script` run are unchanged, because latch poses are bay-local.
- [ ] **Step 5: Check the look by eye.** Render each CCTV camera to a PNG in the scratchpad with a short throwaway script (not committed), and look at it. The neighbouring bays must not be visible, the shadows must read, and the robot must not be clipped. Iterate on light and camera values.
- [ ] **Step 6: Verify and commit.** Run the full verification command, then `git commit -m "feat(sim): enclosed bay cells with lighting and per-bay cctv cameras"`.

---

### Task T-171: Canned background cycle for Bays 3-4

**Files:**
- Create: `src/teleop/console/__init__.py`, `src/teleop/console/background.py`, `src/teleop/console/data/background_cycle.npz`, `scripts/record_background_cycle.py`, `tests/console/__init__.py` (only if the tests tree uses package `__init__`s; follow `tests/atc`), `tests/console/test_background.py`

**Interfaces:**
- Produces:
  - `DEFAULT_CYCLE_PATH = Path(__file__).parent / "data" / "background_cycle.npz"`
  - `BLEND_S = 0.5`, `PHASE_OFFSETS = {3: 0.37, 4: 0.71}` (fractions of the cycle)
  - `@dataclass(frozen=True) class BackgroundCycle`: `t: NDArray[float64]` with shape (N,) starting at 0; `joint_names: tuple[str, ...]` unprefixed (e.g. `"joint1"`, `"if_mcp"`, latch joints); `qpos_widths: tuple[int, ...]`; `qpos: NDArray` with shape (N, sum(widths)); `confidence: NDArray` with shape (N,). Plus a `duration_s` property.
  - `record_cycle(max_sim_s: float = 30.0) -> BackgroundCycle` builds a `StageController` headless (local-mock UNO Q, local compliance and reasoner, `NullSpeaker`, no background bays), ticks at 60 Hz, and samples every `bay1/` joint's qpos plus Bay 1's `BaySnapshot.confidence`. It samples from the first tick until the controller starts Bay 1's second cycle, so the clip includes the dwell. It raises `RuntimeError` if `max_sim_s` passes first.
  - `save_cycle(cycle: BackgroundCycle, path: Path) -> None` and `load_cycle(path: Path = DEFAULT_CYCLE_PATH) -> BackgroundCycle` (NumPy `.npz`).
  - `class BackgroundPlayer`: `__init__(self, cycle: BackgroundCycle, model: Any, bays: Sequence[int] = (3, 4))` resolves `bay{k}/<joint>` qpos addresses by name and skips bays absent from the model. `apply(self, data: Any, wall_t: float) -> dict[int, float]` writes each bay's interpolated qpos into `data` (it does not call `mj_forward`) and returns `{bay: confidence}`. Phase is `(wall_t / duration + PHASE_OFFSETS[bay]) % 1`. In the last `BLEND_S` seconds it linearly blends towards frame 0 so the wrap doesn't jump, except for latch joints, which snap (a new pallet).

- [ ] **Step 1: Write the failing tests**

```python
def test_roundtrip(tmp_path):
    c = BackgroundCycle(
        t=np.array([0.0, 1.0]),
        joint_names=("joint1",),
        qpos_widths=(1,),
        qpos=np.array([[0.0], [1.0]]),
        confidence=np.array([0.9, 0.95]),
    )
    save_cycle(c, tmp_path / "c.npz")
    back = load_cycle(tmp_path / "c.npz")
    assert back.joint_names == ("joint1",) and np.allclose(back.qpos, c.qpos)


def test_committed_cycle_is_plausible():
    c = load_cycle()
    assert 2.0 < c.duration_s < 30.0
    assert c.confidence.min() > 0.85  # a routine, never-stalling cycle
    assert "joint1" in c.joint_names


@needs_models
def test_player_moves_bay3_and_differs_from_bay4():
    model = build_four_bay_scene()
    data = mj.MjData(model)
    player = BackgroundPlayer(load_cycle(), model)
    conf = player.apply(data, wall_t=1.0)
    assert set(conf) == {3, 4}
    q3 = data.qpos[model.joint("bay3/joint1").qposadr[0]]
    q4 = data.qpos[model.joint("bay4/joint1").qposadr[0]]
    assert q3 != pytest.approx(q4)


@needs_models
def test_player_skips_absent_bays():
    model = build_four_bay_scene(include_background_bays=False)
    assert BackgroundPlayer(load_cycle(), model).apply(mj.MjData(model), 0.0) == {}


def test_wrap_is_continuous_for_arm_joints():
    # With a synthetic 2-frame cycle, the arm qpos at phase 1-ε and at phase 0 differ by < 1e-3.
    ...
```

- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/console/test_background.py -v`. Expected: ImportError.
- [ ] **Step 3: Implement `background.py`** as specified, and write `scripts/record_background_cycle.py` (calls `record_cycle()` then `save_cycle(..., DEFAULT_CYCLE_PATH)` and prints duration and minimum C(t)).
- [ ] **Step 4: Generate the data file.** Run `uv run python scripts/record_background_cycle.py`. Expected: it prints a duration between 2 and 30 s and a minimum C above 0.85. Commit the resulting `.npz`; it should be under 1 MB.
- [ ] **Step 5: Run the tests and confirm they pass.** Run `uv run pytest tests/console/test_background.py -v`.
- [ ] **Step 6: Verify and commit.** `git commit -m "feat(atc): canned routine cycle for background bays 3-4"`

---

### Task T-172: Console protocol: feed layout, telemetry adapter and server

**Files:**
- Create: `src/teleop/console/layout.py`, `telemetry.py`, `server.py`, `tests/console/test_layout.py`, `test_telemetry.py`, `test_server.py`
- (Creates `src/teleop/console/__init__.py` if T-171 hasn't landed yet. Keep it docstring-only so both tasks merge cleanly.)

**Interfaces:**
- Consumes: `teleop.atc.hud.HudView`, `STATE_LABELS`, `teleop.atc.types.BayState`, and `teleop.retarget.clutch.rescue_camera_names`. CCTV camera names are `f"bay{k}/cctv"`. `layout.py` builds them itself from a `CCTV_SUFFIX = "cctv"` constant, so it doesn't depend on T-170. T-173's rendering tests catch any mismatch.
- Produces:
  - `layout.py`: `FeedRole = Literal["tile", "main", "strip"]`; `@dataclass(frozen=True) class FeedSpec(slot: int, camera: str, bay: int, role: FeedRole, width: int, height: int)`; `TILE = (640, 360)`, `MAIN = (960, 540)`, `STRIP = (384, 216)`; `feed_layout(bays: Sequence[int], rescue_bay: int | None) -> tuple[FeedSpec, ...]`. In the grid, slot i is the CCTV tile of `sorted(bays)[i]`. In rescue, slot 0 is the rescued bay's `top_ortho` (main), slot 1 is its `side_ortho` (main), and slots 2.. are the CCTV strip of every bay in `sorted(bays)`.
  - `telemetry.py`: `THRESHOLD = 0.3`, `HISTORY_LEN = 150`, `EVENT_LOG_LEN = 50`; `class TelemetryAdapter(bays_present: Sequence[int], background: Sequence[int], clock: Callable[[], datetime] = datetime.now)` with `build(self, view: HudView, sim_t: float, feeds: Sequence[FeedSpec], background_confidence: Mapping[int, float]) -> dict[str, Any]`, which returns the protocol telemetry dict. Every call appends one history sample per bay (the caller calls it at 10 Hz). `layout_id` increments whenever `feeds` differs from the previous call.
  - Event rules (state transitions per bay, from the previous `build` call):

    | New state | level | text |
    |---|---|---|
    | STALLED | alert | `Bay {k} stalled · C(t) {c:.2f}` |
    | QUEUED | warn | `Bay {k} queued #{pos+1}` |
    | RESCUE | info | `Operator took control of Bay {k}` |
    | VALIDATING | info | `Bay {k}: validating learned skill` |
    | AUTONOMOUS_SKILL | ok | `Bay {k}: replaying learned skill` |
    | OOD_ESCALATED | alert | `Bay {k}: unseen jam, skill transfer refused` |
    | NOMINAL (from anything else) | ok | `Bay {k} back to nominal` |

    A change in `gate.ok`/`passed` gives `Skill gate {passed}/{total} passed in {elapsed:.2f} s` (ok) or `... FAILED` (alert). A change in the `compliance` or `reasoning` line gives `info` with that line as the text.
  - Background bays: `replayed: true`, state `NOMINAL`, confidence taken from `background_confidence`. Bays 1..4 absent from `bays_present` get state `STANDBY` and confidence 0 with empty history.
  - `server.py`: `WEB_DIR = Path(__file__).parent / "web"`; `class ConsoleServer(web_dir: Path = WEB_DIR, host: str = "127.0.0.1", port: int = 8765, on_key: Callable[[int], None] | None = None)` with `start() -> None` (runs an asyncio loop on a daemon thread and returns once bound; raises `OSError` if the port is taken), `stop() -> None`, `port: int` (the actual bound port, so `port=0` works in tests), `url: str`, `client_count: int`, `publish_telemetry(msg: dict[str, Any]) -> None`, and `publish_frame(slot: int, layout_id: int, jpeg: bytes) -> None`. Both publish methods are thread-safe and non-blocking. The server keeps the latest telemetry and latest frame per slot, sends them to each new client right away, and holds at most one pending item per slot plus one telemetry message per client.

- [ ] **Step 1: Write the failing tests**

```python
# test_layout.py
def test_grid_layout():
    f = feed_layout([1, 2, 3, 4], None)
    assert [(s.slot, s.camera, s.role) for s in f] == [
        (0, "bay1/cctv", "tile"),
        (1, "bay2/cctv", "tile"),
        (2, "bay3/cctv", "tile"),
        (3, "bay4/cctv", "tile"),
    ]


def test_rescue_layout():
    f = feed_layout([1, 2, 3, 4], 2)
    assert (f[0].camera, f[0].role) == ("bay2/top_ortho", "main")
    assert (f[1].camera, f[1].role) == ("bay2/side_ortho", "main")
    assert [s.camera for s in f[2:]] == [f"bay{k}/cctv" for k in (1, 2, 3, 4)]
    assert all(s.role == "strip" for s in f[2:])


# test_telemetry.py  (build HudView fixtures like tests/atc/test_stage_hud.py::_view)
def test_stall_transition_logs_alert(): ...  # NOMINAL then STALLED -> one alert event, text "Bay 1 stalled · C(t) 0.21"
def test_history_is_bounded(): ...  # 200 builds -> len(history) == HISTORY_LEN
def test_absent_bays_are_standby(): ...  # bays_present=(1,2), background=() -> bays 3,4 STANDBY
def test_background_bays_marked_replayed(): ...  # background=(3,4), background_confidence={3: 0.91} -> replayed true, 0.91
def test_layout_id_bumps_on_layout_change(): ...  # same feeds -> same id; rescue feeds -> id+1
def test_output_is_json_serialisable(): ...  # json.dumps(adapter.build(...)) does not raise


# test_server.py  (asyncio client from websockets.asyncio.client; server on port=0)
def test_serves_index_and_rejects_traversal(): ...  # GET / 200 text/html; GET /../pyproject.toml 404; GET /x.py 404
def test_new_client_gets_latest_state(): ...  # publish telemetry + frame(slot 0) before connecting -> both arrive first
def test_key_messages_reach_callback(): ...  # send {"type":"key","code":49} -> on_key(49); code 9999 and junk text ignored
def test_publish_without_clients_is_noop(): ...  # 10_000 publishes with no client: no exception; client_count == 0
def test_slow_client_keeps_only_latest(): ...  # client connects, doesn't read; 500 frames to slot 0; then read -> latest frame arrives, total received ≤ 3
def test_port_in_use_raises(): ...  # second server on the same bound port -> OSError from start()
```

- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/console/test_layout.py tests/console/test_telemetry.py tests/console/test_server.py -v`. Expected: ImportError.
- [ ] **Step 3: Implement `layout.py`.**
- [ ] **Step 4: Implement `telemetry.py`.** `wall` is `clock().strftime("%H:%M:%S")`; `label` comes from `STATE_LABELS`.
- [ ] **Step 5: Implement `server.py`** with `websockets.asyncio.server.serve`. Serve HTTP through `process_request` returning a `websockets.http11.Response` with the right `Content-Type`, `Cache-Control: no-store`. Each client gets an `asyncio.Event` plus a dict `{slot: (layout_id, bytes)}` and one telemetry slot that a sender task drains. Publishers hand off with `loop.call_soon_threadsafe`.
- [ ] **Step 6: Run the tests and confirm they pass.**
- [ ] **Step 7: Verify and commit.** `git commit -m "feat(atc): console feed layout, telemetry adapter and websocket server"`

---

### Task T-173: Offscreen feed renderer and render worker

**Files:**
- Create: `src/teleop/console/snapshot.py`, `src/teleop/console/render.py`, `tests/console/test_render.py`
- Modify: `pyproject.toml`, `uv.lock` (`uv add pillow` only)

**Interfaces:**
- Consumes: T-170 cameras and `rescue_view_option()`; T-171 `BackgroundPlayer`; T-172 `FeedSpec`, `feed_layout`, `TelemetryAdapter`, `ConsoleServer.publish_frame/publish_telemetry`.
- Produces:
  - `snapshot.py`: `@dataclass(frozen=True) class SimSnapshot(qpos: NDArray, mocap_pos: NDArray, mocap_quat: NDArray, view: HudView, sim_t: float)` with `SimSnapshot.capture(data: Any, view: HudView, sim_t: float) -> SimSnapshot` (copies the arrays). `class LatestSlot(Generic[T])` has `put(item: T) -> None`, `get() -> T | None` and `version: int`, guarded by a lock.
  - `render.py`: `JPEG_QUALITY = 80`. `class FeedRenderer(model: Any, background: BackgroundPlayer | None, quality: int = JPEG_QUALITY)` must be constructed and used on one thread. It owns an `MjData` and one `mujoco.Renderer` per `(width, height)`. `render(self, snap: SimSnapshot, feeds: Sequence[FeedSpec], wall_t: float) -> tuple[list[tuple[int, bytes]], dict[int, float]]` copies qpos/mocap in, applies the background player, runs `mj_kinematics` and `mj_camlight`, renders each feed (`role == "main"` uses `rescue_view_option()`), and returns `[(slot, jpeg)]` plus the background confidences.
  - `class RenderWorker(threading.Thread)`: `__init__(self, model: Any, snapshots: LatestSlot[SimSnapshot], server: ConsoleServer, adapter: TelemetryAdapter, bays_present: Sequence[int], background: BackgroundCycle | None, fps: int = 30, telemetry_hz: float = 10.0)`, plus `stop() -> None` and `stats: RenderStats` (`frames: int`, `errors: int`, `mean_frame_ms: float`). The worker builds the `FeedRenderer` and `BackgroundPlayer` inside `run()` (the GL context belongs to this thread). Each frame it gets the latest snapshot, computes `feed_layout(bays_present, snap.view.rescue.bay_id)`, renders and publishes. At `telemetry_hz` it calls `adapter.build(...)` and publishes telemetry. Any exception in a frame is logged with `logger.exception`, increments `errors`, sleeps 1 s and continues. Telemetry keeps flowing even when rendering keeps failing.

- [ ] **Step 1: Write the failing tests** (rendering tests use `offscreen_render_ok` and `needs_models`)

```python
def test_latest_slot_returns_newest():
    s = LatestSlot[int]()
    assert s.get() is None
    s.put(1)
    s.put(2)
    assert s.get() == 2 and s.version == 2


def test_snapshot_copies(): ...  # mutate data.qpos after capture -> snapshot unchanged


def test_renders_grid_feeds_as_jpeg(offscreen_render_ok):
    model = build_four_bay_scene()
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    r = FeedRenderer(model, BackgroundPlayer(load_cycle(), model))
    frames, conf = r.render(
        SimSnapshot.capture(data, VIEW, 0.0), feed_layout([1, 2, 3, 4], None), 0.0
    )
    assert [s for s, _ in frames] == [0, 1, 2, 3]
    assert all(b[:2] == b"\xff\xd8" for _, b in frames)  # JPEG SOI
    assert set(conf) == {3, 4}


def test_renders_rescue_feeds(
    offscreen_render_ok,
): ...  # feed_layout([1,2,3,4], 1) -> 6 frames, slot 0 is 960x540 (decode with PIL)


def test_worker_publishes_frames_and_telemetry(
    offscreen_render_ok,
): ...  # FakeServer records calls; run ~0.5 s; ≥1 telemetry, ≥1 frame per grid slot


def test_worker_survives_render_error(): ...  # FeedRenderer.render monkeypatched to raise; worker still publishes telemetry, stats.errors ≥ 1, thread alive
```

- [ ] **Step 2: Run them and confirm they fail.** Run `uv run pytest tests/console/test_render.py -v`.
- [ ] **Step 3: Add Pillow.** Run `uv add pillow`.
- [ ] **Step 4: Implement `snapshot.py` and `render.py`.**
- [ ] **Step 5: Run the tests and confirm they pass.**
- [ ] **Step 6: Measure.** Write a throwaway scratchpad script (not committed) that runs a `RenderWorker` against a fake server for 10 s in grid and in rescue layout and prints `stats.mean_frame_ms`. Expected: the mean frame time is at most 33 ms in both layouts on the RTX 4060 laptop. If not, lower `STRIP`/`TILE` sizes or the JPEG quality, record the numbers in the PR description, and only then commit.
- [ ] **Step 7: Verify and commit.** `git commit -m "feat(atc): offscreen cctv feed renderer and render worker thread"`

---

### Task T-174: Console front end

**Files:**
- Create: `src/teleop/console/web/index.html`, `console.css`, `console.js`, `scripts/console_preview.py`, `tests/console/test_web_assets.py`

**Interfaces:**
- Consumes: the wire protocol above and T-172's `ConsoleServer`.
- Produces: the page. `scripts/console_preview.py [--port 8765] [--rescue]` runs a `ConsoleServer` that publishes synthetic telemetry walking the stage script (nominal → Bay 1 stall → rescue → gate pass → Bay 2 auto skill → OOD), plus placeholder JPEG frames drawn with Pillow (label text on a dark frame). Front-end work and screenshots need no sim.

What the page must do:
- **Layout at 1920×1080:** a header bar with `FLEET OPS`, bay count, operator count, a `● LIVE` indicator and the wall clock. The main area is the 2×2 CCTV grid; on the right is a 360-420 px side panel. It must still work at 1280×720 and 1024×768, with no clipped panels and no page scroll.
- **Each tile:** the image fills the tile. The CSS overlay shows the bay label top-left, the state badge and C(t) value top-right, the `REC ●` blink and timestamp bottom-left, `REPLAY` in small type for background bays, plus a subtle vignette and scanlines. The tile border takes the state colour; STALLED pulses. STANDBY tiles show "NO SIGNAL · STANDBY" with no image.
- **Rescue mode** (`mode == "rescue"`): two `main` feeds side by side, labelled TOP and SIDE, with a `LIVE CONTROL · BAY k` banner, a clutch indicator (`CLUTCH HELD` filled vs `CLUTCH RELEASED` outline) and the rescue timer. The `strip` feeds run as a row below. The switch animates in no more than 250 ms.
- **Side panel:** the queue as chips in order; C(t) sparklines per bay on `<canvas>` with a dashed line at `threshold`; a gate card; the event log (newest first, coloured by `level`); the compliance and reasoning lines; and the UNO Q label in the footer.
- **Frames:** each binary message updates its slot's `<img>` through a Blob URL, revoking the previous one. Frames whose `layout_id` differs from the current one are dropped. A feed with no frame for 2 s shows a `SIGNAL LOST` overlay.
- **Connection:** reconnect with 1 s → 5 s backoff. While disconnected, a full-width `LINK DOWN · reconnecting` bar shows and the last state stays on screen.
- **Keys:** `keydown` without repeat maps digits to 48-57, letters to their uppercase code 65-90 and Space to 32, then sends `{"type":"key","code":n}` and calls `preventDefault` for Space. Other keys are ignored.
- **Status colours** (CSS variables): NOMINAL green, STALLED red, QUEUED amber, RESCUE cyan, VALIDATING violet, AUTO SKILL teal, OOD - HUMAN magenta, STANDBY grey. The theme is a dark industrial ops console (near-black panels, 1 px hairlines, no gradients or glow). Fonts are `Bahnschrift` and `Cascadia Mono`/`Consolas`. Load `frontend-design` guidance before styling.

- [ ] **Step 1: Write the failing tests**

```python
def test_index_references_only_local_assets():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert "console.css" in html and "console.js" in html
    assert "http://" not in html and "https://" not in html


def test_js_speaks_protocol():
    js = (WEB_DIR / "console.js").read_text(encoding="utf-8")
    for token in ('"/ws"', "layout_id", '"key"', "SIGNAL LOST"):
        assert token in js


def test_server_serves_assets():  # ConsoleServer(port=0) -> GET /console.css and /console.js are 200 with css/js content types
    ...
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Write `scripts/console_preview.py`.**
- [ ] **Step 4: Build `index.html`, `console.css` and `console.js`.**
- [ ] **Step 5: Run the tests and confirm they pass.**
- [ ] **Step 6: Check the page in a browser.** Run `uv run python scripts/console_preview.py`, open it with the Playwright browser tools, and take screenshots at 1920×1080, 1280×720 and 1024×768 in grid and rescue mode. Expected: nothing clipped, labels legible, layout switch clean. Also kill the preview server and check that the LINK DOWN bar appears, then restart it and check the page recovers.
- [ ] **Step 7: Verify and commit.** `git commit -m "feat(atc): fleet ops console web front end and preview server"`

---

### Task T-175: Wire the console into `teleop atc`

**Files:**
- Create: `src/teleop/console/run.py`, `tests/console/test_run.py`
- Modify: `src/teleop/commands/atc.py` (`AtcConfig`, `_parser`, `_run`, module docstring), `configs/atc.yaml`

**Interfaces:**
- Consumes: `RenderWorker`, `SimSnapshot`, `LatestSlot` (T-173); `ConsoleServer`, `TelemetryAdapter` (T-172); `load_cycle` (T-171); `StageController` (`tick`, `on_key`, `hud_view`, `coordinator.model/.data/.time_s`, `background_bays`).
- Produces:
  - `AtcConfig` fields `display: str = "console"` (`"console"` or `"viewer"`), `console_port: int = 8765`, `console_fps: int = 30`, `open_browser: bool = True`. The flags are `--display {console,viewer}`, `--console-port N`, `--console-fps N` and `--no-browser` (`dest="open_browser"`). Add the same keys to `configs/atc.yaml`, with comments.
  - `teleop.console.run.run_console(controller: StageController, operator: Callable[[], OperatorInput | None], *, port: int, fps: int, open_browser: bool, steps: int | None = None) -> LoopStats` starts the `ConsoleServer` (with `on_key=controller.on_key`) and the `RenderWorker`. It logs and prints the URL, opens the browser with `webbrowser.open` when asked, then runs `FixedRateLoop(60)`: `controller.tick(operator())`, then `snapshots.put(SimSnapshot.capture(...))`. It stops the worker and server in `finally`. `bays_present` comes from the model (the bays whose `bay{k}/cctv` camera exists); `background` is `load_cycle()` when `controller.background_bays` is non-empty, otherwise `None`.
  - `_run`: when not headless, `display == "console"` calls `run_console(...)` and `"viewer"` calls the existing `_run_viewer`. An `OSError` from a busy port prints `teleop atc: console port {port} in use (try --console-port)` and exits 1.

- [ ] **Step 1: Write the failing tests**

```python
def test_config_defaults_and_flags():
    cfg = load_config(None, {})
    assert (cfg.display, cfg.console_port, cfg.console_fps, cfg.open_browser) == (
        "console",
        8765,
        30,
        True,
    )
    ns = _parser().parse_args(["--display", "viewer", "--no-browser", "--console-port", "9000"])
    assert ns.display == "viewer" and ns.open_browser is False and ns.console_port == 9000


def test_shipped_yaml_loads():
    assert load_config(DEFAULT_CONFIG, {}).display == "console"


@needs_models
def test_run_console_ticks_and_serves(offscreen_render_ok):
    ...
    # controller as in tests/atc/test_stage_hud.py; run_console(..., port=0, open_browser=False, steps=120)
    # -> returns LoopStats with steps == 120 and the coordinator time advanced by ~2 s


def test_headless_does_not_import_console(monkeypatch):
    ...
    # main(["--headless", "--steps", "1", "--no-audio"]) with sys.modules guard -> "teleop.console.render" not imported


def test_busy_port_exits_1(
    capsys,
): ...  # occupy a port, main([... "--console-port", str(p), "--no-browser", "--steps", "1"]) -> 1 and the message
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement `run.py` and the `atc.py`/YAML changes.** Update the module docstring to describe the console and the `--display viewer` fallback.
- [ ] **Step 4: Run the tests and confirm they pass.** Then run `uv run pytest tests/atc -v` to confirm the 6-beat `--script` headless run still passes.
- [ ] **Step 5: Do an end-to-end run on the laptop.** Run `uv run teleop atc --script`. The browser opens and the full 6-beat script plays in the console: four live feeds with Bays 3-4 moving, the Bay 1 stall shown red and logged, the rescue view with two angles, gate 3/3, Bay 2 auto skill, and the OOD alert. Watch it at 1080p and confirm smooth video with no hitch at the gate. Then run `uv run teleop atc --display viewer` and confirm the old viewer still works.
- [ ] **Step 6: Verify and commit.** `git commit -m "feat(cli): run teleop atc through the fleet ops console by default"`
