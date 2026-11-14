# Console Pages and Always-On Data Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the fleet ops console into Operate / Data & training / Setup & health pages, record every rescue by default with dashboard switches, and persist the DMP skill library across runs, with a real board-side "start fresh".

**Architecture:** A new board op `clear_skills` makes the UNO Q forgettable. `atc/skill_store.py` owns `data/skills/library.json` (load at startup, atomic save on deploy, reset). `EpisodeSink` gains per-source pause switches and session stats; `record/pile.py` totals all sessions. `atc/data_controls.py` joins sink, store and pile into one `DataControls` that the console reads (`data` telemetry object) and drives (`record` / `skills` client messages). The console becomes one HTML document with hash-routed tabs over the existing single WebSocket.

**Tech Stack:** Python 3.12, uv, MuJoCo, LeRobot 0.7 (optional `train` group), `websockets`, plain HTML/CSS/JS.

**Spec:** `docs/superpowers/specs/2026-11-14-console-pages-and-data-collection-design.md`

## Global Constraints

- The M0/M1 plan's Global Constraints apply (Python 3.12, uv only, xyzw quaternions, joints by name, no hardware in tests).
- Do not edit `src/teleop/cli.py`. New behaviour goes through `teleop atc` flags in `src/teleop/commands/atc.py`.
- With no saved skills, `recovery_backend: dmp` stays byte-for-byte today's behaviour; the headless DMP stage golden (`tests/atc/golden`) must not change.
- Headless runs: no default recording, no default skill library. `--record-episodes DIR` keeps today's `session_root` behaviour.
- Live defaults: episodes root `data/episodes/`, always a new `<YYYYmmdd-HHMMSS>` subfolder; skill library `data/skills/library.json`.
- No disk I/O on the 60 Hz control thread or inside `TelemetryAdapter.build`.
- `lerobot` missing: recording shows unavailable with `uv sync --group train`; `teleop atc` still starts.
- No new JS/npm deps; one WebSocket; one page load.
- Per-fault training target: `PILE_TARGET_PER_FAULT = 50`.
- Verify before PR: `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`.
- Conventional Commits; no `Co-Authored-By` or "Generated with" lines.

## Review Focus

1. **Corrupt or foreign `library.json`** (truncated write, older version, other hand's actuators): `teleop atc` starts, skips the bad skills with a warning, and the file is not overwritten until a new skill deploys. Test in Task 2.
2. **Board on old firmware** (`clear_skills` answers "unknown op"): laptop library still resets, Data page shows "board not cleared: redeploy the UNO Q app". Test in Task 2.
3. **Switch flipped mid-rescue**: the open episode still saves; only the next one is skipped. Test in Task 3.
4. **Start fresh while a bay is running a DMP skill**: the running recovery finishes with its own skill object; no crash, the skill is not re-saved. Test in Task 2.
5. **A session folder with no sidecar or a half-written last line** under `data/episodes/`: the all-sessions scan skips it or the bad line, never raises. Test in Task 3.

---

## File Ownership

| Task | Files |
|---|---|
| T-230 | `src/teleop/unoq/{protocol,server,classifier,client,switch}.py`, `tests/unoq/test_clear_skills.py` |
| T-231 | new `src/teleop/atc/skill_store.py`, `src/teleop/atc/coordinator.py` (hook + reset only), `src/teleop/commands/atc.py` (skills flags), `tests/atc/test_skill_store.py` |
| T-232 | `src/teleop/atc/episodes.py`, new `src/teleop/record/pile.py`, `src/teleop/commands/atc.py` (recording defaults), `tests/atc/test_episodes.py`, `tests/record/test_pile.py` |
| T-233 | `src/teleop/console/web/{index.html,console.js,console.css,health.js}`, `tests/console/test_web_assets.py` |
| T-234 | new `src/teleop/atc/data_controls.py`, new `src/teleop/console/web/data.js`, `src/teleop/console/{server,run,telemetry}.py`, `src/teleop/commands/atc.py` (wiring only), `tests/console/*`, `tests/atc/test_data_controls.py` |

T-231 and T-232 both touch `commands/atc.py` argument parsing; whichever merges second rebases. T-197 and T-215 (pending console tasks) build on T-233's pages: recovery chip on Operate, copilot call-outs on Operate, reviews on Data.

---

### Task 1 (T-230): Board `clear_skills` op

**Files:**
- Modify: `src/teleop/unoq/protocol.py`, `server.py`, `classifier.py`, `client.py`, `switch.py`
- Test: `tests/unoq/test_clear_skills.py`

**Interfaces:**
- Produces: `P.OP_CLEAR_SKILLS = "clear_skills"` (TCP op); `MahalanobisStallClassifier.clear(part_sku_id: str) -> int`; `UnoQClient.clear_skills(part_sku_id: str) -> int` and `SwitchableUnoQ.clear_skills(part_sku_id: str) -> int`, both returning the number of skills removed and raising `UnoQError` on an "unknown op" reply.

- [ ] **Step 1: Write the failing tests** (use the in-process `UnoQServer` + `UnoQClient` fixture pattern from `tests/unoq/test_edge_dmp.py`)

```python
def test_clear_skills_forgets_skills_and_prototypes(board):
    client, skill = board.client, board.registered_skill()  # registered for SKU "pack-a"
    assert client.clear_skills("pack-a") == 1
    match, skill_id = client.classify_stall("pack-a", skill.prototype)
    assert match.is_ood and skill_id is None


def test_clear_skills_resets_signature_numbering(board):
    board.registered_skill()
    board.client.clear_skills("pack-a")
    assert board.fit_new_skill().signature_id == "Jam_Sig_1"


def test_clear_skills_leaves_other_skus(board): ...  # skill on "pack-b" still classifies


def test_switch_clear_empties_replay_list(switch_with_two_nodes):
    sw = switch_with_two_nodes
    sw.register_skill(skill)
    sw.clear_skills("pack-a")
    sw.swap(second_node)
    assert second_node.server.skills == {}


def test_old_firmware_raises_unoq_error(server_without_clear_op):
    with pytest.raises(UnoQError, match="unknown op"):
        client.clear_skills("pack-a")
```

- [ ] **Step 2: Run** `uv run pytest tests/unoq/test_clear_skills.py -v` → FAIL (`OP_CLEAR_SKILLS` / `clear_skills` missing).
- [ ] **Step 3: Implement.** Add the op to `protocol.py` and its TCP op set. `classifier.clear` drops `self._protos[part_sku_id]` and returns how many signatures went. `server._clear_skills` removes every `self.skills` entry with that SKU, calls `classifier.clear`, replies `P.ok(removed=n)`. `client.clear_skills` sends `{"op": OP_CLEAR_SKILLS, "part_sku_id": sku}` and returns `int(reply["removed"])`. `switch.clear_skills` runs through `_call` and, under the same pointer lock as `register_skill`, removes that SKU's skills from `self._skills`.
- [ ] **Step 4: Run** `uv run pytest tests/unoq -v` → PASS.
- [ ] **Step 5: Commit** `feat(unoq): clear_skills op to forget a SKU's skills and prototypes (T-230)`.
- [ ] **Step 6 (hardware, manual):** redeploy the board app to the UNO Q per `README` / `docs/` UNO Q deploy notes, then `uv run teleop unoq` ping and a `clear_skills` call against the real board succeed. Record the result in the PR description.

### Task 2 (T-231): Skill library on disk

**Files:**
- Create: `src/teleop/atc/skill_store.py`
- Modify: `src/teleop/atc/coordinator.py` (after `self.library[skill.skill_id] = skill` at ~line 1471; new `reset_library`), `src/teleop/commands/atc.py`
- Test: `tests/atc/test_skill_store.py`

**Interfaces:**
- Consumes: `clear_skills(part_sku_id) -> int` (Task 1), `SkillPrimitive.to_dict/from_dict`, `UnoQClient.register_skill`.
- Produces:
  - `SkillStore(path: Path, *, session: Callable[[], str | None] = lambda: None)`; `.load(coordinator: AtcCoordinator) -> LoadReport`; `.on_deployed(skill: SkillPrimitive) -> None` (queues an atomic rewrite on its own daemon thread); `.reset(coordinator) -> ResetReport`; `.entries() -> list[SkillEntry]`; `.close()` (flushes the writer).
  - `SkillEntry(skill_id: str, signature_id: str, learned_at: str, session: str | None)`; `LoadReport(loaded: int, skipped: list[str])`; `ResetReport(board_cleared: bool, backup: Path | None)`.
  - `AtcCoordinator.skill_listeners: list[Callable[[SkillPrimitive], None]]` (called after a gate pass deploys a skill); `AtcCoordinator.reset_library() -> bool` (clears `self.library`, calls `self.unoq.clear_skills(cfg.part_sku_id)`, returns `False` if that raised `UnoQError`).
  - `teleop atc` flags: `--skills PATH` (live default `data/skills/library.json`, headless default none), `--fresh-skills`.
- File format (spec): `{"version": 1, "skills": [{"skill": <to_dict>, "learned_at": ISO, "session": str|null}]}`.

- [ ] **Step 1: Write the failing tests** (fake coordinator with a real loopback `UnoQServer`; `tmp_path` library)

```python
def test_deploy_then_load_restores_library_and_board(tmp_path, coord_factory, skill):
    store = SkillStore(tmp_path / "library.json")
    store.on_deployed(skill); store.close()
    fresh = coord_factory()
    assert SkillStore(tmp_path / "library.json").load(fresh).loaded == 1
    assert fresh.library[skill.skill_id] == skill
    assert fresh.unoq.classify_stall(skill.part_sku_id, skill.prototype)[1] == skill.skill_id

def test_load_clears_board_first(...)  # board had a stale skill not in the file -> gone after load
def test_load_skips_mismatched_actuators_and_bad_json(...)  # LoadReport.skipped names them; file untouched
def test_truncated_file_loads_nothing_and_is_not_overwritten(...)
def test_save_is_atomic(...)  # writer leaves no *.tmp; file always parses
def test_reset_backs_up_file_and_clears_everything(...)  # library-<ts>.json.bak exists; library {}; board empty
def test_reset_with_old_firmware_reports_board_not_cleared(...)  # ResetReport.board_cleared is False
def test_reset_during_autonomous_skill_lets_it_finish(...)  # bay.auto.skill still set; tick runs; nothing re-saved
def test_fresh_skills_ignores_file(...)  # --fresh-skills: load not called, board cleared, file replaced on next deploy
def test_headless_has_no_default_library(...)  # AtcConfig parse: headless -> skills None; live -> data/skills/library.json
```

- [ ] **Step 2: Run** `uv run pytest tests/atc/test_skill_store.py -v` → FAIL.
- [ ] **Step 3: Implement** `skill_store.py`. Rewrite = write `path.with_suffix(".json.tmp")` then `os.replace`. The in-memory entry list is guarded by a lock, because `on_deployed` runs on the gate-apply path and `reset` runs on the WebSocket thread. `load` calls `coordinator.reset_library()` first, then `register_skill` and inserts into `library` for each valid entry. Coordinator: call each `skill_listeners` entry right after `self.library[...] = skill`, wrapped in `try/except Exception: logger.exception`.
- [ ] **Step 4: Wire in `commands/atc.py`.** After `_setup_recovery`: if `cfg.skills` is set, build a `SkillStore`. Call `load` (or `reset_library()` when `--fresh-skills`), append `store.on_deployed` to `skill_listeners`, and add `store.close` to `stack.callback`. Print `teleop atc: loaded N skills from PATH (skipped M)`.
- [ ] **Step 5: Run** `uv run pytest tests/atc -v` → PASS, golden unchanged.
- [ ] **Step 6: Commit** `feat(atc): persist the DMP skill library across runs (T-231)`.

### Task 3 (T-232): Recording on by default, switches, session stats and pile totals

Depends on T-191 merged.

**Files:**
- Modify: `src/teleop/atc/episodes.py`, `src/teleop/commands/atc.py`
- Create: `src/teleop/record/pile.py`
- Test: `tests/atc/test_episodes.py`, `tests/record/test_pile.py`

**Interfaces:**
- Produces:
  - `RecordGroup = Literal["human", "autonomous"]`; `group_of(source: EpisodeStartSource) -> RecordGroup` (`"operator"` → human; `"dmp"`, `"act"` → autonomous).
  - `EpisodeSink.set_recording(group: RecordGroup, on: bool) -> None`; `EpisodeSink.recording() -> dict[RecordGroup, bool]` (both `True` at construction); `EpisodeSink.stats() -> SessionStats`; `EpisodeSink.recording_bays() -> frozenset[int]` (bays with an open episode).
  - `SessionStats(saved: int, writing: int, by_source: dict[str, int], by_fault: dict[str, dict[str, int]], latest: list[LatestEpisode])`; `by_fault[fault] = {"success": n, "aborted": m}`; `latest` holds the last 10, newest first, each `LatestEpisode(time: str, bay: int, fault: str | None, source: str, frames: int, success: bool)`.
  - `new_session_dir(root: Path) -> Path`: always `root/<YYYYmmdd-HHMMSS>[-n]`.
  - `pile.scan(root: Path) -> PileTotals`; `PileTotals(sessions: int, episodes: int, by_fault: dict[str, int], by_source: dict[str, int])`, counting only `success: true` lines; `PILE_TARGET_PER_FAULT = 50`.
  - `teleop atc`: live default `record_episodes = data/episodes` with `new_session_dir`; `--no-record`; headless unchanged.

- [ ] **Step 1: Write the failing tests** (fake recorder/renderer as in the existing `tests/atc/test_episodes.py`)

```python
def test_paused_group_opens_no_episode(sink):
    sink.set_recording("human", False)
    sink.on_episode_start(2, "operator", None, None)
    tick_and_end(sink, 2)
    assert sink.stats().saved == 0


def test_autonomous_unaffected_by_human_switch(sink): ...
def test_switch_flipped_mid_rescue_keeps_open_episode(sink):
    sink.on_episode_start(2, "operator", None, None)
    sink.set_recording("human", False)
    tick_and_end(sink, 2)
    sink.close()
    assert sink.stats().saved == 1


def test_stats_by_source_fault_and_latest(sink): ...  # latest[0] is newest; len capped at 10
def test_recording_bays_tracks_open_episodes(sink): ...
def test_new_session_dir_always_nests(tmp_path):
    assert new_session_dir(tmp_path).parent == tmp_path  # even when tmp_path is empty


# tests/record/test_pile.py
def test_scan_counts_successes_across_sessions(tmp_path): ...
def test_scan_skips_dirs_without_sidecar_and_bad_lines(tmp_path): ...
def test_scan_missing_root_is_empty(tmp_path):
    assert scan(tmp_path / "nope").episodes == 0


def test_atc_defaults(): ...  # live -> data/episodes + new_session_dir; --no-record -> None; headless -> None
```

- [ ] **Step 2: Run** `uv run pytest tests/atc/test_episodes.py tests/record/test_pile.py -v` → FAIL.
- [ ] **Step 3: Implement.** Switches are read only in `on_episode_start`, with no lock needed for a bool read. Stats are updated on the writer thread under `self._lock` after a successful write. `open_episode_sink` takes a `session_dir_fn` parameter (default `session_root`), and live default wiring passes `new_session_dir`. If `lerobot` is missing in the live default path, log one line, do not fail startup, and record `available=False` for Task 5. An explicit `--record-episodes` keeps today's exit-1 behaviour.
- [ ] **Step 4: Run** `uv run pytest tests/atc tests/record -v` → PASS.
- [ ] **Step 5: Commit** `feat(record): record rescues by default with per-source switches and pile totals (T-232)`.

### Task 4 (T-233): Console pages, tabs and rescue banner

**Files:**
- Modify: `src/teleop/console/web/index.html`, `console.js`, `console.css`, `health.js`
- Test: `tests/console/test_web_assets.py`

**Interfaces:**
- Produces (DOM contract for Task 5, T-197, T-215 and the calibration panel): `<nav class="tabs">` with `a[href="#operate"|"#data"|"#setup"]`; page containers `#page-operate`, `#page-data`, `#page-setup` (only the active one visible; `document.body.dataset.page` holds the name); `#tab-data-rec` (REC chip and count text) and `#tab-setup-warn` (dot) inside the tabs; `#rescue-banner` (hidden on Operate); `#brief-rec` chip on the brief; empty `#calib-slot` section on Setup with heading "Glove calibration"; `#data-root` empty container on Data.
- `console.js` dispatches the existing `telemetry` CustomEvent unchanged; adds `setPage(name)` and `hashchange` handling (unknown hash → `operate`).

- [ ] **Step 1: Write the failing tests** (string/DOM-structure checks in the existing style of `test_web_assets.py`)

```python
def test_three_pages_and_tabs():
    html = read("index.html")
    for p in ("operate", "data", "setup"):
        assert f'id="page-{p}"' in html and f'href="#{p}"' in html


def test_panels_moved():
    # Operate rail keeps queue/sparks/gate/log; dummy panel + #health + #unoq on Setup; hooks on Data
    ...


def test_view_bar_only_on_operate(): ...  # .bar__views carries data-page-only="operate"
def test_rescue_banner_and_calib_slot_present(): ...
def test_unknown_hash_falls_back_to_operate(): ...  # console.js contains the fallback
```

- [ ] **Step 2: Run** `uv run pytest tests/console/test_web_assets.py -v` → FAIL.
- [ ] **Step 3: Implement.** Move the existing sections into the page containers and do not rewrite them. Show the rescue banner when any bay in telemetry is in a waiting-for-rescue or human-rescue state, and only when not on Operate. Clicking it calls `setPage("operate")`. The setup warning dot shows when any `health` check level is not OK. The camera wall stays mounted when hidden (`display:none` on the page container), so frames and slots survive page switches.
- [ ] **Step 4: Run** `uv run pytest tests/console -v` → PASS. Manual check: run `uv run teleop atc --source synthetic`, switch pages during a rescue, and confirm the banner shows, the cameras resume instantly and the WebSocket never reconnects (live dot stays green).
- [ ] **Step 5: Commit** `feat(atc): split the console into Operate, Data and Setup pages (T-233)`.

### Task 5 (T-234): Data & training page and its wire protocol

Depends on T-231, T-232, T-233.

**Files:**
- Create: `src/teleop/atc/data_controls.py`, `src/teleop/console/web/data.js`
- Modify: `src/teleop/console/server.py`, `run.py`, `telemetry.py`, `src/teleop/console/web/index.html` (script tag), `src/teleop/commands/atc.py` (pass `DataControls` to `run_console`)
- Test: `tests/atc/test_data_controls.py`, `tests/console/test_server.py`, `tests/console/test_run.py`, `tests/console/test_telemetry.py`, `tests/console/test_web_assets.py`

**Interfaces:**
- Consumes: `EpisodeSink.set_recording/recording/stats/recording_bays`, `pile.scan`, `PILE_TARGET_PER_FAULT`, `SkillStore.entries/reset`, `AtcCoordinator`.
- Produces:
  - `DataControls(sink: EpisodeSink | None, store: SkillStore | None, coordinator, episodes_root: Path | None, *, available: bool, scan_interval_s: float = 10.0)`; `.status() -> dict[str, Any]` (cheap: reads cached pile totals); `.handle(msg: dict[str, Any]) -> None`; `.start()` / `.stop()` for the pile-scan thread (rescans every `scan_interval_s` and when `stats().saved` changes).
  - `status()` shape: `{"available": bool, "session_path": str|null, "switches": {"human": bool, "autonomous": bool}, "recording_bays": [int], "session": SessionStats-as-dict, "pile": {"sessions", "episodes", "by_fault", "by_source", "target": 50}, "skills": {"path": str|null, "entries": [SkillEntry-as-dict], "board_warning": str|null}, "commands": {"process": str, "train": str}}`.
  - `commands.process` = `uv run teleop process --in <session> [--in <session> ...] --out data/train/<YYYYmmdd>` built from the scanned session dirs; `commands.train` = the T-194 runbook command if `docs/` has it, else the runbook path.
  - Client messages: `{"type":"record","source":"human"|"autonomous","on":bool}`, `{"type":"skills","action":"reset"}`. `ConsoleServer._on_message` forwards `record`/`skills` to `on_client_message`; `run_console(..., data: DataControls | None = None)` routes them to `data.handle`; `TelemetryAdapter.build(..., data_status: Mapping | None = None)` adds `"data"` when given.
  - `board_warning` text when reset reports `board_cleared=False`: `"board not cleared: redeploy the UNO Q app"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_record_message_toggles_sink(controls, sink):
    controls.handle({"type": "record", "source": "human", "on": False})
    assert sink.recording()["human"] is False


def test_bad_record_messages_ignored(controls): ...  # unknown source, non-bool on, missing keys
def test_skills_reset_backs_up_and_reports_board_warning(controls_old_firmware):
    controls_old_firmware.handle({"type": "skills", "action": "reset"})
    assert (
        controls_old_firmware.status()["skills"]["board_warning"]
        == "board not cleared: redeploy the UNO Q app"
    )


def test_status_unavailable_without_sink(): ...  # available False, session None, no crash
def test_status_never_scans_disk(
    controls, monkeypatch
): ...  # pile.scan patched to raise; status() still returns
def test_server_forwards_record_and_skills(server): ...
def test_telemetry_includes_data_only_when_given(adapter): ...
def test_data_js_sends_protocol(): ...  # data.js contains '"record"' and '"skills"' sends and listens for "telemetry"
```

- [ ] **Step 2: Run** `uv run pytest tests/atc/test_data_controls.py tests/console -v` → FAIL.
- [ ] **Step 3: Implement** `data_controls.py` and the server, run and telemetry plumbing. `handle` runs on the asyncio thread: `set_recording` is a bool write, and `reset` takes the `SkillStore` lock and calls `coordinator.reset_library()`, which is a TCP round trip with the client's existing timeout. That's acceptable on the WS thread because `on_reconnect` already does the same kind of call there.
- [ ] **Step 4: Implement `data.js`** following `health.js`: listen for `telemetry`, render into `#data-root` the Recording switches (or the unavailable line with `uv sync --group train`), This session, All sessions bars toward `pile.target`, Latest episodes, DMP skill library with **Start fresh** behind `confirm()`, Next steps with copy buttons. Update `#tab-data-rec` and `#brief-rec` (shown when the focused/rescue bay is in `recording_bays`). Rebuild only on change, as `health.js` does.
- [ ] **Step 5: Run** the full verify command → PASS.
- [ ] **Step 6: Manual check on the hardware laptop.** Run `uv run teleop atc` live with two bays in rescue at once. The console holds 30 fps (FPS readout or `LoopStats`). Both episodes appear on the Data page, and Start fresh empties the library and the board. If 30 fps fails, lower the writer's render rate (not the camera set) and note it in the PR.
- [ ] **Step 7: Commit** `feat(atc): Data & training page with recording switches and skill library (T-234)`.

---

## Task files

Create `tasks/T-230-unoq-clear-skills.md` … `tasks/T-234-console-data-page.md` (frontmatter per `tasks/README.md`, `plan:` this file):

| ID | depends_on |
|---|---|
| T-230 | [] |
| T-231 | [T-230] |
| T-232 | [T-191] |
| T-233 | [] |
| T-234 | [T-231, T-232, T-233] |

Add a "Console pages and data collection (`T-230`–`T-234`)" section to `tasks/README.md`, and a line to the file-ownership map in `docs/superpowers/plans/2026-11-14-act-recovery-plan.md` noting that T-232 edits `episodes.py` after T-191, and that T-197 builds on T-233's pages.
