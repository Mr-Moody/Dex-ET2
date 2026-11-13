# M0 + M1: Scaffold and Gloves-to-Sim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A uv-managed repo where `uv run teleop sim --source studio` (live gloves through Rokoko Studio) and `--source synthetic` drive two simulated hands in MuJoCo in real time, with calibration, Coil Pro / IMU wrist, a managed 2.4 GHz hotspot, LeRobot v3 episode writing, and latency logged.

**Architecture:** Shared contracts (`teleop.types`, `teleop.protocols`, `teleop.registry`, `teleop.record.schema`) land first (M0). After that, the input, sim, retarget, wrist, recorder and hotspot tasks are independent and run in parallel worktrees. Each one only *creates* its own modules. Every subcommand and every config-selectable component is pre-registered by import path in M0, so parallel tasks never edit the same file. T-050 wires everything together.

**Tech Stack:** Python 3.12, uv, mujoco 3.x (MjSpec, passive viewer), numpy, draccus, PyYAML, python-dotenv, lz4, winrt (Windows tethering), lerobot ≥0.4 (dataset v3), pytest + hypothesis, ruff, pyright, pre-commit, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-13-glove-teleop-booth-design.md` (and background `docs/KICKOFF_PROMPT.md`)

M2 (booth app, processing/QC) and M3 (train/eval/attract) get their own plan, written once M1's interfaces exist. Their task files are seeded in T-002 with status `blocked`.

## Global Constraints

- Python **3.12** (`.python-version`), uv only (`uv add`, `uv run`, `uv sync`), `src/` layout, `uv.lock` committed. Never pip.
- Groups: `dev` (ruff, pyright, pytest, pytest-cov, hypothesis, pre-commit, conventional-pre-commit), `train` (lerobot>=0.4), `viz` (rerun-sdk). Windows-only deps get the `sys_platform == 'win32'` marker.
- All M1 runtime dependencies are added in T-001. A later task that needs a new dependency must say so in its PR, because `uv.lock` conflicts are expensive.
- Joints are mapped **by name, never by index**. Human joints use canonical OpenXR names (T-003 `OPENXR_JOINTS`). Robot joints and actuators use the names in the hand YAML. The sim prefixes them with `left/` and `right/`.
- Quaternions are **xyzw**. Positions are in metres. The sim world is Z-up.
- Control 60 Hz; physics ≥500 Hz via substeps; recording 30 fps; clock `time.monotonic_ns()`.
- Gloves rate ≤83 Hz. Studio Custom Streaming is UDP on port `14043`, JSON v3, optionally LZ4. RGMP is `127.0.0.1:12276`, the hand solver `127.0.0.1:12277`, RCSP `45451`.
- Hotspot SSID exactly `Router? I barely know 'er`, band 2.4 GHz, password from `.env` key `HOTSPOT_PASSWORD`.
- Robot, task, input source, wrist source and sink are chosen by config string through `teleop.registry`. Never hard-code them.
- No test or CI job may need gloves, Studio, the hotspot, a GPU or a display. Hardware-only checks are marked `@pytest.mark.hardware` and deselected by default.
- Never commit `.env`, licences, `data/`, `outputs/`. The exception is small trimmed captures under `tests/fixtures/` (each <200 KB).
- Conventional Commits `type(scope): summary`, scopes `input|sim|retarget|record|train|cli|infra|docs|tasks|net`. Branch `<type>/T-NNN-<kebab>`. No AI co-author trailers or "Generated with" lines in commits or PRs.
- Verification before any PR: `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest` all pass. Paste the output in the PR.

## Review Focus

1. **Two agents on different laptops claim the same task from the same `HEAD`.** Exactly one wins. Claim refs point at a unique commit per claim, so the loser's push is rejected and it releases locally. Pinned in T-002 `test_claim_race_two_clones`.
2. **The gloves go silent mid-session** (out of hotspot range, battery, Studio closed). The source reports `connected=False` after 250 ms with no frames. The sink holds the last targets. Nothing raises. Pinned in T-012 `test_latest_buffer_staleness` and T-050 `test_sim_survives_source_dropout`.
3. **Device timestamps jump back or repeat** (Studio restarts, the clock steps). Frames are still accepted and ordered by host receive time, and the jump is counted in `SourceStatus.timestamp_anomalies`. Pinned in T-011 `test_parser_tolerates_timestamp_jump`.
4. **Ctrl-C or an exception while an episode is being recorded.** The partial episode is discarded, `finalize()` still runs, and the dataset loads. Pinned in T-040 `test_interrupt_mid_episode_dataset_loads`.
5. **The hotspot is turned off externally, its config drifts (another SSID or band), or `HOTSPOT_PASSWORD` is missing or shorter than 8 characters.** The watchdog restores the config and restarts the hotspot. A bad or missing password fails fast with a message naming `.env`. Pinned in T-010 `test_watchdog_restarts_when_off`, `test_reconfigures_on_drift` and `test_password_validation`.

## File Map

```
pyproject.toml  uv.lock  .python-version  .env.example  .pre-commit-config.yaml
.github/workflows/ci.yml  .github/pull_request_template.md
CLAUDE.md  CONTRIBUTING.md  tasks/README.md  tasks/T-*.md
docs/adr/0001-studio-primary-inspire-default.md
configs/sim.yaml  configs/hands/{inspire,leap}.yaml
scripts/task.py  scripts/fetch_models.py
third_party/  (models fetched by script; gitignored except README)
src/teleop/
  __init__.py  cli.py  registry.py  types.py  protocols.py  config.py  latency.py
  devtools/task.py                       # task CLI
  commands/{hotspot,capture,sim,calibrate,record}.py
  net/hotspot.py  net/winrt_backend.py
  input/{buffer,synthetic,replay,studio,rgmp}.py
  wrist/{coil,imu,auto}.py
  retarget/{flexion,calibration,direct}.py
  sim/{hands,scene,sink,loop}.py  sim/tasks/lift_cube.py
  record/{schema,recorder}.py
tests/  (mirrors src; tests/fixtures/studio/*.jsonl)
```

---

## M0 (sequential, lead agent on the hardware laptop)

### Task T-001: Project scaffold, tooling, CI, docs

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.env.example`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `.github/pull_request_template.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `docs/adr/0001-studio-primary-inspire-default.md`, `src/teleop/__init__.py`, `src/teleop/cli.py`, `tests/test_cli.py`
- Modify: `.gitignore` (append `data/`, `outputs/`, `.env`, `third_party/*` with `!third_party/README.md`, `.venv/`)

**Interfaces:**
- Produces: the `teleop` console script → `teleop.cli:main(argv: list[str] | None = None) -> int`, and the dict `COMMANDS: dict[str, str]` mapping subcommand → module path, pre-populated with: `hotspot→teleop.commands.hotspot`, `capture→teleop.commands.capture`, `sim→teleop.commands.sim`, `calibrate→teleop.commands.calibrate`, `record→teleop.commands.record`, `replay→teleop.commands.replay`, `process→teleop.commands.process`, `eval→teleop.commands.eval`. Each command module exposes `main(argv: list[str]) -> int`. Also the console script `task` → `teleop.devtools.task:main` (T-002 creates the module).

- [ ] **Step 1:** `uv init --package --name teleop --python 3.12` (keep the existing `.gitignore`). Add the deps: `uv add mujoco numpy draccus pyyaml python-dotenv lz4` and `uv add "winrt-runtime; sys_platform == 'win32'" "winrt-Windows.Networking.NetworkOperators; sys_platform == 'win32'" "winrt-Windows.Networking.Connectivity; sys_platform == 'win32'" "winrt-Windows.Foundation; sys_platform == 'win32'"`, plus the groups from Global Constraints (`uv add --group dev ...`, `--group train "lerobot>=0.4"`, `--group viz rerun-sdk`). Configure ruff (line length 100, rules `E,F,I,UP,B,SIM`), pyright (`strict` for `src/teleop`, excluding `src/teleop/net/winrt_backend.py`), pytest (`-m "not hardware"`, markers `hardware`, `lerobot`).
- [ ] **Step 2: Write the failing test** `tests/test_cli.py`:
  - `test_unknown_command_returns_2`: `main(["nope"]) == 2`.
  - `test_unimplemented_command_reports_cleanly`: with `COMMANDS` monkeypatched to `{"x": "teleop.commands.does_not_exist"}`, `main(["x"]) == 3` and stderr contains `"not implemented yet"`.
  - `test_dispatch_passes_remaining_args`: a fake module registered via monkeypatch receives `["--a", "1"]`.
- [ ] **Step 3:** Run `uv run pytest tests/test_cli.py -v`. Expected: FAIL (`teleop.cli` has no `main`).
- [ ] **Step 4:** Implement `main` in `src/teleop/cli.py`: argparse with the first positional as the command, `importlib.import_module` inside try/except `ModuleNotFoundError` (only when the missing module *is* the command module), and the rest forwarded.
- [ ] **Step 5:** Run `uv run pytest tests/test_cli.py -v`. Expected: 3 PASS.
- [ ] **Step 6:** Write the CI workflow: `astral-sh/setup-uv`, `uv sync --all-groups`, then `ruff check`, `ruff format --check`, `pyright` and `pytest` as separate steps. Set `MUJOCO_GL: egl` and install `libegl1` on Ubuntu. Write `.pre-commit-config.yaml` with ruff, ruff-format and `conventional-pre-commit` at the commit-msg stage, using the scopes above. Write the PR template with the fields from brief §4: task ID, spec link, plan link, summary, test evidence, checklist, `Closes T-xxx`.
- [ ] **Step 7:** Write `CLAUDE.md`. It covers: run `uv run task list --ready` first; never work without `uv run task claim`; one active claim per agent; worktree per task; Global Constraints (link this plan); the verification command; this laptop is the hardware host. Write `CONTRIBUTING.md` (brief §4 rules, plus the weekend relaxation from spec §6: merge on green CI with a quick lead check). Write ADR 0001, recording the spec §9 deviations. Write `.env.example` with `HOTSPOT_PASSWORD=` and a comment about the 8–63 character limit.
- [ ] **Step 8:** Run the full verification command. Expected: all pass (ruff, format, pyright 0 errors, pytest 3 passed).
- [ ] **Step 9: Commit** `build(infra): scaffold uv project, tooling, CI and contributor docs`. Branch `build/T-001-scaffold`, PR, merge on green.

### Task T-002: Task checklist CLI with atomic claims, seed tasks

**Files:**
- Create: `src/teleop/devtools/__init__.py`, `src/teleop/devtools/task.py`, `scripts/task.py` (a 3-line shim calling `main`), `tasks/README.md`, `tasks/T-0NN-*.md` (one per task in this plan, plus the M2/M3 placeholders `T-060-booth-app`, `T-061-process-qc`, `T-070-train-eval-attract`, `T-080-real-inspire-sink`, `T-081-vlm-annotator`, `T-013` marked stretch)
- Test: `tests/devtools/test_task.py`

**Interfaces:**
- Produces: `main(argv: list[str] | None = None) -> int` with subcommands `list [--status S] [--ready]`, `claim ID --agent NAME`, `release ID`, `status ID S`, `done ID`. `TaskFile` dataclass with `id, title, status, owner, branch, depends_on: list[str], plan, claimed_at` and the body, plus `load_tasks(dir: Path) -> list[TaskFile]` and `save_task(t: TaskFile) -> None`. These round-trip YAML frontmatter and preserve the body byte-for-byte. Status values: `todo|claimed|in_review|done|blocked`.

Claim algorithm (this exact sequence; the unique commit is what makes the race in Review Focus #1 safe):
```
git fetch origin "+refs/claims/*:refs/claims/*"   (ignore failure if offline; warn)
sha = git commit-tree HEAD^{tree} -p HEAD -m "claim <ID> by <agent> at <iso-utc>"
git update-ref refs/claims/<ID> <sha> ""          (fails if ref exists → exit 1 "already claimed by …")
git push origin refs/claims/<ID>                  (rejected → git update-ref -d refs/claims/<ID>; exit 1 "lost race")
then set status=claimed, owner, branch=<current branch>, claimed_at in tasks/<ID>-*.md
```
`release`: delete the ref locally and `git push origin :refs/claims/<ID>`, then set status to `todo` and clear owner. `done`: set status `done` and release. `list` shows the claim owner and age (taken from the claim commit message and time). `--ready` means status `todo` and every `depends_on` is `done`.

- [ ] **Step 1: Write the failing tests** (use `tmp_path`: a bare repo as `origin`, plus two clones `a` and `b` with an initial commit):
  - `test_frontmatter_roundtrip_preserves_body`
  - `test_ready_requires_all_deps_done`: T-2 depends on T-1. T-2 isn't ready while T-1 is `todo`, and is ready once T-1 is `done`.
  - `test_claim_creates_ref_and_updates_file`: after a claim in `a`, `git show-ref refs/claims/T-1` succeeds in `a` and on `origin`, and the file has `status: claimed`, `owner: agentA`.
  - `test_second_claim_same_clone_fails`: exit 1, and the message contains `already claimed`.
  - `test_claim_race_two_clones`: `a` claims, then `b` (without fetching first, simulated with `--no-fetch`) claims and gets exit 1 containing `lost race`. `b` has no local `refs/claims/T-1` afterwards.
  - `test_release_removes_local_and_remote_ref`
  - `test_done_sets_status_and_releases`
- [ ] **Step 2:** Run `uv run pytest tests/devtools -v`. Expected: FAIL (import error).
- [ ] **Step 3:** Implement `src/teleop/devtools/task.py` using `subprocess.run(["git", ...], check=...)` and PyYAML. Add a `--no-fetch` flag for tests only. Write `tasks/README.md` documenting the protocol and the rule that the lead may reclaim a claim older than 6 h with no branch activity.
- [ ] **Step 4:** Run `uv run pytest tests/devtools -v`. Expected: 7 PASS.
- [ ] **Step 5:** Seed the task files with each task's acceptance criteria copied as `- [ ]` items from its section of this plan. Set `depends_on` as in the dependency graph below. Run `uv run task list --ready`. Expected: only T-003 is listed (T-001 and T-002 are marked `done` in this PR).
- [ ] **Step 6:** Full verification, then commit `feat(tasks): add task CLI with atomic claim refs and seed M1 tasks`, then PR and merge.

### Task T-003: Shared contracts, registry, config, dataset schema

**Files:**
- Create: `src/teleop/types.py`, `src/teleop/protocols.py`, `src/teleop/registry.py`, `src/teleop/config.py`, `src/teleop/latency.py`, `src/teleop/record/__init__.py`, `src/teleop/record/schema.py`, empty `__init__.py` for `input/`, `wrist/`, `retarget/`, `sim/`, `sim/tasks/`, `net/`, `commands/`, plus `configs/sim.yaml`
- Test: `tests/test_types.py`, `tests/test_registry.py`, `tests/record/test_schema.py`, `tests/test_latency.py`

**Interfaces (produced; every M1 task consumes these exactly):**
```python
# types.py
Side = Literal["left", "right"]; SIDES: tuple[Side, Side] = ("left", "right")
Quat = tuple[float, float, float, float]   # xyzw
Vec3 = tuple[float, float, float]
OPENXR_JOINTS: tuple[str, ...]  # 26 names in XrHandJointEXT order:
  # palm, wrist, thumb_metacarpal, thumb_proximal, thumb_distal, thumb_tip,
  # index_metacarpal, index_proximal, index_intermediate, index_distal, index_tip,
  # (same 5 for middle, ring, little)
@dataclass(frozen=True, slots=True) class Pose: orientation: Quat; position: Vec3 | None = None
@dataclass(frozen=True, slots=True) class HandFrame:
    side: Side; host_time_ns: int; seq: int; source: str
    joints: Mapping[str, Quat]          # local rotations, keys ⊆ OPENXR_JOINTS
    device_time_us: int | None = None
    wrist: Pose | None = None           # world pose if the source provides it (Coil Pro)
@dataclass(frozen=True, slots=True) class BimanualFrame: left: HandFrame | None; right: HandFrame | None
@dataclass(frozen=True, slots=True) class SourceStatus:
    connected: bool; last_rx_ns: int | None; frames: int; dropped: int; timestamp_anomalies: int
@dataclass(frozen=True, slots=True) class JointTargets: side: Side; host_time_ns: int; positions: Mapping[str, float]  # unprefixed actuator names
@dataclass(frozen=True, slots=True) class WristTarget: side: Side; pose: Pose   # pose.position required
@dataclass(frozen=True, slots=True) class RobotState: qpos: Mapping[str, float]; wrist: Mapping[Side, Pose]  # prefixed joint names
@dataclass(frozen=True, slots=True) class TaskScore: success: bool; score: float; elapsed_s: float

# protocols.py  (typing.Protocol, runtime_checkable)
class InputSource:  name: str; def start(self) -> None; def stop(self) -> None
                    def latest(self) -> BimanualFrame; def status(self) -> SourceStatus
class PacketParser: def parse(self, payload: bytes, host_time_ns: int) -> list[HandFrame]
class WristSource:  mode: str; def wrist(self, frame: HandFrame) -> WristTarget
class Retargeter:   def retarget(self, frame: HandFrame) -> JointTargets
class RobotSink:    def apply(self, targets: Sequence[JointTargets], wrists: Sequence[WristTarget]) -> None
                    def state(self) -> RobotState
class Task:         name: str; instruction: str
                    def add_to_spec(self, spec: Any) -> None   # mujoco.MjSpec
                    def reset(self, model: Any, data: Any, seed: int) -> None
                    def score(self, model: Any, data: Any) -> TaskScore
class AuxSensorSource: name: str; def features(self) -> dict[str, dict[str, Any]]; def read(self) -> Mapping[str, Any]

# registry.py — lazy "module:attr" strings, so parallel tasks never edit this file
SOURCES = {"synthetic": "teleop.input.synthetic:SyntheticSource", "replay": "teleop.input.replay:ReplaySource",
           "studio": "teleop.input.studio:StudioSource", "rokoko-sdk": "teleop.input.rgmp:SdkSource"}
WRISTS  = {"auto": "teleop.wrist.auto:AutoWrist", "coil": "teleop.wrist.coil:CoilWrist", "imu": "teleop.wrist.imu:ImuWrist"}
RETARGETERS = {"direct": "teleop.retarget.direct:DirectRetargeter"}
TASKS   = {"lift_cube": "teleop.sim.tasks.lift_cube:LiftCube"}
SINKS   = {"sim": "teleop.sim.sink:SimSink"}
def resolve(table: Mapping[str, str], name: str) -> type   # KeyError lists valid names

# config.py (draccus dataclasses)
@dataclass class StudioConfig: host: str = "0.0.0.0"; port: int = 14043; actor: str | None = None; tee_raw: Path | None = None
@dataclass class WristConfig: mode: str = "auto"; home_left: Vec3 = (-0.25, 0.0, 0.25); home_right: Vec3 = (0.25, 0.0, 0.25)
                              studio_to_sim_offset: Vec3 = (0.0, 0.0, 0.0); studio_to_sim_scale: float = 1.0; nudge_step_m: float = 0.01
@dataclass class SimConfig:   source: str = "synthetic"; hand: str = "inspire"; task: str = "lift_cube"; sink: str = "sim"
                              retargeter: str = "direct"; control_hz: int = 60; physics_hz: int = 600; record_fps: int = 30
                              headless: bool = False; steps: int | None = None; calibration: Path | None = None
                              replay_file: Path | None = None
                              studio: StudioConfig; wrist: WristConfig; cameras: list[CameraConfig]  (default: one "front" 224x224)
@dataclass class CameraConfig: name: str = "front"; width: int = 224; height: int = 224

# latency.py
class LatencyStats: def add(self, ns: int) -> None; def summary(self) -> dict[str, float]  # keys p50_ms, p95_ms, max_ms, n
                    (ring buffer of the last 2048 samples)

# record/schema.py
WRIST_FIELDS = ("x", "y", "z", "qx", "qy", "qz", "qw")
def state_names(hand_joint_names: Mapping[Side, Sequence[str]]) -> list[str]
    # [f"{side}/{j}" for side in SIDES for j in joints[side]] + [f"{side}/wrist.{f}" for side in SIDES for f in WRIST_FIELDS]
def action_names(hand_actuator_names: Mapping[Side, Sequence[str]]) -> list[str]   # same pattern with actuators
def build_features(state: list[str], action: list[str], cameras: Sequence[CameraConfig]) -> dict[str, dict[str, Any]]
    # {"observation.state": {"dtype":"float32","shape":(len(state),),"names":state},
    #  "action": {... action ...},
    #  f"observation.images.{c.name}": {"dtype":"video","shape":(c.height,c.width,3),"names":["height","width","channels"]}}
```

- [ ] **Step 1: Write the failing tests:**
  - `test_openxr_joints_len_and_order`: length 26, `[0] == "palm"`, `[1] == "wrist"`, `[-1] == "little_tip"`.
  - `test_frames_are_frozen`: assigning to a field raises.
  - `test_resolve_unknown_lists_valid`: `KeyError` whose message contains `"synthetic"`.
  - `test_registry_entries_are_well_formed`: every value matches `^[\w.]+:\w+$`. Don't import them, since the modules don't exist yet.
  - `test_build_features_bimanual_order`: with left joints `["a"]` and right joints `["b"]`, `state[:2] == ["left/a", "right/b"]`, the length is `2 + 14`, the image key `observation.images.front` has shape `(224, 224, 3)`, and the dtype is `"video"`.
  - `test_latency_summary_percentiles`: adding 1..100 ms gives `p50_ms ≈ 50.5` (±1) and `max_ms == 100`.
  - `test_config_defaults_parse`: `draccus.parse(SimConfig, args=[])` gives `control_hz == 60` and `studio.port == 14043`.
- [ ] **Step 2:** Run `uv run pytest tests -v`. Expected: FAIL (imports).
- [ ] **Step 3:** Implement the files above exactly as specified. `configs/sim.yaml` mirrors the defaults.
- [ ] **Step 4:** Run `uv run pytest tests -v`. Expected: all PASS. `uv run pyright` reports 0 errors.
- [ ] **Step 5:** Commit `feat(infra): add shared contracts, registry, config and dataset schema`. PR, merge, then `uv run task done T-003`. **This unblocks M1 fan-out.**

---

## M1 (parallel after T-003)

Dependency graph: T-010, T-011, T-012, T-020, T-030, T-040 depend only on T-003. T-031 depends on T-012 (it uses `SyntheticSource` in its tests). T-021 depends on T-020. T-013 is a stretch task that depends on T-012. T-050 depends on T-011, T-012, T-021, T-030, T-031 and T-040.
**Hardware laptop only:** T-010, T-011 Step 1, T-050 Step 6. Teammates' agents take the others first.

### Task T-010: Hotspot manager + keep-alive watchdog (Windows)

**Files:** Create `src/teleop/net/hotspot.py`, `src/teleop/net/winrt_backend.py`, `src/teleop/commands/hotspot.py`; Test `tests/net/test_hotspot.py`

**Interfaces:**
- Produces:
  - `DEFAULT_SSID = "Router? I barely know 'er"`.
  - `class TetheringBackend(Protocol)` with: `state() -> Literal["on","off","transitioning","unknown"]`, `client_count() -> int`, `current_config() -> tuple[str, str]` (ssid, band where band is `"2.4GHz"|"5GHz"|"auto"`), `band_supported(band: str) -> bool`, `configure(ssid: str, password: str, band: str) -> None`, `start() -> None`, `stop() -> None`, `disable_idle_timeout() -> None`.
  - `class Hotspot(backend, ssid=DEFAULT_SSID, password: str, band="2.4GHz")` with `ensure_up() -> list[str]` (returns the event strings, such as `"reconfigured"` and `"started"`) and `status() -> dict`.
  - `class HotspotWatchdog(hotspot, interval_s: float = 5.0, on_event: Callable[[str], None])` with `start()`, `stop()` and `events: list[tuple[int, str]]` (monotonic ns, event).
  - `load_password(env_file: Path = Path(".env")) -> str`. Raises `HotspotConfigError` mentioning `.env` and `HOTSPOT_PASSWORD` if the password is missing or not 8–63 characters.
- CLI: `teleop hotspot up|status|down|watch`.

- [ ] **Step 1: Write the failing tests** with an in-memory `FakeBackend`:
  - `test_password_validation`: missing, 7 characters and 64 characters all raise `HotspotConfigError` with `".env"` in the message. 8 characters passes.
  - `test_ensure_up_configures_and_starts`: the fake starts `off` with SSID `"Other"`. Afterwards it is `on` with `(DEFAULT_SSID, "2.4GHz")`, and the events are `["reconfigured", "started"]`.
  - `test_ensure_up_noop_when_correct`: the events are `[]`.
  - `test_band_unsupported_raises`: `band_supported` returns False, so `ensure_up` raises `HotspotConfigError` containing `"2.4GHz"`.
  - `test_watchdog_restarts_when_off`: the fake flips to `off` after the first poll. Within 3 polls (using an injected `sleep` and clock) the events include `"started"`.
  - `test_reconfigures_on_drift`: the fake's SSID changes mid-run, and the watchdog emits `"reconfigured"`.
  - `test_disable_idle_timeout_called_once`.
- [ ] **Step 2:** Run `uv run pytest tests/net -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `hotspot.py` (pure logic, no winrt imports). Implement `winrt_backend.py` with `NetworkOperatorTetheringManager.create_from_connection_profile(profile)`. The profile is `NetworkInformation.get_internet_connection_profile()`; if that's None, fall back to the first entry of `get_connection_profiles()` that has a WLAN adapter, and raise a clear error if none exists. Set the access point configuration's `ssid`, `passphrase` and `band = TetheringWiFiBand.TWO_POINT_FOUR_GIGAHERTZ`, checking `is_band_supported` first. Use `configure_access_point_async`, `start_tethering_async`/`stop_tethering_async`, `tethering_operational_state`, `client_count` and `NetworkOperatorTetheringManager.disable_no_connections_timeout_async()`. Await WinRT ops with `asyncio.run` inside the backend.
- [ ] **Step 4:** Run `uv run pytest tests/net -v`. Expected: 7 PASS.
- [ ] **Step 5 (hardware laptop, manual):** run `uv run teleop hotspot status` and compare the reported SSID character by character with `DEFAULT_SSID`, especially the apostrophe (straight `'` vs curly `’`). If they differ, **stop and ask the user** which is correct. Then `uv run teleop hotspot up` and confirm the gloves reconnect in Studio. Turn the hotspot off in Windows Settings with `teleop hotspot watch` running, and confirm it comes back within 10 s. Paste the output in the PR.
- [ ] **Step 6:** Commit `feat(net): manage 2.4GHz hotspot with keep-alive watchdog`. PR.

### Task T-011: Studio raw capture + StudioSource parser (hardware laptop for Step 1)

**Files:** Create `src/teleop/commands/capture.py`, `src/teleop/input/studio.py`, `tests/fixtures/studio/*.jsonl`, `docs/studio-json-v3.md`; Test `tests/input/test_studio.py`

**Interfaces:**
- Consumes: T-003 types and `PacketParser`. From T-012: `LatestBuffer` and `ThreadedSource`. If T-012 hasn't merged, implement against their Interfaces block below and rebase.
- Produces:
  - Raw capture format (shared with ReplaySource): JSON Lines, one datagram per line, `{"t_ns": <host monotonic_ns>, "src": "studio", "payload_b64": "<base64>"}`.
  - `teleop capture --port 14043 --seconds 30 --out data/raw/<name>.jsonl`.
  - `class StudioParser(PacketParser)` with `__init__(actor: str | None = None)`.
  - `class StudioSource(ThreadedSource)` with `__init__(cfg: StudioConfig)`, which binds UDP and feeds `StudioParser` output into the buffer. With `cfg.tee_raw` set, it also writes captures in the format above.

- [ ] **Step 1 (hardware):** Implement `capture` first (raw UDP recv loop, no parsing). In Studio, enable Custom Streaming → JSON v3 → this laptop's IP, port 14043. Capture 30 s of each: (a) both gloves, Coil Pro on; (b) both gloves, Coil Pro off; (c) LZ4 compression on. Inspect the files and write `docs/studio-json-v3.md`. It records the observed top-level keys, the actor and glove joint names, the rotation and position fields and their units, the timestamp field, the coordinate frame (expected Y-up), whether Coil Pro adds world positions, and how LZ4 is framed. Trim each capture to ≤200 KB into `tests/fixtures/studio/{coil_on,coil_off,lz4}.jsonl`.
- [ ] **Step 2: Write the failing tests** from the fixtures:
  - `test_parses_both_hands_from_coil_on`: frames for `left` and `right`, each with ≥15 of the `OPENXR_JOINTS` keys, unit quaternions (|q| within 1e-3 of 1), and `wrist.position is not None`.
  - `test_coil_off_has_no_wrist_position`: `wrist is None` or `wrist.position is None`.
  - `test_lz4_payload_parses_same_as_plain`.
  - `test_parser_tolerates_timestamp_jump`: two payloads whose device timestamps go backwards both parse, and `host_time_ns` is preserved.
  - `test_garbage_payload_returns_empty`: hypothesis `st.binary()` never raises, and returns `[]` or valid frames.
  - `test_studio_source_receives_udp`: a localhost UDP socket sends a fixture payload to an ephemeral port, and `latest().right` is not None within 0.5 s.
- [ ] **Step 3:** Run `uv run pytest tests/input/test_studio.py -v`. Expected: FAIL.
- [ ] **Step 4:** Implement `StudioParser`. Detect LZ4 when the payload doesn't start with `{`. Map Studio joint names to OpenXR names with an explicit dict in `studio.py`, so unknown names are ignored. Convert Studio Y-up to Z-up only for the wrist *position*; joint local rotations stay as-is and the retargeter deals with axes.
- [ ] **Step 5:** Run the tests. Expected: 6 PASS.
- [ ] **Step 6:** Commit `feat(input): add Studio UDP capture and JSON v3 parser`. PR.

### Task T-012: Latest-value buffer, threaded source base, Synthetic + Replay sources

**Files:** Create `src/teleop/input/buffer.py`, `src/teleop/input/synthetic.py`, `src/teleop/input/replay.py`, `src/teleop/commands/replay.py`; Test `tests/input/test_buffer.py`, `test_synthetic.py`, `test_replay.py`

**Interfaces:**
- Produces:
  - `class LatestBuffer(stale_after_ns: int = 250_000_000)` with `put(frame: HandFrame) -> None`, `get() -> BimanualFrame` and `status(now_ns: int | None = None) -> SourceStatus`. It's thread-safe (a lock), counts `timestamp_anomalies` whenever `device_time_us` doesn't increase per side, and reports `connected=False` when the newest frame is older than `stale_after_ns`.
  - `class ThreadedSource(InputSource, ABC)` with `__init__(name: str)`, `start()`/`stop()` (daemon thread runs `_run(stop: threading.Event)`), `latest()`, `status()` and a protected `self._buffer: LatestBuffer`.
  - `class SyntheticSource(ThreadedSource)` with `__init__(rate_hz: float = 60.0, period_s: float = 2.0)`. Both hands open and close sinusoidally: the flexion joints rotate about their local x axis from 0 to 1.4 rad. The wrist pose is fixed at the config home positions, with a position.
  - `class ReplaySource(ThreadedSource)` with `__init__(path: Path, parser: PacketParser | None = None, speed: float = 1.0, loop: bool = True)`. It reads the raw capture format from T-011 and sleeps to reproduce the `t_ns` deltas. The default parser is `StudioParser` when `src == "studio"`, imported lazily.

- [ ] **Step 1: Write the failing tests:**
  - `test_latest_buffer_keeps_newest_per_side`.
  - `test_latest_buffer_staleness`: put a frame, then `status(now_ns=t + 300ms).connected is False`.
  - `test_timestamp_anomaly_counted`: device times 10, 20, 15 for the same side give `timestamp_anomalies == 1`, and the frame is still stored.
  - `test_synthetic_produces_both_hands`: `start()`, wait 0.2 s, both sides non-None with joints ⊆ `OPENXR_JOINTS`, then `stop()` returns within 1 s.
  - `test_replay_uses_injected_parser`: a fake parser yielding one right frame per line, and a 3-line temp file with `speed=100` gives `status().frames >= 3`.
  - `test_replay_missing_file_raises_filenotfound`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: 6 PASS.
- [ ] **Step 5:** Commit `feat(input): add latest-value buffer, synthetic and replay sources`. PR.

### Task T-013 (stretch): Rokoko SDK RGMP/RCSP source

**Files:** Create `src/teleop/input/rgmp.py`; Test `tests/input/test_rgmp.py`
**Interfaces:**
- Produces:
  - `parse_frames(buf: bytearray) -> list[tuple[int, bytes]]`, which consumes complete `[u32 prefix][u32 len][payload]` frames (little-endian) and leaves any partial tail in `buf`.
  - `class StreamDefinition`, parsed from the prefix-1 JSON (`groups[i].name`, `streams`, `timestamp_epoch`).
  - `decode_data(defn, payload) -> tuple[device_id, group_id, timestamp_us, dict[str, tuple]]`, using `struct.unpack_from` with no alignment assumptions.
  - `class SdkSource(ThreadedSource)`, which connects to the hand-solver output on `127.0.0.1:12277` (26-joint OpenXR order) and sends RCSP `SubscribeToData` on `45451` for USB devices.
- [ ] Tests: `test_split_frames_across_reads` (feed bytes one at a time), `test_decode_packed_unaligned_floats`, `test_disconnect_prefix_3`, and a hypothesis fuzz test that `parse_frames` never raises on random bytes. The implementation follows the RGMP spec values in Global Constraints. Commit `feat(input): add RGMP/RCSP SDK source`.

### Task T-020: Hand models and per-hand config

**Files:** Create `scripts/fetch_models.py`, `third_party/README.md`, `configs/hands/inspire.yaml`, `configs/hands/leap.yaml`, `src/teleop/sim/hands.py`; Test `tests/sim/test_hands.py`

**Interfaces:**
- Produces:
  - `@dataclass class FlexMap: actuator: str; joints: list[str]; axis: Literal["x","y","z"]; invert: bool = False; weights: list[float] | None = None`.
  - `@dataclass class HandSpec: name: str; mjcf: dict[Side, Path]; mirror_left: bool; base_body: str; flex: list[FlexMap]`.
  - `load_hand(name: str, configs_dir: Path = Path("configs/hands")) -> HandSpec`.
  - `actuator_names(spec, side) -> list[str]` and `joint_names(spec, side) -> list[str]`, both read from the loaded MJCF and unprefixed.
- `fetch_models.py`:
  - Clones `google-deepmind/mujoco_menagerie` (sparse checkout `leap_hand`) into `third_party/menagerie`.
  - Downloads the Inspire RH56DFX MJCF from the repository linked by arXiv 2603.08988 into `third_party/inspire`. It records the source URL, the commit and the licence in `third_party/README.md`.
  - If the Inspire model can't be obtained and loaded within ~1 h, set the `SimConfig.hand` default to `"leap"` in this PR and add `docs/adr/0002-leap-default.md`.
- `inspire.yaml` FlexMap targets the 6 actuators: thumb rotation, thumb bend, index, middle, ring, little. Thumb rotation ← `thumb_metacarpal` about the axis that bends toward the palm; each finger ← `[*_proximal, *_intermediate]`.

- [ ] **Step 1:** Write `fetch_models.py` and run `uv run python scripts/fetch_models.py`. Expected: the files exist under `third_party/`.
- [ ] **Step 2: Write the failing tests** (skip with a clear reason if `third_party/` is missing, and have CI run `fetch_models.py` before the tests):
  - `test_hand_loads_in_mujoco[inspire|leap]`: `mujoco.MjModel.from_xml_path` succeeds for both sides.
  - `test_flexmap_actuators_exist`: every `FlexMap.actuator` is in `actuator_names`.
  - `test_flexmap_joints_are_openxr`: every `FlexMap.joints` entry is in `OPENXR_JOINTS`.
- [ ] **Step 3:** Implement. **Step 4:** Run. Expected: PASS. **Step 5:** Add the CI step `uv run python scripts/fetch_models.py` (this is the one exception to the file-ownership rule: append-only to `ci.yml`). Commit `feat(sim): fetch hand models and add per-hand configs`. PR.

### Task T-021: Bimanual scene, lift-cube task, SimSink, fixed-rate loop

**Files:** Create `src/teleop/sim/scene.py`, `src/teleop/sim/sink.py`, `src/teleop/sim/loop.py`, `src/teleop/sim/tasks/lift_cube.py`; Test `tests/sim/test_scene.py`, `test_sink.py`, `test_loop.py`, `test_lift_cube.py`

**Interfaces:**
- Consumes: `HandSpec`, `load_hand`, `actuator_names`, `joint_names` from T-020; `JointTargets`, `WristTarget`, `RobotState`, `TaskScore` and `CameraConfig` from T-003.
- Produces:
  - `build_scene(hand: HandSpec, task: Task, cameras: Sequence[CameraConfig]) -> mujoco.MjModel`. It uses `mujoco.MjSpec`: a floor, a table top at z=0, and per side a mocap body `"{side}/mocap"` welded to the hand attached with prefix `"{side}/"`. Each camera from `cameras` looks at the table, and `task.add_to_spec(spec)` adds the task objects.
  - `class SimSink(RobotSink)` with `__init__(model, data, physics_hz: int, control_hz: int)`. `apply` maps `JointTargets.positions[a]` → `data.ctrl[actuator id of f"{side}/{a}"]`, clipped to `ctrlrange`, and silently ignores unknown names (counted in `self.unknown_names`). It sets `mocap_pos`/`mocap_quat` (MuJoCo quats are **wxyz**, so convert from xyzw) and runs `physics_hz // control_hz` substeps of `mj_step`. `state()` returns the prefixed joint qpos and the wrist poses.
  - `class FixedRateLoop(hz: int, clock: Callable[[], int] = time.monotonic_ns, sleep: Callable[[float], None] = time.sleep)` with `run(step: Callable[[int], bool], max_steps: int | None = None) -> LoopStats` (`overruns: int`, `steps: int`). It schedules against absolute deadlines. When it falls more than one period behind, it skips ahead to the next deadline instead of bursting, and counts an overrun.
  - `class LiftCube(Task)`: `name = "lift_cube"`, `instruction = "pick up the cube"`, `add_to_spec(spec)` (a 4 cm free cube at (0, 0.1, 0.02)), `reset` (randomizes the cube xy by ±3 cm with the seed), and `score`: success when cube z > 0.12 for 0.5 s, `score = max height reached × 100`.
- [ ] **Step 1: Write the failing tests:**
  - `test_scene_has_both_hands_prefixed`: names starting with `left/` and `right/` exist.
  - `test_sink_maps_by_name_and_clips`: an out-of-range target is clipped to `ctrlrange[1]`.
  - `test_sink_ignores_unknown_actuator`.
  - `test_wrist_mocap_quat_converted`: an xyzw identity becomes wxyz `[1, 0, 0, 0]`.
  - `test_substeps`: `data.time` advances by `1/control_hz` per `apply`.
  - `test_loop_absolute_deadlines_no_drift`: with a fake clock, 60 steps take 1.0 s ± 1 ms.
  - `test_loop_skips_when_late`: a step that takes 3 periods gives `overruns == 1` and no burst of catch-up calls.
  - `test_lift_cube_success_when_lifted`: set the cube qpos z to 0.2 and hold for 0.5 s of sim time, then `score().success`.
- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement. **Step 4:** Run. Expected: 8 PASS (headless and offscreen-free). **Step 5:** Commit `feat(sim): add bimanual scene, lift-cube task, sim sink and fixed-rate loop`. PR.

### Task T-030: Flexion, calibration, DirectRetargeter

**Files:** Create `src/teleop/retarget/flexion.py`, `calibration.py`, `direct.py`; Test `tests/retarget/test_flexion.py`, `test_calibration.py`, `test_direct.py`

**Interfaces:**
- Consumes: `FlexMap` and `HandSpec` from T-020. If T-020 hasn't merged, use these dataclasses as specified there and import them from `teleop.sim.hands`.
- Produces:
  - `twist_angle(q: Quat, axis: Literal["x","y","z"]) -> float`: the swing-twist decomposition. It's the signed angle in (−π, π] of the rotation component about `axis`.
  - `flex_value(frame: HandFrame, fm: FlexMap) -> float | None`: the (weighted) sum of `twist_angle` over `fm.joints` that are present, negated if `invert`, or None if none of the joints are present.
  - `@dataclass class Calibration: ranges: dict[Side, dict[str, tuple[float, float]]]` keyed by actuator (open, closed), with `to_json(path)`, `from_json(path)` and `default(spec) -> Calibration` (open 0.0, closed 1.4 per joint × number of joints).
  - `calibrate(spec: HandSpec, open_frames: Sequence[HandFrame], closed_frames: Sequence[HandFrame]) -> Calibration`, using the median `flex_value` per actuator for each pose.
  - `class DirectRetargeter(Retargeter)` with `__init__(spec: HandSpec, ctrlrange: Mapping[Side, Mapping[str, tuple[float, float]]], calib: Calibration)`. `retarget` normalises `n = clip((v - open) / (closed - open), 0, 1)`, then `target = lo + n * (hi - lo)`. If a flex value is missing, it keeps the previous target for that actuator. If `|closed - open| < 0.05 rad`, that actuator falls back to the default range and is flagged in `self.degenerate`.
- [ ] **Step 1: Write the failing tests:**
  - `test_twist_angle_pure_axis`: a rotation of 0.7 rad about x gives 0.7 ± 1e-6.
  - `test_twist_ignores_swing`: a rotation about x composed with a rotation about y still gives the x twist ± 1e-6.
  - `test_twist_sign_and_wrap`: hypothesis angles in (−π, π) round-trip.
  - `test_calibrate_median`.
  - `test_calibration_json_roundtrip`.
  - `test_retarget_open_maps_lo_closed_maps_hi`.
  - `test_retarget_clips_beyond_range`.
  - `test_degenerate_range_falls_back`: identical open and closed frames produce no ZeroDivisionError and flag the actuator in `degenerate`.
  - `test_missing_joint_holds_previous`.
- [ ] **Step 2–4:** Run (FAIL), implement, run (PASS: 9).
- [ ] **Step 5:** Commit `feat(retarget): add flexion, calibration and direct retargeter`. PR.

### Task T-031: Wrist sources (Coil Pro, IMU contingency, auto)

**Files:** Create `src/teleop/wrist/coil.py`, `imu.py`, `auto.py`; Test `tests/wrist/test_wrist.py`
**Interfaces:**
- Consumes: `WristConfig`, `HandFrame` and `WristTarget` from T-003.
- Produces:
  - `CoilWrist(cfg)`: `mode = "coil"`. It returns `frame.wrist` with `position × scale + offset`, and raises `ValueError` if `frame.wrist` or its position is None.
  - `ImuWrist(cfg)`: `mode = "imu"`. Orientation comes from `frame.wrist.orientation`, else the `wrist` joint, else identity. Position is the per-side home plus a keyboard nudge offset. It has `nudge(side: Side, dx: float, dy: float, dz: float) -> None` and `reset_nudge()`.
  - `AutoWrist(cfg, hold_s: float = 0.5)`: `mode` is `"coil"` or `"imu"`, whichever was used last. It uses Coil whenever positions were seen within `hold_s`, otherwise IMU. When switching from Coil to IMU, the home position becomes the last Coil position, so the hand doesn't jump.
- [ ] **Tests:**
  - `test_coil_applies_scale_offset`.
  - `test_coil_without_position_raises`.
  - `test_imu_uses_home_plus_nudge`.
  - `test_auto_switches_to_imu_after_hold`: with a fake clock, positions stop and after 0.6 s `mode == "imu"`.
  - `test_auto_no_jump_on_switch`: the first IMU position equals the last Coil position.
- Then implement, get 5 passing, and commit `feat(sim): add coil, imu and auto wrist sources`.

### Task T-040: LeRobot episode recorder

**Files:** Create `src/teleop/record/recorder.py`; Test `tests/record/test_recorder.py` (marked `lerobot`, run in CI with the `train` group)
**Interfaces:**
- Consumes: `build_features`, `state_names` and `action_names` from T-003.
- Produces `class EpisodeRecorder(repo_id: str, root: Path, fps: int, features: dict, robot_type: str)`. It calls `LeRobotDataset.create(...)` lazily; check the exact signature against the installed `lerobot` source first and note it in a comment. It has:
  - `start_episode(task: str) -> None`.
  - `add_frame(state: np.ndarray, action: np.ndarray, images: Mapping[str, np.ndarray]) -> None`, which adds `"task"` to the frame dict.
  - `end_episode(meta: Mapping[str, Any]) -> int`, which calls `save_episode()`, appends `{"episode_index", **meta}` to `<root>/meta/teleop_episodes.jsonl` and returns the index.
  - `discard_episode() -> None`, which clears the episode buffer.
  - `close() -> None`, which calls `finalize()` and is idempotent.
  - Context manager support: `__exit__` discards the open episode, then calls `close()`, even on an exception.
- [ ] **Step 1: Write the failing tests** (state dimension 4, action dimension 3, one camera 32×32, fps 10):
  - `test_roundtrip_two_episodes`: 2 episodes of 12 frames, `close()`, then `LeRobotDataset(repo_id, root=root)` has `num_episodes == 2` and `len == 24`, and `ds[0]` has the keys `observation.state`, `action` and `observation.images.front`.
  - `test_teleop_meta_sidecar`: 2 lines with `episode_index` 0 and 1.
  - `test_interrupt_mid_episode_dataset_loads`: inside `with EpisodeRecorder(...)`, save 1 episode, add 5 frames to a second, then raise `KeyboardInterrupt`. Catch it outside, then the dataset loads with `num_episodes == 1`.
  - `test_close_idempotent`.
- [ ] **Step 2–4:** Run (FAIL), implement, run (PASS: 4). **Step 5:** Commit `feat(record): add LeRobot v3 episode recorder`. PR.

### Task T-050: `teleop sim` / `calibrate` / `record` wiring, latency, live check

**Files:** Create `src/teleop/commands/sim.py`, `src/teleop/commands/calibrate.py`, `src/teleop/commands/record.py`, `src/teleop/app.py`; Test `tests/test_app.py`
**Interfaces:**
- Consumes: everything above, plus `HotspotWatchdog` (optional, `--hotspot` flag, Windows only).
- Produces:
  - `class TeleopApp(cfg: SimConfig)` with `step(i: int) -> bool` (one control tick: `source.latest()` → per side, if the frame isn't None: `retarget` + `wrist`, then `sink.apply`, then latency `now - frame.host_time_ns` into `LatencyStats`) and `run() -> LoopStats`.
  - `latency_log`: every 2 s it logs `p50/p95/max` and `SourceStatus` at INFO.
  - `on_key(keycode)`: the passive-viewer `key_callback`. Arrow keys and PgUp/PgDn nudge the right wrist; WASD and Q/E nudge the left; `R` resets the task.
  - `teleop sim [--source ...] [--hand ...] [--headless --steps N]` opens `mujoco.viewer.launch_passive` unless headless.
  - `teleop calibrate --out data/calib/<name>.json` prints "flat hand" and collects for 2 s, then "fist" for 2 s, then writes `Calibration`. The on-screen version is M2.
  - `teleop record --task "pick up the cube" --episodes N` records each episode until task success or 60 s, using offscreen `mujoco.Renderer` per camera at `record_fps`. Frames are sampled every `control_hz // record_fps` ticks. Episode `meta` = `{score, success, calibration (path + ranges), wrist_mode history, source SourceStatus at end, hotspot watchdog events during the episode, latency summary}`.
- [ ] **Step 1: Write the failing tests** (headless):
  - `test_sim_synthetic_moves_actuators`: `TeleopApp` with synthetic source and `hand="leap"` (always fetched). After 120 steps the ctrl variance is > 0 for both sides, and `latency.summary()["n"] > 0`.
  - `test_sim_survives_source_dropout`: a source that returns frames for 30 steps and then `BimanualFrame(None, None)` with `connected=False` runs 120 steps without an exception, and ctrl after the dropout equals ctrl at step 30.
  - `test_record_two_short_episodes` (marked `lerobot`): `record --episodes 2` with a 1 s cap writes a loadable dataset with 2 episodes.
- [ ] **Step 2–4:** Run (FAIL), implement, run (PASS).
- [ ] **Step 5:** Run `uv run teleop sim --source synthetic` locally. Expected: the viewer opens and both hands open and close.
- [ ] **Step 6 (hardware laptop):** Run `uv run teleop hotspot up`, put on the gloves with Studio streaming, then `uv run teleop calibrate --out data/calib/lead.json` and `uv run teleop sim --source studio --calibration data/calib/lead.json`. Expected: both hands follow the fingers, the wrist follows the Coil Pro (or IMU mode, shown in the log), and the logged p95 latency is < 50 ms. Paste the latency log lines into the PR. If p95 ≥ 50 ms, open a `perf` task with the numbers rather than blocking the merge.
- [ ] **Step 7:** Commit `feat(cli): wire sim, calibrate and record commands with latency logging`. PR. **M1 complete.**
