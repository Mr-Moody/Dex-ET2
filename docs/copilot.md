# Operator copilot and active grasp gate

The copilot is the three-tier Gemini intelligence and physical grasp-verification layer for the
4-bay EV-battery disassembly fleet. It runs in two ways:

- **In-process inside `uv run teleop atc`** (default showcase path when `GEMINI_API_KEY` or
  `COPILOT_LOCAL_URL` is configured and neither `--headless` nor `--no-gemini` is passed): drives
  instant SSE-streamed voice alerts, co-gates DMP skill promotion in parallel with `SkillValidator`,
  and runs silent background rescue audits for the Fleet Ops Console and ACT dataset curation.
- **Standalone via `uv run python -m teleop.copilot {probe,watch,run}`**: connects to the Fleet Ops
  Console WebSocket (`ws://127.0.0.1:8765/ws`) to probe models, watch events, speak call-outs, and
  publish background audit second opinions.

Plan and task list: `docs/superpowers/plans/2026-11-14-operator-copilot-plan.md` (`T-210` to
`T-215`).

## Three-tier architecture

| Tier | Latency target / measured | Model & endpoint | Role & authority |
|---|---|---|---|
| **Tier 1 — Instant Voice & SSE TTS** (`tts_stream.py`, `speech.py`) | `< 0.7 s` warm TTFA (`483–653 ms` live; `< 50 ms` pre-cached cues; `1.8 s` first-chunk fallback) | `gemini-3.8-flash-lite-tts` (voice `Kore`), `POST /v1beta/models/gemini-3.8-flash-lite-tts:streamGenerateContent?alt=sse` over persistent IPv4 `HTTPSConnection` (`keep-alive`) | Zero-VLM deterministic telemetry cues (pre-cached at startup) plus dynamic spoken verdicts piped as raw 24 kHz 16-bit s16le mono PCM to `paplay --raw` / `aplay` (cross-platform WAV fallback; `espeak-ng -s 230` / system-voice timeout fallback). |
| **Tier 2 — Inline Fast Grasp Gate** (`gate.py`, `atc/coordinator.py`) | `~3.3–5.7 s` gate (`4.97–6.32 s` incl. warm SSE TTS TTFA) | `gemini-4-hydrogen` (`thinkingLevel="LOW"`, `mediaResolution="MEDIA_RESOLUTION_LOW"`), `POST /v1beta/models/gemini-4-hydrogen:generateContent` | **Active co-gate on skill promotion:** evaluates 3 `<= 512 px` JPEG rescue keyframes (`stall`, `mid-hold`, `bolt-clear`) + contact torque / bolt / HV battery terminal clearance / flap depression telemetry in parallel with 20-rollout `SkillValidator` before `unoq.register_skill(skill)`. Speaks `verdict.say` immediately via Tier 1 SSE TTS. |
| **Tier 3 — Silent Deep Audit** (`review.py`, `console/*`, `atc/episodes.py`, `process/atc_qc.py`) | `~8–16 s` background (`10.45–16.46 s` live on 6 frames) | `gemini-4-argon` (`thinkingLevel="LOW"`), `POST /v1beta/models/gemini-4-argon:generateContent` | **Strictly silent & fire-and-forget** (zero TTS calls, never blocks Bay 1 gate or Bay 2 autonomous replay). Populates Fleet Ops Console `#audit-panel` second-opinion card and writes `dataset_tag` (`"approve"` \| `"flag"`) to `meta/teleop_episodes.jsonl` for ACT QC (`T-191` / `T-193`). |

- It never sends key presses or any other control message to the console. The one thing it
  writes is display text for the console's Copilot panel (`{"type": "copilot", ...}`, see
  [Copilot panel](#copilot-panel-t-215)).
- It never imports `teleop.atc.coordinator`.
- It never blocks or changes the 60 Hz loop, stall detection, the DMP fit, the 3-rollout gate or
  any safety abort.

The human and the existing gates decide. `tests/copilot/test_gemini.py` checks the package
source for coordinator imports, `"key"` messages and `send` calls. The only module allowed a
`send` is `publish.py`, and `tests/copilot/test_publish.py` checks it can only carry `copilot`
messages.

## Active grasp gating and retained safety boundaries

Earlier milestones restricted the copilot to passive console commentary. That restriction is lifted
specifically for **DMP skill promotion** and **ACT dataset curation**:

- **Dual-approval skill gate:** When the operator releases the clutch with `bolt >= 5.2 cm`
  (`0.052 m`), `AtcCoordinator` fits the DMP via `unoq.fit_dmp(...)` and runs `SkillValidator`
  (20 randomized headless MuJoCo rollouts, `>= 18/20` required) **in parallel with** Tier 2
  `GraspGate` (`gemini-4-hydrogen`, or configured local Gemma fallback). `unoq.register_skill(skill)`
  promotes the skill to the Fleet Library **if and only if both** Sim and `GraspGate` approve.
- **Episode curation:** Tier 3 `AuditReviewer` (`gemini-4-argon`) tags recorded episodes
  (`dataset_tag`: `"approve"` or `"flag"`) so `teleop process` (`atc_qc.py`) drops `"flag"`
  episodes (`"copilot flagged"`) before ACT policy training.

The hard real-time and control safety boundaries remain strictly enforced:

- **Never sends raw operator control keys:** No code path in `teleop.copilot` ever sends `"key"`
  messages (`{"type": "key", ...}`) or invokes `.on_key(...)`. `tests/copilot/test_gemini.py`
  enforces this AST boundary while permitting grasp-gate types and `"copilot_audit"` WebSocket
  status updates.
- **Never blocks the 60 Hz physics loop:** `GraspGate` and `AuditReviewer` run on dedicated
  background worker executors with private off-thread MuJoCo `MjModel` / `MjData` / `mj.Renderer`
  instances, never touching the live control step, stall detector, or torque-tube safety aborts.
- **Never reads simulator ground truth:** Control and gating never read `flap_hinge` `qpos`,
  `true_world_latch()`, or `data.contact`.
- **100% deterministic offline fallback:** When `GEMINI_API_KEY` is unset, `--no-gemini` is passed
  to `teleop atc`, or `--headless` is active (and `COPILOT_LOCAL_URL` is unset), `GraspGate`
  defaults to deterministic physics-only pass-through with zero network calls, preserving the exact
  `2386`-tick golden trajectory in `tests/atc/test_recovery.py`.

## Setup

1. Get a Gemini API key from Google AI Studio.
2. Put it in the repo's `.env`, which is gitignored:

   ```dotenv
   GEMINI_API_KEY=your-key-here
   ```

   The copilot reads, in order of precedence:

   1. the process environment;
   2. the `.env` of the current worktree;
   3. the `.env` at the main checkout's root.

   So a key saved once in the main checkout also works inside every git worktree.
   `--env-file PATH` reads one file instead.
3. Check the key, the models and the latency:

   ```bash
   uv run python -m teleop.copilot probe
   uv run python -m teleop.copilot probe --no-images   # text-only requests
   ```

   `probe` exits with:
   - `0` when the call-out, review and speech roles all work;
   - `1` when one of them failed (the report says which, and what to use instead);
   - `2` when no key is set.
4. Run the integrated three-tier showcase in `teleop atc` (or force offline physics-only mode):

   ```bash
   uv run teleop atc --source synthetic              # 3-tier copilot + active grasp gate enabled when GEMINI_API_KEY is set
   uv run teleop atc --source synthetic --no-gemini  # deterministic physics-only gate (or local Gemma if COPILOT_LOCAL_URL is set)
   ```

### Environment variables

These are documented here rather than in `.env.example`, which `T-100` owns. All of them are
optional except the key.

| Variable | Default | Used for |
|---|---|---|
| `GEMINI_API_KEY` | (none) | Every Gemini call (Tier 1 SSE TTS, Tier 2 Hydrogen gate, Tier 3 Argon audit, standalone call-outs). Never printed or logged. |
| `COPILOT_FAST_MODEL` | `gemini-4-hydrogen` | Tier 2 inline fast grasp gate (`GraspGate`) and live console call-outs |
| `COPILOT_REVIEW_MODEL` | `gemini-4-argon` | Tier 3 silent background engineering audit (`AuditReviewer`) and ACT dataset tagging |
| `COPILOT_TTS_MODEL` | `gemini-3.8-flash-lite-tts` | Tier 1 SSE-streamed speech (`GeminiStreamingTTS`) and startup cue pre-cache (`precache_cues`) |
| `COPILOT_VOICE` | `Kore` | Prebuilt Gemini TTS voice (`Kore` default; `Charon` alternative) |
| `COPILOT_LOCAL_URL` | (none) | OpenAI-compatible multimodal endpoint (`POST {url}/chat/completions`) for offline/local VLM fallback (`T-214`) |
| `COPILOT_LOCAL_MODEL` | `gemma4:e4b` | Local fallback model used for Tier 2 grasp gating and call-outs when Gemini is disabled (`--no-gemini`) or `GEMINI_API_KEY` is unset |
| `COPILOT_CONSOLE_URL` | `ws://127.0.0.1:8765/ws` | The console stream watched/updated by standalone `teleop.copilot` (`T-211`, `T-215`) |
| `COPILOT_MAX_REVIEWS_PER_HOUR` | `20` | Sliding-window hourly budget for Tier 3 `AuditReviewer` (`T-213`) |
| `COPILOT_FAST_THINKING` | `low` | Thinking level (`LOW` on wire) for Tier 2 Hydrogen gate and live call-outs: `low`, `medium`, `high` or `xhigh` (`xhigh` on Hydrogen only) |
| `COPILOT_REVIEW_THINKING` | `low` | Thinking level (`LOW` on wire) for Tier 3 Argon deep audits (`low`, `medium` or `high`) |

The flags `--no-images` and `--mute` become `CopilotConfig.no_images` and `CopilotConfig.mute`;
`--no-gemini` on `teleop atc` disables cloud Gemini calls for offline determinism.

## Privacy

- **Camera frames are sent to Google** when images are on. With `--no-images`, the client drops
  every image part before building the request across Tier 2 `GraspGate`, Tier 3 `AuditReviewer`,
  and live call-outs, so no image bytes leave the machine. Automated tests verify this.
- **Direct TLS & header-only key:** Both unary `UrllibTransport` and streaming `IPv4HTTPSConnection`
  connect directly over TLS (`server_hostname="generativelanguage.googleapis.com"`) with the key
  only in the `x-goog-api-key` header (never in URLs, and scrubbed from every exception string).
- **Redirects are refused.** `urllib` would otherwise copy the key header to the host a redirect
  names, and `IPv4HTTPSConnection` never follows HTTP redirects.
- **The probe never sends a camera frame.** It sends only synthetic 64×64 test patterns.
- **Logs** (`runs/copilot/*.jsonl` and `meta/teleop_episodes.jsonl`) store SHA-256 frame hashes and
  structured verdicts, never raw images or API keys.

## Models

| Role | Model | Why |
|---|---|---|
| **Tier 1 — Instant Voice & SSE TTS** | `gemini-3.8-flash-lite-tts`, voice `Kore` | Streamed via `POST /v1beta/models/gemini-3.8-flash-lite-tts:streamGenerateContent?alt=sse` over a persistent IPv4 `HTTPSConnection` (`keep-alive`). Emits raw 24 kHz, 16-bit signed LE mono PCM (`audio/l16; rate=24000; channels=1`) piped straight to `paplay --raw` / `aplay` (with WAV fallback), achieving `~0.48–0.65 s` warm TTFA (`< 0.7 s` target). Falls back to `espeak-ng -s 230` / system voice if chunk 1 exceeds `1.8 s` (`1.5–2.0 s`). Fixed telemetry cues are pre-cached at startup (`< 50 ms`, zero VLM wait). |
| **Tier 2 — Inline fast grasp gate & live call-outs** | `gemini-4-hydrogen`, thinking `low` | Evaluates 3 `<= 512 px` rescue keyframes (`stall`, `mid-hold`, `bolt-clear`) + `GraspTelemetrySummary` at `mediaResolution: "MEDIA_RESOLUTION_LOW"` (`798` image tokens total vs `3,192` default) in parallel with `SkillValidator` (`~4.5–5.7 s`), returning structured `GraspGateVerdict` JSON (`approve` \| `reject`, `say`, `reason`, physical booleans, `confidence`). `minimal` and `thinkingBudget` are HTTP 400 on Gemini 4. Cloud fallback: `gemini-3.8-flash`. |
| **Tier 3 — Silent deep audit & dataset curation** | `gemini-4-argon`, thinking `low` | Strongest Gemini 4 model for multi-frame spatial/engineering audit on up to 6 `<= 512 px` rescue JPEGs + full bay rescue timeline (`~10.5–16.5 s`, 90 s timeout, max `COPILOT_MAX_REVIEWS_PER_HOUR=20`/hr). **Strictly silent (zero TTS calls)** and off the critical path; populates Console `#audit-panel` and tags `meta/teleop_episodes.jsonl` (`dataset_tag`: `"approve"` \| `"flag"`). |
| **Offline / local VLM fallback (`T-214`)** | `gemma4:e4b` | Served via `LocalModelClient` (`COPILOT_LOCAL_URL`) when `GEMINI_API_KEY` is unset or `--no-gemini` is passed; backs Tier 2 `GraspGate` and live call-outs before falling back to deterministic physics-only / templates. |

Speech uses a dedicated TTS model rather than the text models.

Other models were considered and rejected for `T-210`:
- **DeepSeek Flash and ElevenLabs** each need a second vendor key and a second client.

`T-214`'s OpenAI-compatible client can point at other providers later without code changes.

## Gemini clients and streaming TTS transport

### Unary JSON & WAV client (`teleop.copilot.gemini.GeminiClient`)

`GeminiClient` uses only the standard library, through an injectable `Transport`, so tests run
offline:

- `list_models(timeout_s)` lists every model the key can see, following `nextPageToken`.
- `generate_json(model, system, parts, schema, timeout_s, *, thinking_level=None, media_resolution=None)`
  returns a `dict`.
  - **Parts** are `str`, `TextPart` or `ImagePart`. An image is sent as base64 `inlineData`.
  - **Schema:** a JSON Schema (lower-case `"type": "object"`) is sent as `responseJsonSchema`. A
    Gemini OpenAPI schema (upper-case `"OBJECT"`) is sent as `responseSchema`.
  - **Answer:** thought parts (`p.get("thought") is True`) and Markdown fences are stripped, and
    the result must be one JSON object.
- `synthesize(text, model, voice, timeout_s)` returns a WAV file for unary probes/caching. It
  requests `responseModalities: ["AUDIO"]` and a prebuilt voice.
- **Errors:**
  - `GeminiHttpError` (with `status_code`), `GeminiTimeoutError` and `GeminiMalformedJsonError`.
  - All of them subclass `GeminiError`, and none contains the key.
  - A refusal from the model's safety filter raises `GeminiError` too.

### Persistent IPv4 SSE streaming TTS (`teleop.copilot.tts_stream`)

- **`IPv4HTTPSConnection`:** Subclasses `http.client.HTTPSConnection` (`host="generativelanguage.googleapis.com"`,
  `port=443`), resolves `socket.AF_INET` via `socket.getaddrinfo` (avoiding broken/slow IPv6 routes),
  sets `TCP_NODELAY=1` and `SO_KEEPALIVE=1`, and wraps TLS with `server_hostname=self.host`.
- **`GeminiStreamingTTS.stream_pcm(text, *, voice="Kore", first_chunk_timeout_s=1.8, total_timeout_s=8.0)`:**
  - Sends `POST /v1beta/models/{tts_model}:streamGenerateContent?alt=sse` with `Connection: keep-alive`
    and `generationConfig={"responseModalities": ["AUDIO"], "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}`
    (**no `thinkingConfig`**).
  - Parses SSE `data: {...}\r\n\r\n` events, base64-decoding `inlineData.data` (`audio/l16; rate=24000; channels=1`)
    into raw 24 kHz 16-bit signed little-endian mono PCM chunks.
  - Drains the HTTP response cleanly through `finishReason == "STOP"` and `readline() == b""` so the
    underlying TLS socket remains warm and reusable across utterances; auto-reconnects once on stale
    idle sockets; calls `conn.close()` on any mid-stream timeout/abort so unread SSE bytes never
    corrupt `keep-alive` state.
  - Uses a dedicated connection instance separate from VLM calls so Tier 2 Hydrogen completion never
    contends with Tier 1 audio streaming.

### Notes about the API

- **Interactions API vs. `generateContent`:** Google's Gemini 4 developer pages document
  `/v1beta/interactions`, while `generateContent` and `:streamGenerateContent?alt=sse` remain fully
  supported across Gemini 4 Hydrogen, Gemini 4 Argon, Gemini 3.8 Flash, and `gemini-3.8-flash-lite-tts`
  on an AI Studio key.
- **Wire constraints verified by live probing:**
  1. `thinkingConfig` is **forbidden** on TTS models (`HTTP 400: Thinking level is not supported for this model`).
  2. `thinkingLevel: "MINIMAL"` is **forbidden** on Gemini 4 (`HTTP 400`); `"LOW"` is the fastest supported level.
  3. `responseSchema` (OpenAPI uppercase `"OBJECT"`) **must omit `additionalProperties`** (`HTTP 400: Unknown name "additionalProperties"` on Hydrogen and Argon), whereas `responseJsonSchema` (JSON Schema lowercase `"object"`) accepts `"additionalProperties": false`.
  4. `mediaResolution: "MEDIA_RESOLUTION_LOW"` in `generationConfig` cuts 3-frame JPEG prompt image tokens by 75% (`3,192 -> 798`), keeping Tier 2 Hydrogen + warm SSE TTS TTFA well under 10 s (`4.97–6.32 s`).
  5. Gemini 4 answer parts carry `thoughtSignature` alongside `text`; filtering `p.get("thought") is not True` cleanly extracts structured JSON.

## Probe results

Measured on 2026-11-14 from this workstation with a Gemini API key from AI Studio. Rerun with:

```bash
uv run python -m teleop.copilot probe
uv run python -m teleop.copilot probe --no-images
```

### Three-tier SSE & active grasp gate benchmark (2026-11-14)

Live host verification across the active three-tier pipeline (`IPv4HTTPSConnection` + SSE raw PCM
streaming, `MEDIA_RESOLUTION_LOW` Hydrogen 3-frame gate, and 6-frame Argon background audit):

| Tier / Path | Endpoint & Config | Measured Latency | Notes |
|---|---|---|---|
| Tier 1 TTS (warm `keep-alive`) | `gemini-3.8-flash-lite-tts:streamGenerateContent?alt=sse`, `Kore`, `IPv4HTTPSConnection` | **`483.4 – 652.6 ms` TTFA** (`1049–1389 ms` full stream) | Meets `< 0.7 s` target on 100% of warm calls; single `AF_INET` TLS socket reused continuously; `1920*k` byte (`40 ms`) `s16le` 24 kHz mono PCM chunks |
| Tier 1 TTS (cold connect) | Same endpoint, including `423.2 ms` IPv4 TCP+TLSv1.3 handshake | **`1348.1 ms` TTFA** (`2058.9 ms` full stream) | `< 1.5 s` (`< 1.8 s` timeout); startup `precache_cues` warms the socket before the operator session begins |
| Tier 1 TTS (unary baseline comparison) | `:generateContent` (full WAV buffer before byte 1) | `1952.9 ms` warm / `3384–12600 ms` `urllib` | Explains why initial `T-210` / `T-212` unary WAV synthesis missed the 1.5 s deadline |
| Tier 2 Fast Grasp Gate | `gemini-4-hydrogen:generateContent`, `thinkingLevel="LOW"`, `MEDIA_RESOLUTION_LOW`, 3 × `512×384` JPEGs + telemetry | **`4474.2 – 5741.9 ms`** (`798` image tokens) | Runs in parallel with `SkillValidator` on clutch release (`bolt >= 5.2 cm`) |
| Tier 2 Gate + Tier 1 Dynamic Voice E2E | Hydrogen 3-frame verdict -> warm `tts_conn` SSE `verdict.say` | **`4970.6 – 6316.4 ms` total to first spoken audio** | Well under the `< 10 s` showcase budget |
| Tier 3 Silent Deep Audit | `gemini-4-argon:generateContent`, `thinkingLevel="LOW"`, 6 × `512×384` JPEGs + timeline | **`10,450.6 – 16,455.2 ms`** | Fire-and-forget daemon worker; implicit prefix caching (`cachedContentTokenCount`); 0 TTS calls |

### Initial `T-210` unary `:generateContent` baseline

The measurements below come from the initial `T-210` unary `:generateContent` probe (before Tier 1
switched to persistent IPv4 SSE streaming and before Tier 2 enabled `MEDIA_RESOLUTION_LOW`), kept
here as baseline context for the `probe` command output:

| Role | Model | Calls | Results (ms) | Median |
|---|---|---|---|---|
| Call-out | `gemini-4-hydrogen`, thinking `low` | 7 | 3826, timeout (> 8 s), 5418, 5769, 4114, 3377, 2904 | 3970 (6 answered) |
| Call-out | `gemini-3.8-flash`, thinking `low` | 7 | 6955, 3000, 3715, 4922, 3320, 3341, 2885 | 3341 |
| Review | `gemini-4-argon`, thinking `high`, 3 frames | 2 | 23977, 84445 | — |
| Review | `gemini-4-argon`, thinking `low`, 3 frames | 5 | 12423, 39679, 12621, 17834, 38445 | 17834 |
| Speech (unary WAV) | `gemini-3.8-flash-lite-tts`, Kore | 2 | 3914 (3.2 s of audio), 4271 (4.1 s of audio) | — |

- **Why Tier 1 uses pre-cached cues + SSE streaming:** Unary WAV synthesis (`3.4–12.6 s` over
  fresh `urllib` connections) waited for the entire utterance before returning byte 1, motivating
  startup `precache_cues` (`< 50 ms`) and persistent IPv4 SSE streaming (`483–653 ms` warm TTFA).
- **Why Tier 2 runs Hydrogen in parallel with `SkillValidator` at `MEDIA_RESOLUTION_LOW`:**
  Evaluating 3 rescue keyframes at `MEDIA_RESOLUTION_LOW` completes in `~4.5–5.7 s` alongside the
  20 headless simulation rollouts, streaming Hydrogen's spoken `"say"` verdict immediately over
  warm SSE TTS in `~5.0–6.3 s` total.
- **Why Tier 3 Argon stays strictly off the critical path:** Multi-frame Argon audits take
  `~10.5–40 s` (and up to `84 s` at `high`), so `AuditReviewer` runs fire-and-forget on a daemon
  worker with zero TTS calls.
- With synthetic probe images, Argon answered `insufficient_evidence`, confirming conservative
  grounding when visual evidence is synthetic or incomplete.

### Thinking levels

Every text/VLM model runs at thinking level `low` by default:

- **Tier 3 Argon moved from `high` to `low`.** Over five 3-frame unary calls, Argon took 12–40 s
  (median 17.8 s) vs. 24 s and 84 s at `high`, and 10.5–16.5 s on 6 rescue keyframes with prefix
  caching.
- **Tier 2 Hydrogen and call-outs run at `low`,** which is the fastest level Gemini 4 accepts
  (`minimal` is HTTP 400).
- **The TTS models have no effort setting.** A request with any `thinkingConfig` gets HTTP 400:
  "Thinking level is not supported for this model".

In two earlier unary `urllib` probe runs at `low` (images on), fresh-connection unary latency varied
with server load:

| Model | Earlier | At `low` (unary `urllib`) |
|---|---|---|
| Hydrogen call-out | 2.9–5.8 s | 5.7 s, 5.7 s |
| Flash call-out | 2.9–7.0 s | 7.7 s, 9.8 s |
| Speech (unary WAV) | 3.4–4.3 s | 12.6 s, 9.3 s |

Persistent IPv4 `keep-alive` SSE streaming eliminates per-call TCP+TLS setup and full-utterance
buffering on speech, while the `1.8 s` first-chunk timeout (`espeak-ng` / system voice) and `1 s`
callout templates bound worst-case tail latency.

### Run 1: images on (verbatim, review at `high` before the change)

Images: on (synthetic 64x64 JPEGs). Times are wall-clock ms from `time.monotonic_ns()`.

### Models the key can use

`GET /v1beta/models` returned 64 models in 433 ms; 49 support `generateContent`.

| Model | Listed | generateContent |
|---|---|---|
| `gemini-4-hydrogen` | yes | yes |
| `gemini-4-argon` | yes | yes |
| `gemini-3.8-flash` | yes | yes |
| `gemini-3.8-flash-lite-tts` | yes | yes |
| `gemini-3.8-flash-tts` | yes | yes |
| `gemini-3.1-flash-tts-preview` | yes | yes |

### Timed calls

| Role | Model | Result | ms | Notes |
|---|---|---|---|---|
| call-out | `gemini-4-hydrogen` | ok | 3826 | thinking low; answer {"say": "Bay 1 stalled during armed phase due to a flap fault.", "detail": "Bay 1 confidence dropped to 0.21, below the 0.3 threshold. Bolt is at 1.4 cm of 5.2 cm required. Fault: flap in armed phase.", "severity": "alarm"}; slower than the 3000 ms target |
| call-out (comparison) | `gemini-3.8-flash` | ok | 6955 | thinking low; answer {"say": "Bay 1 stalled due to low contact confidence during flap hold.", "detail": "Bay 1 confidence dropped to 0.21, below the 0.3 threshold with bolt at 1.4 cm. A flap fault occurred in armed phase.", "severity": "alarm"}; slower than the 3000 ms target |
| review | `gemini-4-argon` | ok | 23977 | thinking high; answer {"verdict": "insufficient_evidence", "reasons": ["Telemetry only reports the initial stall state for Bay 1: contact confidence is 0.21 (staying below the 0.3 threshold at 0.27 and 0.21), bolt travel is 1.4 cm (below the required 5.2 cm clearance), and a flap fault is armed.", "The three camera frames are synthetic test patterns for a connectivity probe, so the physical positions of the stopper flap and bolt during the clutch hold and after the rescue are unclear.", "No post-rescue telemetry is provided to verify that the stopper flap remained held down or that the bolt slid at least 5.2 cm clear."], "operator_note": "Bay 1 stalled with an armed flap fault, contact confidence at 0.21 (below 0.3), and bolt position at 1.4 cm (below the 5.2 cm target). Because the images are synthetic test patterns and post-rescue telemetry is missing, the rescue status is unclear. Please verify physical latch state before proceeding.", "confidence": 0.95} |
| speech | `gemini-3.8-flash-lite-tts` | ok | 3914 | voice Kore; WAV 161308 bytes, 3200 ms of audio |

### Recommended `.env` values

```dotenv
COPILOT_FAST_MODEL=gemini-4-hydrogen
COPILOT_REVIEW_MODEL=gemini-4-argon
COPILOT_TTS_MODEL=gemini-3.8-flash-lite-tts
COPILOT_VOICE=Kore
```

### Run 2: `--no-images` (timed calls, verbatim, review at `high`)

Listing and recommendations matched run 1.

### Timed calls

| Role | Model | Result | ms | Notes |
|---|---|---|---|---|
| call-out | `gemini-4-hydrogen` | ok | 4174 | thinking low; answer {"say": "Bay 1 stalled during the armed phase due to a flap fault.", "detail": "Bay 1 is STALLED with contact confidence at 0.21, below the 0.3 threshold. The bolt is at 1.4 cm, short of the 5.2 cm clear target. Fault: flap in armed phase.", "severity": "alarm"}; slower than the 3000 ms target |
| call-out (comparison) | `gemini-3.8-flash` | ok | 2170 | thinking low; answer {"say": "Bay 1 stalled due to low contact confidence on the flap.", "detail": "Bay 1 contact confidence dropped to 0.21, below the 0.3 threshold during the flap armed phase. Bolt reached only 1.4 cm of the required 5.2 cm clearance.", "severity": "alarm"} |
| review | `gemini-4-argon` | ok | 21526 | thinking high; answer {"verdict": "insufficient_evidence", "reasons": ["Visual frames for the stall, clutch hold, and after-rescue states are missing or synthetic test patterns, making visual verification unclear.", "Provided telemetry only shows Bay 1 in a STALLED state with contact confidence at 0.21 (below the 0.3 threshold).", "Bolt position is at 1.4 cm, which has not reached the required 5.2 cm clearance, and an armed flap fault remains present."], "operator_note": "Rescue outcome is unclear due to synthetic/missing frame data and lack of post-rescue telemetry. Telemetry shows Bay 1 stalled with an armed flap fault, confidence at 0.21 (below 0.3), and bolt at 1.4 cm (below 5.2 cm required). Do not bypass safety stops or alter settings; inspect Bay 1 directly.", "confidence": 0.9} |
| speech | `gemini-3.8-flash-lite-tts` | ok | 3384 | voice Kore; WAV 184348 bytes, 3680 ms of audio |

### Recommended `.env` values

```dotenv
COPILOT_FAST_MODEL=gemini-4-hydrogen
COPILOT_REVIEW_MODEL=gemini-4-argon
COPILOT_TTS_MODEL=gemini-3.8-flash-lite-tts
COPILOT_VOICE=Kore
```

## Console feed and events (T-211)

In standalone mode, `FeedClient` watches the console's `/ws` stream for telemetry and camera frames
and never sends robot control (`"key"`) or hardware `"reconnect"` messages. When `T-215` audit
publishing is active, standalone `python -m teleop.copilot run` only sends
`{"type": "copilot_audit", "audit": ...}` metadata updates so the Fleet Ops Console `#audit-panel`
can display Argon's second opinion (while in-process `teleop atc` publishes `gate` and `audit`
state directly through `TelemetryAdapter`).

- **`FeedClient`** (`teleop.copilot.feed`) connects to `COPILOT_CONSOLE_URL`.
  - It reads telemetry and camera frames.
  - It reconnects after 1 s, doubling each time up to 5 s. A successful connection resets the
    delay.
  - Each frame's slot is mapped to a camera through the latest telemetry's `feeds`. A frame is
    dropped when it arrives before any telemetry or carries an old layout's byte.
- **`FrameStore`** (`teleop.copilot.frames`) keeps the newest frame per bay and camera.
  - While a bay is in `RESCUE` with the clutch held, it also samples that bay's frames: at most
    one per camera every 0.5 s, and at most 12 in all. These are for the review.
  - The samples are kept after the clutch is released and cleared at that bay's next stall.
  - `downscale_jpeg` shrinks a frame to at most 512 px.
  - Logs use `StoredFrame.sha256`, never the image.
- **`EventDetector`** (`teleop.copilot.events`) turns telemetry changes into events: `stall`,
  `refused`, `rescue_started`, `rescue_released`, `validating`, `gate_result`,
  `autonomous_started` and `autonomous_finished`.
  - The first message only sets a baseline, so a reconnect never repeats an event.
  - A backwards jump in `sim_t` means the console restarted, so the detector starts a new
    baseline.
  - Replayed background bays never fire.

Facts from the recorded stage (`tests/copilot/fixtures/`, about 35 s of `--script`):
- A jam goes straight from `NOMINAL` to `RESCUE` within one 10 Hz message.
- The skill gate runs 20 randomized rollouts (`passed`/`total`), not 3.
- The refusal is Bay 2 in `OOD_ESCALATED`.

### Try it

```bash
uv run teleop atc --source synthetic            # terminal 1; press F to jam Bay 1
uv run python -m teleop.copilot watch           # terminal 2
```

Example output from the scripted stage:

```text
[16:38:33] BAY 1 STALL                NOMINAL -> RESCUE, C=0.24, bolt 3.4 cm, frame bay1/cctv ed788796739f8ad2
[16:38:38] BAY 1 RESCUE_STARTED       RESCUE -> RESCUE, C=0.74, bolt 3.4 cm, frame bay1/side_ortho 75659194be5399f8
[16:38:41] BAY 1 RESCUE_RELEASED      RESCUE -> VALIDATING, C=0.55, bolt 5.6 cm, frame bay1/side_ortho 8ae094806c786223, 10 rescue frames
[16:38:41] BAY 1 VALIDATING           RESCUE -> VALIDATING, C=0.55, bolt 5.6 cm, frame bay1/side_ortho 8ae094806c786223, 10 rescue frames
[16:39:14] BAY 1 GATE_RESULT          VALIDATING -> NOMINAL, C=0.64, bolt 5.6 cm, gate PASS 20/20, frame bay1/side_ortho 7441cd24b6794bb7, 10 rescue frames
[16:39:20] BAY 2 AUTONOMOUS_STARTED   NOMINAL -> AUTONOMOUS_SKILL, C=0.36, bolt 3.4 cm, frame bay2/cctv 2e55add3d416dc4f
```

## Live call-outs and Tier 1 instant voice (T-212)

```bash
uv run teleop atc --source synthetic                     # integrated 3-tier voice + grasp gate + audit
uv run python -m teleop.copilot run                      # standalone sidecar watching console /ws
uv run python -m teleop.copilot run --mute --no-images   # print and log only, text-only requests
```

Flags on `python -m teleop.copilot run`: `--console URL` (default `COPILOT_CONSOLE_URL`),
`--no-images`, `--mute`, `--gemini-tts` / `--no-gemini-tts`, `--env-file PATH`, and `--log-dir DIR`
(default `runs/copilot`, gitignored). Without `GEMINI_API_KEY` (or when `--no-gemini` is passed to
`teleop atc`), deterministic template cues and `espeak-ng` / system voice run offline without
external calls.

### Call-outs (`teleop.copilot.callouts`, `teleop.copilot.prompts`)

- **Model events:** `stall`, `refused`, `gate_result` and `autonomous_finished` ask the fast model
  (`COPILOT_FAST_MODEL`, thinking `low`) with the bay's telemetry slice and its latest frame,
  downscaled to at most 512 px (no frame with `--no-images`). The answer follows
  `{"say", "detail", "severity"}`; `say` is clipped to 15 words and `detail` to 40.
- **Templated / zero-VLM events:** `rescue_started` (`clutch_engage`), `bolt_cleared`,
  `rescue_released` (`clutch_release_short`), `validating`, and `autonomous_started` (`autonomous`)
  fire directly from deterministic telemetry transitions with **zero VLM calls**.
- **The 1 s template.** If the model has not answered within 1 s, the deterministic template
  (for example "Bay 1 stalled on a flap jam. Rescue needed.") is spoken at once. If the model's
  line then arrives while the template is still waiting in the speech queue, it replaces it there;
  if the template is already being spoken, the model's line is only printed and logged
  (`(update)`). Model errors and malformed answers speak the template straight away.
- **8 s hard timeout** (wall clock, on top of the client's per-socket timeout): a later answer is
  dropped.
- **One request per bay.** A newer model event for a bay supersedes an older pending model request
  and cancels queued lines for that bay, while templated events let an in-flight model call finish.

### Tier 1 Instant Voice & SSE Speech (`teleop.copilot.speech`, `teleop.copilot.tts_stream`)

Tier 1 targets `< 0.7 s` warm time-to-first-audio (`483–653 ms` measured live) with zero VLM wait
on telemetry edges, active by default whenever `GEMINI_API_KEY` is set (unless `--mute` or
`--no-gemini`):

1. **Startup cue pre-caching (`Speaker.precache_cues`):** Fixed short telemetry cues (`stall`,
   `ood`, `clutch_engage`, `bolt_cleared`, `clutch_release_short`, `validating`, `gate_passed`,
   `gate_failed`, `autonomous`) are pre-synthesized with `gemini-3.8-flash-lite-tts` (`Kore` voice)
   into an in-memory/disk PCM/WAV cache at startup—simultaneously warming the persistent
   `IPv4HTTPSConnection`—so deterministic alerts play in `< 50 ms` with zero VLM or network wait.
2. **SSE raw PCM streaming (`GeminiStreamingTTS.stream_pcm`):** Uncached dynamic lines (such as
   Tier 2 Hydrogen's `verdict.say` or live event call-outs) stream 24 kHz, 16-bit signed LE mono
   PCM chunks over `POST /v1beta/models/gemini-3.8-flash-lite-tts:streamGenerateContent?alt=sse`
   directly into:
   - **Primary (Linux PulseAudio / PipeWire):**
     `paplay --raw --format=s16le --rate=24000 --channels=1 --latency-msec=40`
   - **Secondary (Linux ALSA):**
     `aplay -q -f S16_LE -r 24000 -c 1 -t raw -`
   - **Cross-platform WAV fallback (`pcm_to_wav` -> `play_wav`):** wraps PCM in WAV for `paplay`,
     `aplay`, macOS `afplay`, or Windows PowerShell `System.Media.SoundPlayer` when raw PCM stdin
     players are unavailable. (After `proc.stdin.close()`, `proc.stdin = None` is set before
     `proc.wait()` / `communicate()` per Python 3.12 `subprocess` semantics.)
3. **First-chunk timeout fallback (`first_chunk_timeout_s = 1.8`, within `1.5–2.0 s`):** If chunk 1
   exceeds `1.8 s` or raises `GeminiError` / `BrokenPipeError`, `Speaker` closes the aborted TTS
   socket and any unplayed raw pipe, and immediately speaks the line with the system voice
   (`espeak-ng -s 230` tried before `espeak -s 230` and `spd-say -w -r 30` on Linux; `say -r 230`
   on macOS; persistent PowerShell `SpeechSynthesizer` at rate 2 on Windows). Late audio is never
   played out of order.
4. **Queue & voice:** At most 2 lines wait (`MAX_WAITING = 2`). Alarms jump ahead of non-alarms;
   lines older than `10.0 s` (`STALE_AFTER_S`) are dropped; only `say` is spoken, never `detail`.
   Default voice is `Kore` (`COPILOT_VOICE=Charon` alternative). `--mute` silences audio while
   keeping terminal and JSONL logging active.

### Outputs

```text
[12:01:07] BAY 1 ALARM     Bay 1 stalled on a flap jam. Rescue needed.
           NOMINAL -> RESCUE, C=0.24, bolt 3.4 of 5.2 cm, fault flap  [template; model slower than 1 s]
```

`runs/copilot/<date>.jsonl` records `event`, `callout`, `speech` (`route`, `start_ms`, `speak_ms`),
`gate`, and `review` entries. It never holds raw images or the API key.

### Historical `T-212` stage checks (pre-SSE baseline, 2026-11-14)

- **Offline template check:** `teleop atc --source synthetic --script` + `python -m teleop.copilot run --mute`
  without a Gemini key emitted all eight scripted events in 2–3 ms in stage order (Bay 1 stall,
  clutch engage/release, validating, 20/20 gate, Bay 2 autonomous start/finish, Bay 2 refusal) with
  frame hashes only.
- **Initial unary WAV TTS runs (superseded by SSE streaming):** Before `tts_stream.py` introduced
  persistent IPv4 `:streamGenerateContent?alt=sse`, unary `:generateContent` WAV synthesis missed a
  `1.5–4.0 s` deadline on live runs (`3.4–12.6 s`), causing system-voice fallback (`espeak-ng -s 230`
  / `say -r 230` / `spd-say -r 30` / Windows `SpeechSynthesizer`). Switching to persistent IPv4 SSE
  raw PCM streaming reduced warm TTFA to `483–653 ms` (`1348 ms` cold), making `Kore` Gemini TTS fast
  enough to serve as the primary live voice while retaining `espeak-ng` / system voice as the `1.8 s`
  first-chunk safety net.

## Tier 2: Inline fast grasp gate (`gemini-4-hydrogen`, `teleop.copilot.gate`)

When the operator releases the clutch on a rescued bay with `bolt_x >= 0.052 m` (`5.2 cm`),
`AtcCoordinator` validates both **mechanical rollout generalization** and **physical grasp quality**
in parallel before promoting the skill:

- **Lightweight pose capture & off-thread keyframe rendering:** During the rescue clutch hold,
  `AtcCoordinator` records `< 50 µs` `PoseSnapshot` structs (`qpos`, `mocap_pos`, `mocap_quat`,
  `body_pos`, `body_quat`) at semantic milestones on the 60 Hz thread. `GraspGate` renders 3
  `<= 512 px` JPEG keyframes (`stall`, `mid-hold`, `bolt-clear`) on its own background worker using
  a private `MjModel` / `MjData` / `mj.Renderer` copy so the 60 Hz physics loop and viewer OpenGL
  context are never blocked or shared across threads.
- **Parallel gate submission:** `AtcCoordinator._submit_gate` fits the DMP via `unoq.fit_dmp(...)`
  and submits `SkillValidator.validate(skill)` (20 randomized rollouts on `coordinator._executor`)
  **in parallel** with `GraspGate.submit(...)` (on `GraspGate`'s dedicated executor).
- **Physical assessment & structured JSON (`GraspGateVerdict`):** `GraspGate` calls
  `POST /v1beta/models/gemini-4-hydrogen:generateContent` with `thinkingConfig: {"thinkingLevel": "LOW"}`
  and `mediaResolution: "MEDIA_RESOLUTION_LOW"`, passing the 3 rescue keyframes (omitted under
  `--no-images`) plus `GraspTelemetrySummary` (`bay_id`, `sku`, `skill_id`, `signature_id`,
  `duration_s`, `bolt_start_m`, `bolt_final_m`, `peak_tau_hand_nm`, `mean_tau_hand_nm`,
  `peak_arm_ratio`, `min_wrist_x_latch_m` for `-X` HV battery terminal clearance,
  `min_wrist_z_latch_m` for flap depression geometry, `touch_offset_m`). Hydrogen returns:
  `{"verdict": "approve" | "reject", "say": str, "reason": str, "flap_depressed": bool, "bolt_clearance_ok": bool, "hv_clearance_ok": bool, "torque_controlled": bool, "confidence": float}`.
- **Immediate spoken verdict:** As soon as Hydrogen resolves—without waiting for the 20-rollout
  `SkillValidator` or the `10.0` sim-s gate deadline—`GraspGate` dispatches `verdict.say` straight
  to Tier 1 SSE `gemini-3.8-flash-lite-tts` (`Kore`), delivering spoken AI feedback in `~5.0–6.3 s`
  from clutch release.
- **Dual-approval promotion rule:** At the gate deadline, `AtcCoordinator._apply_gate` computes
  `approved = sim_report.passed and (grasp_verdict is None or grasp_verdict.approved)` and calls
  `unoq.register_skill(skill)` **if and only if** `approved` is `True`. If either Sim or Hydrogen
  rejects, the skill is withheld from the Fleet Library, `"VALIDATION_GATE"` logs `passed=False`
  with Hydrogen metadata, and the Console `#gate` panel displays the VLM verdict and reason.

## Offline determinism and local Gemma fallback (T-214, `teleop.copilot.local`)

 Tier 2 `GraspGate` and live call-outs follow a strict three-level hierarchy so offline development,
CI, and `--headless` runs remain 100% deterministic:

1. **Cloud Gemini** (`GEMINI_API_KEY` set, and neither `--no-gemini` nor `--headless` passed):
   `GeminiClient` (`gemini-4-hydrogen` for Tier 2 gate & call-outs, `gemini-4-argon` for Tier 3
   silent audit) + `GeminiStreamingTTS` (`gemini-3.8-flash-lite-tts`, `Kore`).
2. **Local Gemma (`LocalModelClient`)** (`GEMINI_API_KEY` unset or `--no-gemini` passed, while
   `COPILOT_LOCAL_URL` is configured and not `--headless`): Calls OpenAI-compatible
   `POST {COPILOT_LOCAL_URL}/chat/completions` (`COPILOT_LOCAL_MODEL`, default `gemma4:e4b`) with
   the same structured JSON `generate_json` contract for Tier 2 `GraspGate` (`source="local"`) and
   call-outs; speech uses `espeak-ng` / system voice; Tier 3 Argon audits report `status="skipped"`.
3. **Deterministic physics-only** (`--headless`, or `--no-gemini` / unset `GEMINI_API_KEY` when
   `COPILOT_LOCAL_URL` is unset, including all default `pytest` runs): Zero external network calls
   and zero audio subprocesses; `SkillValidator` governs alone (`source="physics_only"`) and
   `tests/atc/test_recovery.py` preserves its exact `2386`-tick golden trajectory
   (`tests/atc/golden/dmp_stage.json`).

Example local Gemma setup (e.g. Ollama on `http://127.0.0.1:11434/v1`):

```bash
COPILOT_LOCAL_URL=http://127.0.0.1:11434/v1 COPILOT_LOCAL_MODEL=gemma4:e4b \
  uv run teleop atc --source synthetic --no-gemini
```

## Tier 3: Silent deep audit, console panel and ACT dataset curation (T-213, T-215, T-191/T-193)

## Copilot panel (T-215)

`python -m teleop.copilot run` also shows every call-out in the fleet ops console, in a
**Copilot** panel at the top of the rail. The panel is hidden until the first message.

- **What it shows:** per bay, the latest call-out (severity as the row colour, the spoken line,
  the detail, the time) and, when a review has been sent, the latest verdict (approved, flagged,
  not enough evidence). Call-outs that refine a template line replace it.
- **Protocol:** the copilot posts `{"type": "copilot", "bay", "severity", "say", "detail",
  "source", "verdict"?}` on its console connection. The server validates it (bay 1-4, severity
  `info`/`attention`/`alarm`, verdict `approve`/`flag`/`insufficient_evidence`, bounded text),
  adds `seq` and `wall`, keeps the last 20 and sends it to every *other* client. A page that
  connects or reloads gets the kept messages first. Copilot messages never reach the key handler.
- **Advisory only:** no key or control message is involved. The page shows the text with
  `textContent`, never as markup.
- **Turn it off:** `--no-panel`. If the console is down, messages wait (at most 20) and are sent
  after the feed reconnects.
- **Reviews:** the `verdict` field is ready for `T-213`; nothing sends one yet.

- **Fire-and-forget `AuditReviewer` (`teleop.copilot.review`):** Triggered asynchronously on a
  daemon worker after each rescue or refusal (bounded by a sliding-window budget of
  `COPILOT_MAX_REVIEWS_PER_HOUR = 20`). Sends up to 6 `<= 512 px` rescue JPEGs (`2 stall / early`,
  `2 mid-hold`, `2 clear / release`, omitted under `--no-images`) plus the full bay rescue telemetry
  timeline to `gemini-4-argon` (`thinkingLevel: "LOW"`, `90 s` timeout).
- **Strictly silent & non-blocking:** `AuditReviewer` **never** calls `Speaker` / TTS (**zero TTS
  calls**) and **never** blocks `AtcCoordinator._Gate`, `unoq.register_skill(skill)`, or Bay 2
  autonomous recovery.
- **Structured `AuditReport`:** `audit_id`, `status` (`"pending"` -> `"complete"` \| `"skipped"` \|
  `"error"`), `bay`, `skill_id`, `episode_uid`, `verdict` (`"approve"` \| `"flag"`), `dataset_tag`
  (`"approve"` \| `"flag"`), `confidence`, `reasons`, `operator_note`, `model`, `frame_count`,
  `latency_ms`, `wall`.
- **Fleet Ops Console `#audit-panel` (`T-215`, `src/teleop/console/web/audit.js`, `index.html`,
  `console.css`, `telemetry.py`, `server.py`):** Displays live `pending` / `complete` engineering
  second-opinion cards (`approve` / `flag` badge, `ACT: approve` / `ACT: flag` dataset tag,
  confidence, engineering reasons, operator note, model, frame count, and latency) in the right-hand
  rail alongside the Hydrogen VLM badge in `#gate`. Supports both in-process `TelemetryAdapter`
  updates (`msg["audit"]`) and standalone WebSocket ingress (`{"type": "copilot_audit", "audit": ...}`)
  with zero external URLs in `src/teleop/console/web/*`.
- **Async ACT dataset curation (`T-191` / `T-193`, `src/teleop/atc/episodes.py`,
  `src/teleop/process/atc_qc.py`):** `EpisodeSink` assigns a monotonic `episode_uid`
  (`latest_episode_uid(bay_id)`) and exposes thread-safe
  `EpisodeSink.tag_episode(episode_uid, *, dataset_tag="approve"|"flag", audit=...)`, handling both
  episodes still queued/encoding in `episode-writer` (pre-AV1 encode) and atomic locked rewrites of
  `<root>/meta/teleop_episodes.jsonl` (post-encode). Downstream `teleop process` (`atc_qc.py`)
  includes `"copilot flagged"` in `DROP_REASONS`, dropping episodes where
  `meta.get("dataset_tag") == "flag"` while retaining `"approve"` and untagged (`None`) offline
  episodes.
