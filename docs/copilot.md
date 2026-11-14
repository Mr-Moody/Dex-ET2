# Operator copilot

The copilot is a control-room assistant for the human operator. It runs as its own process on
the operator's computer, watches the fleet ops console as one more viewer, and helps in two
ways:

- **Call-outs:** short spoken lines when something happens, for example when a bay stalls.
- **Reviews:** a careful second opinion on a rescue or a refusal.

Plan and task list: `docs/superpowers/plans/2026-11-14-operator-copilot-plan.md` (`T-210` to
`T-215`). This page covers `T-210`: the configuration, the Gemini client, and the live probe.
Later tasks add their own sections below.

## Advisory only

The copilot never controls a robot.

- It never sends key presses or any other control message to the console.
- It never imports `teleop.atc.coordinator`.
- It never blocks or changes the 60 Hz loop, stall detection, the DMP fit, the 3-rollout gate or
  any safety abort.

The human and the existing gates decide. `tests/copilot/test_gemini.py` checks the package
source for coordinator imports, `"key"` messages and `send` calls.

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

### Environment variables

These are documented here rather than in `.env.example`, which `T-100` owns. All of them are
optional except the key.

| Variable | Default | Used for |
|---|---|---|
| `GEMINI_API_KEY` | (none) | Every Gemini call. Never printed or logged. |
| `COPILOT_FAST_MODEL` | `gemini-4-hydrogen` | Live call-outs |
| `COPILOT_REVIEW_MODEL` | `gemini-4-argon` | Deep reviews |
| `COPILOT_TTS_MODEL` | `gemini-3.8-flash-lite-tts` | Speech |
| `COPILOT_VOICE` | `Kore` | A prebuilt Gemini TTS voice (`Charon` is an alternative) |
| `COPILOT_LOCAL_URL` | (none) | OpenAI-compatible local server for the Gemma fallback (`T-214`) |
| `COPILOT_LOCAL_MODEL` | `gemma4:e4b` | Local fallback model (`T-214`) |
| `COPILOT_CONSOLE_URL` | `ws://127.0.0.1:8765/ws` | The console stream to watch (`T-211`) |
| `COPILOT_MAX_REVIEWS_PER_HOUR` | `20` | Review budget (`T-213`) |
| `COPILOT_FAST_THINKING` | `low` | Thinking level for call-outs: `low`, `medium`, `high` or `xhigh` (`xhigh` on Hydrogen only) |
| `COPILOT_REVIEW_THINKING` | `low` | Thinking level for reviews (Argon takes `low`, `medium` or `high`) |

The flags `--no-images` and `--mute` become `CopilotConfig.no_images` and `CopilotConfig.mute`.

## Privacy

- **Camera frames are sent to Google** when images are on. With `--no-images`, the client drops
  every image part before building the request, so no image bytes leave the machine. A test
  checks this.
- **The key** travels only in the `x-goog-api-key` header, never in a URL. Every error message is
  scrubbed of it.
- **Redirects are refused.** urllib would otherwise copy the key header to the host a redirect
  names.
- **The probe never sends a camera frame.** It sends only synthetic 64×64 test patterns.
- **Later logs** (`runs/copilot/*.jsonl`, `T-212`) store frame hashes, never images or keys.

## Models

| Role | Model | Why |
|---|---|---|
| Live call-outs: telemetry plus one JPEG → JSON | `gemini-4-hydrogen`, thinking `low` | The fast Gemini 4 model, multimodal, with structured output. Thinking defaults to `high`, which is too slow for a call-out, so the client sends `low`. `minimal` and `thinkingBudget` are HTTP 400 on Gemini 4, so the client offers neither. Fallback: `gemini-3.8-flash`. |
| Deep review: telemetry plus up to 6 JPEGs → JSON | `gemini-4-argon`, thinking `low` | The strongest Gemini 4 model for multi-frame spatial reasoning: telling a clean latch clearance from a fragile scrape or a pinned bolt. It runs off the critical path (90 s timeout, at most 20 reviews an hour). Thinking is `low` for speed; set `COPILOT_REVIEW_THINKING=high` for harder cases. Paid tier only. The introductory price ends 30 Nov 2026. |
| Speech | `gemini-3.8-flash-lite-tts`, voice `Kore` | The low-latency Gemini TTS tier, on the same key and client. It returns 24 kHz, 16-bit mono PCM, which the client wraps in a WAV. If speech takes more than 1.5 s, `T-212` falls back to the system voice. |
| Offline backup (`T-214`) | `gemma4:e4b` | The largest multimodal Gemma that runs on this laptop's CPU and Iris Xe next to the simulation. Used for call-outs only, never for reviews. |

Speech uses a dedicated TTS model rather than the text models.

Other models were considered and rejected for `T-210`:
- **DeepSeek Flash and ElevenLabs** each need a second vendor key and a second client.

`T-214`'s OpenAI-compatible client can point at other providers later without code changes.

## Gemini client

`teleop.copilot.gemini.GeminiClient` uses only the standard library, through an injectable
`Transport`, so tests run offline:

- `list_models(timeout_s)` lists every model the key can see, following `nextPageToken`.
- `generate_json(model, system, parts, schema, timeout_s, *, thinking_level=None)` returns a
  `dict`.
  - **Parts** are `str`, `TextPart` or `ImagePart`. An image is sent as base64 `inlineData`.
  - **Schema:** a JSON Schema (lower-case `"type": "object"`) is sent as `responseJsonSchema`. A
    Gemini OpenAPI schema (upper-case `"OBJECT"`) is sent as `responseSchema`.
  - **Answer:** thought parts and Markdown fences are stripped, and the result must be one JSON
    object.
- `synthesize(text, model, voice, timeout_s)` returns a WAV file. It requests
  `responseModalities: ["AUDIO"]` and a prebuilt voice.
- **Errors:**
  - `GeminiHttpError` (with `status_code`), `GeminiTimeoutError` and `GeminiMalformedJsonError`.
  - All of them subclass `GeminiError`, and none contains the key.
  - A refusal from the model's safety filter raises `GeminiError` too.

### Notes about the API

- **Interactions API.** Google's newest Gemini 4 developer pages document the newer Interactions
  API (`/v1beta/interactions`). This client uses the long-standing `generateContent` endpoint
  that the plan specifies.
  - The probe confirmed that Gemini 4 Hydrogen and Argon, the 3.8 models and the TTS models all
    accept `generateContent`, `responseJsonSchema` and the upper-case thinking levels on an AI
    Studio key.
- **Timeout scope.** The urllib timeout applies to each socket operation, not to the whole
  request. A hard wall-clock limit for call-outs is `T-212`'s job.

## Probe results

Measured on 2026-11-14 from this laptop, with a Gemini API key from AI Studio. Rerun with:

```bash
uv run python -m teleop.copilot probe
uv run python -m teleop.copilot probe --no-images
```

### Summary

- **The key and endpoint work.** It can see and call all six candidate models through
  `generateContent`, with `responseJsonSchema` and the upper-case thinking levels. Every role
  answered with valid JSON or audio.
- **Latency varies a lot between calls.** The table counts every call with images on: three probe
  runs plus a five-round alternating benchmark of the two fast models.

| Role | Model | Calls | Results (ms) | Median |
|---|---|---|---|---|
| Call-out | `gemini-4-hydrogen`, thinking `low` | 7 | 3826, timeout (> 8 s), 5418, 5769, 4114, 3377, 2904 | 3970 (6 answered) |
| Call-out | `gemini-3.8-flash`, thinking `low` | 7 | 6955, 3000, 3715, 4922, 3320, 3341, 2885 | 3341 |
| Review | `gemini-4-argon`, thinking `high`, 3 frames | 2 | 23977, 84445 | — |
| Review | `gemini-4-argon`, thinking `low`, 3 frames | 5 | 12423, 39679, 12621, 17834, 38445 | 17834 |
| Speech | `gemini-3.8-flash-lite-tts`, Kore | 2 | 3914 (3.2 s of audio), 4271 (4.1 s of audio) | — |

- **Neither fast model meets the 3 s call-out target reliably.**
  - Hydrogen stays the default because it is the preferred model. Its median, 4.0 s, is about
    0.6 s slower than Flash's, and it timed out once in seven calls.
  - `COPILOT_FAST_MODEL=gemini-3.8-flash` is a measured, drop-in alternative.
  - `T-212`'s 1 s template call-out covers either one.
- **Speech sits right at the 4 s fallback cut-off.** Expect the system voice to take over often
  unless `T-212` streams the audio or relaxes the limit.
- **Argon can take most of the 90 s review budget.** One review took 84 s. Keep reviews off the
  critical path, as planned.
- With synthetic images, Argon correctly answered `insufficient_evidence`, which is the
  behaviour wanted.

### Thinking levels

Every text model now runs at thinking level `low`:

- **The review moved from `high` to `low`.** Over five calls, Argon took 12–40 s (median 17.8 s),
  against 24 s and 84 s at `high`. On these synthetic frames the verdict was the same
  (`insufficient_evidence`) at both levels.
- **The call-outs were already at `low`,** which is the lowest level Gemini 4 accepts
  (`minimal` is HTTP 400).
- **The TTS models have no effort setting.** A request with any thinking level gets HTTP 400:
  "Thinking level is not supported for this model".

In the two probe runs at `low` (images on), the calls were slower than earlier the same day:

| Model | Earlier | At `low` |
|---|---|---|
| Hydrogen call-out | 2.9–5.8 s | 5.7 s, 5.7 s |
| Flash call-out | 2.9–7.0 s | 7.7 s, 9.8 s |
| Speech | 3.4–4.3 s | 12.6 s, 9.3 s |

Speech does no thinking at all, so much of this delay is load on Google's side, not effort.
Plan the call-out and speech paths (`T-212`) around the 1 s template and the system-voice
fallback, not around the model's speed.

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

The copilot watches the console as one more read-only viewer. It never sends anything.

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

## Live call-outs and speech (T-212)

```bash
uv run teleop atc --source synthetic                     # terminal 1; press F to jam Bay 1
uv run python -m teleop.copilot run                      # terminal 2
uv run python -m teleop.copilot run --mute --no-images   # print and log only, text-only requests
```

Flags: `--console URL` (default `COPILOT_CONSOLE_URL`), `--no-images`, `--mute`, `--env-file PATH`
and `--log-dir DIR` (default `runs/copilot`, gitignored). Without `GEMINI_API_KEY` the command
still runs: every call-out is the template, spoken with the system voice.

### Call-outs (`teleop.copilot.callouts`, `teleop.copilot.prompts`)

- **Model events:** `stall`, `refused`, `gate_result` and `autonomous_finished` ask the fast model
  (`COPILOT_FAST_MODEL`, thinking `low`) with the bay's telemetry slice and its latest frame,
  downscaled to at most 512 px (no frame with `--no-images`). The answer follows
  `{"say", "detail", "severity"}`; `say` is clipped to 15 words and `detail` to 40.
- **Templated events:** `rescue_started`, `rescue_released`, `validating` (its deep review is
  `T-213`) and `autonomous_started` are fully described by telemetry, so they never call a model.
- **The 1 s template.** If the model has not answered within 1 s, the deterministic template
  (for example "Bay 1 stalled on a flap jam. Rescue needed.") is spoken at once. If the model's
  line then arrives while the template is still waiting in the speech queue, it replaces it there;
  if the template is already being spoken, the model's line is only printed and logged
  (`(update)`). Model errors and malformed answers speak the template straight away.
- **8 s hard timeout** (wall clock, on top of the client's per-socket timeout): a later answer is
  dropped.
- **One request per bay.** A newer event for a bay drops the older event's pending answer and
  cancels that bay's lines that are still waiting to be spoken. While the older request is still
  running, the newer event gets the template.

### Speech (`teleop.copilot.speech`)

- **Queue:** at most 2 lines wait. Alarms jump ahead of other lines; when the queue is full the
  oldest non-alarm line is dropped (an info line never pushes out an alarm). Lines older than
  10 s are dropped. Only `say` is spoken, never `detail`.
- **Gemini TTS** (`COPILOT_TTS_MODEL`) returns a WAV, played with the first available of
  `paplay`, `aplay`, `afplay` or PowerShell `System.Media.SoundPlayer`.
- **Fallback:** if TTS fails or takes longer than 1.5 s, the same line is spoken with the system
  voice (`System.Speech` on Windows, `say` on macOS, `spd-say`/`espeak` on Linux). The late WAV is
  never played, so lines are never out of order, but it is cached: template lines repeat, so the
  next "Clutch engaged on bay 1." plays the Gemini voice at once. The cache holds 32 lines.
- **Voice:** `Kore`, a clear, firm and neutral prebuilt voice that stays intelligible over a noisy
  booth. `Charon` is a calmer alternative (`COPILOT_VOICE=Charon`).
- **`--mute`** speaks nothing; call-outs are still printed and logged.

### Outputs

```text
[12:01:07] BAY 1 ALARM     Bay 1 stalled on a flap jam. Rescue needed.
           NOMINAL -> RESCUE, C=0.24, bolt 3.4 of 5.2 cm, fault flap  [template; model slower than 1 s]
```

`runs/copilot/<date>.jsonl` gets one `event` record per event (bay, states, frame hash) and one
`callout` record per call-out (source, model, latency, severity, text, frame hash). It never holds
an image or the API key.

### Demo check (2026-11-14)

`teleop atc --source synthetic --script` and `python -m teleop.copilot run --mute`, run together on
this laptop without a Gemini key (template path only). All eight scripted events produced their
call-out 2–3 ms after the event record, in stage order: the Bay 1 stall, the clutch engage and
release, validating, the gate (20 of 20), Bay 2's autonomous start and finish, and the Bay 2
refusal. The JSONL log held only frame hashes.

### Live run with a key (2026-11-14, scripted stage, images on)

| Event | Template spoken | Hydrogen answer | Outcome |
|---|---|---|---|
| Bay 1 stall | at the event (model > 1 s) | 5.5 s | printed as an update |
| Bay 1 gate result | at the event | 4.2 s | printed as an update |
| Bay 2 autonomous finished | at the event | 3.8 s | replaced the queued template |
| Bay 2 refused | at the event | 6.2 s | printed as an update |

- **Hydrogen never beat 1 s,** so in practice the operator hears the template and reads the model's
  line. It answered every time, within the 8 s limit.
- **Gemini TTS missed the 4 s cut-off on every line** (5 of 5), so the system voice spoke them all.
  Because each fallback first waits the full 4 s, spoken lines trailed their events by 4–9 s.
  The cut-off is now **1.5 s** (`TTS_DEADLINE_S`), so the system voice starts about 1.5 s after a
  line is dequeued. Repeated lines still get the Gemini voice once their late audio is cached.
- **A clutch flip about 1 s after a stall** used to cancel the stall's pending model request. Now
  templated events let a pending model answer finish; only a newer model event supersedes it.
- `runs/copilot/*.jsonl` now also has a `speech` record per line (`route`: `tts`, `system`,
  `dropped_stale` or `muted`, plus the TTS time), so the TTS hit rate can be measured.
