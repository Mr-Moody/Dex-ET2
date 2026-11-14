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
| `COPILOT_FAST_MODEL` | `gemini-4-hydrogen` | Live call-outs, at thinking level `low` |
| `COPILOT_REVIEW_MODEL` | `gemini-4-argon` | Deep reviews, at thinking level `high` |
| `COPILOT_TTS_MODEL` | `gemini-3.8-flash-lite-tts` | Speech |
| `COPILOT_VOICE` | `Kore` | A prebuilt Gemini TTS voice (`Charon` is an alternative) |
| `COPILOT_LOCAL_URL` | (none) | OpenAI-compatible local server for the Gemma fallback (`T-214`) |
| `COPILOT_LOCAL_MODEL` | `gemma4:e4b` | Local fallback model (`T-214`) |
| `COPILOT_CONSOLE_URL` | `ws://127.0.0.1:8765/ws` | The console stream to watch (`T-211`) |
| `COPILOT_MAX_REVIEWS_PER_HOUR` | `20` | Review budget (`T-213`) |

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
| Deep review: telemetry plus up to 6 JPEGs → JSON | `gemini-4-argon`, thinking `high` | The strongest Gemini 4 model for multi-frame spatial reasoning: telling a clean latch clearance from a fragile scrape or a pinned bolt. It runs off the critical path (90 s timeout, at most 20 reviews an hour), so the extra thinking costs nothing live. Paid tier only. The introductory price ends 30 Nov 2026. |
| Speech | `gemini-3.8-flash-lite-tts`, voice `Kore` | The low-latency Gemini TTS tier, on the same key and client. It returns 24 kHz, 16-bit mono PCM, which the client wraps in a WAV. If speech takes more than 4 s, `T-212` falls back to the system voice. |
| Offline backup (`T-214`) | `gemma4:e4b` | The largest multimodal Gemma that runs on this laptop's CPU and Iris Xe next to the simulation. Used for call-outs only, never for reviews. |

Speech uses a dedicated TTS model rather than the text models.

Other models were considered and rejected for `T-210`:
- **DeepSeek Flash and ElevenLabs** each need a second vendor key and a second client.
- **Kimi** already backs the fleet reasoner (`teleop.reasoning`). Using another model family here
  gives an independent second opinion.

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

### Caveats about the API (the probe settles these)

- **Interactions API.** Google's newest Gemini 4 developer pages document the newer Interactions
  API (`/v1beta/interactions`). This client uses the long-standing `generateContent` endpoint
  that the plan specifies.
  - Run against Google with an invalid key, the endpoint and header auth answer correctly:
    `HTTP 400: INVALID_ARGUMENT: API key not valid`.
  - Whether a real key's Gemini 4 calls accept `generateContent`, `responseJsonSchema` and the
    upper-case thinking levels is what the probe reports.
- **Timeout scope.** The urllib timeout applies to each socket operation, not to the whole
  request. A hard wall-clock limit for call-outs is `T-212`'s job.

## Probe results

> **Pending.** No `GEMINI_API_KEY` was set when `T-210` was implemented, so the live probe has not
> run yet. Add the key, run the two commands below, and paste their Markdown output here.
> Then update the defaults in `src/teleop/copilot/config.py` if the probe recommends different
> models.

```bash
uv run python -m teleop.copilot probe
uv run python -m teleop.copilot probe --no-images
```

The report has three parts:
1. Which of the candidate models the key can see and call with `generateContent`.
2. One timed call per role, with each model's JSON answer, the WAV size and the audio length.
   Times come from `time.monotonic_ns()`.
3. The `.env` values it recommends.

The targets are a call-out in ≤ 3 s (8 s hard timeout) and speech in ≤ 4 s.
