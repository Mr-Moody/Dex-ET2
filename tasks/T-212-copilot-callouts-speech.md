---
id: T-212
title: Tier 1 SSE Gemini TTS voice and Tier 2 inline Hydrogen grasp gate
status: done
owner: null
branch: null
depends_on: [T-211]
plan: docs/superpowers/plans/2026-11-14-operator-copilot-plan.md#t-212-live-call-outs-and-speech
claimed_at: null
---

Implements Tier 1 instant voice (`< 0.7 s` warm target, zero VLM wait) and Tier 2 active inline grasp gating (`~3.3-5.5 s`):

- **Tier 1 (`tts_stream.py`, `speech.py`, `callouts.py`)**: Streams `gemini-3.8-flash-lite-tts` (voice `Kore`, `responseModalities: ["AUDIO"]`, no `thinkingConfig`) via `POST /v1beta/models/gemini-3.8-flash-lite-tts:streamGenerateContent?alt=sse` over persistent IPv4 `HTTPSConnection`, piping 24 kHz 16-bit mono s16le PCM directly into `paplay --raw` / `aplay` (WAV fallback `pcm_to_wav`), falling back to `espeak-ng` / system voice if chunk 1 exceeds `~1.8 s` (`1.5-2.0 s`), and pre-caching deterministic telemetry cues at startup (`precache_cues`).
- **Tier 2 (`gate.py`, `atc/coordinator.py`, `commands/atc.py`)**: On clutch release (`bolt >= 5.2 cm`), `AtcCoordinator._Gate` runs `GraspGate` (`gemini-4-hydrogen`, `thinkingLevel="LOW"`, `mediaResolution="MEDIA_RESOLUTION_LOW"`) on 3 `<= 512 px` rescue keyframes (`stall`, `mid-hold`, `bolt-clear`) + contact torque/bolt/HV terminal clearance/flap telemetry in parallel with `SkillValidator`. Both Sim and `GraspGate` must approve before `unoq.register_skill(skill)` deploys the DMP to the Fleet Library, and Hydrogen's dynamic `"say"` verdict is spoken immediately via Tier 1 SSE TTS. `--no-gemini`, `--headless`, or unset `GEMINI_API_KEY` fall back cleanly to `LocalModelClient` or 100% deterministic physics-only (`2386` golden ticks).
