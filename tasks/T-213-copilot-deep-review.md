---
id: T-213
title: Tier 3 silent deep audit (Gemini 4 Argon) and ACT dataset curation tagging
status: todo
owner: null
branch: null
depends_on: [T-212]
plan: docs/superpowers/plans/2026-11-14-operator-copilot-plan.md#t-213-deep-review-of-rescues-and-refusals-gemini-4-argon
claimed_at: null
---

Fire-and-forget Tier 3 background engineering audit (`AuditReviewer`, `gemini-4-argon`, `thinkingLevel="LOW"`, up to 6 `<= 512 px` JPEG rescue frames + bay rescue timeline, rate-limited by `COPILOT_MAX_REVIEWS_PER_HOUR`):

- **Strictly silent and non-blocking**: Invokes **zero TTS calls** (`Speaker` is never called) and never blocks `AtcCoordinator._Gate`, `unoq.register_skill`, or Bay 2 autonomous recovery.
- **UI + ACT dataset curation (`T-191`/`T-193`)**: Emits structured `AuditReport` (`verdict: "approve" | "flag"`, `dataset_tag: "approve" | "flag"`, `reasons`, `operator_note`, `confidence`) to the Fleet Ops Console `#audit-panel` card and asynchronously tags `<root>/meta/teleop_episodes.jsonl` via `EpisodeSink.tag_episode(...)` so `teleop.process.atc_qc` drops `"flag"` episodes (`"copilot flagged"`).

Owned files: `src/teleop/copilot/review.py`, `src/teleop/copilot/prompts.py`, `src/teleop/atc/episodes.py`, `src/teleop/process/atc_qc.py`, `tests/copilot/test_review.py`, `tests/atc/test_episodes.py`, `tests/process/test_atc_qc.py`.
