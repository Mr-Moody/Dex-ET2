# ADR 0001: Studio UDP primary, Inspire default hand

Status: accepted (2026-11-13). Source: spec section 9, deviations from the kickoff brief.

## Context

The kickoff brief assumed the Rokoko SDK as the main input, a LEAP/Shadow hand, and left the
wrist source open. Investigation found otherwise (spec section 3).

## Decisions

- Studio Custom Streaming (UDP 14043) is the primary input; the SDK is secondary. The SDK driver
  streams sensor orientations only, so joints need the hand solver.
- The default hand is Inspire (not in Menagerie), with LEAP as the fallback.
- The wrist source is the Coil Pro.
- Added: hotspot management, booth app, processing/QC stage and annotator hook.
- Removed for v1: Diffusion Policy, mink/arm, the per-PR reviewer agent.

## Consequences

Inspire needs its own model and hand YAML; the Studio parser is on the critical path.
