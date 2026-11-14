---
id: T-160
title: Solana compliance & alert chain integration
status: todo
owner: null
branch: null
depends_on: [T-100]
plan: docs/superpowers/plans/2026-11-14-atc-implementation-plan.md#task-t-160-solana-compliance--alert-chain-integration-teammate-task
claimed_at: null
---

**Reserved for Solana teammate. Implement inside `src/teleop/compliance/solana.py` and `tests/compliance/test_solana.py` without changing `ComplianceSink` / `ComplianceEvent` signatures in `protocol.py`.**

Acceptance (from the plan):

- [ ] Async background worker anchors SHA-256 event digests + metadata (`STALL_ALERT`, `RESCUE_LOGGED`, `VALIDATION_GATE`, `SKILL_DEPLOYED`, `OOD_REFUSAL`) to Solana (`SOLANA_RPC_URL`, `SOLANA_KEYPAIR_PATH`)
- [ ] `emit()` never blocks the 60 Hz control thread (`< 0.5 ms` queue put)
- [ ] Graceful local signed/JSONL audit fallback when RPC is offline or unset so CI and live stage demo never stall
- [ ] Unit tests in `tests/compliance/test_solana.py` pass without network access
