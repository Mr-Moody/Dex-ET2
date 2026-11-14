"""Solana compliance sink: everything runs offline against an in-memory fake RPC."""

from __future__ import annotations

import base64
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from solders.hash import Hash
from solders.keypair import Keypair
from solders.transaction import Transaction

from teleop.compliance import solana as sol
from teleop.compliance.protocol import ComplianceEvent, ComplianceSink
from teleop.compliance.solana import (
    MEMO_PROGRAM_ID,
    AuditEntry,
    RpcError,
    SolanaCompliancePlaceholder,
    SolanaComplianceSink,
    build_memo,
    canonical_json,
    digest_record,
    explorer_url,
    load_keypair,
    memo_digest,
    read_audit_log,
    read_receipts,
    save_keypair,
    to_json_value,
    verify_audit_log,
)

DEVNET = "https://api.devnet.solana.com"


class FakeSolana:
    """In-memory Solana RPC: decodes and signature-checks real transactions, stores memos."""

    def __init__(self) -> None:
        self.offline = False
        self.send_delay_s = 0.0
        self.txs: dict[str, Transaction] = {}
        self.tampered: dict[str, str] = {}
        self.calls: list[str] = []
        self.slot = 1000
        self._lock = threading.Lock()

    def call(self, method: str, params: list[Any]) -> Any:
        with self._lock:
            self.calls.append(method)
        if self.offline:
            raise RpcError(f"{method}: connection refused")
        if method == "getLatestBlockhash":
            value = {"blockhash": str(Hash.new_unique()), "lastValidBlockHeight": 2000}
            return {"context": {"slot": self.slot}, "value": value}
        if method == "sendTransaction":
            time.sleep(self.send_delay_s)
            assert params[1]["encoding"] == "base64"
            tx = Transaction.from_bytes(base64.b64decode(params[0]))
            tx.verify()  # raises if the signature does not match the message
            sig = str(tx.signatures[0])
            with self._lock:
                self.slot += 1
                self.txs[sig] = tx
            return sig
        if method == "getSignatureStatuses":
            out: list[Any] = []
            for sig in params[0]:
                ok = sig in self.txs
                status = {"slot": self.slot, "err": None, "confirmationStatus": "confirmed"}
                out.append(status if ok else None)
            return {"context": {"slot": self.slot}, "value": out}
        if method == "getTransaction":
            tx = self.txs.get(params[0])
            if tx is None:
                return None
            msg = tx.message
            keys = msg.account_keys
            n_signers = msg.header.num_required_signatures
            ix = msg.instructions[0]
            memo = self.tampered.get(params[0], bytes(ix.data).decode("utf-8"))
            return {
                "slot": self.slot,
                "blockTime": 1_763_000_000,
                "transaction": {
                    "message": {
                        "accountKeys": [
                            {"pubkey": str(k), "signer": i < n_signers, "writable": i == 0}
                            for i, k in enumerate(keys)
                        ],
                        "instructions": [
                            {
                                "programId": str(keys[ix.program_id_index]),
                                "program": "spl-memo",
                                "parsed": memo,
                            }
                        ],
                    }
                },
                "meta": {"err": None},
            }
        if method == "getBalance":
            return {"context": {"slot": self.slot}, "value": 2 * sol.LAMPORTS_PER_SOL}
        raise AssertionError(f"unexpected RPC method {method}")


def make_event(
    kind: str = "STALL_ALERT", bay: int = 1, summary: str = "flap jam", **metadata: Any
) -> ComplianceEvent:
    meta: dict[str, Any] = {"confidence": 0.21, "x_bolt_m": 0.035, **metadata}
    return ComplianceEvent(
        kind=kind,  # type: ignore[arg-type]
        bay_id=bay,
        part_sku_id="EV-MOD-7",
        timestamp_ns=123_456_789,
        payload_sha256="ab" * 32,
        summary=summary,
        metadata=meta,
    )


def solana_sink(
    tmp_path: Path, rpc: FakeSolana, keypair: Keypair, **kw: Any
) -> SolanaComplianceSink:
    return SolanaComplianceSink(
        keypair=keypair,
        transport=rpc,
        rpc_url=DEVNET,
        audit_path=tmp_path / "audit.jsonl",
        anchors_path=tmp_path / "anchors.jsonl",
        poll_interval_s=0.001,
        **kw,
    )


@pytest.fixture
def keypair() -> Keypair:
    return Keypair()


# ── hashing ──


def test_canonical_json_is_key_order_independent_and_normalises_numpy_and_nan() -> None:
    a = to_json_value({"b": np.float32(1.5), "a": [np.int64(3), np.array([1.0, 2.0])]})
    b = to_json_value({"a": (3, [1.0, 2.0]), "b": 1.5})
    assert canonical_json(a) == canonical_json(b)
    assert to_json_value({"x": float("nan"), "y": float("inf")}) == {"x": "nan", "y": "inf"}
    assert to_json_value({"s": {3, 1, 2}}) == {"s": [1, 2, 3]}


def test_memo_roundtrip_and_explorer_clusters() -> None:
    entry = AuditEntry(seq=7, digest="cd" * 32, record={"kind": "OOD_REFUSAL", "bay_id": 2})
    memo = build_memo(entry)
    assert memo_digest(memo) == "cd" * 32
    assert "OOD_REFUSAL bay=2 seq=7" in memo and len(memo.encode()) < 200
    assert memo_digest("something else entirely") is None
    assert explorer_url("SIG", DEVNET).endswith("/tx/SIG?cluster=devnet")
    assert explorer_url("SIG", "https://api.mainnet-beta.solana.com").endswith("/tx/SIG")
    assert "customUrl=http%3A%2F%2F127.0.0.1%3A8899" in explorer_url("SIG", "http://127.0.0.1:8899")


# ── protocol + modes ──


def test_sinks_satisfy_compliance_sink_protocol(tmp_path: Path) -> None:
    placeholder = SolanaCompliancePlaceholder(capacity=4)
    sink = SolanaComplianceSink(audit_path=tmp_path / "a.jsonl", anchors_path=None)
    try:
        assert isinstance(placeholder, ComplianceSink)
        assert isinstance(sink, ComplianceSink)
    finally:
        placeholder.close()
        sink.close()


def test_from_env_picks_mode_and_never_touches_network_at_init(tmp_path: Path) -> None:
    kp_path = tmp_path / "station.json"
    save_keypair(Keypair(), kp_path)
    rpc = FakeSolana()
    cases = {
        "local": {},
        "local-signed": {"SOLANA_KEYPAIR_PATH": str(kp_path)},
        "solana": {"SOLANA_KEYPAIR_PATH": str(kp_path), "SOLANA_RPC_URL": DEVNET},
    }
    for expected, env in cases.items():
        sink = SolanaComplianceSink.from_env(
            env, audit_path=tmp_path / f"{expected}.jsonl", anchors_path=None, transport=rpc
        )
        assert sink.mode == expected
        sink.close()
    assert rpc.calls == []
    # an unreadable keypair degrades to an unsigned local log instead of crashing
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    sink = SolanaComplianceSink.from_env(
        {"SOLANA_KEYPAIR_PATH": str(bad), "SOLANA_RPC_URL": DEVNET}, audit_path=None
    )
    assert sink.mode == "local"
    sink.close()


def test_keypair_file_roundtrip_is_owner_only_and_not_overwritten(tmp_path: Path) -> None:
    kp = Keypair()
    path = tmp_path / "k" / "station.json"
    save_keypair(kp, path)
    assert load_keypair(path).pubkey() == kp.pubkey()
    assert path.stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        save_keypair(Keypair(), path)


# ── local logging ──


def test_local_mode_writes_a_verifiable_hash_chain(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    sink = SolanaComplianceSink(audit_path=audit, anchors_path=None)
    for kind in ("STALL_ALERT", "RESCUE_LOGGED", "VALIDATION_GATE", "SKILL_DEPLOYED"):
        sink.emit(make_event(kind))
    sink.close()
    entries = read_audit_log(audit)
    assert [e.seq for e in entries] == [1, 2, 3, 4]
    assert entries[0].record["prev_digest"] == sol.GENESIS_DIGEST
    assert all(
        b.record["prev_digest"] == a.digest for a, b in zip(entries, entries[1:], strict=False)
    )
    assert all(e.signature is None for e in entries)
    results = verify_audit_log(audit)
    assert all(r.ok for r in results) and all(r.onchain_ok is None for r in results)
    assert [s.state for s in sink.recent_status()] == ["local"] * 4


def test_signed_local_fallback_when_rpc_unset(tmp_path: Path, keypair: Keypair) -> None:
    audit = tmp_path / "audit.jsonl"
    sink = SolanaComplianceSink(keypair=keypair, audit_path=audit, anchors_path=None)
    sink.emit(make_event())
    sink.close()
    (entry,) = read_audit_log(audit)
    assert entry.signer == str(keypair.pubkey()) and entry.signature is not None
    (result,) = verify_audit_log(audit)
    assert result.signature_ok is True and result.ok


def test_restart_continues_the_chain(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    for _ in range(2):
        sink = SolanaComplianceSink(audit_path=audit, anchors_path=None)
        sink.emit(make_event())
        sink.close()
    assert [e.seq for e in read_audit_log(audit)] == [1, 2]
    assert all(r.ok for r in verify_audit_log(audit))


def test_torn_last_line_is_skipped_and_repaired(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    sink = SolanaComplianceSink(audit_path=audit, anchors_path=None)
    sink.emit(make_event())
    sink.close()
    with audit.open("a", encoding="utf-8") as fh:
        fh.write('{"digest": "trunc')  # simulated crash mid-write
    sink = SolanaComplianceSink(audit_path=audit, anchors_path=None)
    sink.emit(make_event())
    sink.close()
    assert [e.seq for e in read_audit_log(audit)] == [1, 2]
    assert all(r.ok for r in verify_audit_log(audit))


def test_emit_never_raises_and_ignores_events_after_close(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    sink = SolanaComplianceSink(audit_path=audit, anchors_path=None)
    sink.emit(make_event(blob=object(), arr=np.zeros(3)))  # unserialisable -> repr fallback
    sink.emit("not an event")  # type: ignore[arg-type]
    sink.close()
    sink.emit(make_event())
    sink.close()  # idempotent
    entries = read_audit_log(audit)
    assert len(entries) == 1
    meta = entries[0].record["metadata"]
    assert isinstance(meta, dict) and meta["arr"] == [0.0, 0.0, 0.0]


def test_real_sink_matches_placeholder_hud_api(tmp_path: Path) -> None:
    """events()/latest() behave the same on both sinks, so the HUD can take either."""
    real = SolanaComplianceSink(audit_path=tmp_path / "a.jsonl", anchors_path=None, capacity=3)
    placeholder = SolanaCompliancePlaceholder(capacity=3)
    events = [make_event(bay=bay) for bay in range(1, 6)]
    for sink in (real, placeholder):
        for event in events:
            sink.emit(event)
        sink.close()
        assert [e.bay_id for e in sink.events()] == [3, 4, 5]
        assert sink.latest() == events[-1]
    assert [s.bay_id for s in real.recent_status(10)] == [3, 4, 5]
    status = real.latest_status()
    assert status is not None and status.seq == 5


# ── real-time guarantee ──


def test_emit_stays_under_half_a_millisecond_while_rpc_is_slow(
    tmp_path: Path, keypair: Keypair
) -> None:
    rpc = FakeSolana()
    rpc.send_delay_s = 0.5  # every anchor blocks the anchor thread for 500 ms
    sink = solana_sink(tmp_path, rpc, keypair, close_grace_s=0.1)
    n = 2000
    durations: list[int] = []
    for i in range(n):
        event = make_event(bay=1 + i % 4)
        t0 = time.perf_counter_ns()
        sink.emit(event)
        durations.append(time.perf_counter_ns() - t0)
    sink.close()
    mean_ms = sum(durations) / n / 1e6
    assert mean_ms < 0.5, f"mean emit {mean_ms:.4f} ms"
    assert float(np.percentile(durations, 99)) / 1e6 < 0.5


# ── anchoring ──


def test_anchors_each_digest_in_a_signed_memo_transaction(tmp_path: Path, keypair: Keypair) -> None:
    rpc = FakeSolana()
    sink = solana_sink(tmp_path, rpc, keypair)
    sink.emit(make_event("STALL_ALERT", bay=1))
    sink.emit(make_event("OOD_REFUSAL", bay=2))
    sink.close()
    entries = read_audit_log(tmp_path / "audit.jsonl")
    receipts = read_receipts(tmp_path / "anchors.jsonl")
    assert set(receipts) == {e.digest for e in entries}
    for entry in entries:
        tx = rpc.txs[receipts[entry.digest].tx_signature]
        assert tx.message.account_keys[0] == keypair.pubkey()  # station pays and signs
        ix = tx.message.instructions[0]
        assert tx.message.account_keys[ix.program_id_index] == MEMO_PROGRAM_ID
        assert memo_digest(bytes(ix.data).decode()) == entry.digest
    assert [s.state for s in sink.recent_status()] == ["anchored", "anchored"]
    latest = sink.latest_status()
    assert latest is not None and latest.explorer_url is not None
    assert latest.explorer_url.endswith("?cluster=devnet")
    anchor = sink.anchor
    assert anchor is not None
    results = verify_audit_log(tmp_path / "audit.jsonl", tmp_path / "anchors.jsonl", anchor)
    assert all(r.ok and r.onchain_ok is True for r in results)


def test_rpc_offline_keeps_local_log_then_backfills_on_restart(
    tmp_path: Path, keypair: Keypair
) -> None:
    rpc = FakeSolana()
    rpc.offline = True
    sink = solana_sink(tmp_path, rpc, keypair, close_grace_s=0.2)
    sink.emit(make_event())
    sink.emit(make_event(bay=2))
    sink.close()
    assert len(read_audit_log(tmp_path / "audit.jsonl")) == 2  # local trail is complete
    assert read_receipts(tmp_path / "anchors.jsonl") == {}
    assert sink.counts()["errors"] >= 1

    rpc.offline = False
    sink = solana_sink(tmp_path, rpc, keypair)
    sink.close()
    assert len(read_receipts(tmp_path / "anchors.jsonl")) == 2
    assert all(r.ok for r in verify_audit_log(tmp_path / "audit.jsonl", tmp_path / "anchors.jsonl"))


# ── tamper detection ──


def test_editing_a_record_breaks_its_digest_and_signature(tmp_path: Path, keypair: Keypair) -> None:
    rpc = FakeSolana()
    sink = solana_sink(tmp_path, rpc, keypair)
    sink.emit(make_event(summary="flap jam"))
    sink.emit(make_event(summary="second"))
    sink.close()
    audit = tmp_path / "audit.jsonl"
    audit.write_text(audit.read_text().replace("flap jam", "all fine"), encoding="utf-8")
    anchor = sink.anchor
    assert anchor is not None
    first, second = verify_audit_log(audit, tmp_path / "anchors.jsonl", anchor)
    assert not first.ok
    assert first.digest_ok is False and first.signature_ok is False and first.onchain_ok is False
    assert second.ok


def test_deleting_a_record_breaks_the_chain(tmp_path: Path) -> None:
    audit = tmp_path / "audit.jsonl"
    sink = SolanaComplianceSink(audit_path=audit, anchors_path=None)
    for bay in (1, 2, 3):
        sink.emit(make_event(bay=bay))
    sink.close()
    lines = audit.read_text().splitlines()
    audit.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    first, third = verify_audit_log(audit)
    assert first.ok and not third.ok and third.chain_ok is False


def test_digest_recomputed_with_forged_digest_field_still_fails_on_chain(
    tmp_path: Path, keypair: Keypair
) -> None:
    rpc = FakeSolana()
    sink = solana_sink(tmp_path, rpc, keypair)
    sink.emit(make_event())
    sink.close()
    audit = tmp_path / "audit.jsonl"
    doc = json.loads(audit.read_text())
    doc["record"]["summary"] = "rewritten history"
    doc["digest"] = digest_record(doc["record"])  # attacker recomputes the local digest
    audit.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    anchor = sink.anchor
    assert anchor is not None
    (result,) = verify_audit_log(audit, tmp_path / "anchors.jsonl", anchor)
    assert result.digest_ok is True  # locally self-consistent…
    assert result.signature_ok is False  # …but the station key never signed it
    assert result.onchain_ok is None  # receipt is keyed by the original digest
    assert not result.ok


def test_onchain_memo_mismatch_is_reported(tmp_path: Path, keypair: Keypair) -> None:
    rpc = FakeSolana()
    sink = solana_sink(tmp_path, rpc, keypair)
    sink.emit(make_event())
    sink.close()
    (receipt,) = read_receipts(tmp_path / "anchors.jsonl").values()
    rpc.tampered[receipt.tx_signature] = f"{sol.MEMO_PREFIX} {'ee' * 32} STALL_ALERT bay=1 seq=1"
    anchor = sink.anchor
    assert anchor is not None
    (result,) = verify_audit_log(tmp_path / "audit.jsonl", tmp_path / "anchors.jsonl", anchor)
    assert result.onchain_ok is False and "on-chain memo" in result.problem


# ── CLI ──


def test_cli_keygen_demo_and_verify_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)  # keep load_dotenv away from the repo's .env
    monkeypatch.delenv("SOLANA_RPC_URL", raising=False)
    kp_path = tmp_path / "station.json"
    monkeypatch.setenv("SOLANA_KEYPAIR_PATH", str(kp_path))
    logs = ["--audit-log", str(tmp_path / "a.jsonl"), "--anchors-log", str(tmp_path / "b.jsonl")]
    assert sol.main(["keygen"]) == 0
    assert sol.main(["keygen"]) == 1  # refuses to overwrite
    assert sol.main([*logs, "demo", "--bay", "3", "--kind", "OOD_REFUSAL"]) == 0
    assert sol.main([*logs, "verify", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "mode: local-signed" in out and "1/1 entries verified - all OK" in out
    audit = tmp_path / "a.jsonl"
    audit.write_text(audit.read_text().replace("DEMO-SKU", "OTHER"), encoding="utf-8")
    assert sol.main([*logs, "verify", "--offline"]) == 1
    assert sol.main(["airdrop"]) == 1  # no RPC configured
