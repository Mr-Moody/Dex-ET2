"""Solana compliance sink: canonical hashing, Ed25519 hash-chain, SPL Memo, and tamper checks."""

from __future__ import annotations

import base64
import json
import sys
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
    """In-memory Solana RPC that verifies Ed25519 transaction signatures and stores SPL memos."""

    def __init__(self) -> None:
        self.offline = False
        self.send_delay_s = 0.0
        self.txs: dict[str, Transaction] = {}
        self.tampered: dict[str, str] = {}
        self.slot = 1000
        self._lock = threading.Lock()

    def call(self, method: str, params: list[Any]) -> Any:
        if self.offline:
            raise RpcError(f"{method}: connection refused")
        if method == "getLatestBlockhash":
            return {
                "context": {"slot": self.slot},
                "value": {"blockhash": str(Hash.new_unique()), "lastValidBlockHeight": 2000},
            }
        if method == "sendTransaction":
            if self.send_delay_s:
                time.sleep(self.send_delay_s)
            tx = Transaction.from_bytes(base64.b64decode(params[0]))
            tx.verify()
            sig = str(tx.signatures[0])
            with self._lock:
                self.slot += 1
                self.txs[sig] = tx
            return sig
        if method == "getSignatureStatuses":
            status = {"slot": self.slot, "err": None, "confirmationStatus": "confirmed"}
            return {
                "context": {"slot": self.slot},
                "value": [status if sig in self.txs else None for sig in params[0]],
            }
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
        raise AssertionError(f"unexpected RPC method {method}")


def _event(kind: str = "STALL_ALERT", bay: int = 1, summary: str = "flap jam") -> ComplianceEvent:
    return ComplianceEvent(
        kind=kind,  # type: ignore[arg-type]
        bay_id=bay,
        part_sku_id="EV-MOD-7",
        timestamp_ns=123_456_789,
        payload_sha256="ab" * 32,
        summary=summary,
        metadata={"confidence": 0.21, "x_bolt_m": 0.035},
    )


def _sink(tmp_path: Path, rpc: FakeSolana, keypair: Keypair, **kw: Any) -> SolanaComplianceSink:
    return SolanaComplianceSink(
        keypair=keypair,
        transport=rpc,
        rpc_url=DEVNET,
        audit_path=tmp_path / "audit.jsonl",
        anchors_path=tmp_path / "anchors.jsonl",
        poll_interval_s=0.001,
        **kw,
    )


def test_canonical_json_memo_keypair_and_env_modes(tmp_path: Path) -> None:
    a = to_json_value({"b": np.float32(1.5), "a": [np.int64(3), np.array([1.0, 2.0])]})
    b = to_json_value({"a": (3, [1.0, 2.0]), "b": 1.5})
    assert canonical_json(a) == canonical_json(b)
    assert to_json_value({"x": float("nan"), "y": float("inf")}) == {"x": "nan", "y": "inf"}

    entry = AuditEntry(seq=7, digest="cd" * 32, record={"kind": "OOD_REFUSAL", "bay_id": 2})
    memo = build_memo(entry)
    assert memo_digest(memo) == "cd" * 32 and memo_digest("unrelated") is None
    assert explorer_url("SIG", DEVNET).endswith("/tx/SIG?cluster=devnet")

    kp = Keypair()
    kp_path = tmp_path / "k" / "station.json"
    save_keypair(kp, kp_path)
    assert load_keypair(kp_path).pubkey() == kp.pubkey()
    if sys.platform != "win32":
        assert kp_path.stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        save_keypair(Keypair(), kp_path)

    placeholder = SolanaCompliancePlaceholder(capacity=2)
    assert isinstance(placeholder, ComplianceSink)
    placeholder.close()

    rpc = FakeSolana()
    for mode, env in (
        ("local", {}),
        ("local-signed", {"SOLANA_KEYPAIR_PATH": str(kp_path)}),
        ("solana", {"SOLANA_KEYPAIR_PATH": str(kp_path), "SOLANA_RPC_URL": DEVNET}),
    ):
        s = SolanaComplianceSink.from_env(
            env, audit_path=tmp_path / f"{mode}.jsonl", anchors_path=None, transport=rpc
        )
        assert isinstance(s, ComplianceSink) and s.mode == mode
        s.close()


def test_hash_chain_signed_local_restart_repair_and_nonblocking_emit(tmp_path: Path) -> None:
    kp = Keypair()
    audit = tmp_path / "audit.jsonl"
    sink = SolanaComplianceSink(keypair=kp, audit_path=audit, anchors_path=None)
    sink.emit(_event("STALL_ALERT"))
    sink.emit(_event("RESCUE_LOGGED"))
    sink.close()

    # Simulate a torn trailing line from a crash, then continue the chain on restart.
    with audit.open("a", encoding="utf-8") as fh:
        fh.write('{"digest": "trunc')
    sink2 = SolanaComplianceSink(keypair=kp, audit_path=audit, anchors_path=None)
    sink2.emit(_event("SKILL_DEPLOYED"))
    sink2.close()

    entries = read_audit_log(audit)
    assert [e.seq for e in entries] == [1, 2, 3]
    assert entries[0].record["prev_digest"] == sol.GENESIS_DIGEST
    assert all(
        b.record["prev_digest"] == a.digest for a, b in zip(entries, entries[1:], strict=False)
    )
    results = verify_audit_log(audit)
    assert all(r.ok and r.signature_ok is True for r in results)

    # Real-time guarantee: emit stays < 0.5 ms p99 even when RPC blocks 300 ms per tx.
    rpc = FakeSolana()
    rpc.send_delay_s = 0.3
    rt_sink = _sink(tmp_path / "rt", rpc, kp, close_grace_s=0.05)
    durations_ns: list[int] = []
    for i in range(500):
        ev = _event(bay=1 + i % 4)
        t0 = time.perf_counter_ns()
        rt_sink.emit(ev)
        durations_ns.append(time.perf_counter_ns() - t0)
    rt_sink.close()
    assert float(np.percentile(durations_ns, 99)) / 1e6 < 0.5


def test_onchain_memo_anchoring_offline_backfill_and_tamper_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kp = Keypair()
    rpc = FakeSolana()
    rpc.offline = True

    # Offline: local trail is complete; restart when online backfills SPL Memo receipts.
    offline_dir = tmp_path / "backfill"
    s_off = _sink(offline_dir, rpc, kp, close_grace_s=0.1)
    s_off.emit(_event("STALL_ALERT", bay=1))
    s_off.emit(_event("OOD_REFUSAL", bay=2))
    s_off.close()
    assert len(read_audit_log(offline_dir / "audit.jsonl")) == 2
    assert read_receipts(offline_dir / "anchors.jsonl") == {}

    rpc.offline = False
    s_on = _sink(offline_dir, rpc, kp)
    s_on.close()
    entries = read_audit_log(offline_dir / "audit.jsonl")
    receipts = read_receipts(offline_dir / "anchors.jsonl")
    assert set(receipts) == {e.digest for e in entries}
    for entry in entries:
        tx = rpc.txs[receipts[entry.digest].tx_signature]
        ix = tx.message.instructions[0]
        assert tx.message.account_keys[ix.program_id_index] == MEMO_PROGRAM_ID
        assert memo_digest(bytes(ix.data).decode()) == entry.digest
    assert s_on.anchor is not None
    verified = verify_audit_log(
        offline_dir / "audit.jsonl", offline_dir / "anchors.jsonl", s_on.anchor
    )
    assert all(r.ok and r.onchain_ok is True for r in verified)

    # Tamper 1: recomputing local digest after editing summary still fails Ed25519 signature check.
    tamper_dir = tmp_path / "tamper"
    s_t = _sink(tamper_dir, rpc, kp)
    s_t.emit(_event(summary="flap jam"))
    s_t.emit(_event(summary="second"))
    s_t.close()
    audit_file = tamper_dir / "audit.jsonl"
    lines = audit_file.read_text(encoding="utf-8").splitlines()
    doc0 = json.loads(lines[0])
    doc0["record"]["summary"] = "forged"
    doc0["digest"] = digest_record(doc0["record"])
    audit_file.write_text(json.dumps(doc0) + "\n" + lines[1] + "\n", encoding="utf-8")
    assert s_t.anchor is not None
    r0, r1 = verify_audit_log(audit_file, tamper_dir / "anchors.jsonl", s_t.anchor)
    assert r0.digest_ok is True and r0.signature_ok is False and not r0.ok
    assert r1.chain_ok is False and not r1.ok

    # Offline CLI roundtrip: keygen -> demo -> verify --offline -> detect tamper.
    cli_dir = tmp_path / "cli"
    cli_dir.mkdir()
    monkeypatch.chdir(cli_dir)
    monkeypatch.delenv("SOLANA_RPC_URL", raising=False)
    monkeypatch.setenv("SOLANA_KEYPAIR_PATH", str(cli_dir / "station.json"))
    logs = ["--audit-log", str(cli_dir / "a.jsonl"), "--anchors-log", str(cli_dir / "b.jsonl")]
    assert sol.main(["keygen"]) == 0
    assert sol.main([*logs, "demo", "--bay", "1", "--kind", "OOD_REFUSAL"]) == 0
    assert sol.main([*logs, "verify", "--offline"]) == 0
    (cli_dir / "a.jsonl").write_text(
        (cli_dir / "a.jsonl").read_text().replace("DEMO-SKU", "TAMPERED"), encoding="utf-8"
    )
    assert sol.main([*logs, "verify", "--offline"]) == 1
