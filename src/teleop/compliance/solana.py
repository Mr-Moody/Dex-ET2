"""Solana compliance sink (T-160): a hash-chained local audit log anchored with Memo transactions.

Pipeline for every ``ComplianceEvent``::

    emit()          control thread: one deque.append + one SimpleQueue.put. No I/O, never raises.
      |
    ledger thread   canonical JSON record -> SHA-256 digest chained to the previous digest
      |             -> Ed25519 signature by the station key (if configured) -> one JSONL line.
      |             No network on this thread, so the local trail is never delayed by the RPC.
      |
    anchor thread   Memo transaction carrying the digest -> sendTransaction -> wait "confirmed"
                    -> receipt line in the anchors log. RPC errors back off and retry; entries
                    still unanchored at close() are re-queued on the next start.

Modes, picked by ``SolanaComplianceSink.from_env``:

* ``solana``        ``SOLANA_RPC_URL`` + ``SOLANA_KEYPAIR_PATH``: signed log + on-chain anchors.
* ``local-signed``  keypair only: signed local log, no network.
* ``local``         neither (CI default): unsigned hash-chained local log.

Only the digest, event kind, bay and sequence number go on-chain. The full record stays local,
so anyone holding the log can recompute the digest and compare it with the chain.

``SolanaCompliancePlaceholder`` (from T-100) is kept unchanged: a no-network ring buffer with an
optional plain JSONL log. Use ``SolanaComplianceSink.from_env()`` for the real chain; both share
``emit``/``close``/``events``/``latest``.

CLI: ``uv run python -m teleop.compliance.solana {keygen,airdrop,status,verify,demo}``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import math
import os
import queue
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import IO, Literal, Protocol, Self, cast, get_args
from urllib import request as urlrequest
from urllib.error import HTTPError
from urllib.parse import quote

import numpy as np
from dotenv import find_dotenv, load_dotenv
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import Message
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.transaction import Transaction

from teleop.compliance.protocol import ComplianceEvent, ComplianceKind

log = logging.getLogger(__name__)

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type AnchorState = Literal["local", "pending", "anchored"]

MEMO_PROGRAM_ID = Pubkey.from_string("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")
DEVNET_RPC_URL = "https://api.devnet.solana.com"
DEFAULT_AUDIT_PATH = Path("outputs/compliance_chain.jsonl")  # hash-chained, signed entries
DEFAULT_ANCHORS_PATH = Path("outputs/compliance_anchors.jsonl")
PLACEHOLDER_AUDIT_PATH = Path("outputs/compliance_audit.jsonl")  # T-100 placeholder, plain JSONL
DEFAULT_KEYPAIR_PATH = Path.home() / ".config" / "teleop" / "solana-compliance.json"
RECORD_SCHEMA = "atc.compliance/v1"
MEMO_PREFIX = "atc-compliance/v1"
GENESIS_DIGEST = "0" * 64
LAMPORTS_PER_SOL = 1_000_000_000


# ── Canonical hashing ────────────────────────────────────────────────────────────────────────


def to_json_value(value: object) -> JsonValue:
    """Convert ``value`` into plain JSON types deterministically (numpy, enums, sets, paths…).

    Non-finite floats become strings (``"nan"``, ``"inf"``) so canonical JSON never fails.
    Unknown objects fall back to ``repr``; the stored record is what gets hashed, so
    verification recomputes from the stored text, not from the original object.
    """
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, Enum):
        return to_json_value(cast(object, value.value))
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, np.ndarray):
        return to_json_value(cast(object, value.tolist()))
    if isinstance(value, np.generic):
        return to_json_value(cast(object, value.item()))
    if isinstance(value, Mapping):
        items = cast(Mapping[object, object], value)
        return {str(k): to_json_value(v) for k, v in items.items()}
    if isinstance(value, list | tuple):
        return [to_json_value(v) for v in cast(Sequence[object], value)]
    if isinstance(value, set | frozenset):
        members = [to_json_value(v) for v in cast(set[object], value)]
        return sorted(members, key=_dumps)
    if isinstance(value, bytes | bytearray):
        return bytes(value).hex()
    if isinstance(value, Path):
        return value.as_posix()
    return repr(value)


def _dumps(value: JsonValue) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def canonical_json(value: JsonValue) -> bytes:
    """Sorted keys, no whitespace, UTF-8, NaN rejected: byte-identical for equal values."""
    return _dumps(value).encode("utf-8")


def digest_record(record: JsonObject) -> str:
    """Lowercase hex SHA-256 of the record's canonical JSON."""
    return hashlib.sha256(canonical_json(record)).hexdigest()


def build_record(
    event: ComplianceEvent,
    metadata: Mapping[str, object],
    *,
    seq: int,
    prev_digest: str,
    emitted_unix_ns: int,
) -> JsonObject:
    """The hashed audit record. ``prev_digest`` links it to the previous entry (hash chain)."""
    return {
        "schema": RECORD_SCHEMA,
        "seq": seq,
        "prev_digest": prev_digest,
        "kind": str(event.kind),
        "bay_id": int(event.bay_id),
        "part_sku_id": str(event.part_sku_id),
        "timestamp_ns": int(event.timestamp_ns),
        "emitted_unix_ns": emitted_unix_ns,
        "payload_sha256": str(event.payload_sha256),
        "summary": str(event.summary),
        "metadata": to_json_value(metadata),
    }


def sign_digest(keypair: Keypair, digest: str) -> str:
    """Base58 Ed25519 signature over the 32 raw digest bytes."""
    return str(keypair.sign_message(bytes.fromhex(digest)))


def verify_signature(signer: str, signature: str, digest: str) -> bool:
    try:
        sig = Signature.from_string(signature)
        return sig.verify(Pubkey.from_string(signer), bytes.fromhex(digest))
    except ValueError:
        return False


def _is_hex64(text: str) -> bool:
    return len(text) == 64 and all(c in "0123456789abcdef" for c in text)


# ── Log entries ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """One line of the audit log."""

    seq: int
    digest: str
    record: JsonObject
    signer: str | None = None
    signature: str | None = None

    @property
    def kind(self) -> str:
        return str(self.record.get("kind"))

    @property
    def bay_id(self) -> int:
        bay = self.record.get("bay_id")
        return bay if isinstance(bay, int) else -1

    def to_line(self) -> str:
        return _dumps(
            {
                "digest": self.digest,
                "record": self.record,
                "signer": self.signer,
                "signature": self.signature,
            }
        )

    @classmethod
    def from_line(cls, line: str) -> AuditEntry:
        doc = _require_obj(json.loads(line), "audit line")
        record = _require_obj(doc.get("record"), "audit record")
        seq = record.get("seq")
        digest = doc.get("digest")
        if not isinstance(seq, int) or not isinstance(digest, str):
            raise ValueError("audit line needs an integer record.seq and a string digest")
        signer = doc.get("signer")
        signature = doc.get("signature")
        return cls(
            seq=seq,
            digest=digest,
            record=record,
            signer=signer if isinstance(signer, str) else None,
            signature=signature if isinstance(signature, str) else None,
        )


@dataclass(frozen=True, slots=True)
class AnchorReceipt:
    """One line of the anchors log: where a digest landed on-chain."""

    digest: str
    seq: int
    tx_signature: str
    slot: int | None
    explorer_url: str
    anchored_unix_ns: int

    def to_line(self) -> str:
        return _dumps(
            {
                "digest": self.digest,
                "seq": self.seq,
                "tx_signature": self.tx_signature,
                "slot": self.slot,
                "explorer_url": self.explorer_url,
                "anchored_unix_ns": self.anchored_unix_ns,
            }
        )

    @classmethod
    def from_line(cls, line: str) -> AnchorReceipt:
        doc = _require_obj(json.loads(line), "anchor line")
        digest, seq, sig = doc.get("digest"), doc.get("seq"), doc.get("tx_signature")
        url, at = doc.get("explorer_url"), doc.get("anchored_unix_ns")
        if not (isinstance(digest, str) and isinstance(seq, int) and isinstance(sig, str)):
            raise ValueError("anchor line needs digest, seq and tx_signature")
        return cls(
            digest=digest,
            seq=seq,
            tx_signature=sig,
            slot=_opt_int(doc.get("slot")),
            explorer_url=url if isinstance(url, str) else "",
            anchored_unix_ns=at if isinstance(at, int) else 0,
        )


@dataclass(frozen=True, slots=True)
class AuditStatus:
    """What the HUD shows for one recent event."""

    seq: int
    kind: str
    bay_id: int
    digest: str
    state: AnchorState
    tx_signature: str | None = None
    explorer_url: str | None = None


def read_audit_log(path: Path) -> list[AuditEntry]:
    """All parseable entries in file order; malformed lines (e.g. a torn last write) are skipped."""
    if not path.exists():
        return []
    entries: list[AuditEntry] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(AuditEntry.from_line(line))
        except ValueError as exc:
            log.warning("compliance: skipping malformed audit line %d in %s: %s", n, path, exc)
    return entries


def read_receipts(path: Path | None) -> dict[str, AnchorReceipt]:
    """Receipts keyed by digest (last one wins)."""
    if path is None or not path.exists():
        return {}
    receipts: dict[str, AnchorReceipt] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            receipt = AnchorReceipt.from_line(line)
        except ValueError as exc:
            log.warning("compliance: skipping malformed anchor line in %s: %s", path, exc)
            continue
        receipts[receipt.digest] = receipt
    return receipts


# ── JSON-RPC ─────────────────────────────────────────────────────────────────────────────────


class RpcError(RuntimeError):
    """Any Solana RPC failure: transport, JSON-RPC error, failed or unconfirmed transaction."""


class RpcTransport(Protocol):
    def call(self, method: str, params: list[JsonValue]) -> JsonValue: ...


class HttpJsonRpc:
    """Minimal JSON-RPC 2.0 client over ``urllib`` (no extra dependency)."""

    def __init__(self, url: str, timeout_s: float = 10.0) -> None:
        self.url = url
        self.timeout_s = timeout_s

    def call(self, method: str, params: list[JsonValue]) -> JsonValue:
        body = _dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        req = urlrequest.Request(
            self.url,
            data=body.encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "teleop-compliance/0.1"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self.timeout_s) as resp:
                raw = cast(bytes, resp.read())
            doc = _require_obj(json.loads(raw), method)
        except HTTPError as exc:  # RPC errors often arrive as HTTP 4xx with a JSON-RPC body
            raise RpcError(f"{method}: HTTP {exc.code}: {_http_error_message(exc)}") from exc
        except (OSError, ValueError) as exc:
            raise RpcError(f"{method}: {exc}") from exc
        if doc.get("error") is not None:
            raise RpcError(f"{method}: {_dumps(doc.get('error'))}")
        return doc.get("result")


def _http_error_message(exc: HTTPError) -> str:
    try:
        doc: object = json.loads(exc.read())
        if isinstance(doc, dict):
            err = cast(JsonObject, doc).get("error")
            if isinstance(err, dict) and isinstance(err.get("message"), str):
                return str(err["message"])
    except (OSError, ValueError):
        pass
    return str(exc.reason)


def _require_obj(value: object, what: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{what}: expected a JSON object")
    return cast(JsonObject, value)


def _rpc_obj(value: JsonValue, what: str) -> JsonObject:
    if not isinstance(value, dict):
        raise RpcError(f"{what}: expected a JSON object, got {type(value).__name__}")
    return value


def _rpc_list(value: JsonValue, what: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise RpcError(f"{what}: expected a JSON array, got {type(value).__name__}")
    return value


def _rpc_str(value: JsonValue, what: str) -> str:
    if not isinstance(value, str):
        raise RpcError(f"{what}: expected a string, got {type(value).__name__}")
    return value


def _opt_int(value: JsonValue) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def explorer_url(tx_signature: str, rpc_url: str) -> str:
    base = f"https://explorer.solana.com/tx/{tx_signature}"
    lowered = rpc_url.lower()
    if "devnet" in lowered:
        return f"{base}?cluster=devnet"
    if "testnet" in lowered:
        return f"{base}?cluster=testnet"
    if "localhost" in lowered or "127.0.0.1" in lowered:
        return f"{base}?cluster=custom&customUrl={quote(rpc_url, safe='')}"
    return base


def is_faucet_cluster(rpc_url: str) -> bool:
    lowered = rpc_url.lower()
    return any(k in lowered for k in ("devnet", "testnet", "localhost", "127.0.0.1"))


# ── Memo anchoring ───────────────────────────────────────────────────────────────────────────


def build_memo(entry: AuditEntry) -> str:
    """``atc-compliance/v1 <sha256> <KIND> bay=<n> seq=<n>``: ~110 bytes, no personal data."""
    return f"{MEMO_PREFIX} {entry.digest} {entry.kind} bay={entry.bay_id} seq={entry.seq}"


def memo_digest(memo: str) -> str | None:
    parts = memo.split()
    if len(parts) >= 2 and parts[0] == MEMO_PREFIX and _is_hex64(parts[1]):
        return parts[1]
    return None


@dataclass(frozen=True, slots=True)
class OnChainMemo:
    memo: str
    signers: tuple[str, ...]
    slot: int | None
    block_time: int | None


def parse_memo_transaction(doc: JsonObject) -> OnChainMemo | None:
    """Extract the memo text and signers from a ``getTransaction`` (jsonParsed) result."""
    message = _rpc_obj(_rpc_obj(doc.get("transaction"), "transaction").get("message"), "message")
    signers: list[str] = []
    for key in _rpc_list(message.get("accountKeys"), "accountKeys"):
        if isinstance(key, dict) and key.get("signer") is True:
            signers.append(_rpc_str(key.get("pubkey"), "accountKeys.pubkey"))
    memo: str | None = None
    for ix in _rpc_list(message.get("instructions"), "instructions"):
        if not isinstance(ix, dict) or ix.get("programId") != str(MEMO_PROGRAM_ID):
            continue
        parsed = ix.get("parsed")
        if isinstance(parsed, str):
            memo = parsed
            break
    if memo is None:  # older RPCs: fall back to the Memo program's log line
        meta = doc.get("meta")
        logs = meta.get("logMessages") if isinstance(meta, dict) else None
        for line in logs if isinstance(logs, list) else []:
            if isinstance(line, str) and "Memo (len" in line and '): "' in line:
                memo = line.split('): "', 1)[1].removesuffix('"')
                break
    if memo is None:
        return None
    return OnChainMemo(
        memo, tuple(signers), _opt_int(doc.get("slot")), _opt_int(doc.get("blockTime"))
    )


class MemoAnchor:
    """Signs and submits Memo transactions with the station keypair; reads them back."""

    def __init__(
        self,
        keypair: Keypair,
        transport: RpcTransport,
        *,
        rpc_url: str = DEVNET_RPC_URL,
        confirm_timeout_s: float = 30.0,
        poll_interval_s: float = 0.5,
        blockhash_ttl_s: float = 30.0,
        sleep: Callable[[float], object] = time.sleep,
    ) -> None:
        self._keypair = keypair
        self._transport = transport
        self.rpc_url = rpc_url
        self._confirm_timeout_s = confirm_timeout_s
        self._poll_interval_s = poll_interval_s
        self._blockhash_ttl_s = blockhash_ttl_s
        self._sleep = sleep
        self._blockhash: tuple[Hash, float] | None = None

    @property
    def pubkey(self) -> Pubkey:
        return self._keypair.pubkey()

    def build_transaction(self, memo: str, blockhash: Hash) -> Transaction:
        payer = self._keypair.pubkey()
        ix = Instruction(MEMO_PROGRAM_ID, memo.encode("utf-8"), [AccountMeta(payer, True, True)])
        message = Message.new_with_blockhash([ix], payer, blockhash)
        return Transaction([self._keypair], message, blockhash)

    def _latest_blockhash(self) -> Hash:
        now = time.monotonic()
        if self._blockhash is not None and now - self._blockhash[1] < self._blockhash_ttl_s:
            return self._blockhash[0]
        res = self._transport.call("getLatestBlockhash", [{"commitment": "confirmed"}])
        value = _rpc_obj(_rpc_obj(res, "getLatestBlockhash").get("value"), "blockhash value")
        blockhash = Hash.from_string(_rpc_str(value.get("blockhash"), "blockhash"))
        self._blockhash = (blockhash, now)
        return blockhash

    def submit(self, memo: str) -> tuple[str, int | None]:
        """Send one memo tx and wait until it is ``confirmed``. Returns (signature, slot)."""
        try:
            tx = self.build_transaction(memo, self._latest_blockhash())
            wire = base64.b64encode(bytes(tx)).decode("ascii")
            opts: JsonObject = {"encoding": "base64", "preflightCommitment": "confirmed"}
            sig = _rpc_str(self._transport.call("sendTransaction", [wire, opts]), "signature")
            return sig, self._wait_confirmed(sig)
        except Exception:
            self._blockhash = None  # a stale blockhash is the usual cause; refetch next time
            raise

    def _wait_confirmed(self, sig: str) -> int | None:
        deadline = time.monotonic() + self._confirm_timeout_s
        while True:
            opts: JsonObject = {"searchTransactionHistory": False}
            res = self._transport.call("getSignatureStatuses", [[sig], opts])
            statuses = _rpc_list(_rpc_obj(res, "getSignatureStatuses").get("value"), "statuses")
            status = statuses[0] if statuses else None
            if isinstance(status, dict):
                if status.get("err") is not None:
                    raise RpcError(f"transaction {sig} failed: {_dumps(status.get('err'))}")
                if status.get("confirmationStatus") in ("confirmed", "finalized"):
                    return _opt_int(status.get("slot"))
            if time.monotonic() >= deadline:
                raise RpcError(f"transaction {sig} not confirmed in {self._confirm_timeout_s}s")
            self._sleep(self._poll_interval_s)

    def fetch(self, tx_signature: str) -> OnChainMemo | None:
        opts: JsonObject = {
            "encoding": "jsonParsed",
            "commitment": "confirmed",
            "maxSupportedTransactionVersion": 0,
        }
        res = self._transport.call("getTransaction", [tx_signature, opts])
        if res is None:
            return None
        return parse_memo_transaction(_rpc_obj(res, "getTransaction"))

    def balance_lamports(self) -> int:
        res = self._transport.call("getBalance", [str(self.pubkey), {"commitment": "confirmed"}])
        value = _opt_int(_rpc_obj(res, "getBalance").get("value"))
        if value is None:
            raise RpcError("getBalance: missing value")
        return value

    def request_airdrop(self, lamports: int) -> str:
        res = self._transport.call("requestAirdrop", [str(self.pubkey), lamports])
        return _rpc_str(res, "requestAirdrop")


# ── Keypair files (solana-keygen JSON format) ────────────────────────────────────────────────


def load_keypair(path: Path) -> Keypair:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array of 64 bytes (solana-keygen format)")
    data = cast(list[object], raw)
    if len(data) != 64 or not all(isinstance(b, int) and 0 <= b <= 255 for b in data):
        raise ValueError(f"{path}: expected a JSON array of 64 bytes (solana-keygen format)")
    return Keypair.from_bytes(bytes(cast(list[int], data)))


def save_keypair(keypair: Keypair, path: Path, *, overwrite: bool = False) -> None:
    """Write ``path`` with owner-only permissions. Refuses to overwrite unless asked."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(list(bytes(keypair)), fh)


# ── The sink ─────────────────────────────────────────────────────────────────────────────────

type _LedgerItem = tuple[ComplianceEvent, dict[str, object], int]


def _open_append(path: Path | None) -> IO[str] | None:
    """Open for appending; repair a torn last line so the next record starts on its own line."""
    if path is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torn = False
        if path.exists() and path.stat().st_size > 0:
            with path.open("rb") as fh:
                fh.seek(-1, os.SEEK_END)
                torn = fh.read(1) != b"\n"
        out = path.open("a", encoding="utf-8")
        if torn:
            out.write("\n")
        return out
    except OSError:
        log.exception("compliance: cannot open %s; continuing without it", path)
        return None


class SolanaComplianceSink:
    """``ComplianceSink`` that logs every event locally and anchors its digest on Solana."""

    def __init__(
        self,
        *,
        keypair: Keypair | None = None,
        transport: RpcTransport | None = None,
        rpc_url: str = DEVNET_RPC_URL,
        audit_path: Path | None = DEFAULT_AUDIT_PATH,
        anchors_path: Path | None = DEFAULT_ANCHORS_PATH,
        capacity: int = 256,
        max_backoff_s: float = 30.0,
        close_grace_s: float = 5.0,
        confirm_timeout_s: float = 30.0,
        poll_interval_s: float = 0.5,
        fsync: bool = False,
    ) -> None:
        if transport is not None and keypair is None:
            raise ValueError("on-chain anchoring needs a keypair (SOLANA_KEYPAIR_PATH)")
        self._keypair = keypair
        self._anchor = (
            MemoAnchor(
                keypair,
                transport,
                rpc_url=rpc_url,
                confirm_timeout_s=confirm_timeout_s,
                poll_interval_s=poll_interval_s,
            )
            if keypair is not None and transport is not None
            else None
        )
        self._audit_path = audit_path
        self._anchors_path = anchors_path
        self._capacity = capacity
        self._max_backoff_s = max_backoff_s
        self._close_grace_s = close_grace_s
        self._fsync = fsync

        self._events: deque[ComplianceEvent] = deque(maxlen=capacity)
        self._status: OrderedDict[str, AuditStatus] = OrderedDict()
        self._counts = {"logged": 0, "anchored": 0, "errors": 0}
        self._lock = threading.Lock()
        self._ledger_q: queue.SimpleQueue[_LedgerItem | None] = queue.SimpleQueue()
        self._anchor_q: queue.SimpleQueue[AuditEntry | None] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._closed = False
        self._seq = 0
        self._prev = GENESIS_DIGEST

        self._resume()
        self._ledger_thread = threading.Thread(
            target=self._ledger_loop, name="compliance-ledger", daemon=True
        )
        self._ledger_thread.start()
        self._anchor_thread: threading.Thread | None = None
        if self._anchor is not None:
            self._anchor_thread = threading.Thread(
                target=self._anchor_loop, name="compliance-anchor", daemon=True
            )
            self._anchor_thread.start()

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        audit_path: Path | None = DEFAULT_AUDIT_PATH,
        anchors_path: Path | None = DEFAULT_ANCHORS_PATH,
        transport: RpcTransport | None = None,
        close_grace_s: float = 5.0,
    ) -> Self:
        """Build from ``SOLANA_RPC_URL`` / ``SOLANA_KEYPAIR_PATH``; degrade to local modes."""
        env = os.environ if env is None else env
        rpc_url = env.get("SOLANA_RPC_URL", "").strip()
        keypair_path = env.get("SOLANA_KEYPAIR_PATH", "").strip()
        keypair: Keypair | None = None
        if keypair_path:
            try:
                keypair = load_keypair(Path(keypair_path).expanduser())
            except (OSError, ValueError) as exc:
                log.warning("compliance: cannot load SOLANA_KEYPAIR_PATH (%s); unsigned log", exc)
        if keypair is None or not rpc_url:
            transport = None
        elif transport is None:
            transport = HttpJsonRpc(rpc_url)
        return cls(
            keypair=keypair,
            transport=transport,
            rpc_url=rpc_url or DEVNET_RPC_URL,
            audit_path=audit_path,
            anchors_path=anchors_path,
            close_grace_s=close_grace_s,
        )

    # ── public API ──

    @property
    def mode(self) -> Literal["solana", "local-signed", "local"]:
        if self._anchor is not None:
            return "solana"
        return "local-signed" if self._keypair is not None else "local"

    @property
    def pubkey(self) -> str | None:
        return str(self._keypair.pubkey()) if self._keypair is not None else None

    @property
    def anchor(self) -> MemoAnchor | None:
        return self._anchor

    def emit(self, event: ComplianceEvent) -> None:
        """Enqueue and return immediately. Never blocks on I/O and never raises."""
        if self._closed:
            return
        try:
            self._events.append(event)
            self._ledger_q.put((event, dict(event.metadata), time.time_ns()))
        except Exception:  # a bad event must never take down the control loop
            log.exception("compliance: emit failed")

    def close(self) -> None:
        """Flush the local log, give pending anchors ``close_grace_s`` to land, then stop."""
        if self._closed:
            return
        self._closed = True
        self._ledger_q.put(None)
        self._ledger_thread.join(timeout=self._close_grace_s)
        if self._anchor_thread is not None:
            self._anchor_q.put(None)  # after the ledger drained, so it follows every entry
            self._anchor_thread.join(timeout=self._close_grace_s)
            self._stop.set()
            self._anchor_thread.join(timeout=1.0)

    # events() / latest() match SolanaCompliancePlaceholder, so HUD code works with either sink.

    def events(self) -> list[ComplianceEvent]:
        """Most recent emitted events (up to ``capacity``), newest last."""
        return list(self._events)

    def latest(self) -> ComplianceEvent | None:
        return self._events[-1] if self._events else None

    def recent_status(self, n: int = 10) -> list[AuditStatus]:
        """Digest and anchor state of the most recent logged entries, newest last."""
        with self._lock:
            return list(self._status.values())[-n:]

    def latest_status(self) -> AuditStatus | None:
        with self._lock:
            return next(reversed(self._status.values()), None)

    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    # ── internals ──

    def _resume(self) -> None:
        """Continue the hash chain from the existing log and re-queue unanchored entries."""
        if self._audit_path is None:
            return
        entries = read_audit_log(self._audit_path)
        if not entries:
            return
        self._seq, self._prev = entries[-1].seq, entries[-1].digest
        receipts = read_receipts(self._anchors_path)
        me = self.pubkey
        for entry in entries:
            receipt = receipts.get(entry.digest)
            if receipt is not None:
                self._track(entry, "anchored", receipt, count=False)
            elif self._anchor is not None and entry.signer in (None, me):
                self._track(entry, "pending", count=False)
                self._anchor_q.put(entry)
            else:
                self._track(entry, "local", count=False)

    def _make_entry(self, item: _LedgerItem) -> AuditEntry:
        event, metadata, emitted_unix_ns = item
        seq = self._seq + 1
        record = build_record(
            event, metadata, seq=seq, prev_digest=self._prev, emitted_unix_ns=emitted_unix_ns
        )
        digest = digest_record(record)
        if self._keypair is None:
            return AuditEntry(seq, digest, record)
        return AuditEntry(
            seq, digest, record, str(self._keypair.pubkey()), sign_digest(self._keypair, digest)
        )

    def _ledger_loop(self) -> None:
        out = _open_append(self._audit_path)
        try:
            while (item := self._ledger_q.get()) is not None:
                try:
                    entry = self._make_entry(item)
                    if out is not None:
                        out.write(entry.to_line() + "\n")
                        out.flush()
                        if self._fsync:
                            os.fsync(out.fileno())
                except Exception:
                    log.exception("compliance: could not log event")
                    with self._lock:
                        self._counts["errors"] += 1
                    continue
                self._seq, self._prev = entry.seq, entry.digest
                self._track(entry, "pending" if self._anchor is not None else "local")
                if self._anchor is not None:
                    self._anchor_q.put(entry)
        finally:
            if out is not None:
                out.close()

    def _anchor_loop(self) -> None:
        anchor = self._anchor
        assert anchor is not None
        pending: deque[AuditEntry] = deque()
        draining = False
        backoff = 0.5
        out: IO[str] | None = None
        try:
            while True:
                if not pending and not draining:
                    item = self._anchor_q.get()
                    if item is None:
                        draining = True
                    else:
                        pending.append(item)
                while True:
                    try:
                        item = self._anchor_q.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        draining = True
                    else:
                        pending.append(item)
                if not pending:
                    if draining:
                        return
                    continue
                if self._stop.is_set():
                    return  # leftovers are re-queued from the audit log on the next start
                entry = pending[0]
                try:
                    sig, slot = anchor.submit(build_memo(entry))
                except Exception as exc:
                    log.warning(
                        "compliance: anchoring seq=%d failed (%s); retry in %.1fs",
                        entry.seq,
                        exc,
                        backoff,
                    )
                    with self._lock:
                        self._counts["errors"] += 1
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2.0, self._max_backoff_s)
                    continue
                backoff = 0.5
                receipt = AnchorReceipt(
                    digest=entry.digest,
                    seq=entry.seq,
                    tx_signature=sig,
                    slot=slot,
                    explorer_url=explorer_url(sig, anchor.rpc_url),
                    anchored_unix_ns=time.time_ns(),
                )
                if out is None:
                    out = _open_append(self._anchors_path)
                if out is not None:
                    try:
                        out.write(receipt.to_line() + "\n")
                        out.flush()
                    except OSError:
                        log.exception("compliance: could not write anchor receipt")
                self._track(entry, "anchored", receipt)
                pending.popleft()
        finally:
            if out is not None:
                out.close()

    def _track(
        self,
        entry: AuditEntry,
        state: AnchorState,
        receipt: AnchorReceipt | None = None,
        *,
        count: bool = True,
    ) -> None:
        status = AuditStatus(
            seq=entry.seq,
            kind=entry.kind,
            bay_id=entry.bay_id,
            digest=entry.digest,
            state=state,
            tx_signature=receipt.tx_signature if receipt is not None else None,
            explorer_url=receipt.explorer_url if receipt is not None else None,
        )
        with self._lock:
            if count:
                self._counts["anchored" if state == "anchored" else "logged"] += 1
            if state == "anchored" and entry.digest not in self._status and count:
                return  # already evicted from the ring buffer; keep the newest entries only
            self._status[entry.digest] = status
            while len(self._status) > self._capacity:
                self._status.popitem(last=False)


_STOP = object()


class SolanaCompliancePlaceholder:
    """T-100 placeholder, kept as-is: ring buffer for the HUD plus an optional JSONL log.

    ``emit()`` never touches the disk; a daemon thread writes the log. Nothing goes to Solana.
    """

    def __init__(
        self,
        capacity: int = 256,
        jsonl_path: Path | None = None,
        rpc_url: str | None = None,
        keypair_path: str | None = None,
    ) -> None:
        self.rpc_url = rpc_url
        self.keypair_path = keypair_path
        self.jsonl_path = jsonl_path
        self._lock = threading.Lock()
        self._ring: deque[ComplianceEvent] = deque(maxlen=capacity)
        self._closed = False
        self._queue: queue.SimpleQueue[ComplianceEvent | object] = queue.SimpleQueue()
        self._writer: threading.Thread | None = None
        if jsonl_path is not None:
            self._writer = threading.Thread(
                target=self._write_loop, args=(jsonl_path,), name="compliance-jsonl", daemon=True
            )
            self._writer.start()

    @classmethod
    def from_env(
        cls, capacity: int = 256, jsonl_path: Path | None = PLACEHOLDER_AUDIT_PATH
    ) -> SolanaCompliancePlaceholder:
        return cls(
            capacity=capacity,
            jsonl_path=jsonl_path,
            rpc_url=os.environ.get("SOLANA_RPC_URL") or None,
            keypair_path=os.environ.get("SOLANA_KEYPAIR_PATH") or None,
        )

    def emit(self, event: ComplianceEvent) -> None:
        with self._lock:
            if self._closed:
                return
            self._ring.append(event)
        if self._writer is not None:
            self._queue.put(event)

    def events(self) -> list[ComplianceEvent]:
        with self._lock:
            return list(self._ring)

    def latest(self) -> ComplianceEvent | None:
        with self._lock:
            return self._ring[-1] if self._ring else None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self._writer is not None:
            self._queue.put(_STOP)
            self._writer.join(timeout=5.0)

    def _write_loop(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                while (item := self._queue.get()) is not _STOP:
                    assert isinstance(item, ComplianceEvent)
                    fh.write(json.dumps(item.to_dict(), default=str) + "\n")
                    if self._queue.empty():
                        fh.flush()
        except OSError:
            log.exception("compliance: audit log %s failed; continuing in memory only", path)


# ── Verification ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class VerifyResult:
    seq: int
    kind: str
    bay_id: int
    digest: str
    digest_ok: bool
    chain_ok: bool
    signature_ok: bool | None  # None: entry is unsigned
    onchain_ok: bool | None  # None: not anchored, or not checked (offline)
    tx_signature: str | None = None
    explorer_url: str | None = None
    problem: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.digest_ok
            and self.chain_ok
            and self.signature_ok is not False
            and self.onchain_ok is not False
        )


def verify_audit_log(
    audit_path: Path, anchors_path: Path | None = None, anchor: MemoAnchor | None = None
) -> list[VerifyResult]:
    """Recompute every digest, check the chain and signatures, and (with ``anchor``) the chain.

    Detects an edited record (digest mismatch), a deleted or reordered record (chain break), a
    forged signature, and a record whose content no longer matches its on-chain memo.
    """
    receipts = read_receipts(anchors_path)
    results: list[VerifyResult] = []
    prev = GENESIS_DIGEST
    for entry in read_audit_log(audit_path):
        recomputed = digest_record(entry.record)
        digest_ok = recomputed == entry.digest
        chain_ok = entry.record.get("prev_digest") == prev
        signature_ok: bool | None = None
        if entry.signer is not None and entry.signature is not None:
            signature_ok = verify_signature(entry.signer, entry.signature, recomputed)
        problems: list[str] = []
        if not digest_ok:
            problems.append("record content does not match its digest")
        if not chain_ok:
            problems.append("hash chain broken (entry missing, reordered or edited before it)")
        if signature_ok is False:
            problems.append("station signature invalid")
        receipt = receipts.get(entry.digest)
        onchain_ok: bool | None = None
        if receipt is not None and anchor is not None:
            try:
                onchain = anchor.fetch(receipt.tx_signature)
            except RpcError as exc:
                onchain = None
                problems.append(f"could not fetch transaction: {exc}")
            if onchain is None:
                onchain_ok = False
                problems.append("anchor transaction not found on-chain")
            else:
                onchain_ok = memo_digest(onchain.memo) == recomputed and (
                    entry.signer is None or entry.signer in onchain.signers
                )
                if not onchain_ok:
                    problems.append("on-chain memo does not match this record")
        results.append(
            VerifyResult(
                seq=entry.seq,
                kind=entry.kind,
                bay_id=entry.bay_id,
                digest=entry.digest,
                digest_ok=digest_ok,
                chain_ok=chain_ok,
                signature_ok=signature_ok,
                onchain_ok=onchain_ok,
                tx_signature=receipt.tx_signature if receipt is not None else None,
                explorer_url=receipt.explorer_url if receipt is not None else None,
                problem="; ".join(problems),
            )
        )
        prev = entry.digest
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────────────────────


def _mark(value: bool | None) -> str:
    return {True: "OK", False: "FAIL", None: "-"}[value]


def _cmd_keygen(path: Path, force: bool) -> int:
    keypair = Keypair()
    try:
        save_keypair(keypair, path, overwrite=force)
    except FileExistsError:
        print(f"{path} already exists (use --force to replace it)")
        return 1
    print(f"station pubkey: {keypair.pubkey()}")
    print(f"wrote {path}; set SOLANA_KEYPAIR_PATH={path} in .env")
    return 0


def _station_anchor(env: Mapping[str, str]) -> MemoAnchor | None:
    rpc_url = env.get("SOLANA_RPC_URL", "").strip()
    keypair_path = env.get("SOLANA_KEYPAIR_PATH", "").strip()
    if not rpc_url or not keypair_path:
        return None
    return MemoAnchor(
        load_keypair(Path(keypair_path).expanduser()), HttpJsonRpc(rpc_url), rpc_url=rpc_url
    )


def _cmd_airdrop(env: Mapping[str, str], sol: float) -> int:
    anchor = _station_anchor(env)
    if anchor is None:
        print("set SOLANA_RPC_URL and SOLANA_KEYPAIR_PATH first")
        return 1
    if not is_faucet_cluster(anchor.rpc_url):
        print(f"refusing: {anchor.rpc_url} is not devnet/testnet/localnet")
        return 1
    try:
        sig = anchor.request_airdrop(round(sol * LAMPORTS_PER_SOL))
    except RpcError as exc:
        print(f"airdrop failed: {exc}")
        print(f"Request devnet SOL for {anchor.pubkey} at https://faucet.solana.com")
        return 1
    print(f"airdrop requested for {anchor.pubkey}: {explorer_url(sig, anchor.rpc_url)}")
    return 0


def _cmd_status(env: Mapping[str, str], audit_path: Path, anchors_path: Path) -> int:
    entries = read_audit_log(audit_path)
    receipts = read_receipts(anchors_path)
    anchor = _station_anchor(env)
    keypair_path = env.get("SOLANA_KEYPAIR_PATH", "").strip()
    mode = "solana" if anchor else ("local-signed" if keypair_path else "local")
    print(f"mode: {mode}")
    if anchor is not None:
        print(f"station: {anchor.pubkey}  rpc: {anchor.rpc_url}")
        try:
            print(f"balance: {anchor.balance_lamports() / LAMPORTS_PER_SOL:.4f} SOL")
        except RpcError as exc:
            print(f"balance: unavailable ({exc})")
    anchored = sum(1 for e in entries if e.digest in receipts)
    print(f"log: {len(entries)} entries, {anchored} anchored, {len(entries) - anchored} not")
    if entries:
        last = entries[-1]
        print(f"latest: seq {last.seq} {last.kind} bay {last.bay_id} {last.digest[:16]}…")
    return 0


def _cmd_verify(env: Mapping[str, str], audit_path: Path, anchors_path: Path, offline: bool) -> int:
    anchor = None if offline else _station_anchor(env)
    results = verify_audit_log(audit_path, anchors_path, anchor)
    if not results:
        print(f"no entries in {audit_path}")
        return 0
    for r in results:
        line = (
            f"seq {r.seq:>4} {r.kind:<15} bay {r.bay_id} {r.digest[:12]}…  "
            f"digest {_mark(r.digest_ok)}  chain {_mark(r.chain_ok)}  "
            f"sig {_mark(r.signature_ok)}  on-chain {_mark(r.onchain_ok)}"
        )
        print(line + (f"  {r.explorer_url}" if r.explorer_url else ""))
        if r.problem:
            print(f"       -> {r.problem}")
    bad = sum(1 for r in results if not r.ok)
    print(f"{len(results) - bad}/{len(results)} entries verified" + ("" if bad else " - all OK"))
    return 1 if bad else 0


def _cmd_demo(audit_path: Path, anchors_path: Path, bay: int, kind: str) -> int:
    sink = SolanaComplianceSink.from_env(
        audit_path=audit_path, anchors_path=anchors_path, close_grace_s=45.0
    )
    metadata: dict[str, object] = {
        "source": "demo",
        "confidence_min": 0.21,
        "x_bolt_m": 0.035,
        "tau_ext_inf_nm": 7.4,
    }
    payload = hashlib.sha256(canonical_json(to_json_value(metadata))).hexdigest()
    event = ComplianceEvent(
        kind=cast(ComplianceKind, kind),
        bay_id=bay,
        part_sku_id="DEMO-SKU",
        timestamp_ns=time.monotonic_ns(),
        payload_sha256=payload,
        summary=f"demo {kind} on bay {bay}",
        metadata=metadata,
    )
    print(f"mode: {sink.mode}")
    sink.emit(event)
    sink.close()
    status = sink.latest_status()
    if status is None:
        print("nothing was logged (see warnings above)")
        return 1
    print(f"seq {status.seq} {status.kind} bay {status.bay_id} digest {status.digest}")
    print(f"state: {status.state}" + (f"  {status.explorer_url}" if status.explorer_url else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    load_dotenv(find_dotenv(usecwd=True))  # the .env where the command is run, not the module
    kinds = [str(k) for k in get_args(ComplianceKind)]
    p = argparse.ArgumentParser(
        prog="python -m teleop.compliance.solana",
        description="Compliance audit log anchored on Solana (Memo program)",
    )
    p.add_argument("--audit-log", type=Path, default=DEFAULT_AUDIT_PATH)
    p.add_argument("--anchors-log", type=Path, default=DEFAULT_ANCHORS_PATH)
    sub = p.add_subparsers(dest="cmd", required=True)
    kg = sub.add_parser("keygen", help="create the station keypair")
    kg.add_argument("--path", type=Path, default=None)
    kg.add_argument("--force", action="store_true")
    ad = sub.add_parser("airdrop", help="request devnet SOL for the station key")
    ad.add_argument("--sol", type=float, default=1.0)
    sub.add_parser("status", help="mode, balance and log counts")
    vf = sub.add_parser("verify", help="recompute digests and check them against the chain")
    vf.add_argument("--offline", action="store_true", help="skip the on-chain check")
    dm = sub.add_parser("demo", help="emit one synthetic event and anchor it")
    dm.add_argument("--bay", type=int, default=1)
    dm.add_argument("--kind", choices=kinds, default="STALL_ALERT")
    a = p.parse_args(argv)
    env = os.environ
    match a.cmd:
        case "keygen":
            default = env.get("SOLANA_KEYPAIR_PATH", "").strip()
            path = a.path or (Path(default).expanduser() if default else DEFAULT_KEYPAIR_PATH)
            return _cmd_keygen(path, a.force)
        case "airdrop":
            return _cmd_airdrop(env, a.sol)
        case "status":
            return _cmd_status(env, a.audit_log, a.anchors_log)
        case "verify":
            return _cmd_verify(env, a.audit_log, a.anchors_log, a.offline)
        case _:
            return _cmd_demo(a.audit_log, a.anchors_log, a.bay, a.kind)


if __name__ == "__main__":
    raise SystemExit(main())
