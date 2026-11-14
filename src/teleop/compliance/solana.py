"""Solana compliance sink. T-100 placeholder; T-160 (Solana teammate) owns this file.

``SolanaCompliancePlaceholder`` keeps the last events in a ring buffer for the HUD and tests
and, optionally, appends them to a local JSONL audit log from a daemon thread so ``emit()``
never touches the disk on the 60 Hz loop. Nothing is sent to Solana.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from collections import deque
from pathlib import Path

from teleop.compliance.protocol import ComplianceEvent

log = logging.getLogger(__name__)

DEFAULT_AUDIT_PATH = Path("outputs/compliance_audit.jsonl")

_STOP = object()


class SolanaCompliancePlaceholder:
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
        cls, capacity: int = 256, jsonl_path: Path | None = DEFAULT_AUDIT_PATH
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
