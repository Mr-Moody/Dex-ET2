"""``teleop capture``: record raw Studio UDP datagrams to JSON Lines (no parsing)."""

from __future__ import annotations

import argparse
import base64
import json
import socket
import sys
import time
from pathlib import Path


def record_raw(host: str, port: int, seconds: float, out: Path) -> int:
    """Receive datagrams for ``seconds`` and write one ``{t_ns, src, payload_b64}`` per line."""
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, port))
        sock.settimeout(0.2)
        deadline = time.monotonic() + seconds
        with out.open("w", encoding="utf-8", newline="\n") as fh:
            while time.monotonic() < deadline:
                try:
                    data, _addr = sock.recvfrom(65535)
                except TimeoutError:
                    continue
                except ConnectionResetError:  # Windows: ICMP unreachable from an earlier send
                    continue
                rec = {
                    "t_ns": time.monotonic_ns(),
                    "src": "studio",
                    "payload_b64": base64.b64encode(data).decode("ascii"),
                }
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
                count += 1
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
    return count


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="teleop capture")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=14043)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", type=Path, required=True)
    ns = ap.parse_args(argv)
    print(f"capturing UDP {ns.host}:{ns.port} for {ns.seconds:g}s -> {ns.out}", flush=True)
    try:
        n = record_raw(ns.host, ns.port, ns.seconds, ns.out)
    except OSError as exc:
        print(
            f"teleop capture: cannot bind UDP {ns.host}:{ns.port} ({exc}); "
            "is another capture or teleop process using the port?",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(f"wrote {n} datagrams to {ns.out}", flush=True)
    if n == 0:
        print("teleop capture: no packets received; check Studio streaming target/IP", flush=True)
        return 1
    return 0
