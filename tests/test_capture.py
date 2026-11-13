import base64
import json
import socket
import threading
import time
from pathlib import Path

from teleop.commands.capture import main


def test_capture_writes_raw_jsonl(tmp_path: Path):
    out = tmp_path / "sub" / "cap.jsonl"
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    def send():
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            tx.sendto(b'{"hello": 1}', ("127.0.0.1", port))
            time.sleep(0.02)
        tx.close()

    t = threading.Thread(target=send)
    t.start()
    rc = main(["--port", str(port), "--seconds", "0.6", "--out", str(out), "--host", "127.0.0.1"])
    t.join()
    assert rc == 0
    recs = [json.loads(x) for x in out.read_text().splitlines()]
    assert recs
    assert recs[0]["src"] == "studio"
    assert base64.b64decode(recs[0]["payload_b64"]) == b'{"hello": 1}'
    assert recs[0]["t_ns"] <= recs[-1]["t_ns"]
