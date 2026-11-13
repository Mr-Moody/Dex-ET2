from teleop.input.buffer import LatestBuffer
from teleop.types import HandFrame, Side


def _f(side: Side, t_ns: int, dev: int | None = None, seq: int = 0) -> HandFrame:
    return HandFrame(
        side=side, host_time_ns=t_ns, seq=seq, source="t", joints={}, device_time_us=dev
    )


def test_latest_buffer_keeps_newest_per_side():
    b = LatestBuffer()
    b.put(_f("left", 1, seq=1))
    b.put(_f("left", 2, seq=2))
    b.put(_f("right", 3, seq=3))
    got = b.get()
    assert got.left is not None and got.left.seq == 2
    assert got.right is not None and got.right.seq == 3


def test_latest_buffer_staleness():
    b = LatestBuffer()
    t = 1_000_000_000
    assert b.status(now_ns=t).connected is False
    b.put(_f("left", t))
    assert b.status(now_ns=t + 100_000_000).connected is True
    assert b.status(now_ns=t + 300_000_000).connected is False


def test_timestamp_anomaly_counted():
    b = LatestBuffer()
    for i, dev in enumerate((10, 20, 15)):
        b.put(_f("left", i, dev=dev, seq=i))
    st = b.status(now_ns=3)
    assert st.timestamp_anomalies == 1
    assert st.frames == 3
    got = b.get().left
    assert got is not None and got.device_time_us == 15


def test_run_exception_is_logged_and_stored(caplog):
    import threading
    import time

    from teleop.input.buffer import ThreadedSource

    class Boom(ThreadedSource):
        def _run(self, stop: threading.Event) -> None:
            raise RuntimeError("boom")

    src = Boom("boom")
    with caplog.at_level("ERROR"):
        src.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and src.last_error is None:
            time.sleep(0.01)
    src.stop()
    assert isinstance(src.last_error, RuntimeError)
    assert any("boom" in r.getMessage() or r.exc_info for r in caplog.records)
