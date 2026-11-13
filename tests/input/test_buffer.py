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
