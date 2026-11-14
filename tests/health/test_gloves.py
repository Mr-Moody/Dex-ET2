from teleop.health.gloves import GloveCheck
from teleop.health.model import Level
from teleop.input.studio import StudioLiveness

NOW = 100_000_000_000
S = 1_000_000_000


class FakeSource:
    def __init__(self, live: StudioLiveness) -> None:
        self.live = live
        self.calls: list[str] = []

    def liveness(self) -> StudioLiveness:
        return self.live

    def start(self) -> None:
        self.calls.append("start")

    def stop(self) -> None:
        self.calls.append("stop")


def live(**kw: object) -> StudioLiveness:
    base: dict[str, object] = {
        "bound": True,
        "port": 14043,
        "bind_error": None,
        "last_datagram_ns": None,
        "last_right_frame_ns": None,
        "datagram_hz": 0.0,
    }
    base.update(kw)
    return StudioLiveness(**base)  # type: ignore[arg-type]


def check(src: FakeSource) -> GloveCheck:
    return GloveCheck(src, 14043, clock=lambda: NOW)  # type: ignore[arg-type]


def test_ok_when_recent_right_frame():
    h = check(
        FakeSource(
            live(
                last_datagram_ns=NOW - int(0.2 * S),
                last_right_frame_ns=NOW - int(0.2 * S),
                datagram_hz=83.0,
            )
        )
    ).status()
    assert h.level is Level.OK
    assert h.summary == "streaming 83 Hz"


def test_warn_packets_without_right_hand():
    h = check(FakeSource(live(last_datagram_ns=NOW - int(0.1 * S)))).status()
    assert h.level is Level.WARN
    assert "no right hand" in h.detail


def test_warn_stalling():
    h = check(
        FakeSource(live(last_datagram_ns=NOW - S, last_right_frame_ns=NOW - S, datagram_hz=1.0))
    ).status()
    assert h.level is Level.WARN
    assert h.summary == "glove stream stalling"


def test_fail_no_packets():
    h = check(FakeSource(live(last_datagram_ns=NOW - int(4.2 * S)))).status()
    assert h.level is Level.FAIL
    assert h.summary == "no packets for 4 s"
    assert "127.0.0.1:14043" in h.detail
    assert h.can_reconnect is True


def test_fail_never_received():
    h = check(FakeSource(live())).status()
    assert h.level is Level.FAIL
    assert h.summary == "waiting for Studio"


def test_fail_bind_error():
    h = check(FakeSource(live(bound=False, bind_error="port 14043 in use"))).status()
    assert h.level is Level.FAIL
    assert "port 14043 in use" in h.detail


def test_off_without_source():
    h = GloveCheck(None, 14043).status()
    assert h.level is Level.OFF
    assert h.summary == "synthetic glove"
    assert h.can_reconnect is False


def test_reconnect_restarts_source():
    src = FakeSource(live())
    check(src).reconnect()
    assert src.calls == ["stop", "start"]
