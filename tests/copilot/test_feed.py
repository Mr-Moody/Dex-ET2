"""T-211: console feed client and frame store, with real recorded frames and a fake socket."""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from teleop.copilot.feed import (
    CameraFrame,
    Connection,
    FeedClient,
    FeedMapper,
    parse_feeds,
    parse_telemetry,
    split_frame,
)
from teleop.copilot.frames import FrameStore, downscale_jpeg

FIXTURES = Path(__file__).parent / "fixtures"
MAIN_FRAME = (FIXTURES / "frame-0-1.bin").read_bytes()  # slot 0, layout 1: bay1/top_ortho
STRIP_FRAME = (FIXTURES / "frame-2-1.bin").read_bytes()  # slot 2, layout 1: a CCTV strip tile


def stage() -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in (FIXTURES / "stage_telemetry.jsonl").read_text().splitlines()
    ]


def rescue_msg() -> dict[str, Any]:
    """The first telemetry of layout 1: Bay 1 is in RESCUE and the main feeds are its close-ups."""
    return next(m for m in stage() if m["layout_id"] == 1)


def clock_from(start: int = 0, step: int = 100_000_000) -> Any:
    ticks = iter(range(start, 10**15, step))
    return lambda: next(ticks)


# -- parsing ----------------------------------------------------------------------------------


def test_parse_telemetry_accepts_only_telemetry() -> None:
    assert parse_telemetry(json.dumps(rescue_msg()))
    assert parse_telemetry('{"type": "other"}') is None
    assert parse_telemetry("[1, 2]") is None
    assert parse_telemetry("not json") is None


def test_parse_feeds_of_the_rescue_layout() -> None:
    feeds = parse_feeds(rescue_msg())
    assert feeds[0].camera == "bay1/top_ortho" and feeds[0].bay == 1 and feeds[0].role == "main"
    assert feeds[1].camera == "bay1/side_ortho"
    assert parse_feeds({"feeds": [{"slot": "x"}, 3, {"slot": 1, "camera": "c", "bay": 2}]}) == {
        1: parse_feeds({"feeds": [{"slot": 1, "camera": "c", "bay": 2}]})[1]
    }


def test_split_frame() -> None:
    parts = split_frame(MAIN_FRAME)
    assert parts is not None
    slot, layout_byte, jpeg = parts
    assert (slot, layout_byte) == (0, 1) and jpeg[:2] == b"\xff\xd8"
    assert split_frame(b"\x00\x01") is None
    assert split_frame(b"\x00\x01not a jpeg") is None


def test_mapper_maps_slots_and_drops_stale_layouts() -> None:
    mapper = FeedMapper(clock=clock_from(5))
    assert mapper.on_bytes(MAIN_FRAME) is None  # no telemetry yet: slot meaning unknown
    assert mapper.on_text(json.dumps(rescue_msg())) is not None
    frame = mapper.on_bytes(MAIN_FRAME)
    assert frame is not None
    assert (frame.bay, frame.camera, frame.role) == (1, "bay1/top_ortho", "main")
    assert frame.jpeg == MAIN_FRAME[2:] and frame.t_ns == 5
    stale = bytes((0, 0)) + MAIN_FRAME[2:]  # same slot, previous layout's byte
    assert mapper.on_bytes(stale) is None
    assert mapper.dropped_frames == 2
    mapper.reset()
    assert mapper.on_bytes(MAIN_FRAME) is None


def test_mapper_handles_layout_ids_past_255() -> None:
    mapper = FeedMapper()
    msg = dict(rescue_msg(), layout_id=257)
    mapper.on_text(json.dumps(msg))
    assert mapper.on_bytes(MAIN_FRAME) is not None  # 257 % 256 == 1


# -- client -------------------------------------------------------------------------------------


class FakeSocket:
    """Yields scripted messages. It has no ``send``: the client must never need one."""

    def __init__(self, messages: list[str | bytes]) -> None:
        self._messages = messages

    def __aiter__(self) -> AsyncIterator[str | bytes]:
        return self._gen()

    async def _gen(self) -> AsyncIterator[str | bytes]:
        for m in self._messages:
            await asyncio.sleep(0)
            yield m


class Script:
    """A fake ``connect``: each call takes the next item, a socket's messages or an error."""

    def __init__(
        self, sessions: list[list[str | bytes] | BaseException], client_ref: list[FeedClient]
    ):
        self.sessions = sessions
        self.calls = 0
        self.client_ref = client_ref

    def __call__(self, url: str) -> Any:
        @asynccontextmanager
        async def cm() -> AsyncIterator[Connection]:
            self.calls += 1
            if not self.sessions:
                self.client_ref[0].stop()
                raise OSError("console gone")
            item = self.sessions.pop(0)
            if isinstance(item, BaseException):
                raise item
            yield FakeSocket(item)

        return cm()


def run_client(
    sessions: list[list[str | bytes] | BaseException],
) -> tuple[FeedClient, list[Any], list[float]]:
    got: list[Any] = []
    delays: list[float] = []

    async def sleep(s: float) -> None:
        delays.append(s)

    ref: list[FeedClient] = []
    script = Script(sessions, ref)
    client = FeedClient(
        "ws://test/ws",
        on_telemetry=lambda m: got.append(("t", m["sim_t"])),
        on_frame=lambda f: got.append(("f", f.camera)),
        connect=script,
        sleep=sleep,
    )
    ref.append(client)
    asyncio.run(asyncio.wait_for(client.run(), timeout=5))
    return client, got, delays


def test_client_reads_telemetry_and_frames_then_reconnects_with_backoff() -> None:
    msg = json.dumps(rescue_msg())
    client, got, delays = run_client(
        [
            OSError("refused"),
            OSError("refused"),
            [MAIN_FRAME, msg, MAIN_FRAME, STRIP_FRAME, "garbage"],
            TimeoutError("open timeout"),
            [MAIN_FRAME, msg, MAIN_FRAME],  # after reconnect the layout must be learned again
        ]
    )
    sim_t = rescue_msg()["sim_t"]
    strip = parse_feeds(rescue_msg())[2].camera
    assert got == [
        ("t", sim_t),
        ("f", "bay1/top_ortho"),
        ("f", strip),
        ("t", sim_t),
        ("f", "bay1/top_ortho"),
    ]
    # 1 s, doubling, reset after each successful connect; the last attempt calls stop().
    assert delays == [1.0, 2.0, 1.0, 2.0, 1.0]
    assert client.sessions == 2 and client.telemetry_count == 2 and not client.connected
    assert client.mapper.dropped_frames == 2  # the first frame of each session


def test_backoff_is_capped_at_5_s() -> None:
    _, _, delays = run_client([OSError("x")] * 6)
    assert delays == [1.0, 2.0, 4.0, 5.0, 5.0, 5.0]


def test_client_lets_unexpected_errors_through() -> None:
    with pytest.raises(ValueError):
        run_client([ValueError("bug")])


# -- frame store ----------------------------------------------------------------------------------


def frame(bay: int, camera: str, t_s: float, jpeg: bytes = b"\xff\xd8jpeg") -> CameraFrame:
    return CameraFrame(bay, camera, "main", jpeg, round(t_s * 1e9))


def telemetry(state: str, clutch: bool | None, sim_t: float = 1.0) -> dict[str, Any]:
    rescue = None if clutch is None else {"bay": 1, "clutch": clutch, "elapsed_s": 0.0}
    return {
        "type": "telemetry",
        "sim_t": sim_t,
        "rescue": rescue,
        "bays": [{"id": 1, "state": state}, {"id": 2, "state": "NOMINAL"}],
    }


def test_store_keeps_the_latest_frame_per_camera() -> None:
    store = FrameStore()
    store.on_telemetry(telemetry("NOMINAL", None, sim_t=3.5))
    store.add(frame(1, "bay1/cctv", 1.0, b"\xff\xd8old"))
    store.add(frame(1, "bay1/cctv", 2.0, b"\xff\xd8new"))
    store.add(frame(1, "bay1/top_ortho", 1.5))
    latest = store.latest(1, "bay1/cctv")
    assert latest is not None and latest.jpeg == b"\xff\xd8new" and latest.sim_t == 3.5
    newest = store.latest(1)
    assert newest is not None and newest.camera == "bay1/cctv"
    assert store.latest(2) is None and store.latest(1, "bay1/wrist") is None
    assert len(latest.sha256) == 16 and "jpeg" not in repr(latest)
    assert store.rescue_frames(1) == []  # not in a rescue: nothing recorded


def test_store_records_the_clutch_hold_sampled_and_capped() -> None:
    store = FrameStore(ring_frames=12, sample_interval_s=0.5)
    store.on_telemetry(telemetry("NOMINAL", None))
    store.on_telemetry(telemetry("RESCUE", False))
    store.add(frame(1, "bay1/top_ortho", 0.0))  # clutch not held yet
    assert store.rescue_frames(1) == [] and not store.is_recording(1)
    store.on_telemetry(telemetry("RESCUE", True))
    assert store.is_recording(1)
    for i in range(40):  # 10 Hz for 4 s on two cameras
        t = 1.0 + i * 0.1
        store.add(frame(1, "bay1/top_ortho", t))
        store.add(frame(1, "bay1/side_ortho", t))
        store.add(frame(2, "bay2/cctv", t))  # another bay: never recorded
    ring = store.rescue_frames(1)
    assert len(ring) == 12
    times = [f.t_ns for f in ring if f.camera == "bay1/top_ortho"]
    assert all(b - a >= 500_000_000 for a, b in zip(times, times[1:], strict=False))
    store.on_telemetry(telemetry("VALIDATING", False))  # released: kept for the review
    store.add(frame(1, "bay1/top_ortho", 9.0))
    assert len(store.rescue_frames(1)) == 12 and not store.is_recording(1)
    store.on_telemetry(telemetry("NOMINAL", None))
    store.on_telemetry(telemetry("RESCUE", False))  # the next stall clears the old rescue
    assert store.rescue_frames(1) == []


def test_store_with_real_frames_from_the_feed() -> None:
    mapper = FeedMapper(clock=clock_from())
    store = FrameStore(sample_interval_s=0.0)
    msg = rescue_msg()
    held = dict(msg, rescue={"bay": 1, "clutch": True, "elapsed_s": 1.0})
    for m in (msg, held):
        mapper.on_text(json.dumps(m))
        store.on_telemetry(m)
    for _ in range(3):
        f = mapper.on_bytes(MAIN_FRAME)
        assert f is not None
        store.add(f)
    ring = store.rescue_frames(1)
    assert len(ring) == 3 and all(f.camera == "bay1/top_ortho" for f in ring)


def test_downscale_real_and_large_frames() -> None:
    real = MAIN_FRAME[2:]
    with Image.open(io.BytesIO(real)) as img:
        size = img.size
    small = downscale_jpeg(real, max_side_px=128)
    with Image.open(io.BytesIO(small)) as img:
        assert max(img.size) == 128
        assert abs(img.size[0] / img.size[1] - size[0] / size[1]) < 0.05
    assert downscale_jpeg(real, max_side_px=max(size)) is real  # already fits: untouched
