from teleop.sim.loop import FixedRateLoop


class FakeTime:
    def __init__(self) -> None:
        self.now = 0

    def clock(self) -> int:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += round(s * 1e9)


def test_loop_absolute_deadlines_no_drift():
    t = FakeTime()
    work_ns = 3_000_000

    def step(i: int) -> bool:
        t.now += work_ns  # per-step work must not accumulate as drift
        return True

    stats = FixedRateLoop(60, t.clock, t.sleep).run(step, max_steps=60)
    assert stats.steps == 60
    assert stats.overruns == 0
    assert abs(t.now - 1_000_000_000) <= 1_000_000


def test_loop_skips_when_late():
    t = FakeTime()
    calls: list[int] = []
    period = 1_000_000_000 // 60

    def step(i: int) -> bool:
        calls.append(t.now)
        if i == 2:
            t.now += 3 * period
        return True

    stats = FixedRateLoop(60, t.clock, t.sleep).run(step, max_steps=6)
    assert stats.overruns == 1
    gaps = [b - a for a, b in zip(calls, calls[1:], strict=False)]
    assert all(g >= period - 1000 for g in gaps)  # no catch-up burst
