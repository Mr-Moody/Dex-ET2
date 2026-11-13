import draccus

from teleop.config import SimConfig
from teleop.latency import LatencyStats


def test_latency_summary_percentiles() -> None:
    s = LatencyStats()
    for ms in range(1, 101):
        s.add(ms * 1_000_000)
    out = s.summary()
    assert abs(out["p50_ms"] - 50.5) <= 1
    assert out["max_ms"] == 100
    assert out["n"] == 100


def test_latency_empty() -> None:
    assert LatencyStats().summary() == {"p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "n": 0.0}


def test_config_defaults_parse() -> None:
    cfg = draccus.parse(SimConfig, args=[])
    assert cfg.control_hz == 60
    assert cfg.studio.port == 14043
    assert len(cfg.cameras) == 1
