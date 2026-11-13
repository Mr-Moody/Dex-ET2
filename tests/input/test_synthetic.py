import time

from teleop.input.synthetic import SyntheticSource
from teleop.types import OPENXR_JOINTS


def test_synthetic_produces_both_hands():
    src = SyntheticSource()
    src.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            fr = src.latest()
            if fr.left is not None and fr.right is not None:
                break
            time.sleep(0.02)
        time.sleep(0.2)
        fr = src.latest()
        assert fr.left is not None and fr.right is not None
        assert set(fr.left.joints) <= set(OPENXR_JOINTS)
        assert fr.left.wrist is not None and fr.left.wrist.position is not None
        assert src.status().frames > 0
    finally:
        t0 = time.monotonic()
        src.stop()
        assert time.monotonic() - t0 < 1.0
