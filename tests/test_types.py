import dataclasses

import pytest

from teleop.types import OPENXR_JOINTS, HandFrame, Pose


def test_openxr_joints_len_and_order() -> None:
    assert len(OPENXR_JOINTS) == 26
    assert OPENXR_JOINTS[0] == "palm"
    assert OPENXR_JOINTS[1] == "wrist"
    assert OPENXR_JOINTS[-1] == "little_tip"


def test_frames_are_frozen() -> None:
    f = HandFrame(side="left", host_time_ns=0, seq=0, source="t", joints={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.seq = 1  # pyright: ignore[reportAttributeAccessIssue]
    p = Pose(orientation=(0.0, 0.0, 0.0, 1.0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.position = (0.0, 0.0, 0.0)  # pyright: ignore[reportAttributeAccessIssue]
