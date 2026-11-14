"""Console feed layout: which camera goes in which slot."""

from __future__ import annotations

from teleop.console.layout import MAIN, STRIP, TILE, feed_layout


def test_grid_layout() -> None:
    f = feed_layout([1, 2, 3, 4], None)
    assert [(s.slot, s.camera, s.bay, s.role) for s in f] == [
        (0, "bay1/cctv", 1, "tile"),
        (1, "bay2/cctv", 2, "tile"),
        (2, "bay3/cctv", 3, "tile"),
        (3, "bay4/cctv", 4, "tile"),
    ]
    assert all((s.width, s.height) == TILE for s in f)


def test_grid_layout_sorts_and_skips_absent_bays() -> None:
    f = feed_layout([2, 1], None)
    assert [(s.slot, s.camera) for s in f] == [(0, "bay1/cctv"), (1, "bay2/cctv")]


def test_rescue_layout() -> None:
    f = feed_layout([1, 2, 3, 4], 2)
    assert (f[0].slot, f[0].camera, f[0].role) == (0, "bay2/top_ortho", "main")
    assert (f[1].slot, f[1].camera, f[1].role) == (1, "bay2/side_ortho", "main")
    assert (f[0].width, f[0].height) == MAIN
    assert [s.camera for s in f[2:]] == [f"bay{k}/cctv" for k in (1, 2, 3, 4)]
    assert [s.slot for s in f[2:]] == [2, 3, 4, 5]
    assert all(s.role == "strip" and (s.width, s.height) == STRIP for s in f[2:])


def test_rescue_of_absent_bay_falls_back_to_grid() -> None:
    assert feed_layout([1, 2], 3) == feed_layout([1, 2], None)
