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


def test_pin_focus_and_camera_selection() -> None:
    from teleop.console.layout import ViewState

    pinned = feed_layout([1, 2, 3, 4], None, ViewState(mode="pin", pinned_bay=3))
    assert (pinned[0].slot, pinned[0].camera, pinned[0].bay, pinned[0].role) == (
        0,
        "bay3/cctv",
        3,
        "main",
    )
    assert [s.camera for s in pinned[1:]] == [f"bay{k}/cctv" for k in (1, 2, 4)]

    split = feed_layout(
        [1, 2, 3, 4], None, ViewState(mode="pin", pinned_bay=4, cameras={4: "split"})
    )
    assert [(s.slot, s.camera, s.role) for s in split[:2]] == [
        (0, "bay4/top_ortho", "main"),
        (1, "bay4/side_ortho", "main"),
    ]

    wrist_split = feed_layout(
        [1, 2, 3, 4], None, ViewState(mode="pin", pinned_bay=1, cameras={1: "wrist_split"})
    )
    assert [(s.slot, s.camera, s.role) for s in wrist_split[:2]] == [
        (0, "bay1/wrist_cam", "main"),
        (1, "bay1/top_ortho", "main"),
    ]


def test_force_equal_grid_and_custom_bay_order() -> None:
    from teleop.console.layout import ViewState

    vs = ViewState(mode="grid", cameras={1: "top_ortho"}, bay_order=(3, 1, 4, 2))
    # Even during an active rescue on Bay 1, mode="grid" keeps 4 equal tiles in bay_order.
    f = feed_layout([1, 2, 3, 4], 1, vs)
    assert [(s.slot, s.camera, s.bay, s.role) for s in f] == [
        (0, "bay3/cctv", 3, "tile"),
        (1, "bay1/top_ortho", 1, "tile"),
        (2, "bay4/cctv", 4, "tile"),
        (3, "bay2/cctv", 2, "tile"),
    ]


def test_warehouse_overview_layout() -> None:
    from teleop.console.layout import ViewState

    vs = ViewState(mode="overview", zooms={"overview": (1.5, 0.1, -0.1, 0.2, 0.0)})
    f = feed_layout([1, 2, 3, 4], None, vs)
    assert (f[0].slot, f[0].camera, f[0].bay, f[0].role) == (0, "overview", 0, "main")
    assert (f[0].width, f[0].height) == MAIN
    assert f[0].zoom == 1.5
    assert [(s.slot, s.camera, s.bay, s.role) for s in f[1:]] == [
        (1, "bay1/cctv", 1, "strip"),
        (2, "bay2/cctv", 2, "strip"),
        (3, "bay3/cctv", 3, "strip"),
        (4, "bay4/cctv", 4, "strip"),
    ]
