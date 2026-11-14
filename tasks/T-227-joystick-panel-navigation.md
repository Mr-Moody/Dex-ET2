---
id: T-227
title: Joystick and arrow keys move between the camera panels instead of jumping to fixed bays
status: done
owner: null
branch: null
depends_on: []
plan: null
claimed_at: null
---

Operators asked for the console joystick to navigate between the dashboard's camera panels.
Each stick direction used to pick one fixed bay (left 1, right 2, up 3, down 4).

- [ ] The stick moves the selection one panel in its direction on the 2x2 camera grid
  (Bay 1 | Bay 2 over Bay 3 | Bay 4), stopping at the edges; with nothing selected yet it
  selects Bay 1. The firmware (both sketch copies) and `SoftwareMcu` share this behaviour, so
  the MCU's E-stop and clutch gate keep acting on the panel the operator moved to
- [ ] The joystick map and calibration name directions (left, right, up, down), not bays; the
  stored 4-digit map format is unchanged
- [ ] Arrow keys (stage, dashboard key bar, browser) navigate the same way; digits `1`-`4`
  still pick a bay directly, and clicking a feed selects its bay with its digit
- [ ] Hints and docs updated (console README, wiring page, key bar)
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`
