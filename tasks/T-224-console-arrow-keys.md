---
id: T-224
title: Arrow keys select bays; feed controls ignore double-click focus
status: claimed
owner: claude-console-arrow-keys
branch: feat/console-arrow-keys
depends_on: []
plan: null
claimed_at: '2026-11-14T22:03:34+00:00'
---

Operators asked for bay select to match the physical joystick, and for double clicks on a feed's
buttons or overlay text to stop pinning the pane.

- [ ] Arrow keys select bays: Left 1, Right 2, Up 3, Down 4 (GLFW 263/262/265/264) in the stage
  `KEYMAP`, the console key bar, the browser key handler and the hints; `1`-`4` stay a silent alias
- [ ] Arrows never scroll the page and are left alone inside inputs, sliders and the wrist pad
- [ ] Joystick default map (host and firmware) matches: left 1, right 2, up 3, down 4
- [ ] A double click on a feed's buttons, zoom badge or overlay text never pins/unpins the pane
  and never selects text; only the bare picture pins
- [ ] Touched UI follows the Vercel Web Interface Guidelines (named icon buttons, hit targets,
  `touch-action: manipulation`, `<kbd>` caps with accessible names)
- [ ] Full verify passes
