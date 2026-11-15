---
id: T-225
title: Physical UNO Q clutch and E-stop buttons behave like the keyboard (SPACE, E)
status: claimed
owner: console-button-parity
branch: fix/console-button-parity
depends_on: []
plan: docs/hardware/console/README.md
claimed_at: '2026-11-14T22:02:19+00:00'
---

The dashboard's keyboard path (SPACE = clutch, E = E-stop) works; the physical console buttons
(`CLUTCH:<0|1>`, `ESTOP:<0|1>`, `HB:...`) do not behave the same.

- [x] Physical button presses reach the operator displays (action log, toasts) like key presses
- [x] A bay cut by either E-stop (key or button) is restored by the other one
- [x] The keyboard E-stop and clutch act on the bay the dashboard shows as selected
- [x] Joystick and keyboard bay selections compare on one clock (board timestamps restamped)
- [x] A held physical clutch survives one late network poll; bay-state sends never delay a poll
- [ ] Full verify passes (handed to the orchestrator)
