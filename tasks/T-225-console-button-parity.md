---
id: T-225
title: Physical UNO Q clutch and E-stop buttons behave like the keyboard (SPACE, E)
status: todo
owner: null
branch: null
depends_on: []
plan: docs/hardware/console/README.md
claimed_at: null
---

The dashboard's keyboard path (SPACE = clutch, E = E-stop) works; the physical console buttons
(`CLUTCH:<0|1>`, `ESTOP:<0|1>`, `HB:...`) do not behave the same.

- [ ] Physical button presses reach the operator displays (action log, toasts) like key presses
- [ ] A bay cut by either E-stop (key or button) is restored by the other one
- [ ] The keyboard E-stop and clutch act on the bay the dashboard shows as selected
- [ ] Joystick and keyboard bay selections compare on one clock (board timestamps restamped)
- [ ] A held physical clutch survives one late network poll; bay-state sends never delay a poll
- [ ] Full verify passes
