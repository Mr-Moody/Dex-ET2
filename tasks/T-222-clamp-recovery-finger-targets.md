---
id: T-222
title: Clamp recovery finger targets to the hand's ctrlrange so DMP episodes pass QC
status: done
owner: null
branch: null
depends_on: []
plan: null
claimed_at: null
---

Found by an end-to-end run (`teleop atc --headless --script --record-episodes`, then
`teleop process`): the DMP replay's `thumb_yaw` overshoots its 0 target to -0.0053 rad on 23 of
112 frames. MuJoCo clamps `ctrl` to the ctrlrange, so the robot never sees it, but the episode
records the unclamped value and ATC QC drops every DMP episode as `action out of range`.

- [x] Recovery finger targets (DMP, hybrid, ACT) are clamped to the scene's hand ctrlrange
  before they are recorded and applied
- [x] `dmp` physics is unchanged (MuJoCo already clamps `ctrl`); the stage golden still passes
- [x] End-to-end: the demo's DMP episode passes `teleop process`
- [x] Full verify passes
