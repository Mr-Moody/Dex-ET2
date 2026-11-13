# Rokoko Studio Custom Streaming, JSON v3 (observed)

Observed from a real 40 s capture (`coil_on`: both gloves, Coil Pro ON, 1203 datagrams, ~9.9 KB each,
plain JSON, one scene per UDP datagram). Trimmed fixture: `tests/fixtures/studio/coil_on.jsonl`.
Capture with `teleop capture --port 14043 --seconds 30 --out data/raw/<name>.jsonl`.

Status of the other captures: `coil_off` and `lz4` are not captured yet. Their tests skip until
`tests/fixtures/studio/{coil_off,lz4}.jsonl` exist. TODO: record whether Coil Pro off still emits
positions and how LZ4 is framed, then update the parser if needed.

## Top level

```
{"version": "3,0", "fps": 30.0,
 "scene": {"timestamp": <float seconds>, "actors": [...], "newtons": [], "props": [], "characters": []}}
```

- `version` is the string `"3,0"` (comma). `fps` is 30.0: the stream is 30 Hz, not the glove rate.
- `scene.timestamp` is seconds (float), monotonic in this capture (0.035 s .. 40.05 s). It restarts
  or can jump when Studio restarts, so the pipeline never relies on it for ordering.

## Actor

`name` ("newton"), `color` [r,g,b], `meta` {hasGloves, hasLeftGlove, hasRightGlove, hasBody (false),
hasFace}, `dimensions` {totalHeight, hipHeight}, `body`.

`body` has 63 bones keyed in camelCase, each `{"position": {x,y,z}, "rotation": {x,y,z,w}}`:

- Body: `hip spine chest neck head`, `{left|right}{Shoulder UpperArm LowerArm Hand UpLeg Leg Foot Toe ToeEnd}`.
- Fingers: `{left|right}{Thumb|Index|Middle|Ring|Little}{Proximal|Medial|Distal|Tip}` (4 per finger).

## Units and frame

- Positions are metres, world space. Y is up (hip at y = 0.85). Hand positions vary by about 0.5 m over
  the take with the Coil Pro on, so the Coil Pro does provide world positions.
- Rotations are unit quaternions in `x,y,z,w` order and are **world/global**, not local: the
  index proximal bone's world angle swings 107..177 degrees as the hand rotates, while its angle
  relative to the hand stays in 7..78.

## Mapping to `HandFrame`

`teleop.input.studio.JOINT_MAP` is the explicit name table; unknown bones are ignored.

| Studio | OpenXR (`OPENXR_JOINTS`) |
| --- | --- |
| `{side}Hand` | `wrist` and `palm` (same rotation) |
| `{side}{Index,Middle,Ring,Little}Proximal` | `*_proximal` |
| `...Medial` | `*_intermediate` |
| `...Distal` | `*_distal` |
| `...Tip` | `*_tip` |
| `{side}ThumbProximal` | `thumb_metacarpal` |
| `{side}ThumbMedial` | `thumb_proximal` |
| `{side}ThumbDistal` | `thumb_distal` |
| `{side}ThumbTip` | `thumb_tip` |

Finger metacarpals have no Studio bone and are omitted. Thumb justification: Studio's thumb
(as in Unity's humanoid skeleton) has three bones plus a tip, the first one hanging directly off the
hand at the carpometacarpal joint. OpenXR names that bone `thumb_metacarpal`, so the thumb is shifted
one slot relative to the fingers; matching by suffix would leave `thumb_metacarpal` empty and
mislabel every thumb joint.

`HandFrame.joints` are LOCAL rotations: `q_local = conj(q_parent_world) * q_child_world`. Parents:
Hand -> Proximal -> Medial -> Distal -> Tip. `wrist` and `palm` are the hand relative to
`{side}LowerArm`. These are raw bone-relative rotations: Studio's rest pose is not identity, so a flat
hand is not zero. Calibration/retargeting must subtract the rest pose, and left-hand axes may be mirrored.

`HandFrame.wrist` is the hand's world pose in Z-up: Studio (x, y, z) -> (x, -z, y) (a +90 degree
rotation about X, right-handed) applied to the position, and the same conjugation to the orientation
quaternion ((x, y, z, w) -> (x, -z, y, w)). `position` is None if the bone has no `position` key.
`device_time_us = int(scene.timestamp * 1e6)`, `seq` is a per-parser counter, `source = "studio"`.
The actor is the one named by `StudioConfig.actor`, else the first with `meta.hasGloves`.

## LZ4

Payloads not starting with `{` are tried as LZ4 frame, then LZ4 block (4-byte LE size prefix, as in
python-lz4), then raw block. The real framing from Studio is unverified until the `lz4` capture exists.
