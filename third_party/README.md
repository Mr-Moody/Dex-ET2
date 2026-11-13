# third_party

Models here are fetched by `uv run python scripts/fetch_models.py` (idempotent, commit-pinned,
needs only `git`). Everything except this README is gitignored; nothing is redistributed in-repo.

| Dir | Source | Pinned commit | Licence |
| --- | --- | --- | --- |
| `menagerie/` (sparse: `leap_hand`) | https://github.com/google-deepmind/mujoco_menagerie | `370c43965323c0944b433b202aac5058b5f907a9` | MIT (Ananye Agarwal 2023, `leap_hand/LICENSE`) |
| `inspire/` (sparse: `inspire/` assets + MJCF) | https://github.com/correlllab/h1_mujoco | `fa8795a1ce1c2be418e8d87b72761c3a452efcd6` | MIT (Correll Robotics Lab, `LICENSE`) |

## Inspire RH56DFX provenance

arXiv 2603.08988 ("Characterization, Analytical Planning, and Hybrid Force Control for the
Inspire RH56DFX Hand") points to https://correlllab.github.io/rh56dfx.html and from there to
https://github.com/correlllab/rh56_controller (MIT). That repo's `h1_mujoco` submodule
(`fa8795a`) holds `inspire/inspire_left.xml` and `inspire/inspire_right.xml`. Per the paper repo,
hand geometry is derived from the community URDF by Omkar Kshirsagar
(https://github.com/ookkshirsagar/rh56dfx_description); meshes originate from the vendor model.
The MIT licence covers the Correll repo; the upstream mesh licence was not stated, so we only
fetch, never commit, them.

The MJCFs already ship 6 `position` actuators (`pinky, ring, middle, index, thumb_proximal,
thumb_yaw`) and `polycoef` joint equality constraints coupling the distal joints, so no
MjSpec modifications were needed. Left and right are separate files (`mirror_left: false`).
Both files also contain a skybox texture, light and keyframes, which the bimanual scene
builder (T-021) must be prepared to drop when attaching.

## LEAP

Menagerie `leap_hand` ships `left_hand.xml` and `right_hand.xml` (16 position actuators,
3 fingers + thumb, no little finger). Menagerie is MIT as a whole; each model keeps its own
licence file.
