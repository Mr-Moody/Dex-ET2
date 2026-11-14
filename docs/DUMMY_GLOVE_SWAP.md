# Temporary Dummy Glove Process — Swap-Out Note

> **Status:** Active temporary stand-in while physical Rokoko Smartgloves II are not connected (`TODO(rokoko-gloves-swap)`).

## What is currently running as a stand-in
1. **Standalone UDP Streamer Process (`scripts/dummy_glove_process.py` + `src/teleop/input/dummy_glove.py`):**
   - Runs as a separate OS process streaming **Rokoko Studio JSON v3 UDP datagrams** at `60 Hz` to `127.0.0.1:14043` so `teleop atc --source studio` exercises the real `StudioSource` $\rightarrow$ `AutoWrist` (`CoilWrist`) $\rightarrow$ `DirectRetargeter` $\rightarrow$ `GloveOperator` pipeline end-to-end without hardware.
   - Accepts live control datagrams on `127.0.0.1:14044` (`DummyControlClient`).
2. **UI Manual Override + Auto-Rescue Co-Pilot (`DummyOperatorSupervisor` + `Dummy Glove (TEMP)` panel in `src/teleop/console/web/`):**
   - **Manual UI Teleop:** 2D latch-frame wrist drag pad (`X` bolt slide / `Z` height), lateral `Y` & finger sliders (`Thumb flap`, `Index hook`, `Grip`), plus a 1-click `Guided Solve` helper.
   - **Auto-Rescue Co-Pilot:** Automatically selects a queued rescue bay, toggles the clutch, drives the latch-frame rescue over the dummy UDP stream, and releases the clutch to trigger UNO Q DMP fitting + the 3-rollout MuJoCo gate.
   - **Auto-Inject Jams:** Optional non-scripted periodic jam generator so the fleet cycles and learns without hardcoded stage-script keys.

## How to swap in the real Rokoko Smartgloves later
1. Connect the Rokoko Smartgloves II +Rokoko Studio and enable **Custom Streaming $\rightarrow$ JSON v3** to `<this-laptop-ip>:14043`.
2. Calibrate if needed: `uv run teleop calibrate --out outputs/calib.json`.
3. Launch `teleop atc` **without** `--dummy-glove` (and stop any standalone `scripts/dummy_glove_process.py`):
   ```bash
   uv run teleop atc --source studio --calibration outputs/calib.json
   ```
4. All temporary hooks are tagged with `TODO(rokoko-gloves-swap)` in:
   - `src/teleop/input/dummy_glove.py`
   - `scripts/dummy_glove_process.py`
   - `src/teleop/commands/atc.py`
   - `src/teleop/console/run.py`
   - `src/teleop/console/web/console.js` / `index.html`
