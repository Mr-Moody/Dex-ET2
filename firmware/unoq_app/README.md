# UNO Q booth console service (App Lab app)

Runs `teleop unoq --serve` on the UNO Q Linux side. It attaches the MCU operator console through
`arduino-router` (`RouterBridgeTransport`), so the laptop sees real buttons and a heartbeat and
can drive the bay LEDs (`set_bay_state` op). The sketch is a copy of
`firmware/unoq_console/unoq_console.ino`; keep the two in sync.

## Layout

    unoq_app/
      app.yaml            manifest (name, icon, description, ports)
      python/main.py      calls teleop.commands.unoq.main(["--serve"])
      python/requirements.txt
      sketch/sketch.ino   console firmware (copy)
      sketch/sketch.yaml  board profile and libraries

## Getting `teleop` onto the board

`main.py` imports `teleop`. Pick one; neither was tested on hardware:

1. Copy `src/teleop/` into the app's `python/` folder (`main.py` puts its own folder on
   `sys.path`). The edge service only needs the stdlib plus `numpy` (in `requirements.txt`);
   it does not import mujoco or torch.
2. Or, over SSH on the board, `pip install --no-deps` a wheel built with `uv build`, into the
   interpreter App Lab uses for the app.

## Deploy

1. Copy this folder to the board's App Lab apps directory, or import it in Arduino App Lab
   (connected over USB or network).
2. Open the app in App Lab and press Run. App Lab compiles and flashes the sketch and starts
   `python/main.py`.

## Run at startup

In App Lab, open the app's settings and enable "Run at startup" (per the Waveshare guide
below). Reboot the board to confirm the service comes back.

## Check from the laptop (manual, hardware)

    uv run teleop unoq --host <board-ip>

Expect `unoq: unoq-qrb2210 answered via <board-ip>:15052`, not "local loopback". Then press a bay
button and hold the clutch on the console and confirm the booth console follows. Not automated;
needs the board.

## Sources and unconfirmed details

Layout (`app.yaml`, mandatory `python/main.py`, optional `python/requirements.txt`, optional
`sketch/sketch.ino` plus `sketch/sketch.yaml`, `ports`, `icon`, `name`, `description`) follows
the arduino-app-cli app-spec PR: https://github.com/arduino/arduino-app-cli/pull/67/files
"Run at startup" is from https://docs.waveshare.com/Arduino-UNO-Q-02016/App-Lab-Usage

Not confirmed, please verify on the board:

- Where the "Run at startup" setting is stored. Neither source says it is an `app.yaml` key, so
  it is documented as a UI setting only and `app.yaml` does not set it.
- The exact `sketch.yaml` schema: the `fqbn`, `platforms` and `libraries` values are from
  memory of Arduino sketch profiles, not from a fetched UNO Q example.
- Whether App Lab's Python environment can import a copied `teleop` package or a `--no-deps`
  wheel, and which interpreter version it uses (teleop needs Python 3.12).
- Whether App Lab's app directory path and the Docker/venv isolation allow binding ports
  15051/15052 on the LAN (`ports:` is declared in `app.yaml`).
- Whether `sketch.ino` needs changes beyond the copy (it already uses `Arduino_RouterBridge`).
