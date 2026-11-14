"""App Lab entry point: run the UNO Q edge service with the MCU console attached.

The ``teleop`` package must be importable on the board (see ../README.md).
"""

import sys
from pathlib import Path

# Allow a copy of the ``teleop`` package placed next to this file (see README).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from teleop.commands.unoq import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["--serve"]))
