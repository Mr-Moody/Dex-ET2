"""``teleop`` console entry point: dispatches subcommands to ``teleop.commands.*`` modules."""

from __future__ import annotations

import argparse
import importlib
import sys

# subcommand -> module path. Each module exposes ``main(argv: list[str]) -> int``.
# Pre-populated so parallel tasks add command modules without editing this file.
COMMANDS: dict[str, str] = {
    "hotspot": "teleop.commands.hotspot",
    "capture": "teleop.commands.capture",
    "sim": "teleop.commands.sim",
    "calibrate": "teleop.commands.calibrate",
    "record": "teleop.commands.record",
    "replay": "teleop.commands.replay",
    "process": "teleop.commands.process",
    "eval": "teleop.commands.eval",
    "atc": "teleop.commands.atc",
    "unoq": "teleop.commands.unoq",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="teleop", description="Glove teleoperation booth")
    parser.add_argument("command", help="one of: " + ", ".join(COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER, help="forwarded to the command")
    ns = parser.parse_args(sys.argv[1:] if argv is None else argv)

    module_path = COMMANDS.get(ns.command)
    if module_path is None:
        print(
            f"teleop: unknown command {ns.command!r}; choose from: {', '.join(COMMANDS)}",
            file=sys.stderr,
        )
        return 2
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        # exc.name may be the command module or a missing parent package of it
        if exc.name is not None and (
            module_path == exc.name or module_path.startswith(exc.name + ".")
        ):
            print(f"teleop: command {ns.command!r} is not implemented yet", file=sys.stderr)
            return 3
        raise
    return int(module.main(list(ns.args)))


if __name__ == "__main__":
    raise SystemExit(main())
