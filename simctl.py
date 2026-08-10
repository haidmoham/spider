"""Send one command to a running interact.py process."""

from __future__ import annotations

import argparse
import json
import socket
import sys

from interact import HOST, PORT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("state")
    step = commands.add_parser("step")
    step.add_argument("n", nargs="?", type=int, default=1)
    run = commands.add_parser("run")
    run.add_argument("seconds", type=float)
    commands.add_parser("reset")
    power = commands.add_parser("power")
    power.add_argument("front", type=float)
    power.add_argument("middle", type=float)
    power.add_argument("rear", type=float)
    args = parser.parse_args()
    request = vars(args)
    if args.command == "power":
        request["values"] = [request.pop("front"), request.pop("middle"), request.pop("rear")]

    try:
        with socket.create_connection((HOST, PORT), timeout=3) as connection:
            connection.sendall((json.dumps(request) + "\n").encode())
            response = connection.makefile("rb").readline()
    except ConnectionRefusedError:
        print("interact.py is not running. Start it with: python interact.py", file=sys.stderr)
        raise SystemExit(1)
    except OSError as error:
        print(f"Could not reach interact.py: {error}", file=sys.stderr)
        raise SystemExit(1)

    result = json.loads(response)
    print(json.dumps(result, indent=2))
    if "error" in result:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
