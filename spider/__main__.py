"""Run C-1N, send a live command, replay a recording, or render the model."""

import argparse
import json
from pathlib import Path
import socket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    run = actions.add_parser("run", help="live or headless simulation")
    run.add_argument("--headless", action="store_true")
    run.add_argument("--seconds", type=float, default=1.0)
    run.add_argument("--experiment", choices=("none", "stand", "shuffle"), default="stand")
    run.add_argument("--trace", type=Path)
    run.add_argument("--shove-suite", type=Path, metavar="DIRECTORY")
    command = actions.add_parser("command", help="send a command to the live viewer")
    commands = command.add_subparsers(dest="command", required=True)
    commands.add_parser("state")
    commands.add_parser("reset")
    commands.add_parser("step").add_argument("n", type=int, nargs="?", default=1)
    commands.add_parser("run").add_argument("seconds", type=float)
    commands.add_parser("power").add_argument("values", type=float, nargs=3)
    perturb = commands.add_parser("perturb")
    perturb.add_argument("force_n", type=float, nargs=3)
    perturb.add_argument("--seconds", type=float, required=True)
    replay = actions.add_parser("replay", help="play saved states without rerunning physics")
    replay.add_argument("directory", type=Path)
    replay.add_argument("--speed", type=float, default=1.0)
    replay.add_argument("--actuator", default="")
    grid = actions.add_parser("replay-grid", help="four recorded treatments in one window")
    grid.add_argument("directories", type=Path, nargs=4)
    grid.add_argument("--speed", type=float, default=0.5)
    grid.add_argument("--ready", type=Path)
    grid.add_argument("--screenshot", type=Path)
    grid.add_argument("--frames", type=int)
    grid.add_argument("--presentation", choices=("original", "stalk"), default="original")
    policy = actions.add_parser("policy", help="execute a saved PPO policy; no training or notebook dependency")
    policy.add_argument("--checkpoint", type=Path)
    policy.add_argument("--treatment", choices=("baseline", "lower", "smooth", "stalk", "stride"), default="baseline")
    policy.add_argument("--compare", action="store_true", help="record baseline/lower/smooth/stalk on one seed")
    policy.add_argument("--seed", type=int, default=201)
    policy.add_argument("--sampled", action="store_true", help="use saved Gaussian exploration; default is mean action")
    policy.add_argument("--seconds", type=float, default=5.0)
    policy.add_argument("--output", type=Path)
    policy.add_argument("--headless", action="store_true", help="record only; do not open replay")
    policy.add_argument("--presentation", choices=("original", "stalk"), default="stalk")
    tune = actions.add_parser("tune-rate", help="run three independent 10-update PPO command-rate trials")
    tune.add_argument("--output", type=Path, required=True)
    train_stride = actions.add_parser("train-stride", help="run the approved three-seed, 50-update stride round")
    train_stride.add_argument("--output", type=Path, required=True)
    render = actions.add_parser("render", help="render matched model views")
    render.add_argument("--output", type=Path, default=Path("artifacts/c1n_redesign"))
    render.add_argument("--before-directory", type=Path)
    args = parser.parse_args()
    if args.action == "run":
        from .runtime import run_headless, run_shove_suite

        if args.seconds <= 0:
            parser.error("--seconds must be greater than zero")
        if args.headless:
            result = (
                run_shove_suite(args.seconds, args.shove_suite)
                if args.shove_suite
                else run_headless(args.seconds, args.experiment, args.trace)
            )
            print(json.dumps(result, indent=2))
        else:
            from .viewing.live import run_viewer, run_shove_suite_viewer

            if args.shove_suite:
                run_shove_suite_viewer(args.seconds, args.shove_suite)
            else:
                run_viewer(args.experiment)
    elif args.action == "command":
        from .runtime import HOST, PORT

        request = vars(args).copy()
        request.pop("action")
        try:
            with socket.create_connection((HOST, PORT), timeout=3) as connection:
                connection.sendall((json.dumps(request) + "\n").encode())
                with connection.makefile("rb") as response:
                    result = json.loads(response.readline())
        except OSError as error:
            parser.exit(1, f"Cannot reach the viewer: {error}\nStart it with python -m spider run\n")
        print(json.dumps(result, indent=2))
        if "error" in result:
            raise SystemExit(1)
    elif args.action == "replay":
        from .viewing.replay import replay
        import math

        if not math.isfinite(args.speed) or args.speed <= 0:
            parser.error("--speed must be finite and greater than zero")
        replay(args.directory, args.speed, args.actuator)
    elif args.action == "replay-grid":
        from .viewing.replay_grid import replay_grid

        replay_grid(args.directories, args.speed, ready=args.ready,
                    screenshot=args.screenshot, max_frames=args.frames,
                    presentation=args.presentation)
    elif args.action == "policy":
        from datetime import datetime, timezone
        from .policy_run import DEFAULT_CHECKPOINT, compare_policies, run_policy
        from .simulation import ROOT

        checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
        directory = args.output or ROOT / "telemetry/policy" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        try:
            if args.compare:
                if args.treatment != "baseline":
                    parser.error("--compare selects all four treatments; omit --treatment")
                result = compare_policies(checkpoint, directory=directory, seed=args.seed,
                    sampled=args.sampled, seconds=args.seconds, watch=not args.headless,
                    presentation=args.presentation)
                print(f"Saved comparison: {result}")
            else:
                result = run_policy(checkpoint, directory=directory, treatment=args.treatment,
                    seed=args.seed, sampled=args.sampled, seconds=args.seconds)
                print(json.dumps(result, indent=2))
                if not args.headless:
                    from .viewing.replay import replay
                    replay(directory, speed=1, presentation=args.presentation)
        except (ValueError, FileNotFoundError, FileExistsError) as error:
            parser.error(str(error))
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
            parser.error("PPO inference needs PyTorch: python -m pip install -r requirements-policy.txt")
    elif args.action == "tune-rate":
        from .tuning_round import run_rate_round
        run_rate_round(args.output)
    elif args.action == "train-stride":
        from .stride_round import run_stride_round
        run_stride_round(args.output)
    else:
        from .viewing.render import main as render_main

        options = ["--output", str(args.output)]
        if args.before_directory:
            options += ["--before-directory", str(args.before_directory)]
        render_main(options)


if __name__ == "__main__":
    main()
