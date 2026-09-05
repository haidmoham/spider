"""Run the canonical C-1N simulation with optional viewer and socket control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Compatibility imports keep existing scripts and notebooks working.
# New callers should import from the module that owns the behavior.
from commands import HOST, PORT, Request, execute, listener
from live_viewer import (
    _make_stand_figures, _update_figure, _update_stand_figures,
    run_shove_suite_viewer, run_viewer,
)
from runtime import (
    JOINT_NAMES, SHOVE_ANGLES_DEG, SHOVE_DURATION_S, SHOVE_MULTIPLES,
    StancePower, TorsoForcePulse, advance, build_simulation, run_headless,
    run_shove_suite, shove_cases, state, targets_for_step,
)
from simulation import FOOT_NAMES, JOINTS_PER_LEG, MODEL_PATH, ROOT, load_model, measured_state, reset, step
from standing import SupportAwareStanceController
from telemetry import (
    TELEMETRY_HISTORY_SECONDS, TELEMETRY_SAMPLE_INTERVAL_S,
    StandTelemetryRecorder, StandTelemetrySample, write_rollout_trace,
)
from visuals import ResponsivePupils
from walk import GaitCoordinator, apply_gait_control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    parser.add_argument("--seconds", type=float, default=1.0, help="headless duration in seconds")
    parser.add_argument("--experiment", choices=("none", "stand", "shuffle"), default="stand", help="explicit target generator")
    parser.add_argument("--trace", type=Path, help="write compact 50 Hz Telemetry v1 samples to this .npz file")
    parser.add_argument("--shove-suite", type=Path, metavar="DIRECTORY", help="run 0, 0.25, 0.5, 0.75, and 1 mg shoves in eight world-frame directions and save traces here")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be greater than zero")
    if args.headless:
        if args.shove_suite is not None:
            print(json.dumps(run_shove_suite(args.seconds, args.shove_suite), indent=2))
        else:
            print(json.dumps(run_headless(args.seconds, args.experiment, args.trace), indent=2))
    elif args.shove_suite is not None:
        run_shove_suite_viewer(args.seconds, args.shove_suite)
    else:
        run_viewer(args.experiment)



if __name__ == "__main__":
    main()
