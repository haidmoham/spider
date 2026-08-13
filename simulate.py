"""Run the C-1N v0.0 - SPAWN baseline without a viewer."""

from __future__ import annotations

import argparse
import mujoco

from simulation import MODEL_PATH, ROOT, load_model, measured_state, reset, run as run_simulation


def run(seconds: float) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = load_model()
    data = mujoco.MjData(model)
    reset(model, data)
    run_simulation(model, data, seconds)

    return model, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seconds",
        type=float,
        default=1.0,
        help="simulation duration in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    if args.seconds <= 0:
        parser.error("--seconds must be greater than zero")

    model, data = run(args.seconds)
    print(f"model: {MODEL_PATH.relative_to(ROOT)}")
    print(f"bodies: {model.nbody - 1}, joints: {model.njnt}, actuators: {model.nu}")
    print(f"simulated: {data.time:.3f}s")
    state = measured_state(model, data)
    print(f"torso height: {state.torso_position[2]:.3f}m")
    print(f"torso vertical speed: {state.torso_velocity[2]:.3f}m/s")
    print(f"whole-stance foot residual: {state.foot_position_residual_norm:.6f}m")
    print(f"whole-stance J^T r direction: {state.joint_space_update_direction_norm:.6f}m^2/rad")


if __name__ == "__main__":
    main()
