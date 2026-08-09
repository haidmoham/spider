"""Run the v0.1 Spider model without a viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "spider.xml"


def run(seconds: float) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    steps = round(seconds / model.opt.timestep)
    for _ in range(steps):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)

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
    print(f"torso height: {data.qpos[2]:.3f}m")
    print(f"torso vertical speed: {data.qvel[2]:.3f}m/s")


if __name__ == "__main__":
    main()
