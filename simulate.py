"""Run the C-1N // 00 · POSE baseline without a viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "spider.xml"
STANDING_KNEE_TARGET = 0.8


def set_standing_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Place the robot above the ground and give each motor a static pose target."""
    data.qpos[0:7] = [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]
    for leg in range(6):
        hip = 2 * leg
        knee = hip + 1
        data.qpos[7 + hip] = 0.0
        data.qpos[7 + knee] = STANDING_KNEE_TARGET
        data.ctrl[hip] = 0.0
        data.ctrl[knee] = STANDING_KNEE_TARGET
    mujoco.mj_forward(model, data)


def run(seconds: float) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    set_standing_pose(model, data)

    steps = round(seconds / model.opt.timestep)
    for _ in range(steps):
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
