"""Run saved PPO policies through the canonical C-1N simulation boundary.

No notebook or lab dependency. Importing this module does not run a rollout.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from importlib.metadata import version

import mujoco
import numpy as np

from . import simulation
from .recording import TreatmentReplay


DEFAULT_CHECKPOINT = (
    simulation.ROOT / "artifacts/ppo-crude-baseline-20260915/checkpoint-00100.pt"
)
TREATMENTS = ("baseline", "lower", "smooth", "stalk")


def run_policy(
    checkpoint: Path = DEFAULT_CHECKPOINT,
    *,
    treatment: str = "baseline",
    seed: int = 201,
    sampled: bool = False,
    seconds: float = 5.0,
    directory: Path,
) -> dict:
    """Execute and record one bounded episode only when explicitly called.

    The saved observation/action contract uses 10 physics steps per decision.
    Evaluation never updates the actor or changes model parameters.
    """
    from .policy import PPOPolicy

    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    model = simulation.load_model()
    policy = PPOPolicy(checkpoint, model, seed=seed, sampled=sampled, treatment=treatment)
    interval = policy.settings["physics_steps"] * float(model.opt.timestep)
    decisions = round(seconds / interval)
    if decisions < 1 or not math.isclose(decisions * interval, seconds, abs_tol=1e-9):
        raise ValueError(f"seconds must be a positive multiple of the control interval ({interval:g}s)")
    if decisions > policy.settings["horizon"]:
        raise ValueError("Keep evaluation within the checkpoint's trained episode horizon")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    data = mujoco.MjData(model)
    simulation.reset(model, data)
    observed = simulation.measured_state(model, data)
    origin = np.asarray(observed.torso_position)
    mode = "sampled" if sampled else "mean"
    label = f"PPO n={policy.updates} | {treatment} | {mode} | seed={seed}"
    replay = TreatmentReplay(model, label)
    replay.capture(data)
    metadata = dict(
        treatment=treatment, mode=mode, seed=seed, checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=policy.checkpoint_sha256, updates=policy.updates,
        model_sha256=hashlib.sha256(simulation.MODEL_PATH.read_bytes()).hexdigest(),
        settings=policy.settings, seconds_requested=seconds,
        control_interval_s=interval, status="running", controller=policy.description,
        controller_config=policy.config,
        runtime_versions={name: version(name) for name in ("mujoco", "numpy", "torch")},
        runtime_source_sha256={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("policy.py", "policy_run.py", "simulation.py")
        },
    )
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    rows = []
    for decision in range(decisions):
        targets = policy.targets(observed)
        for _ in range(policy.settings["physics_steps"]):
            simulation.step(model, data, targets)
        observed = simulation.measured_state(model, data)
        replay.capture(data)
        if not np.isfinite(replay.states[-1]).all():
            raise RuntimeError("Non-finite simulation state; discard this episode")
        terminated = observed.torso_position[2] < policy.settings["fall_height_m"]
        rows.append(dict(
            action=decision, time_s=observed.time,
            forward_m=observed.torso_position[0] - origin[0],
            lateral_m=observed.torso_position[1] - origin[1],
            height_m=observed.torso_position[2],
            contacts=len(observed.foot_contacts),
            support_margin_m=observed.support_margin,
            terminated=bool(terminated),
            **{f"target_{i}_rad": float(target) for i, target in enumerate(targets)},
        ))
        if terminated:
            break
    mujoco.mj_saveModel(model, str(directory / "model.mjb"))
    np.savez_compressed(directory / "states.npz", states=np.asarray(replay.states),
                        times=np.asarray(replay.times), label=label)
    with (directory / "trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata.update(
        status="complete", seconds=float(observed.time),
        forward_m=rows[-1]["forward_m"], lateral_m=rows[-1]["lateral_m"],
        minimum_height_m=min(float(origin[2]), *(row["height_m"] for row in rows)),
        terminated=rows[-1]["terminated"],
        truncated=not rows[-1]["terminated"] and len(rows) == decisions,
    )
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def compare_policies(
    checkpoint: Path = DEFAULT_CHECKPOINT, *, directory: Path | None = None,
    seed: int = 201, sampled: bool = False, seconds: float = 5.0,
    watch: bool = True, presentation: str = "stalk",
) -> Path:
    """Record the same seed in four explicit control treatments, then replay."""
    if directory is None:
        directory = simulation.ROOT / "telemetry/policy" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    results = []
    for treatment in TREATMENTS:
        result = run_policy(checkpoint, treatment=treatment, seed=seed, sampled=sampled,
                            seconds=seconds, directory=directory / treatment)
        results.append(result)
        print(f"Recorded {treatment}: {result['seconds']:.2f}s", flush=True)
    (directory / "comparison.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if watch:
        from .viewing.replay_grid import replay_grid

        replay_grid([directory / treatment for treatment in TREATMENTS], speed=1,
                    presentation=presentation)
    return directory
