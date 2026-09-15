"""Run the bounded, user-approved three-seed fresh stride training round."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import time

from .policy_run import DEFAULT_CHECKPOINT, run_policy


TRAINING_SEEDS = (11, 22, 33)
EVALUATION_SEEDS = tuple(range(201, 213))
UPDATES_PER_SEED = 50
FORWARD_EXPLORATION = {
    "lower_exploration": dict(noise_scale=0.4, entropy_coefficient=0.0,
                              target_speed_mps=0.25, velocity_weight=1.5),
    "stronger_forward": dict(noise_scale=1.0, entropy_coefficient=0.003,
                             target_speed_mps=0.4, velocity_weight=4.0),
    "combined": dict(noise_scale=0.4, entropy_coefficient=0.0,
                     target_speed_mps=0.4, velocity_weight=4.0),
}


def _mean(values):
    finite = [value for value in values if value is not None]
    return statistics.mean(finite) if finite else None


def run_stride_round(directory: Path, *, comparison: str = "fresh-seeds") -> Path:
    """Train three fresh policies and evaluate once; never auto-extend."""
    if comparison not in {"fresh-seeds", "forward-exploration"}:
        raise ValueError("unknown stride comparison")
    trials = ([(name, 11, config) for name, config in FORWARD_EXPLORATION.items()]
              if comparison == "forward-exploration" else
              [(f"stride_seed_{seed}", seed, {}) for seed in TRAINING_SEEDS])
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    plan = {
        "training": "fresh phase-guided stride policy",
        "training_seeds": TRAINING_SEEDS,
        "updates_per_seed": UPDATES_PER_SEED,
        "episodes_per_update": 8,
        "decisions_per_training_episode": 256,
        "training_episode_seconds": 10.24,
        "evaluation_seconds": 5.0,
        "target_speed_m_s": 0.25,
        "torch_threads": 1,
        "sampled_evaluation_seeds": EVALUATION_SEEDS,
        "mean_evaluation_seeds": [201],
        "accepted_baseline_checkpoint": str(DEFAULT_CHECKPOINT.resolve()),
        "approval": "User approved three fresh seeds at 50 updates each; review before extension.",
        "notes": [
            "The baseline uses its saved 20 ms control cadence; fresh stride policies use 40 ms.",
            "Each policy has one deterministic mean replay because reset and mean action are fixed.",
            "Mean replays are not independent-seed evidence.",
        ],
    }
    if comparison == "forward-exploration":
        plan.pop("training_seeds")
        plan.pop("target_speed_m_s")
        plan.update(comparison=comparison, initialization_seed=11,
                    treatments=FORWARD_EXPLORATION,
                    expected_policies=["accepted_baseline", *FORWARD_EXPLORATION],
                    approval="User approved lower exploration, stronger forward incentive, and combined; 50 updates each.")
    # Archive the complete fixed plan before importing or starting the trainer.
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    from .stride_training import FreshTrainingSession
    import torch

    torch.set_num_threads(1)

    started = time.monotonic()
    checkpoints = {"accepted_baseline": DEFAULT_CHECKPOINT.resolve()}
    for name, seed, configuration in trials:
        print(f"TRAIN {name}: fresh initialization; {UPDATES_PER_SEED} updates", flush=True)
        session = FreshTrainingSession(directory / name / "training", seed, **configuration)
        checkpoints[name] = Path(session.train(updates=UPDATES_PER_SEED)).resolve()
        (directory / "checkpoints.json").write_text(
            json.dumps({key: str(value) for key, value in checkpoints.items()}, indent=2) + "\n",
            encoding="utf-8",
        )

    results = []
    for name, checkpoint in checkpoints.items():
        treatment = "baseline" if name == "accepted_baseline" else "stride"
        for sampled, seeds in ((False, (201,)), (True, EVALUATION_SEEDS)):
            mode = "sampled" if sampled else "mean"
            for seed in seeds:
                recording = directory / name / "evaluation" / f"{mode}-{seed}"
                result = run_policy(
                    checkpoint,
                    treatment=treatment,
                    seed=seed,
                    sampled=sampled,
                    seconds=5.0,
                    directory=recording,
                    label_suffix=name,
                )
                results.append({"policy": name, "recording": str(recording), **result})
            print(f"EVAL {name}: {mode} complete", flush=True)
        (directory / "evaluations.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )

    rows = []
    scalar_metrics = (
        "speed_m_s",
        "forward_m",
        "minimum_height_m",
        "steady_height_mean_m",
        "steady_height_std_m",
        "action_delta_rms_rad",
        "stance_foot_speed_rms_m_s",
        "vertical_velocity_rms_m_s",
    )
    for name in checkpoints:
        for mode in ("mean", "sampled"):
            subset = [row for row in results if row["policy"] == name and row["mode"] == mode]
            row = {
                "policy": name,
                "mode": mode,
                "episodes": len(subset),
                "falls": sum(result["terminated"] for result in subset),
            }
            row.update({metric: _mean(result.get(metric) for result in subset) for metric in scalar_metrics})
            row["abs_lateral_m"] = _mean(abs(result["lateral_m"]) for result in subset)
            row["joint_rom_mean_rad"] = _mean(
                statistics.mean(result["joint_range_of_motion_rad"])
                for result in subset if result.get("joint_range_of_motion_rad")
            )
            row["joint_rom_max_rad"] = max(
                (max(result["joint_range_of_motion_rad"])
                 for result in subset if result.get("joint_range_of_motion_rad")),
                default=None,
            )
            row["max_foot_clearance_m"] = max(
                (max(result["maximum_foot_clearance_m"].values())
                 for result in subset if result.get("maximum_foot_clearance_m")),
                default=None,
            )
            row["liftoffs"] = _mean(
                sum(result["liftoff_count"].values())
                for result in subset if result.get("liftoff_count")
            )
            rows.append(row)
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    from .policy_acceptance import assess_round
    acceptance = assess_round(directory)
    receipt = {
        "status": "awaiting-user-review" if acceptance["numerical_pass"] else "failed-acceptance",
        "elapsed_seconds": time.monotonic() - started,
        "comparison": comparison,
        "trials": [{"name": name, "seed": seed, **configuration} for name, seed, configuration in trials],
        "updates_per_seed": UPDATES_PER_SEED,
        "total_updates": len(trials) * UPDATES_PER_SEED,
        "evaluation_episodes": len(results),
        "interpretation": "No gait, fall-rate, or speed claim is made by this runner.",
        "acceptance_report": "acceptance.json",
        "numerical_pass": acceptance["numerical_pass"],
    }
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)
    print(f"Stride round complete; review before further training: {directory}", flush=True)
    return directory
