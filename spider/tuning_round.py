"""One bounded, user-approved command-rate tuning round in the robot platform."""

from pathlib import Path
import csv
import json
import statistics
import time

from .policy_run import DEFAULT_CHECKPOINT, run_policy


# Derived from the 12 saved PPO100 sampled episodes, not copied from a paper.
# sum(progress) mean=68.58110025246224; sum(mean(delta_target**2)) mean=.36761249650938477.
RATE_WEIGHTS = {"control": 0.0, "rate_small": 3.731162618445504, "rate_medium": 9.32790654611376}
UPDATES_PER_BRANCH = 10
EVALUATION_SEEDS = tuple(range(201, 213))


def run_rate_round(directory: Path) -> Path:
    """Train three independent continuations and evaluate; never auto-extend."""
    from .policy_training import TrainingSession

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    plan = dict(
        parent_checkpoint=str(DEFAULT_CHECKPOINT), treatment="lower",
        updates_per_branch=UPDATES_PER_BRANCH, action_rate_weights=RATE_WEIGHTS,
        sampled_eval_seeds=EVALUATION_SEEDS, mean_eval_seeds=[201],
        calibration=dict(source="telemetry/ppo/20260915T152940537445Z/checkpoint-comparison-153230927015",
                         progress_mean=68.58110025246224, rate_cost_mean=.36761249650938477,
                         fractions=[0., .02, .05]),
        approval="User approved three independent 10-update branches; review before any extension.",
        notes="Mean episodes have a fixed initial state: one mean replay per policy, not 12 independent repeats.",
    )
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    checkpoints = {"frozen_lower": DEFAULT_CHECKPOINT}
    for name, weight in RATE_WEIGHTS.items():
        print(f"TRAIN {name}: 10 updates from PPO100; rate weight={weight:.8g}", flush=True)
        session = TrainingSession(DEFAULT_CHECKPOINT, directory / name / "training",
                                  action_rate_weight=weight, treatment="lower")
        checkpoints[name] = session.train(updates=UPDATES_PER_BRANCH)
    results = []
    for name, checkpoint in checkpoints.items():
        for sampled, seeds in ((False, (201,)), (True, EVALUATION_SEEDS)):
            for seed in seeds:
                mode = "sampled" if sampled else "mean"
                recording = directory / name / "evaluation" / f"{mode}-{seed}"
                result = run_policy(checkpoint, treatment="lower", seed=seed,
                                    sampled=sampled, seconds=5., directory=recording,
                                    label_suffix=name)
                results.append(dict(branch=name, recording=str(recording), **result))
            print(f"EVAL {name}: {mode} complete", flush=True)
        # Preserve partial completed evaluation if a later branch fails.
        (directory / "evaluations.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    rows = []
    for name in checkpoints:
        for mode in ("mean", "sampled"):
            subset = [row for row in results if row["branch"] == name and row["mode"] == mode]
            row = dict(branch=name, mode=mode, episodes=len(subset), falls=sum(r["terminated"] for r in subset))
            for metric in ("speed_m_s", "forward_m", "minimum_height_m", "steady_height_mean_m",
                           "steady_height_std_m", "action_delta_rms_rad", "stance_foot_speed_rms_m_s",
                           "vertical_velocity_rms_m_s"):
                values = [r[metric] for r in subset if r[metric] is not None]
                row[metric] = statistics.mean(values) if values else None
            row["abs_lateral_m"] = statistics.mean(abs(r["lateral_m"]) for r in subset)
            rows.append(row)
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "receipt.json").write_text(json.dumps(dict(status="awaiting-user-review",
        elapsed_seconds=time.monotonic() - started, total_updates=30, evaluation_episodes=len(results)), indent=2) + "\n")
    print(json.dumps(rows, indent=2), flush=True)
    print(f"Round complete; review before further training: {directory}", flush=True)
    return directory
