"""Fixed-seed evaluation for explicitly selected residual checkpoints.

This command never trains or promotes a policy. A numeric pass still requires
user review of the real-time gait. Model changes are separate from policy gains.
Use --notebook-seeds for Notebook 4's 24-run sampled/mean protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
import torch

from . import simulation
from .gait_feasibility import _tilt_deg
from .policy_acceptance import _references
from .policy_metrics import episode_metrics, transition_metrics
from .recording import TreatmentReplay
from .reference_training import ReferenceResidualPolicy


def summarize(records: list[dict], threshold: float, *, notebook_seeds: bool = False) -> dict:
    mean = [r for r in records if r["mode"] == "mean"]
    sampled = [r for r in records if r["mode"] == "sampled"]
    failures = []
    expected_mean = set(range(201, 213)) if notebook_seeds else {201}
    if len(mean) != len(expected_mean) or {r["seed"] for r in mean} != expected_mean:
        failures.append("requires each expected mean seed exactly once")
    if len(sampled) != 12 or {r["seed"] for r in sampled} != set(range(201, 213)):
        failures.append("requires sampled seeds 201..212 exactly once")
    for r in records:
        if not all(np.isfinite(r[name]) for name in
                   ("seconds", "speed_m_s", "max_joint_limit_violation_rad")):
            failures.append(f"nonfinite summary {r['mode']} {r['seed']}")
        if r["terminated"] or abs(r["seconds"] - 5) > 1e-8:
            failures.append(f"incomplete or fallen {r['mode']} {r['seed']}")
        if r["max_joint_limit_violation_rad"] > 1e-3:
            failures.append(f"joint limit violation {r['mode']} {r['seed']}")
        if not np.isfinite(r["speed_m_s"]):
            failures.append("nonfinite speed")
    sample_speed = float(np.mean([r["speed_m_s"] for r in sampled])) if sampled else None
    if not mean or any(r["speed_m_s"] < threshold for r in mean):
        failures.append("mean policy below accepted PPO-100 mean speed")
    if sample_speed is None or sample_speed < threshold:
        failures.append("sampled average below accepted PPO-100 mean speed")
    return dict(goal_numerical_pass=not failures, failures=failures, accepted=False,
                visual_review_required=True, threshold_m_s=threshold,
                sampled_mean_speed_m_s=sample_speed,
                limitation="Action-noise seeds at fixed initial state; not terrain or push robustness")


def record(checkpoint: Path, output: Path, *, seed: int, sampled: bool) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = mujoco.MjModel.from_xml_string(payload["model_xml"])
    kind = payload.get("policy_kind", "reference_residual")
    if kind == "cadence_action_v1":
        from .cadence_action_training import CadenceResidualPolicy
        policy = CadenceResidualPolicy(checkpoint, model, seed=seed, sampled=sampled)
    elif kind == "reference_residual":
        policy = ReferenceResidualPolicy(checkpoint, model, seed=seed, sampled=sampled)
    else:
        raise ValueError(f"unknown policy kind: {kind}")
    output.mkdir(parents=True, exist_ok=False)
    data = mujoco.MjData(model)
    simulation.reset(model, data)
    data.qpos[7:] = policy.neutral
    data.ctrl[:] = policy.neutral
    mujoco.mj_forward(model, data)
    observed = simulation.measured_state(model, data)
    origin = np.asarray(observed.torso_position)
    previous_target = policy.neutral.copy()
    mode = "sampled" if sampled else "mean"
    treatment = "LEARNED CADENCE" if kind == "cadence_action_v1" else "CANDIDATE"
    label = f"REFERENCE PPO n={payload['updates']} | {treatment} | {mode} | seed={seed}"
    if kind == "reference_residual" and payload.get("treatment_changes"):
        label += f" | cadence={policy.reference.config['frequency_hz']:g} Hz | cadence treatment"
    replay = TreatmentReplay(model, label)
    replay.capture(data)
    interval = model.opt.timestep * policy.settings["physics_steps"]
    decisions = round(5 / interval)
    if abs(decisions * interval - 5) > 1e-9:
        raise ValueError("checkpoint cadence must divide five seconds exactly")
    rows, terminated, max_violation = [], False, 0.0
    for _ in range(decisions):
        before = observed
        target = policy.targets(observed)
        # Diagnostics describe the command held during the following interval.
        command_diagnostics = {f"command_{key}": value
                               for key, value in getattr(policy, "diagnostics", {}).items()}
        if not np.isfinite(target).all() or np.any(target < policy.bounds[:, 0]) or np.any(target > policy.bounds[:, 1]):
            raise ValueError("nonfinite or invalid actuator target")
        for _ in range(policy.settings["physics_steps"]):
            simulation.step(model, data, target)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("nonfinite simulation state: evaluation fails")
            joints = data.qpos[7:]
            max_violation = max(max_violation, float(np.max(np.maximum(
                model.jnt_range[1:, 0] - joints, joints - model.jnt_range[1:, 1]))))
            terminated = data.qpos[2] < .25 or _tilt_deg(data.qpos[3:7]) > 60
            if terminated:
                break
        observed = simulation.measured_state(model, data)
        replay.capture(data)
        rows.append(dict(time_s=observed.time, forward_m=observed.torso_position[0]-origin[0],
                         height_m=observed.torso_position[2], tilt_deg=_tilt_deg(observed.torso_orientation),
                         **transition_metrics(before, observed, target, previous_target,
                                              observed.time-before.time),
                         **{f"joint_{i}_rad": float(q) for i, q in enumerate(observed.joint_positions)},
                         **{f"target_{i}_rad": float(q) for i, q in enumerate(target)}))
        rows[-1].update(command_diagnostics)
        previous_target = target
        if terminated:
            break
    mujoco.mj_saveModel(model, str(output / "model.mjb"))
    np.savez_compressed(output / "states.npz", states=replay.states, times=replay.times, label=label)
    with (output / "trace.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    result = dict(mode=mode, seed=seed, seconds=observed.time, terminated=bool(terminated),
                  max_joint_limit_violation_rad=max_violation, **episode_metrics(rows),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(payload["model_xml"].encode()).hexdigest(),
                  updates=payload["updates"], label=label)
    result["reference_config"] = policy.reference.config
    result["treatment_changes"] = payload.get("treatment_changes", {})
    result["policy_kind"] = kind
    if kind == "cadence_action_v1":
        frequencies = [r["command_current_cadence_hz"] for r in rows]
        result["cadence_hz"] = dict(minimum=min(frequencies), maximum=max(frequencies),
                                    mean=float(np.mean(frequencies)))
    (output / "metadata.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def evaluate(checkpoint: Path, output: Path, *, notebook_seeds: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    sources = output / "sources"; sources.mkdir()
    for name in ("reference_evaluation.py", "reference_training.py", "policy_metrics.py", "simulation.py"):
        shutil.copy2(Path(__file__).with_name(name), sources / name)
    candidate_source = Path(__file__).with_name("cadence_action_training.py")
    if candidate_source.exists():
        shutil.copy2(candidate_source, sources / candidate_source.name)
    records = [record(checkpoint, output / "mean-201", seed=201, sampled=False)]
    if notebook_seeds:
        for seed in range(202, 213):
            records.append(record(checkpoint, output / f"mean-{seed}", seed=seed, sampled=False))
    for seed in range(201, 213):
        records.append(record(checkpoint, output / f"sampled-{seed}", seed=seed, sampled=True))
    result = summarize(records, _references()["mean_speed_m_s"], notebook_seeds=notebook_seeds)
    result["notebook_seeds"] = notebook_seeds
    result["records"] = records
    (output / "acceptance.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--notebook-seeds", action="store_true",
                        help="Match Notebook 4: sampled and mean actions for seeds 201..212")
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.output, notebook_seeds=args.notebook_seeds)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    return 0 if result["goal_numerical_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
