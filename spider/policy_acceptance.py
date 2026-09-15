"""Read-only numerical evidence gate for saved C-1N policy rounds."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import mujoco

from . import simulation


EXPECTED_SAMPLED_SEEDS = set(range(201, 213))
EXPECTED_SECONDS = 5.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _references() -> dict[str, Any]:
    root = simulation.ROOT / "artifacts/ppo-crude-baseline-20260915"
    manifest_path = root / "manifest.json"
    comparison_path = root / "comparison.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _sha256(comparison_path) != manifest["files"]["comparison.csv"]["sha256"]:
        raise ValueError("accepted comparison.csv does not match its manifest")
    if _sha256(simulation.MODEL_PATH) != manifest["files"]["spider.xml"]["sha256"]:
        raise ValueError("canonical model does not match the accepted manifest")
    with comparison_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    accepted = [row for row in rows if int(row["update"]) == 100]
    mean = [row for row in accepted if row["mode"] == "mean" and int(row["seed"]) == 201]
    sampled = [row for row in accepted if row["mode"] == "sampled"]
    return {
        "mean_speed_m_s": float(mean[0]["forward_m"]) / float(mean[0]["seconds"]),
        "sampled_mean_speed_m_s": float(np.mean([
            float(row["forward_m"]) / float(row["seconds"]) for row in sampled
        ])),
        "comparison_sha256": _sha256(comparison_path),
        "manifest_sha256": _sha256(manifest_path),
        "expected_model_sha256": manifest["files"]["spider.xml"]["sha256"],
        "source": "baseline-derived user gate; notebook 04 declares no numeric quality threshold",
    }


def _finite_json(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _same_physics(candidate, canonical) -> bool:
    scalar = ("nq", "nv", "nu", "nbody", "njnt", "ngeom")
    arrays = (
        "jnt_range", "jnt_axis", "jnt_pos", "jnt_type", "dof_damping", "dof_armature",
        "actuator_ctrlrange", "actuator_forcerange", "actuator_trnid", "actuator_gainprm",
        "actuator_biasprm", "actuator_dynprm", "actuator_gear", "body_mass", "body_inertia",
        "body_pos", "body_quat", "body_ipos", "body_iquat", "geom_friction", "geom_size",
        "geom_pos", "geom_quat", "geom_type", "geom_contype", "geom_conaffinity",
        "geom_solmix", "geom_solref", "geom_solimp", "geom_margin", "geom_gap",
    )
    option_scalars = (
        "timestep", "integrator", "solver", "iterations", "ls_iterations",
        "noslip_iterations", "ccd_iterations", "cone", "jacobian", "disableflags",
        "enableflags", "tolerance", "ls_tolerance", "noslip_tolerance",
        "density", "viscosity", "impratio", "ccd_tolerance",
    )
    option_arrays = ("gravity", "wind", "magnetic")
    return (all(getattr(candidate, name) == getattr(canonical, name) for name in scalar)
            and all(getattr(candidate.opt, name) == getattr(canonical.opt, name)
                    for name in option_scalars)
            and all(np.array_equal(getattr(candidate.opt, name), getattr(canonical.opt, name))
                    for name in option_arrays)
            and all(np.array_equal(getattr(candidate, name), getattr(canonical, name))
                    for name in arrays))


def _assess_record(entry: dict, canonical, expected_model_hash: str,
                   model_cache: dict[str, tuple[bool, int]]) -> dict:
    failures = []
    recording = Path(entry.get("recording", ""))
    required = {name: recording / name for name in
                ("metadata.json", "trace.csv", "states.npz", "model.mjb")}
    for name, path in required.items():
        if not path.is_file():
            failures.append(f"missing {name}")
    if failures:
        return {"failures": failures, "speed_m_s": None, "fell": None}
    try:
        metadata = json.loads(required["metadata.json"].read_text(encoding="utf-8"))
        if not _finite_json(metadata):
            failures.append("metadata contains non-finite values")
        if metadata.get("status") != "complete":
            failures.append("recording status is not complete")
        if metadata.get("mode") != entry.get("mode") or metadata.get("seed") != entry.get("seed"):
            failures.append("metadata mode or seed differs from evaluations.json")
        if metadata.get("terminated") is not False or metadata.get("truncated") is not True:
            failures.append("metadata must declare terminated=false and truncated=true")
        if not math.isclose(float(metadata.get("seconds", -1)), EXPECTED_SECONDS, abs_tol=1e-6):
            failures.append("metadata does not declare a complete 5 second episode")
        if metadata.get("model_sha256") != expected_model_hash:
            failures.append("recording model hash differs from accepted model source")
        checkpoint = Path(entry.get("checkpoint", metadata.get("checkpoint", "")))
        if Path(metadata.get("checkpoint", "")).resolve() != checkpoint.resolve():
            failures.append("metadata checkpoint path differs from evaluations.json")
        if not checkpoint.is_file():
            failures.append("checkpoint file is missing")
        else:
            actual_hash = _sha256(checkpoint)
            declared = entry.get("checkpoint_sha256", metadata.get("checkpoint_sha256"))
            if actual_hash != declared:
                failures.append("checkpoint hash does not match checkpoint file")
            if metadata.get("checkpoint_sha256") != declared:
                failures.append("metadata checkpoint hash differs from evaluations.json")
        model_binary_hash = _sha256(required["model.mjb"])
        if model_binary_hash not in model_cache:
            try:
                recorded_model = mujoco.MjModel.from_binary_path(str(required["model.mjb"]))
                model_cache[model_binary_hash] = (
                    _same_physics(recorded_model, canonical),
                    mujoco.mj_stateSize(recorded_model, mujoco.mjtState.mjSTATE_INTEGRATION),
                )
            except (ValueError, OSError):
                model_cache[model_binary_hash] = (False, -1)
        model_matches, state_size = model_cache[model_binary_hash]
        if not model_matches:
            failures.append("recorded model binary is invalid or physics differs from canonical model")
        with required["trace.csv"].open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            failures.append("trace has no rows")
            return {"failures": failures, "speed_m_s": None, "fell": None}
        numeric_columns = ["time_s", "forward_m", "height_m",
                           *[f"target_{i}_rad" for i in range(18)]]
        try:
            numeric = np.asarray([[float(row[name]) for name in numeric_columns] for row in rows])
        except (KeyError, TypeError, ValueError):
            failures.append("trace numeric schema is incomplete")
            return {"failures": failures, "speed_m_s": None, "fell": None}
        if not np.isfinite(numeric).all():
            failures.append("trace contains non-finite values")
        for row in rows:
            for key, value in row.items():
                if key == "support_margin_m" and value in ("", None, "None"):
                    continue
                if key == "terminated":
                    if str(value).lower() not in {"true", "false"}:
                        failures.append("trace terminated field is invalid")
                    continue
                try:
                    if value not in ("", None) and not math.isfinite(float(value)):
                        failures.append(f"trace field {key} contains non-finite values")
                except ValueError:
                    failures.append(f"trace field {key} is not numeric")
        times = numeric[:, 0]
        if np.any(np.diff(times) <= 0) or not math.isclose(times[-1], EXPECTED_SECONDS,
                                                           abs_tol=1e-6):
            failures.append("trace does not contain one monotonic 5 second episode")
        interval = float(metadata.get("control_interval_s", -1))
        if interval <= 0 or len(rows) != round(EXPECTED_SECONDS / interval):
            failures.append("trace count does not match metadata control interval")
        elif not np.allclose(np.diff(np.r_[0.0, times]), interval, atol=1e-9, rtol=0):
            failures.append("trace cadence does not match metadata control interval")
        targets = numeric[:, 3:]
        ctrlrange = np.asarray(canonical.actuator_ctrlrange)
        if np.any(targets < ctrlrange[:, 0] - 1e-12) or np.any(targets > ctrlrange[:, 1] + 1e-12):
            failures.append("trace target exceeds canonical actuator control range")
        fell = any(str(row.get("terminated", "")).lower() == "true" for row in rows)
        if fell or bool(metadata.get("terminated")):
            failures.append("episode contains a fall")
        with np.load(required["states.npz"], allow_pickle=False) as archive:
            if "states" not in archive or "times" not in archive:
                failures.append("state archive schema is incomplete")
            else:
                states, state_times = archive["states"], archive["times"]
                if not np.isfinite(states).all() or not np.isfinite(state_times).all():
                    failures.append("state archive contains non-finite values")
                if len(state_times) < 2 or not math.isclose(float(state_times[-1]), EXPECTED_SECONDS,
                                                             abs_tol=1e-6):
                    failures.append("state archive does not reach 5 seconds")
                if states.shape != (len(rows) + 1, state_size):
                    failures.append("state archive shape does not match trace and recorded model")
                if len(state_times) != len(rows) + 1 or not math.isclose(float(state_times[0]), 0.0,
                                                                          abs_tol=1e-12):
                    failures.append("state archive must start at time zero with one extra frame")
                elif not np.allclose(state_times[1:], times, atol=1e-9, rtol=0):
                    failures.append("state archive times do not align with trace")
                if states.shape == (len(rows) + 1, state_size):
                    origin_x = states[0, 1]
                    if not np.allclose(states[1:, 1] - origin_x, numeric[:, 1], atol=1e-9, rtol=0):
                        failures.append("trace forward displacement differs from recorded qpos")
                    if not np.allclose(states[1:, 3], numeric[:, 2], atol=1e-9, rtol=0):
                        failures.append("trace height differs from recorded qpos")
                    if np.any(states[1:, 3] < 0.25):
                        fell = True
                        failures.append("recorded state falls below the 0.25 m threshold")
        speed = float(numeric[-1, 1]) / float(times[-1])
        return {"failures": failures, "speed_m_s": speed, "fell": fell,
                "recording": str(recording), "mode": entry.get("mode"),
                "seed": entry.get("seed"),
                "file_sha256": {name: _sha256(path) for name, path in required.items()}}
    except (OSError, json.JSONDecodeError, ValueError) as error:
        failures.append(f"unreadable evidence: {type(error).__name__}: {error}")
        return {"failures": failures, "speed_m_s": None, "fell": None}


def assess_round(directory: Path) -> dict:
    """Assess saved evidence and write numerical status without rerunning policy code."""
    directory = Path(directory).resolve()
    evaluations_path = directory / "evaluations.json"
    entries = json.loads(evaluations_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("evaluations.json must contain a list")
    reference = _references()
    canonical = simulation.load_model()
    ctrlrange = np.asarray(canonical.actuator_ctrlrange, dtype=float)
    model_cache = {}
    groups = defaultdict(list)
    for entry in entries:
        name = entry.get("policy", entry.get("branch"))
        groups[str(name) if name is not None else "missing-policy"].append(entry)
    checkpoints_path = directory / "checkpoints.json"
    plan_path = directory / "plan.json"
    expected_groups = set()
    if checkpoints_path.is_file():
        expected_groups |= set(json.loads(checkpoints_path.read_text(encoding="utf-8")))
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        expected_groups |= set(plan.get("expected_policies", []))
        if "action_rate_weights" in plan:
            expected_groups |= set(plan["action_rate_weights"])
            if "parent_checkpoint" in plan:
                expected_groups.add("frozen_lower")
        elif "training_seeds" in plan:
            expected_groups |= {"accepted_baseline", *(
                f"stride_seed_{seed}" for seed in plan["training_seeds"])}
    if not expected_groups:
        expected_groups = set(groups)
    for missing in expected_groups - set(groups):
        groups[missing] = []
    policies = {}
    for name, group in groups.items():
        failures = []
        if any(entry.get("mode") not in {"mean", "sampled"} for entry in group):
            failures.append("evaluation contains an unknown mode")
        checkpoint_hashes = {entry.get("checkpoint_sha256") for entry in group}
        if len(checkpoint_hashes) > 1:
            failures.append("policy group mixes checkpoint hashes")
        mean_entries = [entry for entry in group if entry.get("mode") == "mean"]
        sampled_entries = [entry for entry in group if entry.get("mode") == "sampled"]
        if len(mean_entries) != 1 or mean_entries[0].get("seed") != 201:
            failures.append("requires exactly one mean episode with seed 201")
        sampled_seeds = [entry.get("seed") for entry in sampled_entries]
        if Counter(sampled_seeds) != Counter(EXPECTED_SAMPLED_SEEDS):
            failures.append("requires unique sampled seeds 201 through 212")
        records = [_assess_record(entry, canonical, reference["expected_model_sha256"], model_cache)
                   for entry in group]
        for index, record in enumerate(records):
            failures.extend(f"record {index}: {reason}" for reason in record["failures"])
        mean_speeds = [record["speed_m_s"] for record in records
                       if record.get("mode") == "mean" and record["speed_m_s"] is not None]
        sampled_speeds = [record["speed_m_s"] for record in records
                          if record.get("mode") == "sampled" and record["speed_m_s"] is not None]
        mean_speed = mean_speeds[0] if len(mean_speeds) == 1 else None
        sampled_mean = float(np.mean(sampled_speeds)) if len(sampled_speeds) == 12 else None
        if mean_speed is None or mean_speed < reference["mean_speed_m_s"] - 1e-12:
            failures.append("mean speed is below accepted PPO-100 mean reference")
        if sampled_mean is None or sampled_mean < reference["sampled_mean_speed_m_s"] - 1e-12:
            failures.append("sampled mean speed is below accepted PPO-100 sampled reference")
        # Preserve the historical like-for-like comparison. The thread goal is
        # stronger: the 12-seed sampled average must match the mean-action baseline.
        goal_failures = list(failures)
        if sampled_mean is None or sampled_mean < reference["mean_speed_m_s"] - 1e-12:
            goal_failures.append("12-seed mean speed is below accepted PPO-100 mean-action goal reference")
        policies[name] = {
            "numerical_pass": not failures,
            "failures": failures,
            "goal_numerical_pass": not goal_failures,
            "goal_failures": goal_failures,
            "record_count": len(group),
            "mean_speed_m_s": mean_speed,
            "sampled_mean_speed_m_s": sampled_mean,
            "falls": sum(record.get("fell") is True for record in records),
            "records": records,
        }
    report = {
        "numerical_pass": bool(policies) and all(item["numerical_pass"] for item in policies.values()),
        "accepted": False,
        "visual_review_required": True,
        "goal_sampled_speed_threshold_m_s": reference["mean_speed_m_s"],
        "policies": policies,
        "reference": reference,
        "report_source_sha256": _sha256(Path(__file__)),
    }
    (directory / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, status in policies.items():
        if Path(name).name != name or name in {".", ".."}:
            continue
        policy_directory = directory / name
        policy_directory.mkdir(exist_ok=True)
        (policy_directory / "acceptance-status.json").write_text(
            json.dumps({**status, "accepted": False, "visual_review_required": True}, indent=2) + "\n",
            encoding="utf-8")
    return report
