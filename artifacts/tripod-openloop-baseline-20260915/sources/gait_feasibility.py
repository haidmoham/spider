"""Record untrained open-loop gait references in the canonical physics model.

This runner measures feasibility only. It does not train, accept, or promote a
gait, and it never moves the free torso directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np

from . import simulation
from .gait_reference import GaitReference
from .recording import TreatmentReplay


GAITS = ("wave", "ripple", "tripod")
DURATION_S = 5.0
CONTROL_INTERVAL_S = 0.02
RAMP_DURATION_S = 0.8
FALL_HEIGHT_M = 0.25
MAX_TILT_DEG = 60.0
SOURCE_FILES = ("gait_feasibility.py", "gait_reference.py", "simulation.py")


@dataclass(frozen=True)
class GaitSummary:
    gait: str
    status: str
    requested_duration_s: float
    survival_s: float
    completed_full_duration: bool
    termination_reason: str | None
    dx_m: float
    actual_speed_mps: float
    final_height_m: float
    minimum_height_m: float
    maximum_tilt_deg: float
    foot_slip_m: float
    target_tracking_rmse_rad: float


def _smoothstep(progress: float) -> float:
    value = float(np.clip(progress, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _ramped_target(reference: GaitReference, time_s: float) -> np.ndarray:
    neutral = np.asarray(simulation.neutral_targets(), dtype=float)
    target = np.asarray(reference.pose(time_s), dtype=float)
    if target.shape != neutral.shape or not np.isfinite(target).all():
        raise ValueError(f"{reference.name} pose must contain 18 finite targets")
    blend = _smoothstep(time_s / RAMP_DURATION_S)
    return neutral + blend * (target - neutral)


def _tilt_deg(quaternion: tuple[float, ...] | np.ndarray) -> float:
    quat = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm == 0.0 or not np.isfinite(norm):
        return math.inf
    w, x, y, z = quat / norm
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(float(np.clip(up_z, -1.0, 1.0))))


def _termination(data: mujoco.MjData, height_m: float, tilt_deg: float) -> str | None:
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        return "non_finite_state"
    if height_m < FALL_HEIGHT_M:
        return "torso_below_0.25_m"
    if tilt_deg > MAX_TILT_DEG:
        return "absolute_tilt_above_60_deg"
    return None


def _write_trace(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _copy_sources(output: Path) -> dict[str, str]:
    """Copy the exact evaluator inputs and return their SHA-256 hashes."""
    source_directory = output / "sources"
    source_directory.mkdir()
    paths = [Path(__file__).with_name(name) for name in SOURCE_FILES]
    paths.append(simulation.MODEL_PATH)
    hashes = {}
    for source in paths:
        destination = source_directory / source.name
        shutil.copyfile(source, destination)
        hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes


def run_gait(model: mujoco.MjModel, name: str, output: Path) -> GaitSummary:
    """Run one five-second reference, stopping only on declared safety cutoffs."""
    if name not in GAITS:
        raise ValueError(f"unknown gait {name!r}")
    physics_steps = round(CONTROL_INTERVAL_S / model.opt.timestep)
    if not math.isclose(physics_steps * model.opt.timestep, CONTROL_INTERVAL_S,
                        rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("model timestep must divide the 20 ms control interval")

    output.mkdir(parents=True, exist_ok=False)
    source_hashes = _copy_sources(output)
    data = mujoco.MjData(model)
    simulation.reset(model, data)
    reference = GaitReference(model, name)
    replay = TreatmentReplay(model, f"{name} | UNTRAINED OPEN-LOOP feasibility")
    foot_ids = np.asarray([model.geom(f"{foot}_foot").id for foot in simulation.FOOT_NAMES])
    foot_index = {int(geom_id): index for index, geom_id in enumerate(foot_ids)}
    ground_id = model.geom("ground").id
    origin_x = float(data.qpos[0])
    rows: list[dict[str, object]] = []
    squared_errors: list[float] = []
    cumulative_slip = np.zeros(len(foot_ids))
    minimum_height = float(data.qpos[2])
    maximum_tilt = _tilt_deg(data.qpos[3:7])
    termination_reason: str | None = None
    failure_diagnostic: dict[str, object] | None = None

    def capture(target: np.ndarray) -> None:
        nonlocal minimum_height, maximum_tilt
        observed = simulation.measured_state(model, data)
        tilt = _tilt_deg(observed.torso_orientation)
        error = np.asarray(observed.joint_positions) - target
        squared_errors.extend(np.square(error).tolist())
        minimum_height = min(minimum_height, observed.torso_position[2])
        maximum_tilt = max(maximum_tilt, tilt)
        replay.capture(data)
        rows.append({
            "time_s": observed.time,
            "torso_x_m": observed.torso_position[0],
            "torso_height_m": observed.torso_position[2],
            "tilt_deg": tilt,
            "foot_slip_m": float(np.mean(cumulative_slip)),
            "target_tracking_rmse_rad": float(np.sqrt(np.mean(np.square(error)))),
        })

    target = _ramped_target(reference, 0.0)
    capture(target)
    intervals = round(DURATION_S / CONTROL_INTERVAL_S)
    for interval in range(intervals):
        target = _ramped_target(reference, interval * CONTROL_INTERVAL_S)
        previous_xy = data.geom_xpos[foot_ids, :2].copy()
        for _ in range(physics_steps):
            simulation.step(model, data, target.tolist())
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                termination_reason = "non_finite_state"
                break
            mujoco.mj_forward(model, data)
            current_xy = data.geom_xpos[foot_ids, :2].copy()
            contacts = np.zeros(len(foot_ids), dtype=bool)
            for contact in data.contact[:data.ncon]:
                other = contact.geom2 if contact.geom1 == ground_id else (
                    contact.geom1 if contact.geom2 == ground_id else -1)
                if int(other) in foot_index:
                    contacts[foot_index[int(other)]] = True
            cumulative_slip[contacts] += np.linalg.norm(
                current_xy[contacts] - previous_xy[contacts], axis=1)
            previous_xy = current_xy
            height = float(data.qpos[2])
            tilt = _tilt_deg(data.qpos[3:7])
            minimum_height = min(minimum_height, height)
            maximum_tilt = max(maximum_tilt, tilt)
            termination_reason = _termination(data, height, tilt)
            if termination_reason:
                break
        if termination_reason:
            if termination_reason == "non_finite_state":
                failure_diagnostic = {
                    "reason": termination_reason,
                    "cutoff_time_s": float(data.time),
                    "qpos_finite": bool(np.isfinite(data.qpos).all()),
                    "qvel_finite": bool(np.isfinite(data.qvel).all()),
                }
            else:
                # A finite cutoff is useful evidence even when it falls between
                # regular 20 ms samples. Its timestamp keeps replay duration honest.
                capture(target)
            break
        capture(target)

    survival = float(data.time)
    dx = float(data.qpos[0]) - origin_x
    summary = GaitSummary(
        gait=name,
        status="UNTRAINED OPEN-LOOP feasibility",
        requested_duration_s=DURATION_S,
        survival_s=survival,
        completed_full_duration=termination_reason is None and math.isclose(
            survival, DURATION_S, abs_tol=model.opt.timestep / 2),
        termination_reason=termination_reason,
        dx_m=dx,
        actual_speed_mps=dx / survival if survival > 0.0 else 0.0,
        final_height_m=float(data.qpos[2]),
        minimum_height_m=minimum_height,
        maximum_tilt_deg=maximum_tilt,
        foot_slip_m=float(np.mean(cumulative_slip)),
        target_tracking_rmse_rad=float(np.sqrt(np.mean(squared_errors))),
    )
    mujoco.mj_saveModel(model, str(output / "model.mjb"))
    np.savez_compressed(output / "states.npz", states=np.asarray(replay.states),
                        times=np.asarray(replay.times), label=replay.label)
    _write_trace(output / "trace.csv", rows)
    if failure_diagnostic is not None:
        (output / "failure.json").write_text(
            json.dumps(failure_diagnostic, indent=2) + "\n", encoding="utf-8")
    recorded_duration = float(replay.times[-1])
    metadata = {
        "schema_version": 1,
        "status": summary.status,
        "render_duration_s": recorded_duration,
        "render_note": ("full 5 second replay" if summary.completed_full_duration else
                        "early cutoff: this recording must not be presented as a 5 second replay"),
        "gait_config": reference.config,
        "reference_speed_mps": reference.speed_mps,
        "reference_height_m": reference.height_m,
        "physics_timestep_s": float(model.opt.timestep),
        "control_interval_s": CONTROL_INTERVAL_S,
        "ramp_duration_s": RAMP_DURATION_S,
        "summary": asdict(summary),
        "model_sha256": hashlib.sha256(simulation.MODEL_PATH.read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "versions": {name: version(name) for name in ("mujoco", "numpy")},
        "non_finite_failure_diagnostic": "failure.json" if failure_diagnostic else None,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return summary


def run(output: Path) -> list[GaitSummary]:
    """Record all three references from independent canonical resets."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = simulation.load_model()
    summaries = [run_gait(model, name, output / name) for name in GAITS]
    (output / "summary.json").write_text(
        json.dumps({"status": "UNTRAINED OPEN-LOOP feasibility",
                    "gaits": [asdict(item) for item in summaries]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summaries = run(args.output)
    print(json.dumps([asdict(item) for item in summaries], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
