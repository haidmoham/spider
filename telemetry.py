"""Sample and persist STAND measurements without a viewer."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from simulation import MODEL_PATH, ROOT, measured_state

if TYPE_CHECKING:
    from runtime import TorsoForcePulse

TELEMETRY_SAMPLE_INTERVAL_S = 0.02
TELEMETRY_HISTORY_SECONDS = 6.0


@dataclass(frozen=True)
class StandTelemetrySample:
    """Compact STAND evidence used by both the viewer and notebook."""

    time_s: float
    force_along_shove_n: float
    torso_displacement_along_shove_m: float
    torso_height_m: float
    torso_angular_speed_radps: float
    support_margin_m: float
    declared_contact_count: int
    front_pair_load_n: float
    middle_pair_load_n: float
    rear_pair_load_n: float



class StandTelemetryRecorder:
    """Sample the declared STAND evidence at a fixed rate, independent of display."""

    def __init__(self, direction: tuple[float, float, float], origin_xy: tuple[float, float]) -> None:
        self.direction = np.asarray(direction[:2])
        self.origin_xy = np.asarray(origin_xy)
        self.next_sample_time_s = 0.0
        self.samples: list[StandTelemetrySample] = []

    def sample_if_due(self, model: mujoco.MjModel, data: mujoco.MjData, perturbation: "TorsoForcePulse") -> None:
        if data.time + 1e-12 < self.next_sample_time_s:
            return
        observed = measured_state(model, data)
        torso_xy = np.asarray(observed.torso_position[:2])
        loads = observed.foot_normal_loads
        force_xy = np.asarray(perturbation.force[:2]) if perturbation.remaining_steps else np.zeros(2)
        self.samples.append(StandTelemetrySample(
            time_s=float(data.time),
            force_along_shove_n=float(force_xy @ self.direction),
            torso_displacement_along_shove_m=float((torso_xy - self.origin_xy) @ self.direction),
            torso_height_m=observed.torso_position[2],
            torso_angular_speed_radps=float(np.linalg.norm(observed.torso_angular_velocity)),
            support_margin_m=np.nan if observed.support_margin is None else observed.support_margin,
            declared_contact_count=len(observed.foot_contacts),
            front_pair_load_n=loads["front_left"] + loads["front_right"],
            middle_pair_load_n=loads["middle_left"] + loads["middle_right"],
            rear_pair_load_n=loads["rear_left"] + loads["rear_right"],
        ))
        self.next_sample_time_s += TELEMETRY_SAMPLE_INTERVAL_S



def write_rollout_trace(path: Path, model: mujoco.MjModel, experiment: str, recorder: StandTelemetryRecorder, metadata: dict | None = None) -> None:
    """Persist compact sampled evidence for the viewer and notebook."""
    if not recorder.samples:
        raise ValueError("trace requires at least one telemetry sample")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {field: np.asarray([getattr(sample, field) for sample in recorder.samples]) for field in StandTelemetrySample.__dataclass_fields__}
    payload["metadata_json"] = np.asarray(json.dumps({
        "schema_version": 2,
        "telemetry_standard": "Telemetry v1",
        "c1n_iteration": "v0.11",
        "experiment": experiment,
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "physics_timestep_s": model.opt.timestep,
        "sample_interval_s": TELEMETRY_SAMPLE_INTERVAL_S,
        **(metadata or {}),
    }))
    np.savez_compressed(path, **payload)
