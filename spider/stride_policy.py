"""Fresh phase-guided stride policy for the canonical C-1N model."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .simulation import MeasuredState, neutral_targets


JOINT_LOW = np.tile((-0.65, -0.70, 0.15), 6)
JOINT_HIGH = np.tile((0.65, 0.55, 1.35), 6)
JOINT_CENTER = (JOINT_LOW + JOINT_HIGH) / 2.0
JOINT_HALF_RANGE = (JOINT_HIGH - JOINT_LOW) / 2.0
NEUTRAL = np.asarray(neutral_targets(), dtype=np.float64)
NEUTRAL_LATENT = np.arctanh((NEUTRAL - JOINT_CENTER) / JOINT_HALF_RANGE)
OBSERVATION_SIZE = 68
ACTION_SIZE = 18
HIDDEN_SIZE = 128
PHASE_HZ = 1.25
SETTINGS = {
    "physics_steps": 20,
    "horizon": 256,
    "fall_height_m": 0.25,
    "target_speed_mps": 0.25,
    "target_height_m": 0.43,
    "phase_hz": PHASE_HZ,
    "stance_fraction": 0.60,
}


class StrideActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(OBSERVATION_SIZE, HIDDEN_SIZE), torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN_SIZE, ACTION_SIZE),
        )
        self.log_std = torch.nn.Parameter(torch.log(torch.tensor(
            [0.10, 0.10, 0.12] * 6, dtype=torch.float32
        ) / torch.tensor(JOINT_HALF_RANGE, dtype=torch.float32)))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


def initialize_actor(actor: StrideActor) -> None:
    layers = [module for module in actor.network if isinstance(module, torch.nn.Linear)]
    for layer in layers:
        torch.nn.init.orthogonal_(layer.weight, math.sqrt(2.0))
        torch.nn.init.zeros_(layer.bias)
    torch.nn.init.orthogonal_(layers[-1].weight, 0.01)
    with torch.no_grad():
        layers[-1].bias.copy_(torch.tensor(NEUTRAL_LATENT, dtype=torch.float32))


def build_actor(seed: int | None = None) -> StrideActor:
    with torch.random.fork_rng():
        if seed is not None:
            torch.manual_seed(seed)
        actor = StrideActor()
        initialize_actor(actor)
    return actor


def normalized_target(target: np.ndarray) -> np.ndarray:
    return np.clip((target - JOINT_CENTER) / JOINT_HALF_RANGE, -1.0, 1.0)


def targets_from_latent(latent: torch.Tensor) -> np.ndarray:
    if latent.shape != (ACTION_SIZE,) or not torch.isfinite(latent).all():
        raise ValueError("latent action must contain 18 finite values")
    value = latent.detach().cpu().numpy().astype(np.float64)
    return JOINT_CENTER + JOINT_HALF_RANGE * np.tanh(value)


def observation_from(observed: MeasuredState, previous_target: np.ndarray,
                     target_speed: float, phase: float) -> torch.Tensor:
    values = np.asarray([
        *(np.asarray(observed.torso_velocity) / 1.0),
        *(np.asarray(observed.torso_angular_velocity) / 3.0),
        *observed.torso_orientation,
        *normalized_target(np.asarray(observed.joint_positions)),
        *(np.asarray(observed.joint_velocities) / 10.0),
        (observed.torso_position[2] - SETTINGS["target_height_m"]) / 0.10,
        *normalized_target(np.asarray(previous_target)),
        target_speed,
        math.sin(2.0 * math.pi * phase),
        math.cos(2.0 * math.pi * phase),
    ], dtype=np.float32)
    values = np.clip(values, -5.0, 5.0)
    if values.shape != (OBSERVATION_SIZE,) or not np.isfinite(values).all():
        raise ValueError("stride observation must contain 68 finite values")
    return torch.from_numpy(values)


class StridePolicy:
    """Load and execute one fresh phase-guided stride checkpoint."""

    def __init__(self, checkpoint: str | Path, model: Any, seed: int = 201,
                 sampled: bool = False) -> None:
        self.checkpoint = Path(checkpoint)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.actor = build_actor()
        self.actor.load_state_dict(payload["actor"], strict=True)
        self.actor.eval()
        self.settings = dict(payload["settings"])
        fixed = {key: value for key, value in self.settings.items() if key != "target_speed_mps"}
        expected = {key: value for key, value in SETTINGS.items() if key != "target_speed_mps"}
        target_speed = self.settings.get("target_speed_mps")
        if fixed != expected or not isinstance(target_speed, (int, float)) or isinstance(
            target_speed, bool
        ) or not math.isfinite(target_speed) or target_speed <= 0:
            raise ValueError("checkpoint settings do not match the stride policy contract")
        self.updates = int(payload["updates"])
        self.sampled = bool(sampled)
        self.seed = int(seed)
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        ranges = np.asarray(model.actuator_ctrlrange)
        if ranges.shape != (18, 2) or np.any(JOINT_LOW < ranges[:, 0]) or np.any(
            JOINT_HIGH > ranges[:, 1]
        ):
            raise ValueError("stride target ranges exceed model actuator ranges")
        self._control_dt = self.settings["physics_steps"] * float(model.opt.timestep)
        self.description = (
            f"Fresh phase-guided PPO-{self.updates} "
            f"{'sampled' if sampled else 'mean'} stride policy."
        )
        self.config = {
            "architecture": "68-128-128-18",
            "joint_low_rad": JOINT_LOW.tolist(),
            "joint_high_rad": JOINT_HIGH.tolist(),
            "target_speed_mps": self.settings["target_speed_mps"],
            "phase_hz": self.settings["phase_hz"],
            "training_tuning": payload.get("tuning", {}),
            "sampled": self.sampled,
            "seed": self.seed,
            "checkpoint_sha256": self.checkpoint_sha256,
        }
        self._generator = torch.Generator(device="cpu")
        self.reset()

    def reset(self) -> None:
        self._generator.manual_seed(self.seed)
        self.previous_target = NEUTRAL.copy()

    def targets(self, observed: MeasuredState) -> tuple[float, ...]:
        phase = (observed.time * self.settings["phase_hz"]) % 1.0
        observation = observation_from(
            observed, self.previous_target, self.settings["target_speed_mps"], phase)
        with torch.inference_mode():
            latent = self.actor(observation)
            if self.sampled:
                std = self.actor.log_std.clamp(-3.5, -0.3).exp()
                latent = torch.normal(latent, std, generator=self._generator)
        target = targets_from_latent(latent)
        self.previous_target = target
        return tuple(float(value) for value in target)
