"""Runtime adapter for the preserved PPO-100 C-1N policy.

This module performs policy inference only. The caller owns simulation stepping
and must hold each returned target for ``settings["physics_steps"]`` steps.
"""

from __future__ import annotations

from hashlib import sha256
from math import exp, isfinite
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .simulation import FOOT_NAMES, MeasuredState, neutral_targets


OBSERVATION_SIZE = 47
ACTION_SIZE = 18
HIDDEN_SIZE = 64
LOWER_OFFSET = np.tile((0.0, -0.12, 0.12), len(FOOT_NAMES))
LOWER_RAMP_SECONDS = 2.0
SMOOTH_TAU_SECONDS = 0.08
TREATMENTS = {"baseline", "lower", "smooth", "stalk"}
EXPECTED_ACTUATORS = tuple(
    f"{leg}_{joint}_motor"
    for leg in FOOT_NAMES
    for joint in ("coxa", "hip", "knee")
)


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, Real):
        return isfinite(float(value))
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_tree(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return isinstance(value, str)


class PPOPolicy:
    """Load a frozen C-1N PPO actor and return absolute actuator targets."""

    def __init__(
        self,
        checkpoint: str | Path,
        model: Any,
        seed: int = 201,
        sampled: bool = False,
        treatment: str = "baseline",
    ) -> None:
        self.checkpoint = Path(checkpoint)
        if treatment not in TREATMENTS:
            raise ValueError(f"treatment must be one of {sorted(TREATMENTS)}, got {treatment!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self._validate_payload(payload)
        # Linear constructors initialize parameters from the global RNG. Restore
        # that state because loading saved weights makes the initialization moot.
        with torch.random.fork_rng():
            self.actor = torch.nn.Sequential(
                torch.nn.Linear(OBSERVATION_SIZE, HIDDEN_SIZE),
                torch.nn.Tanh(),
                torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                torch.nn.Tanh(),
                torch.nn.Linear(HIDDEN_SIZE, ACTION_SIZE),
            )
        self.actor.load_state_dict(payload["actor"], strict=True)
        self.actor.eval()

        self.settings = dict(payload["settings"])
        self.updates = int(payload["updates"])
        self.checkpoint_sha256 = sha256(self.checkpoint.read_bytes()).hexdigest()
        self.sampled = bool(sampled)
        self.treatment = treatment
        self.seed = seed
        source = f"Preserved PPO-{self.updates} {'sampled' if self.sampled else 'mean'} policy"
        self.description = {
            "baseline": f"{source} target.",
            "lower": f"{source} with an unvalidated 2 s leg-lowering control intervention.",
            "smooth": f"{source} with an unvalidated 0.08 s low-pass control intervention.",
            "stalk": f"{source} with unvalidated lowering and low-pass control interventions.",
        }[treatment]
        self.config = {
            "treatment": treatment,
            "sampled": self.sampled,
            "seed": seed,
            "physics_steps": int(self.settings["physics_steps"]),
            "bound_rad": float(self.settings["bound_rad"]),
            "std": float(self.settings["std"]),
            "lower_offset_rad_per_leg": (
                [0.0, -0.12, 0.12] if treatment in {"lower", "stalk"} else None
            ),
            "lower_ramp_seconds": (
                LOWER_RAMP_SECONDS if treatment in {"lower", "stalk"} else None
            ),
            "smooth_tau_seconds": (
                SMOOTH_TAU_SECONDS if treatment in {"smooth", "stalk"} else None
            ),
            "checkpoint_sha256": self.checkpoint_sha256,
        }

        self._neutral = np.asarray(neutral_targets(), dtype=np.float64)
        self._ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float64).copy()
        if self._ctrlrange.shape != (ACTION_SIZE, 2) or not np.isfinite(self._ctrlrange).all():
            raise ValueError("model actuator_ctrlrange must be a finite 18 x 2 array")
        if np.any(self._ctrlrange[:, 0] > self._neutral) or np.any(
            self._neutral > self._ctrlrange[:, 1]
        ):
            raise ValueError("neutral targets must lie inside every actuator control range")
        self._validate_actuator_names(model)
        saved_bound = float(self.settings["bound_rad"])
        room = np.minimum(self._neutral - self._ctrlrange[:, 0], self._ctrlrange[:, 1] - self._neutral)
        self._bound = np.minimum(saved_bound, room)
        self._control_dt = int(self.settings["physics_steps"]) * float(model.opt.timestep)
        if not isfinite(self._control_dt) or self._control_dt <= 0.0:
            raise ValueError("model timestep and saved physics_steps must define a positive interval")
        self._generator = torch.Generator(device="cpu")
        self.reset()

    @staticmethod
    def _validate_payload(payload: Any) -> None:
        required = {"actor", "settings", "updates"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError(f"checkpoint must contain {sorted(required)}")
        settings = payload["settings"]
        if not isinstance(settings, dict) or not _finite_tree(settings):
            raise ValueError("checkpoint settings must be a finite dictionary")
        for key in ("bound_rad", "std", "physics_steps", "horizon", "fall_height_m"):
            if key not in settings:
                raise ValueError(f"checkpoint settings are missing {key!r}")
        if float(settings["bound_rad"]) < 0.0 or float(settings["std"]) < 0.0:
            raise ValueError("bound_rad and std must be nonnegative")
        steps = settings["physics_steps"]
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("physics_steps must be a positive integer")
        horizon = settings["horizon"]
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if float(settings["fall_height_m"]) <= 0.0:
            raise ValueError("fall_height_m must be positive")
        updates = payload["updates"]
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("updates must be a nonnegative integer")
        expected_shapes = {
            "0.weight": (HIDDEN_SIZE, OBSERVATION_SIZE),
            "0.bias": (HIDDEN_SIZE,),
            "2.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
            "2.bias": (HIDDEN_SIZE,),
            "4.weight": (ACTION_SIZE, HIDDEN_SIZE),
            "4.bias": (ACTION_SIZE,),
        }
        actor = payload["actor"]
        if not isinstance(actor, dict) or set(actor) != set(expected_shapes):
            raise ValueError("checkpoint actor does not match the 47-64-64-18 topology")
        for name, shape in expected_shapes.items():
            tensor = actor[name]
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise ValueError(f"checkpoint actor tensor {name!r} has the wrong shape")
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError(f"checkpoint actor tensor {name!r} must be finite float32")

    @staticmethod
    def _validate_actuator_names(model: Any) -> None:
        if not hasattr(model, "actuator"):
            return
        names = tuple(str(model.actuator(index).name) for index in range(ACTION_SIZE))
        if names != EXPECTED_ACTUATORS:
            raise ValueError("model actuator names or order do not match the preserved policy")

    def reset(self) -> None:
        """Reset local sampling, ramp, and filter state."""
        self._generator.manual_seed(self.seed)
        self._elapsed = 0.0
        self._smoothed_offset = np.zeros(ACTION_SIZE, dtype=np.float64)

    @staticmethod
    def _observation(observed: MeasuredState) -> torch.Tensor:
        values = np.asarray(
            [
                *observed.torso_velocity,
                *observed.torso_angular_velocity,
                *observed.torso_orientation,
                *observed.joint_positions,
                *observed.joint_velocities,
                observed.torso_position[2],
            ],
            dtype=np.float32,
        )
        if values.shape != (OBSERVATION_SIZE,) or not np.isfinite(values).all():
            raise ValueError("observed state must produce 47 finite policy values")
        return torch.from_numpy(values)

    def targets(self, observed: MeasuredState) -> tuple[float, ...]:
        """Return one absolute 18-actuator command without advancing physics."""
        observation = self._observation(observed)
        with torch.inference_mode():
            latent = self.actor(observation)
            if self.sampled:
                latent = torch.normal(
                    latent,
                    float(self.settings["std"]),
                    generator=self._generator,
                )
            learned = self._neutral + self._bound * torch.tanh(latent).numpy().astype(np.float64)

        learned_offset = learned - self._neutral
        if self.treatment in {"smooth", "stalk"}:
            alpha = 1.0 - exp(-self._control_dt / SMOOTH_TAU_SECONDS)
            self._smoothed_offset += alpha * (learned_offset - self._smoothed_offset)
            learned_offset = self._smoothed_offset.copy()

        command = self._neutral + learned_offset
        if self.treatment in {"lower", "stalk"}:
            progress = min(1.0, self._elapsed / LOWER_RAMP_SECONDS)
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            command += smooth_progress * LOWER_OFFSET
        self._elapsed += self._control_dt
        command = np.clip(command, self._ctrlrange[:, 0], self._ctrlrange[:, 1])
        if command.shape != (ACTION_SIZE,) or not np.isfinite(command).all():
            raise ValueError("policy produced an invalid actuator command")
        return tuple(float(value) for value in command)
