"""PPO cadence-action branch from the locked ``walk_stable_100`` policy.

The first 18 actions remain joint residuals.  Action 19 selects a cadence in
0.8--1.8 Hz around the immutable 1.1 Hz reference: negative actions have a
0.3 Hz range and positive actions have a 0.7 Hz range.  Importing this module,
loading a policy, and preparing a session never starts training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch.distributions import Normal

from . import simulation
from .chassis_candidate import PARAMETER_MANIFEST
from .reference_training import (
    ACTION_SIZE, CONTROL_INTERVAL_S, RAMP_DURATION_S, ReferenceTrainingSession,
    ResidualActor, effective_control_bounds, fallen, model_signature,
    observation_from, residual_target, reward_terms,
)
from .stride_training import gae_episode


POLICY_KIND = "cadence_action_v1"
BASE_FREQUENCY_HZ = 1.1
CADENCE_MIN_HZ = 0.8
CADENCE_MAX_HZ = 1.8
CADENCE_FILTER_TIME_CONSTANT_S = 0.15
CADENCE_ACTION_SIZE = ACTION_SIZE + 1
SOURCE_FILES = (
    "cadence_action_training.py", "reference_training.py", "chassis_candidate.py",
    "gait_reference.py", "simulation.py", "stride_policy.py", "stride_training.py",
)


class CadenceActor(torch.nn.Module):
    """Legacy residual actor plus a zero-initialized cadence head."""

    def __init__(self, legacy: ResidualActor) -> None:
        super().__init__()
        self.legacy = legacy
        # Linear construction initializes parameters before we zero them. Keep
        # that temporary initialization from advancing the process-global RNG.
        with torch.random.fork_rng():
            self.cadence_head = torch.nn.Linear(128, 1)
        torch.nn.init.zeros_(self.cadence_head.weight)
        torch.nn.init.zeros_(self.cadence_head.bias)
        self.cadence_log_std = torch.nn.Parameter(
            torch.tensor([math.log(0.20)], dtype=self.legacy.log_std.dtype))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = self.legacy.network[:4](observation)
        return torch.cat((self.legacy(observation), self.cadence_head(hidden)), dim=-1)

    @property
    def log_std(self) -> torch.Tensor:
        return torch.cat((self.legacy.log_std, self.cadence_log_std))

    def cadence_parameters(self) -> list[torch.nn.Parameter]:
        return [*self.cadence_head.parameters(), self.cadence_log_std]


def cadence_from_latent(latent: float) -> float:
    """Map one unconstrained action to the asymmetric cadence interval."""
    latent = float(latent)
    if not math.isfinite(latent):
        raise ValueError("cadence latent must be finite")
    bounded = math.tanh(latent)
    return BASE_FREQUENCY_HZ + (0.3 if bounded < 0.0 else 0.7) * bounded


class CadenceController:
    """Shared phase-continuous cadence and residual target controller."""

    def __init__(self, reference, neutral: np.ndarray, bounds: np.ndarray,
                 residual_limit_rad: np.ndarray, residual_filter_time_constant_s: float) -> None:
        if not math.isclose(float(reference.config["frequency_hz"]), BASE_FREQUENCY_HZ):
            raise ValueError("cadence policy requires the immutable 1.1 Hz reference")
        self.reference = reference
        self.neutral = np.asarray(neutral, dtype=float).copy()
        self.bounds = np.asarray(bounds, dtype=float)
        self.residual_limit_rad = np.asarray(residual_limit_rad, dtype=float)
        self.residual_filter_time_constant_s = float(residual_filter_time_constant_s)
        self.reset()

    def reset(self) -> None:
        self.phase_offset_cycles = 0.0
        self.last_time_s: float | None = None
        self.current_cadence_hz = BASE_FREQUENCY_HZ
        self.target_cadence_hz = BASE_FREQUENCY_HZ
        self.cadence_latent = 0.0
        self.previous_residual = np.zeros(ACTION_SIZE)
        self.previous_target = self.neutral.copy()

    def sync(self, time_s: float) -> None:
        """Integrate the held frequency exactly once when observed time advances."""
        time_s = float(time_s)
        if not math.isfinite(time_s):
            raise ValueError("observed time must be finite")
        if self.last_time_s is None:
            self.last_time_s = time_s
            return
        elapsed = time_s - self.last_time_s
        if elapsed < -1e-12:
            raise ValueError("observed time moved backwards; reset the cadence controller")
        if elapsed > 0.0:
            self.phase_offset_cycles += (
                self.current_cadence_hz - BASE_FREQUENCY_HZ) * elapsed
            self.last_time_s = time_s

    def observation(self, observed) -> torch.Tensor:
        self.sync(observed.time)
        result = observation_from(
            observed, self.previous_target, self.reference, self.bounds).clone()
        phase = self.phase_cycles(observed.time) % 1.0
        speed = (self.reference.config["stride_length_m"] * self.current_cadence_hz
                 / self.reference.config["stance_fraction"])
        result[-3:] = torch.tensor(
            [speed, math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)],
            dtype=result.dtype)
        return result

    def phase_cycles(self, time_s: float) -> float:
        return float(time_s) * BASE_FREQUENCY_HZ + self.phase_offset_cycles

    def targets(self, observed, latent: np.ndarray) -> np.ndarray:
        self.sync(observed.time)
        action = np.asarray(latent, dtype=float)
        if action.shape != (CADENCE_ACTION_SIZE,) or not np.isfinite(action).all():
            raise ValueError("cadence policy action must contain 19 finite values")
        self.cadence_latent = float(action[-1])
        self.target_cadence_hz = cadence_from_latent(self.cadence_latent)
        smoothing = 1.0 - math.exp(
            -CONTROL_INTERVAL_S / CADENCE_FILTER_TIME_CONSTANT_S)
        self.current_cadence_hz += smoothing * (
            self.target_cadence_hz - self.current_cadence_hz)
        pose_time = observed.time + self.phase_offset_cycles / BASE_FREQUENCY_HZ
        progress = float(np.clip(observed.time / RAMP_DURATION_S, 0.0, 1.0))
        blend = progress * progress * (3.0 - 2.0 * progress)
        pose = np.asarray(self.reference.pose(pose_time), dtype=float)
        base = self.neutral + blend * (pose - self.neutral)
        target, residual = residual_target(
            base, action[:ACTION_SIZE], self.previous_residual, self.bounds,
            self.residual_limit_rad, self.residual_filter_time_constant_s)
        self.previous_target = target
        self.previous_residual = residual
        return target

    @property
    def diagnostics(self) -> dict[str, float]:
        time_s = 0.0 if self.last_time_s is None else self.last_time_s
        return {
            "phase_cycles": self.phase_cycles(time_s),
            "current_cadence_hz": self.current_cadence_hz,
            "target_cadence_hz": self.target_cadence_hz,
            "cadence_latent": self.cadence_latent,
        }


def _validate_payload(payload: dict, model: mujoco.MjModel) -> None:
    from .policy_acceptance import _same_physics
    if payload.get("policy_kind") != POLICY_KIND:
        raise ValueError("checkpoint is not a cadence-action-v1 policy")
    xml = payload["model_xml"]
    saved_model = mujoco.MjModel.from_xml_string(xml)
    if (hashlib.sha256(xml.encode()).hexdigest() != payload["candidate_xml_sha256"]
            or not _same_physics(model, saved_model)
            or model_signature(model) != payload["model_signature"]):
        raise ValueError("checkpoint physical model or XML integrity mismatch")
    if (payload["control_interval_s"] != CONTROL_INTERVAL_S
            or payload["ramp_duration_s"] != RAMP_DURATION_S
            or payload["cadence_config"] != {
                "base_frequency_hz": BASE_FREQUENCY_HZ,
                "minimum_frequency_hz": CADENCE_MIN_HZ,
                "maximum_frequency_hz": CADENCE_MAX_HZ,
                "filter_time_constant_s": CADENCE_FILTER_TIME_CONSTANT_S,
                "negative_delta_hz": 0.3,
                "positive_delta_hz": 0.7,
            }):
        raise ValueError("checkpoint cadence, control, or ramp contract differs from runtime")


class CadenceResidualPolicy:
    """Runtime loader for the cadence checkpoint schema."""

    def __init__(self, checkpoint: str | Path, model: mujoco.MjModel, seed: int = 201,
                 sampled: bool = False) -> None:
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        _validate_payload(payload, model)
        self.policy_kind = POLICY_KIND
        legacy = ResidualActor()
        self.actor = CadenceActor(legacy)
        self.actor.load_state_dict(payload["actor"], strict=True)
        self.actor.eval()
        self.model = model
        self.bounds = effective_control_bounds(model)
        from .gait_reference import GaitReference
        self.reference = GaitReference(
            model, "tripod", foot_centers=np.asarray(payload["reference_foot_centers"]))
        self.reference.config = dict(payload["reference_config"])
        self.reference.height_m = float(self.reference.config["height_m"])
        self.reference.speed_mps = float(payload["reference_speed_mps"])
        self.neutral = np.asarray(payload["neutral_targets"], dtype=float)
        self.residual_limit_rad = np.asarray(payload["residual_limit_rad"], dtype=float)
        self.filter_time_constant_s = float(payload["filter_time_constant_s"])
        self.ppo, self.reward, self.settings = (
            dict(payload["ppo"]), dict(payload["reward"]), dict(payload["settings"]))
        self.updates = int(payload["updates"])
        self.parent_checkpoint = dict(payload["parent_checkpoint"])
        self.sampled, self.seed = bool(sampled), int(seed)
        self.generator = torch.Generator().manual_seed(self.seed)
        self.cadence_generator = torch.Generator().manual_seed(self.seed + 1_000_000_007)
        self.controller = CadenceController(
            self.reference, self.neutral, self.bounds, self.residual_limit_rad,
            self.filter_time_constant_s)

    def reset(self) -> None:
        self.controller.reset()
        self.generator.manual_seed(self.seed)
        self.cadence_generator.manual_seed(self.seed + 1_000_000_007)

    def targets(self, observed) -> np.ndarray:
        obs = self.controller.observation(observed)
        with torch.no_grad():
            mean = self.actor(obs)
            if self.sampled:
                std = self.actor.log_std.clamp(-3.5, -0.5).exp()
                legacy = torch.normal(mean[:ACTION_SIZE], std[:ACTION_SIZE],
                                      generator=self.generator)
                cadence = torch.normal(mean[ACTION_SIZE:], std[ACTION_SIZE:],
                                       generator=self.cadence_generator)
                latent = torch.cat((legacy, cadence))
            else:
                latent = mean
        return self.controller.targets(observed, latent.numpy())

    @property
    def diagnostics(self) -> dict[str, float]:
        return self.controller.diagnostics


class CadenceTrainingSession(ReferenceTrainingSession):
    """Additional PPO updates from the locked stable policy."""

    @classmethod
    def from_stable(cls, checkpoint: str | Path, output_directory: str | Path):
        base = ReferenceTrainingSession.from_checkpoint(checkpoint, output_directory)
        manifest = json.loads((simulation.ROOT / "artifacts/walk_stable_100/manifest.json").read_text())
        if (base.updates != 100 or base.parent_checkpoint["sha256"] !=
                manifest["sha256"]["walk_stable_100.pt"]):
            raise ValueError("parent must be the locked walk_stable_100 checkpoint")
        session = cls.__new__(cls)
        session.__dict__.update(base.__dict__)
        if not math.isclose(float(session.reference.config["frequency_hz"]), BASE_FREQUENCY_HZ):
            raise ValueError("stable parent must use the 1.1 Hz locked cadence")
        legacy = session.actor
        session.actor = CadenceActor(legacy)
        # The loaded Adam object remains bound to the exact legacy parameters and
        # moments.  The new parameters receive a separate fresh state group.
        session.actor_optimizer.add_param_group({
            "params": session.actor.cadence_parameters(), "lr": session.ppo["actor_lr"]})
        session.cadence_seed_offset = 1_000_000_007
        session.parent_payload_parent = base.parent_checkpoint
        session.parent_checkpoint = {
            **base.parent_checkpoint,
            "policy_kind": "reference_residual_v1",
        }
        session.treatment_changes = {
            "policy_action": {"previous": "18 joint residuals",
                              "selected": "18 joint residuals plus cadence"}}
        return session

    def _episode(self, episode_seed: int) -> dict[str, torch.Tensor]:
        data = mujoco.MjData(self.model)
        simulation.reset(self.model, data)
        data.qpos[7:] = self.neutral
        data.ctrl[:] = self.neutral
        mujoco.mj_forward(self.model, data)
        observed = simulation.measured_state(self.model, data)
        controller = CadenceController(
            self.reference, self.neutral, self.bounds, self.residual_limit_rad,
            self.filter_time_constant_s)
        legacy_generator = torch.Generator().manual_seed(episode_seed)
        cadence_generator = torch.Generator().manual_seed(
            episode_seed + self.cadence_seed_offset)
        records = []
        for step in range(self.settings["horizon"]):
            obs = controller.observation(observed)
            with torch.no_grad():
                mean = self.actor(obs)
                std = self.actor.log_std.clamp(-3.5, -0.5).exp()
                legacy = torch.normal(mean[:ACTION_SIZE], std[:ACTION_SIZE],
                                      generator=legacy_generator)
                cadence = torch.normal(mean[ACTION_SIZE:], std[ACTION_SIZE:],
                                       generator=cadence_generator)
                latent = torch.cat((legacy, cadence))
                old_logp = Normal(mean, std).log_prob(latent).sum()
                value = self.critic(obs).squeeze(-1)
            previous_residual = controller.previous_residual.copy()
            target = controller.targets(observed, latent.numpy())
            previous_x = observed.torso_position[0]
            for _ in range(self.settings["physics_steps"]):
                simulation.step(self.model, data, target.tolist())
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("non-finite candidate physics state")
            observed = simulation.measured_state(self.model, data)
            controller.sync(observed.time)
            terms = reward_terms(
                observed, previous_x, controller.previous_residual, previous_residual,
                self.residual_limit_rad, weights=self.reward)
            fell = fallen(observed, self.settings["fall_height_m"])
            total = sum(v for name, v in terms.items() if name.endswith("_weighted"))
            total -= self.reward["fall"] if fell else 0.0
            with torch.no_grad():
                next_obs = controller.observation(observed)
                next_value = self.critic(next_obs).squeeze(-1)
            records.append((obs, latent, old_logp, torch.tensor(total, dtype=torch.float32),
                            value, next_value, torch.tensor(0.0 if fell else 1.0)))
            self.rows.append({"update": self.updates + 1, "episode_seed": episode_seed,
                              "step": step, "total_reward": total, "fell": fell,
                              **controller.diagnostics, **terms})
            if fell:
                break
        names = ("obs", "actions", "old_logp", "rewards", "values", "next_values", "bootstrap")
        columns = list(zip(*records))
        episode = {name: torch.stack(values) for name, values in zip(names, columns)}
        episode["advantages"], episode["targets"] = gae_episode(
            episode["rewards"], episode["values"], episode["next_values"],
            episode["bootstrap"], self.ppo["gamma"], self.ppo["gae_lambda"])
        return episode

    @property
    def cadence_config(self) -> dict[str, float]:
        return {"base_frequency_hz": BASE_FREQUENCY_HZ,
                "minimum_frequency_hz": CADENCE_MIN_HZ,
                "maximum_frequency_hz": CADENCE_MAX_HZ,
                "filter_time_constant_s": CADENCE_FILTER_TIME_CONSTANT_S,
                "negative_delta_hz": 0.3, "positive_delta_hz": 0.7}

    def _save(self) -> Path:
        path = self.output_directory / f"checkpoint-{self.updates:05d}.pt"
        xml = self.model_xml
        torch.save({
            "status": "PREPARED cadence-action branch" if self.updates == 100 else
                      "TRAINED cadence-action checkpoint",
            "policy_kind": POLICY_KIND, "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "updates": self.updates, "seed": self.seed, "ppo": self.ppo,
            "reward": self.reward, "settings": self.settings,
            "optimizer_rng_state": self.generator.get_state(),
            "cadence_seed_offset": self.cadence_seed_offset,
            "parent_checkpoint": self.parent_checkpoint,
            "parent_payload_parent": self.parent_payload_parent,
            "treatment_changes": self.treatment_changes,
            "cadence_config": self.cadence_config,
            "reference_config": self.reference.config,
            "reference_speed_mps": self.reference.speed_mps,
            "reference_foot_centers": self.reference.foot_centers.tolist(),
            "model_xml": xml,
            "candidate_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
            "model_signature": model_signature(self.model),
            "control_interval_s": CONTROL_INTERVAL_S,
            "ramp_duration_s": RAMP_DURATION_S,
            "neutral_targets": self.neutral.tolist(),
            "residual_limit_rad": self.residual_limit_rad.tolist(),
            "filter_time_constant_s": self.filter_time_constant_s,
        }, path)
        return path

    def _prepare_output(self) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=False)
        hashes = {}
        for name in SOURCE_FILES:
            source = Path(__file__).with_name(name)
            shutil.copyfile(source, self.output_directory / name)
            hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        (self.output_directory / "candidate.xml").write_text(self.model_xml, encoding="utf-8")
        config = {
            "status": "RESTORED locked stable checkpoint; no additional updates yet",
            "policy_kind": POLICY_KIND, "seed": self.seed,
            "starting_updates": self.updates, "parent_checkpoint": self.parent_checkpoint,
            "parent_payload_parent": self.parent_payload_parent,
            "treatment_changes": self.treatment_changes,
            "cadence_config": self.cadence_config,
            "reference_config": self.reference.config,
            "reference_speed_mps": self.reference.speed_mps,
            "control_interval_s": CONTROL_INTERVAL_S,
            "ramp_duration_s": RAMP_DURATION_S,
            "chassis_manifest": PARAMETER_MANIFEST, "ppo": self.ppo,
            "reward": self.reward, "settings": self.settings,
            "residual_limit_rad": self.residual_limit_rad.tolist(),
            "filter_time_constant_s": self.filter_time_constant_s,
            "source_sha256": hashes,
            "candidate_xml_sha256": hashlib.sha256(self.model_xml.encode()).hexdigest(),
            "model_signature": model_signature(self.model),
            "versions": {name: version(name) for name in ("torch", "numpy", "mujoco")},
        }
        (self.output_directory / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-stable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--updates", type=int, help="Number of additional PPO updates")
    action.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    session = CadenceTrainingSession.from_stable(args.from_stable, args.output)
    print(session.prepare() if args.prepare_only else session.train(args.updates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
