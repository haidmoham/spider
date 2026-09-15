"""Opt-in PPO residual training around the approved candidate tripod reference.

Importing this module and constructing a session never starts training. The CLI
requires an explicit positive ``--updates`` value.
"""

from __future__ import annotations

import argparse
import csv
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
from .chassis_candidate import PARAMETER_MANIFEST, candidate_xml, load_candidate, reset_candidate
from .gait_reference import GaitReference
from .stride_policy import ACTION_SIZE, HIDDEN_SIZE, OBSERVATION_SIZE
from .stride_training import gae_episode


CONTROL_INTERVAL_S = 0.02
RAMP_DURATION_S = 0.8
RESIDUAL_LIMIT_RAD = np.tile((0.10, 0.12, 0.14), 6)
RESIDUAL_FILTER_TIME_CONSTANT_S = 0.045
PPO = {"gamma": 0.995, "gae_lambda": 0.95, "clip_ratio": 0.2,
       "actor_lr": 3e-4, "critic_lr": 1e-3, "actor_epochs": 5, "critic_epochs": 5,
       "minibatch_size": 512, "max_grad_norm": 0.5, "target_kl": 0.015,
       "entropy_coefficient": 0.002, "episodes_per_update": 8}
REWARD = {"forward": 2.0, "height": 0.5, "lateral": 0.3, "angular": 0.2,
          "tilt": 0.5, "residual": 0.08, "residual_rate": 0.08, "fall": 20.0}
SETTINGS = {"physics_steps": 10, "horizon": 250, "fall_height_m": 0.25}
SOURCE_FILES = ("reference_training.py", "chassis_candidate.py", "gait_reference.py",
                "simulation.py", "stride_policy.py", "stride_training.py")


class ResidualActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(OBSERVATION_SIZE, HIDDEN_SIZE), torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN_SIZE, ACTION_SIZE),
        )
        self.log_std = torch.nn.Parameter(torch.full((ACTION_SIZE,), math.log(0.20)))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


def build_actor(seed: int) -> ResidualActor:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        actor = ResidualActor()
        layers = [module for module in actor.network if isinstance(module, torch.nn.Linear)]
        for layer in layers:
            torch.nn.init.orthogonal_(layer.weight, math.sqrt(2.0))
            torch.nn.init.zeros_(layer.bias)
        # The prepared deterministic policy is exactly the approved reference.
        torch.nn.init.zeros_(layers[-1].weight)
        return actor


def build_critic(seed: int) -> torch.nn.Sequential:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        critic = torch.nn.Sequential(
            torch.nn.Linear(OBSERVATION_SIZE, HIDDEN_SIZE), torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN_SIZE, 1),
        )
        for layer in (module for module in critic if isinstance(module, torch.nn.Linear)):
            torch.nn.init.orthogonal_(layer.weight, math.sqrt(2.0))
            torch.nn.init.zeros_(layer.bias)
        torch.nn.init.orthogonal_(critic[-1].weight, 1.0)
        return critic


def reference_target(reference: GaitReference, time_s: float,
                     neutral: np.ndarray) -> np.ndarray:
    progress = float(np.clip(time_s / RAMP_DURATION_S, 0.0, 1.0))
    blend = progress * progress * (3.0 - 2.0 * progress)
    pose = np.asarray(reference.pose(time_s), dtype=float)
    return neutral + blend * (pose - neutral)


def observation_from(observed, previous_target: np.ndarray, reference: GaitReference,
                     bounds: np.ndarray) -> torch.Tensor:
    center = np.mean(bounds, axis=1)
    half_range = (bounds[:, 1] - bounds[:, 0]) / 2.0
    values = np.asarray([
        *(np.asarray(observed.torso_velocity) / 1.0),
        *(np.asarray(observed.torso_angular_velocity) / 3.0),
        *observed.torso_orientation,
        *((np.asarray(observed.joint_positions) - center) / half_range),
        *(np.asarray(observed.joint_velocities) / 10.0),
        (observed.torso_position[2] - reference.height_m) / 0.10,
        *((np.asarray(previous_target) - center) / half_range),
        reference.speed_mps,
        math.sin(2.0 * math.pi * ((observed.time * reference.config["frequency_hz"]) % 1.0)),
        math.cos(2.0 * math.pi * ((observed.time * reference.config["frequency_hz"]) % 1.0)),
    ], dtype=np.float32)
    values = np.clip(values, -5.0, 5.0)
    if values.shape != (OBSERVATION_SIZE,) or not np.isfinite(values).all():
        raise ValueError("reference observation must contain 68 finite values")
    return torch.from_numpy(values)


def effective_control_bounds(model: mujoco.MjModel) -> np.ndarray:
    """Intersect each position actuator range with its joint range."""
    bounds = np.asarray(model.actuator_ctrlrange, dtype=float).copy()
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        bounds[actuator_id, 0] = max(bounds[actuator_id, 0], model.jnt_range[joint_id, 0])
        bounds[actuator_id, 1] = min(bounds[actuator_id, 1], model.jnt_range[joint_id, 1])
    return bounds


def residual_target(base: np.ndarray, latent: np.ndarray, previous_residual: np.ndarray,
                    ctrlrange: np.ndarray, residual_limit: np.ndarray = RESIDUAL_LIMIT_RAD,
                    filter_time_constant_s: float = RESIDUAL_FILTER_TIME_CONSTANT_S,
                    interval_s: float = CONTROL_INTERVAL_S) -> tuple[np.ndarray, np.ndarray]:
    """Apply a bounded low-pass residual and return target plus residual state."""
    limit = np.asarray(residual_limit, dtype=float)
    if limit.shape != (ACTION_SIZE,) or np.any(limit <= 0) or filter_time_constant_s <= 0:
        raise ValueError("residual limits and filter time constant must be positive")
    smoothing = 1.0 - math.exp(-interval_s / filter_time_constant_s)
    desired = np.tanh(np.asarray(latent, dtype=float)) * limit
    residual = previous_residual + smoothing * (desired - previous_residual)
    target = np.clip(np.asarray(base) + residual, ctrlrange[:, 0], ctrlrange[:, 1])
    return target, residual


def reward_terms(observed, previous_x_m: float, residual: np.ndarray,
                 previous_residual: np.ndarray, residual_limit: np.ndarray = RESIDUAL_LIMIT_RAD,
                 interval_s: float = CONTROL_INTERVAL_S, weights: dict = REWARD,
                 ) -> dict[str, float]:
    velocity = np.asarray(observed.torso_velocity)
    angular = np.asarray(observed.torso_angular_velocity)
    quat = np.asarray(observed.torso_orientation)
    raw = {
        "forward": float((observed.torso_position[0] - previous_x_m) / interval_s),
        "height": math.exp(-((observed.torso_position[2] - 0.37) / 0.06) ** 2),
        "lateral": -min(4.0, float((velocity[1] / 0.25) ** 2)),
        "angular": -min(4.0, float(np.sum(np.square(angular / 0.8)))),
        "tilt": -min(4.0, float(np.sum(np.square(quat[1:3] / math.sin(0.125))))),
        "residual": -float(np.mean(np.square(residual / residual_limit))),
        "residual_rate": -float(np.mean(np.square(
            (residual - previous_residual) / residual_limit))),
    }
    return {f"{name}_{kind}": value if kind == "raw" else weights[name] * value
            for name, value in raw.items() for kind in ("raw", "weighted")}


def fallen(observed, fall_height_m: float) -> bool:
    quat = np.asarray(observed.torso_orientation, dtype=float)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm == 0.0:
        return True
    quat /= norm
    tilt_deg = math.degrees(math.acos(float(np.clip(
        1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2), -1.0, 1.0))))
    return observed.torso_position[2] < fall_height_m or tilt_deg > 60.0


def model_signature(model: mujoco.MjModel) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray([model.nq, model.nv, model.nu, model.nbody, model.ngeom,
                              model.nsite, model.opt.timestep], dtype=np.float64).tobytes())
    for values in (model.body_mass, model.body_inertia, model.geom_size, model.geom_friction,
                   model.jnt_range, model.actuator_ctrlrange, model.actuator_gainprm):
        digest.update(np.ascontiguousarray(values).tobytes())
    for kind, count in ((model.body, model.nbody), (model.geom, model.ngeom),
                        (model.joint, model.njnt), (model.actuator, model.nu)):
        digest.update("\0".join(kind(index).name for index in range(count)).encode())
    return digest.hexdigest()


class ReferenceResidualPolicy:
    """Evaluation-ready deterministic loader for a residual checkpoint."""

    def __init__(self, checkpoint: str | Path, model: mujoco.MjModel, seed: int = 201,
                 sampled: bool = False) -> None:
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        from .policy_acceptance import _same_physics
        saved_model = mujoco.MjModel.from_xml_string(payload["model_xml"])
        if (hashlib.sha256(payload["model_xml"].encode()).hexdigest() != payload["candidate_xml_sha256"]
                or not _same_physics(model, saved_model)):
            raise ValueError("checkpoint physical model or XML integrity mismatch")
        if (payload["control_interval_s"] != CONTROL_INTERVAL_S
                or payload["ramp_duration_s"] != RAMP_DURATION_S):
            raise ValueError("checkpoint cadence or ramp contract differs from this runtime")
        if model_signature(model) != payload["model_signature"]:
            raise ValueError("checkpoint candidate model does not match supplied model")
        self.actor = build_actor(0)
        self.actor.load_state_dict(payload["actor"], strict=True)
        self.actor.eval()
        self.model = model
        self.bounds = effective_control_bounds(model)
        self.reference = GaitReference(
            model, "tripod", foot_centers=np.asarray(payload["reference_foot_centers"]))
        self.reference.config = dict(payload["reference_config"])
        self.reference.height_m = float(self.reference.config["height_m"])
        self.reference.speed_mps = float(payload["reference_speed_mps"])
        self.residual_limit_rad = np.asarray(payload["residual_limit_rad"], dtype=float)
        self.filter_time_constant_s = float(payload["filter_time_constant_s"])
        self.ppo = dict(payload["ppo"])
        self.reward = dict(payload["reward"])
        self.settings = dict(payload["settings"])
        self.updates = int(payload["updates"])
        self.neutral = np.asarray(payload["neutral_targets"], dtype=float)
        self.previous_residual = np.zeros(ACTION_SIZE)
        self.previous_target = self.neutral.copy()
        self.sampled = bool(sampled)
        self.seed = int(seed)
        self.generator = torch.Generator().manual_seed(self.seed)

    def reset(self) -> None:
        self.previous_residual.fill(0.0)
        self.previous_target = self.neutral.copy()
        self.generator.manual_seed(self.seed)

    def targets(self, observed) -> np.ndarray:
        observation = observation_from(
            observed, self.previous_target, self.reference, self.bounds)
        with torch.no_grad():
            mean = self.actor(observation)
            latent = (torch.normal(mean, self.actor.log_std.clamp(-3.5, -0.5).exp(),
                                   generator=self.generator) if self.sampled else mean).numpy()
        base = reference_target(self.reference, observed.time, self.neutral)
        target, self.previous_residual = residual_target(
            base, latent, self.previous_residual, self.bounds,
            self.residual_limit_rad, self.filter_time_constant_s)
        self.previous_target = target
        return target


class ReferenceTrainingSession:
    def __init__(self, output_directory: str | Path, seed: int,
                 residual_limit_rad: np.ndarray = RESIDUAL_LIMIT_RAD,
                 filter_time_constant_s: float = RESIDUAL_FILTER_TIME_CONSTANT_S) -> None:
        self.output_directory = Path(output_directory)
        self.seed = int(seed)
        self.residual_limit_rad = np.asarray(residual_limit_rad, dtype=float).copy()
        self.filter_time_constant_s = float(filter_time_constant_s)
        self.model_xml = candidate_xml()
        self.model = mujoco.MjModel.from_xml_string(self.model_xml)
        self.bounds = effective_control_bounds(self.model)
        self.ppo, self.reward, self.settings = dict(PPO), dict(REWARD), dict(SETTINGS)
        self.generator = torch.Generator().manual_seed(self.seed)
        original_reference = GaitReference(simulation.load_model(), "tripod")
        self.reference = GaitReference(
            self.model, "tripod", foot_centers=original_reference.foot_centers)
        reset_data = mujoco.MjData(self.model)
        self.neutral = np.asarray(reset_candidate(self.model, reset_data))
        self.actor = build_actor(self.seed)
        self.critic = build_critic(self.seed + 1)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.ppo["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.ppo["critic_lr"])
        self.updates = 0
        self.rows: list[dict] = []
        self.update_rows: list[dict] = []

    def _episode(self, episode_seed: int) -> dict[str, torch.Tensor]:
        data = mujoco.MjData(self.model)
        neutral = np.asarray(reset_candidate(self.model, data))
        observed = simulation.measured_state(self.model, data)
        previous_target = neutral.copy()
        previous_residual = np.zeros(ACTION_SIZE)
        generator = torch.Generator().manual_seed(episode_seed)
        records = []
        for step in range(self.settings["horizon"]):
            obs = observation_from(observed, previous_target, self.reference,
                                   self.bounds)
            with torch.no_grad():
                mean = self.actor(obs)
                std = self.actor.log_std.clamp(-3.5, -0.5).exp()
                latent = torch.normal(mean, std, generator=generator)
                old_logp = Normal(mean, std).log_prob(latent).sum()
                value = self.critic(obs).squeeze(-1)
            base = reference_target(self.reference, observed.time, neutral)
            target, residual = residual_target(base, latent.numpy(), previous_residual,
                                               self.bounds,
                                               self.residual_limit_rad,
                                               self.filter_time_constant_s)
            previous_x = observed.torso_position[0]
            for _ in range(self.settings["physics_steps"]):
                simulation.step(self.model, data, target.tolist())
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("non-finite candidate physics state")
            observed = simulation.measured_state(self.model, data)
            terms = reward_terms(observed, previous_x, residual, previous_residual,
                                 self.residual_limit_rad, weights=self.reward)
            fell = fallen(observed, self.settings["fall_height_m"])
            total = sum(value for name, value in terms.items() if name.endswith("_weighted"))
            total -= self.reward["fall"] if fell else 0.0
            with torch.no_grad():
                next_obs = observation_from(observed, target, self.reference,
                                            self.bounds)
                next_value = self.critic(next_obs).squeeze(-1)
            records.append((obs, latent, old_logp, torch.tensor(total, dtype=torch.float32),
                            value, next_value, torch.tensor(0.0 if fell else 1.0)))
            self.rows.append({"update": self.updates + 1, "episode_seed": episode_seed,
                              "step": step, "total_reward": total, "fell": fell, **terms})
            previous_target, previous_residual = target, residual
            if fell:
                break
        names = ("obs", "actions", "old_logp", "rewards", "values", "next_values", "bootstrap")
        columns = list(zip(*records))
        episode = {name: torch.stack(values) for name, values in zip(names, columns)}
        episode["advantages"], episode["targets"] = gae_episode(
            episode["rewards"], episode["values"], episode["next_values"],
            episode["bootstrap"], self.ppo["gamma"], self.ppo["gae_lambda"])
        return episode

    def _optimize(self, batch: dict[str, torch.Tensor]) -> tuple[int, int, float]:
        actor_epochs = 0
        approximate_kl = 0.0
        for epoch in range(self.ppo["actor_epochs"]):
            order = torch.randperm(len(batch["advantages"]), generator=self.generator)
            for indexes in order.split(self.ppo["minibatch_size"]):
                std = self.actor.log_std.clamp(-3.5, -0.5).exp()
                distribution = Normal(self.actor(batch["obs"][indexes]), std)
                new_logp = distribution.log_prob(batch["actions"][indexes]).sum(-1)
                ratio = torch.exp(new_logp - batch["old_logp"][indexes])
                advantage = batch["advantages"][indexes]
                objective = torch.min(ratio * advantage, torch.clamp(
                    ratio, 1-self.ppo["clip_ratio"], 1+self.ppo["clip_ratio"]) * advantage)
                loss = -objective.mean() - self.ppo["entropy_coefficient"] * distribution.entropy().sum(-1).mean()
                self.actor_optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.ppo["max_grad_norm"])
                self.actor_optimizer.step()
            actor_epochs = epoch + 1
            with torch.no_grad():
                std = self.actor.log_std.clamp(-3.5, -0.5).exp()
                logp = Normal(self.actor(batch["obs"]), std).log_prob(batch["actions"]).sum(-1)
                ratio_log = logp - batch["old_logp"]
                approximate_kl = float(((ratio_log.exp() - 1) - ratio_log).mean())
            if approximate_kl > self.ppo["target_kl"]:
                break
        # Critic fitting is independent of the actor KL stop.
        for _ in range(self.ppo["critic_epochs"]):
            for indexes in torch.randperm(
                    len(batch["targets"]), generator=self.generator).split(self.ppo["minibatch_size"]):
                loss = torch.mean((self.critic(batch["obs"][indexes]).squeeze(-1) -
                                   batch["targets"][indexes]) ** 2)
                self.critic_optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.ppo["max_grad_norm"])
                self.critic_optimizer.step()
        return actor_epochs, self.ppo["critic_epochs"], approximate_kl

    def _save(self) -> Path:
        path = self.output_directory / f"checkpoint-{self.updates:05d}.pt"
        xml = self.model_xml
        torch.save({"status": "UNTRAINED PPO residual preparation" if self.updates == 0 else
                    "TRAINED PPO residual checkpoint",
                    "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "actor_optimizer": self.actor_optimizer.state_dict(),
                    "critic_optimizer": self.critic_optimizer.state_dict(),
                    "updates": self.updates, "seed": self.seed, "ppo": self.ppo,
                    "reward": self.reward, "settings": self.settings,
                    "optimizer_rng_state": self.generator.get_state(),
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
                    "filter_time_constant_s": self.filter_time_constant_s}, path)
        return path

    def _prepare_output(self) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=False)
        sources = {}
        for name in SOURCE_FILES:
            source = Path(__file__).with_name(name)
            shutil.copyfile(source, self.output_directory / name)
            sources[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        xml = self.model_xml
        (self.output_directory / "candidate.xml").write_text(xml, encoding="utf-8")
        config = {"status": "UNTRAINED PPO residual preparation", "seed": self.seed,
                  "reference_config": self.reference.config, "chassis_manifest": PARAMETER_MANIFEST,
                  "ppo": self.ppo, "reward": self.reward, "settings": self.settings,
                  "residual_limit_rad": self.residual_limit_rad.tolist(),
                  "filter_time_constant_s": self.filter_time_constant_s,
                  "source_sha256": sources,
                  "candidate_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                  "versions": {name: version(name) for name in ("torch", "numpy", "mujoco")}}
        (self.output_directory / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8")

    def train(self, updates: int) -> Path:
        if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
            raise ValueError("updates must be an explicit positive integer")
        self._prepare_output()
        latest = self._save()
        for _ in range(updates):
            episodes = [self._episode(self.seed * 1_000_000 + self.updates * 8 + index)
                        for index in range(self.ppo["episodes_per_update"])]
            batch = {name: torch.cat([episode[name] for episode in episodes])
                     for name in episodes[0]}
            advantages = batch["advantages"]
            batch["advantages"] = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            actor_epochs, critic_epochs, kl = self._optimize(batch)
            self.updates += 1
            self.update_rows.append({"update": self.updates, "actor_epochs": actor_epochs,
                                     "critic_epochs": critic_epochs, "approx_kl": kl,
                                     "mean_reward": float(batch["rewards"].mean())})
            latest = self._save()
            for filename, rows in (("steps.csv", self.rows), ("updates.csv", self.update_rows)):
                with (self.output_directory / filename).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        return latest

    def prepare(self) -> Path:
        """Write a reviewable, explicitly untrained checkpoint zero without rollouts."""
        self._prepare_output()
        return self._save()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--updates", type=int)
    action.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    session = ReferenceTrainingSession(args.output, args.seed)
    print(session.prepare() if args.prepare_only else session.train(args.updates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

