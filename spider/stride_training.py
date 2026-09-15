"""Fresh phase-guided PPO training for C-1N."""

from __future__ import annotations

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
from .stride_policy import (JOINT_CENTER, JOINT_HALF_RANGE, SETTINGS, StrideActor, build_actor,
                            observation_from, targets_from_latent)


PPO = {"gamma": 0.995, "gae_lambda": 0.95, "clip_ratio": 0.2,
       "actor_lr": 3e-4, "critic_lr": 1e-3, "epochs": 5,
       "minibatch_size": 512, "max_grad_norm": 0.5, "target_kl": 0.015,
       "entropy_coefficient": 0.003}
REWARD = {"velocity": 1.5, "height": 0.50, "clearance": 0.35, "contact": 0.25,
          "slip": 0.40, "action_rate": 0.08, "limits": 0.25,
          "lateral": 0.30, "angular": 0.20, "tilt": 0.50, "fall": 20.0}
TRIPOD_A = {0, 3, 4}


def build_critic(seed: int) -> torch.nn.Sequential:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        network = torch.nn.Sequential(
            torch.nn.Linear(68, 128), torch.nn.Tanh(),
            torch.nn.Linear(128, 128), torch.nn.Tanh(), torch.nn.Linear(128, 1))
        for layer in (module for module in network if isinstance(module, torch.nn.Linear)):
            torch.nn.init.orthogonal_(layer.weight, math.sqrt(2.0))
            torch.nn.init.zeros_(layer.bias)
        torch.nn.init.orthogonal_(network[-1].weight, 1.0)
    return network


def phase_stance(phase: float) -> np.ndarray:
    local = np.asarray([(phase + (0.0 if leg in TRIPOD_A else 0.5)) % 1.0
                        for leg in range(6)])
    return local < SETTINGS["stance_fraction"]


def reward_terms(observed, target, previous_target, phase, foot_heights,
                 contact, stance_slip_speed) -> dict[str, float]:
    stance = phase_stance(phase)
    swing = ~stance
    local_phase = np.asarray([(phase + (0.0 if leg in TRIPOD_A else 0.5)) % 1.0
                              for leg in range(6)])
    swing_progress = ((local_phase[swing] - SETTINGS["stance_fraction"]) /
                      (1.0 - SETTINGS["stance_fraction"]))
    desired_clearance = 0.045 + 0.065 * np.sin(np.pi * swing_progress)
    speed_error = (observed.torso_velocity[0] - SETTINGS["target_speed_mps"]) / 0.20
    target_normalized = (target - JOINT_CENTER) / JOINT_HALF_RANGE
    raw = {
        "velocity": math.exp(-(speed_error * speed_error)),
        "height": math.exp(-((observed.torso_position[2] - 0.43) / 0.05) ** 2),
        "clearance": (float(np.mean(np.exp(-np.square(
            (foot_heights[swing] - desired_clearance) / 0.035)))) if np.any(swing) else 0.0),
        "contact": float(np.mean(contact == stance)),
        "slip": -float(np.mean(np.minimum(
            np.square(stance_slip_speed[stance & contact] / 0.25), 4.0
        ))) if np.any(stance & contact) else 0.0,
        "action_rate": -float(np.mean(np.square(
            (target - previous_target) / JOINT_HALF_RANGE))),
        "limits": -float(np.mean(np.square(np.maximum(
            (np.abs(target_normalized) - 0.85) / 0.15, 0.0)))),
        "lateral": -min(4.0, (observed.torso_velocity[1] / 0.25) ** 2),
        "angular": -min(4.0, float(np.sum(
            np.square(np.asarray(observed.torso_angular_velocity) / 0.8)))),
        "tilt": -min(4.0, float(np.sum(
            np.square(np.asarray(observed.torso_orientation[1:3]) / math.sin(0.125))))),
    }
    result = {}
    for name, value in raw.items():
        result[f"{name}_raw"] = value
        result[f"{name}_weighted"] = REWARD[name] * value
    return result


def gae_episode(rewards, values, next_values, bootstrap, gamma, lam):
    deltas = rewards + gamma * bootstrap * next_values - values
    advantages = torch.empty_like(rewards)
    future = rewards.new_zeros(())
    for index in range(len(rewards) - 1, -1, -1):
        future = deltas[index] + gamma * lam * bootstrap[index] * future
        advantages[index] = future
    return advantages.detach(), (advantages + values).detach()


class FreshTrainingSession:
    def __init__(self, output_directory: str | Path, seed: int) -> None:
        self.output_directory = Path(output_directory)
        self.seed = int(seed)
        self.model = simulation.load_model()
        self.actor: StrideActor = build_actor(self.seed)
        self.critic = build_critic(self.seed + 1)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=PPO["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=PPO["critic_lr"])
        self.updates = 0
        self.rows = []
        self.update_rows = []
        self.foot_ids = np.asarray([self.model.geom(f"{name}_foot").id
                                    for name in simulation.FOOT_NAMES])
        self.ground_id = self.model.geom("ground").id

    def _contacts(self, data) -> np.ndarray:
        contact = np.zeros(6, dtype=bool)
        index = {geom: leg for leg, geom in enumerate(self.foot_ids)}
        for item in data.contact[:data.ncon]:
            other = item.geom2 if item.geom1 == self.ground_id else (
                item.geom1 if item.geom2 == self.ground_id else -1)
            if other in index:
                contact[index[other]] = True
        return contact

    def _episode(self, episode_seed: int):
        data = mujoco.MjData(self.model)
        simulation.reset(self.model, data)
        observed = simulation.measured_state(self.model, data)
        previous_target = np.asarray(simulation.neutral_targets(), dtype=float)
        generator = torch.Generator().manual_seed(episode_seed)
        records = []
        for step in range(SETTINGS["horizon"]):
            phase = (observed.time * SETTINGS["phase_hz"]) % 1.0
            obs = observation_from(observed, previous_target, SETTINGS["target_speed_mps"], phase)
            with torch.no_grad():
                mean = self.actor(obs)
                std = self.actor.log_std.clamp(-3.5, -0.3).exp()
                latent = torch.normal(mean, std, generator=generator)
                old_logp = Normal(mean, std).log_prob(latent).sum()
                value = self.critic(obs).squeeze(-1)
            target = targets_from_latent(latent)
            slip_sum = np.zeros(6)
            slip_count = np.zeros(6)
            positions = data.geom_xpos[self.foot_ids, :2].copy()
            for _ in range(SETTINGS["physics_steps"]):
                simulation.step(self.model, data, target)
                current = data.geom_xpos[self.foot_ids, :2].copy()
                contacts = self._contacts(data)
                speeds = np.linalg.norm(current - positions, axis=1) / self.model.opt.timestep
                # Accumulate all contacting feet. The reward applies one explicit
                # post-action phase mask to these interval measurements.
                active = contacts
                slip_sum[active] += speeds[active]
                slip_count[active] += 1
                positions = current
            observed = simulation.measured_state(self.model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("non-finite physics state; abort this branch")
            contact = self._contacts(data)
            slip = np.divide(slip_sum, slip_count, out=np.zeros(6), where=slip_count > 0)
            after_phase = (observed.time * SETTINGS["phase_hz"]) % 1.0
            terms = reward_terms(observed, target, previous_target, after_phase,
                                 data.geom_xpos[self.foot_ids, 2], contact, slip)
            fell = observed.torso_position[2] < SETTINGS["fall_height_m"]
            total = sum(value for name, value in terms.items() if name.endswith("_weighted"))
            total -= REWARD["fall"] if fell else 0.0
            if not math.isfinite(total) or not all(math.isfinite(value) for value in terms.values()):
                raise RuntimeError("non-finite stride reward; discard this training run")
            with torch.no_grad():
                next_value = self.critic(observation_from(
                    observed, target, SETTINGS["target_speed_mps"], after_phase)).squeeze(-1)
            timeout = step == SETTINGS["horizon"] - 1
            records.append((obs, latent, old_logp, torch.tensor(total, dtype=torch.float32),
                            value, next_value, torch.tensor(0.0 if fell else 1.0)))
            self.rows.append({"update": self.updates + 1, "episode_seed": episode_seed,
                              "step": step, "total_reward": total, **terms, "fell": fell})
            previous_target = target
            if fell or timeout:
                break
        columns = list(zip(*records))
        names = ("obs", "actions", "old_logp", "rewards", "values", "next_values", "bootstrap")
        episode = {name: torch.stack(values) for name, values in zip(names, columns)}
        episode["advantages"], episode["targets"] = gae_episode(
            episode["rewards"], episode["values"], episode["next_values"],
            episode["bootstrap"], PPO["gamma"], PPO["gae_lambda"])
        return episode

    def _update(self, batch, indexes):
        std = self.actor.log_std.clamp(-3.5, -0.3).exp()
        new_logp = Normal(self.actor(batch["obs"][indexes]), std).log_prob(
            batch["actions"][indexes]).sum(-1)
        ratio = torch.exp(new_logp - batch["old_logp"][indexes])
        advantages = batch["advantages"][indexes]
        actor_loss = -torch.min(ratio * advantages, torch.clamp(
            ratio, 1-PPO["clip_ratio"], 1+PPO["clip_ratio"]) * advantages).mean()
        entropy = Normal(self.actor(batch["obs"][indexes]), std).entropy().sum(-1).mean()
        actor_loss = actor_loss - PPO["entropy_coefficient"] * entropy
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), PPO["max_grad_norm"])
        self.actor_optimizer.step()
        with torch.no_grad():
            self.actor.log_std.clamp_(-3.5, -0.3)
        critic_loss = torch.mean((self.critic(batch["obs"][indexes]).squeeze(-1) -
                                  batch["targets"][indexes]) ** 2)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), PPO["max_grad_norm"])
        self.critic_optimizer.step()
        return actor_loss.item(), critic_loss.item(), entropy.item()

    def _save(self):
        path = self.output_directory / f"checkpoint-{self.updates:05d}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "actor_optimizer": self.actor_optimizer.state_dict(),
                    "critic_optimizer": self.critic_optimizer.state_dict(),
                    "updates": self.updates, "settings": SETTINGS, "ppo": PPO,
                    "reward": REWARD, "seed": self.seed}, temporary)
        temporary.replace(path)
        return path

    def _write_csv(self, name, rows):
        path = self.output_directory / name
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        temporary.replace(path)

    def train(self, updates: int = 50) -> Path:
        if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
            raise ValueError("updates must be a positive integer")
        self.output_directory.mkdir(parents=True, exist_ok=False)
        sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in ("stride_policy.py", "stride_training.py", "simulation.py")}
        config = {"seed": self.seed, "settings": SETTINGS, "ppo": PPO, "reward": REWARD,
                  "source_sha256": sources,
                  "model_sha256": hashlib.sha256(simulation.MODEL_PATH.read_bytes()).hexdigest(),
                  "versions": {name: version(name) for name in ("torch", "numpy", "mujoco")}}
        (self.output_directory / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8")
        for name in ("stride_policy.py", "stride_training.py", "simulation.py"):
            shutil.copyfile(Path(__file__).with_name(name), self.output_directory / name)
        latest = self._save()
        for _ in range(updates):
            episodes = [self._episode(self.seed * 1_000_000 + self.updates * 8 + index)
                        for index in range(8)]
            batch = {name: torch.cat([episode[name] for episode in episodes])
                     for name in episodes[0]}
            if not all(torch.isfinite(value).all() for value in batch.values()):
                raise RuntimeError("non-finite training batch; abort this branch")
            advantages = batch["advantages"]
            batch["advantages"] = ((advantages - advantages.mean()) /
                                    (advantages.std(unbiased=False) + 1e-8)).detach()
            generator = torch.Generator().manual_seed(self.seed * 1_000_000 + self.updates)
            actor_losses, critic_losses, entropies = [], [], []
            epochs = 0
            for epoch in range(PPO["epochs"]):
                order = torch.randperm(len(advantages), generator=generator)
                for indexes in order.split(PPO["minibatch_size"]):
                    actor_loss, value_loss, entropy = self._update(batch, indexes)
                    if not all(math.isfinite(value) for value in (actor_loss, value_loss, entropy)):
                        raise RuntimeError("non-finite optimizer result; abort this branch")
                    actor_losses.append(actor_loss); critic_losses.append(value_loss)
                    entropies.append(entropy)
                epochs = epoch + 1
                with torch.no_grad():
                    std = self.actor.log_std.clamp(-3.5, -0.3).exp()
                    logp = Normal(self.actor(batch["obs"]), std).log_prob(batch["actions"]).sum(-1)
                    log_ratio = logp - batch["old_logp"]
                    kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean().item()
                if not math.isfinite(kl):
                    raise RuntimeError("non-finite KL diagnostic; abort this branch")
                if kl > PPO["target_kl"]:
                    break
            self.updates += 1
            if not all(torch.isfinite(parameter).all() for network in (self.actor, self.critic)
                       for parameter in network.parameters()):
                raise RuntimeError("non-finite learned weights; abort this branch")
            self.update_rows.append({"update": self.updates, "epochs": epochs, "approx_kl": kl,
                                     "actor_loss": float(np.mean(actor_losses)),
                                     "critic_loss": float(np.mean(critic_losses)),
                                     "entropy": float(np.mean(entropies)),
                                     "mean_reward": float(batch["rewards"].mean()),
                                     "falls": sum(row["fell"] for row in self.rows
                                                  if row["update"] == self.updates)})
            latest = self._save()
            self._write_csv("steps.csv", self.rows)
            self._write_csv("updates.csv", self.update_rows)
            print(self.update_rows[-1], flush=True)
        return latest
