"""Integrated continuation training for the preserved C-1N PPO checkpoint."""

from __future__ import annotations

import csv
import json
import math
import shutil
import hashlib
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from torch.distributions import Normal

from . import simulation
from .policy import ACTION_SIZE, HIDDEN_SIZE, OBSERVATION_SIZE, PPOPolicy, observation_from


SOURCE = simulation.ROOT / "artifacts/ppo-crude-baseline-20260915/learning-functions.json"


def build_critic() -> torch.nn.Sequential:
    with torch.random.fork_rng():
        return torch.nn.Sequential(
            torch.nn.Linear(OBSERVATION_SIZE, HIDDEN_SIZE), torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN_SIZE, 1),
        )


def reward_terms(before: Any, target: np.ndarray, previous_target: np.ndarray,
                 after: Any, action_rate_weight: float, reward: dict) -> dict[str, float]:
    """Return raw and weighted terms while preserving the accepted base reward."""
    raw_rate = float(np.mean(np.square(target - previous_target)))
    return {
        "progress_raw": float(after.torso_velocity[0]),
        "progress_weighted": float(reward["progress"] * after.torso_velocity[0]),
        "height_raw": float(after.torso_position[2]),
        "height_weighted": float(reward["height"] * after.torso_position[2]),
        "action_rate_raw": raw_rate,
        "action_rate_weighted": -float(action_rate_weight * raw_rate),
    }


def gae_targets(rewards, values, next_values, bootstrap_allowed, episode_end, gamma, lam):
    deltas = rewards + gamma * bootstrap_allowed * next_values - values
    advantages = torch.empty_like(rewards)
    future_advantage = rewards.new_zeros(())
    for index in range(len(rewards) - 1, -1, -1):
        if episode_end[index]:
            future_advantage = deltas[index]
        else:
            future_advantage = deltas[index] + gamma * lam * future_advantage
        advantages[index] = future_advantage
    return advantages.detach(), (advantages + values).detach()


def clipped_policy_loss(new_logp, old_logp, advantages, epsilon):
    ratio = torch.exp(new_logp - old_logp)
    surrogate = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    return -torch.mean(torch.min(clipped, surrogate))


def critic_loss(predicted_values, targets):
    return torch.mean((predicted_values - targets) ** 2)


def approximate_kl(new_logp, old_logp):
    """The frozen full-batch KL diagnostic used by the baseline schedule."""
    log_ratio = new_logp - old_logp
    return (log_ratio.exp() - 1 - log_ratio).mean()


class TrainingSession:
    """Continue one accepted PPO checkpoint using canonical simulation."""

    def __init__(self, checkpoint: str | Path, output_directory: str | Path,
                 action_rate_weight: float, treatment: str = "lower") -> None:
        if treatment != "lower":
            raise ValueError("the first tuning batch requires treatment='lower'")
        if not math.isfinite(action_rate_weight) or action_rate_weight < 0:
            raise ValueError("action_rate_weight must be finite and nonnegative")
        self.checkpoint = Path(checkpoint)
        self.output_directory = Path(output_directory)
        self.action_rate_weight = float(action_rate_weight)
        self.treatment = treatment
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.model = simulation.load_model()
        self.policy = PPOPolicy(self.checkpoint, self.model, sampled=False, treatment=treatment)
        self.actor = self.policy.actor
        self.actor.train()
        self.critic = build_critic()
        self.critic.load_state_dict(payload["critic"], strict=True)
        self.settings = dict(payload["settings"])
        self.ppo = dict(payload["ppo"])
        self.reward = dict(payload["reward"])
        self.absolute_update = int(payload["updates"])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.ppo["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.ppo["critic_lr"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self._rows: list[dict[str, Any]] = []
        self._update_rows: list[dict[str, Any]] = []

    def _episode(self, seed: int) -> dict[str, torch.Tensor]:
        data = mujoco.MjData(self.model)
        simulation.reset(self.model, data)
        observed = simulation.measured_state(self.model, data)
        previous_target = np.asarray(simulation.neutral_targets(), dtype=float)
        self.policy.reset()
        generator = torch.Generator().manual_seed(seed)
        fields = {name: [] for name in (
            "obs", "actions", "old_logp", "rewards", "values", "next_values",
            "bootstrap_allowed", "episode_end",
        )}
        for step in range(self.settings["horizon"]):
            obs = observation_from(observed)
            with torch.no_grad():
                mean = self.actor(obs)
                latent = torch.normal(mean, self.settings["std"], generator=generator)
                logp = Normal(mean, self.settings["std"]).log_prob(latent).sum()
                value = self.critic(obs).squeeze(-1)
            target = np.asarray(self.policy.targets_from_latent(latent))
            before = observed
            for _ in range(self.settings["physics_steps"]):
                simulation.step(self.model, data, target)
            observed = simulation.measured_state(self.model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("Non-finite rollout state; abort this branch")
            terms = reward_terms(before, target, previous_target, observed,
                                 self.action_rate_weight, self.reward)
            total_reward = sum(value for key, value in terms.items() if key.endswith("_weighted"))
            fell = observed.torso_position[2] < self.settings["fall_height_m"]
            horizon = step == self.settings["horizon"] - 1
            with torch.no_grad():
                next_value = self.critic(observation_from(observed)).squeeze(-1)
            for name, value_item in (
                ("obs", obs), ("actions", latent), ("old_logp", logp),
                ("rewards", torch.tensor(total_reward, dtype=torch.float32)),
                ("values", value), ("next_values", next_value),
                ("bootstrap_allowed", torch.tensor(0.0 if fell or horizon else 1.0)),
                ("episode_end", torch.tensor(fell or horizon)),
            ):
                fields[name].append(value_item)
            self._rows.append({"update": self.absolute_update + 1, "seed": seed, "step": step,
                               "reward": total_reward, **terms, "fell": bool(fell)})
            previous_target = target
            if fell:
                break
        return {name: torch.stack(values) for name, values in fields.items()}

    def _update_minibatch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        distribution = Normal(self.actor(batch["obs"]), self.settings["std"])
        new_logp = distribution.log_prob(batch["actions"]).sum(dim=-1)
        actor_loss = clipped_policy_loss(new_logp, batch["old_logp"], batch["advantages"],
                                         self.ppo["clip_ratio"])
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.ppo["max_grad_norm"])
        self.actor_optimizer.step()
        value_loss = critic_loss(self.critic(batch["obs"]).squeeze(-1), batch["targets"])
        self.critic_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.ppo["max_grad_norm"])
        self.critic_optimizer.step()
        return {"actor_loss": actor_loss.detach().item(), "critic_loss": value_loss.detach().item(),
                "actor_grad_norm": actor_norm.detach().item(),
                "critic_grad_norm": critic_norm.detach().item(),
                "approx_kl": (batch["old_logp"] - new_logp.detach()).mean().item()}

    def _save(self) -> Path:
        path = self.output_directory / f"checkpoint-{self.absolute_update:05d}.pt"
        temporary = path.with_suffix(".tmp")
        payload = {
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "updates": self.absolute_update, "settings": self.settings,
            "ppo": self.ppo, "reward": self.reward, "treatment": self.treatment,
            "controller": self.policy.config,
            "tuning": {"action_rate_weight": self.action_rate_weight},
        }
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def _write_logs(self) -> None:
        for name, rows in (("training.csv", self._rows), ("updates.csv", self._update_rows)):
            path = self.output_directory / name
            temporary = path.with_suffix(".tmp")
            with temporary.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(path)

    def train(self, updates: int = 10) -> Path:
        if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
            raise ValueError("updates must be a positive integer")
        self.output_directory.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(SOURCE, self.output_directory / "learning-functions.json")
        for name in ("policy_training.py", "policy.py", "simulation.py"):
            shutil.copyfile(Path(__file__).with_name(name), self.output_directory / name)
        config = {"source_checkpoint": str(self.checkpoint.resolve()),
                  "starting_update": self.absolute_update, "requested_updates": updates,
                  "treatment": self.treatment, "action_rate_weight": self.action_rate_weight,
                  "settings": self.settings, "ppo": self.ppo, "reward": self.reward,
                  "controller": self.policy.config,
                  "parent_sha256": self.policy.checkpoint_sha256,
                  "model_sha256": hashlib.sha256(simulation.MODEL_PATH.read_bytes()).hexdigest(),
                  "versions": {name: version(name) for name in ("torch", "numpy", "mujoco")}}
        (self.output_directory / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8")
        latest = self.checkpoint
        for _ in range(updates):
            episodes = [self._episode(self.ppo["train_seed"] + self.absolute_update * self.settings["batch_episodes"] + episode)
                        for episode in range(self.settings["batch_episodes"])]
            batch = {name: torch.cat([episode[name] for episode in episodes])
                     for name in episodes[0]}
            advantages, targets = gae_targets(
                batch["rewards"], batch["values"], batch["next_values"],
                batch["bootstrap_allowed"], batch["episode_end"],
                self.ppo["gamma"], self.ppo["gae_lambda"])
            batch["advantages"] = (
                (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            ).detach()
            batch["targets"] = targets
            if not all(torch.isfinite(value).all() for value in batch.values()):
                raise RuntimeError("Non-finite frozen batch; abort this branch")
            torch.save(batch, self.output_directory / f"batch-{self.absolute_update + 1:05d}.pt")
            generator = torch.Generator().manual_seed(self.ppo["train_seed"] + self.absolute_update)
            stop = False
            reports = []
            for _epoch in range(self.ppo["epochs"]):
                order = torch.randperm(len(batch["obs"]), generator=generator)
                for start in range(0, len(order), self.ppo["minibatch_size"]):
                    indexes = order[start:start + self.ppo["minibatch_size"]]
                    report = self._update_minibatch({key: value[indexes] for key, value in batch.items()
                                                     if key not in {"rewards", "values", "next_values",
                                                                    "bootstrap_allowed", "episode_end"}})
                    reports.append(report)
                    if not all(math.isfinite(value) for value in report.values()):
                        raise RuntimeError("Non-finite PPO update; abort this branch")
                with torch.no_grad():
                    current_logp = Normal(
                        self.actor(batch["obs"]), self.settings["std"]
                    ).log_prob(batch["actions"]).sum(dim=-1)
                    diagnostic_kl = approximate_kl(current_logp, batch["old_logp"]).item()
                if diagnostic_kl > self.ppo["target_kl"]:
                    stop = True
                    break
            self.absolute_update += 1
            if not all(torch.isfinite(parameter).all() for network in (self.actor, self.critic)
                       for parameter in network.parameters()):
                raise RuntimeError("Non-finite learned weights; abort this branch")
            latest = self._save()
            self._update_rows.append({
                "update": self.absolute_update,
                "samples": len(batch["obs"]),
                "minibatches": len(reports),
                "epochs": _epoch + 1,
                "early_stop": stop,
                "diagnostic_kl": diagnostic_kl,
                **{name: float(np.mean([report[name] for report in reports]))
                   for name in reports[0]},
            })
            self._write_logs()
            print(f"update={self.absolute_update} epochs={_epoch + 1} kl={diagnostic_kl:.5f}", flush=True)
        return latest
