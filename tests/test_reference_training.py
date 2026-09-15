import json
import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import mujoco
    import torch
    from spider import simulation
    from spider.chassis_candidate import reset_candidate
    from spider.reference_training import (
        ACTION_SIZE,
        OBSERVATION_SIZE,
        ReferenceResidualPolicy,
        ReferenceTrainingSession,
        effective_control_bounds,
        fallen,
        gae_episode,
        reference_target,
        residual_target,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional learning dependency")
class ReferenceTrainingTest(unittest.TestCase):
    def test_zero_residual_is_exact_reference_and_is_bounded(self) -> None:
        base = np.linspace(-0.5, 1.0, ACTION_SIZE)
        limits = np.column_stack((np.full(ACTION_SIZE, -2.0), np.full(ACTION_SIZE, 2.0)))
        target, residual = residual_target(base, np.zeros(ACTION_SIZE),
                                           np.zeros(ACTION_SIZE), limits)
        np.testing.assert_array_equal(target, base)
        large_target, large_residual = residual_target(
            base, np.full(ACTION_SIZE, 1e6), residual, limits)
        self.assertTrue(np.all(np.abs(large_residual) <= np.tile((.10, .12, .14), 6)))
        self.assertTrue(np.all(large_target <= limits[:, 1]))

    def test_fall_disables_bootstrap(self) -> None:
        rewards = torch.tensor([1.0, 1.0])
        values = torch.zeros(2)
        next_values = torch.tensor([2.0, 3.0])
        alive, _ = gae_episode(rewards, values, next_values, torch.ones(2), .995, .95)
        fell, _ = gae_episode(rewards, values, next_values, torch.tensor([1.0, 0.0]), .995, .95)
        self.assertGreater(alive[-1], fell[-1])

    def test_actor_kl_stop_does_not_shorten_critic_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = ReferenceTrainingSession(Path(directory) / "unused", seed=7)
            observations = torch.zeros((8, OBSERVATION_SIZE))
            with torch.no_grad():
                distribution = torch.distributions.Normal(
                    session.actor(observations), session.actor.log_std.exp())
                actions = torch.zeros((8, ACTION_SIZE))
                old_logp = distribution.log_prob(actions).sum(-1)
            batch = {"obs": observations, "actions": actions, "old_logp": old_logp,
                     "advantages": torch.linspace(-1.0, 1.0, 8),
                     "targets": torch.ones(8)}
            with patch.dict(session.ppo, {"target_kl": -1.0, "critic_epochs": 3}):
                actor_epochs, critic_epochs, _ = session._optimize(batch)
            self.assertEqual(actor_epochs, 1)
            self.assertEqual(critic_epochs, 3)

    def test_prepared_checkpoint_loads_with_zero_policy_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepared"
            session = ReferenceTrainingSession(output, seed=11)
            checkpoint = session.prepare()
            policy = ReferenceResidualPolicy(checkpoint, session.model)
            data = mujoco.MjData(session.model)
            neutral = np.asarray(reset_candidate(session.model, data))
            observed = simulation.measured_state(session.model, data)
            for time_s in np.linspace(0.0, 5.0, 251):
                sample = replace(observed, time=float(time_s))
                expected = reference_target(session.reference, sample.time, neutral)
                np.testing.assert_array_equal(policy.targets(sample), expected)
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["status"], "UNTRAINED PPO residual preparation")
            self.assertEqual(config["reference_config"],
                             json.loads(json.dumps(session.reference.config)))
            self.assertTrue((output / "candidate.xml").is_file())
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertIn("actor_optimizer", payload)
            self.assertIn("model_xml", payload)
            with self.assertRaises(ValueError):
                ReferenceResidualPolicy(checkpoint, simulation.load_model())

    def test_effective_bounds_intersect_joint_and_actuator_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = ReferenceTrainingSession(Path(directory) / "unused", seed=5)
            bounds = effective_control_bounds(session.model)
            coxa = session.model.actuator("front_left_coxa_motor").id
            self.assertAlmostEqual(bounds[coxa, 0], -np.pi / 4)
            self.assertAlmostEqual(bounds[coxa, 1], np.pi / 4)

    def test_fall_gate_includes_height_and_absolute_tilt(self) -> None:
        upright = SimpleNamespace(torso_position=(0, 0, .37), torso_orientation=(1, 0, 0, 0))
        low = SimpleNamespace(torso_position=(0, 0, .24), torso_orientation=(1, 0, 0, 0))
        tilted = SimpleNamespace(torso_position=(0, 0, .37),
                                 torso_orientation=(np.cos(.6), np.sin(.6), 0, 0))
        self.assertFalse(fallen(upright, .25))
        self.assertTrue(fallen(low, .25))
        self.assertTrue(fallen(tilted, .25))

    def test_updates_are_mandatory_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = ReferenceTrainingSession(Path(directory) / "unused", seed=3)
            for value in (0, -1, True):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    session.train(value)


if __name__ == "__main__":
    unittest.main()
