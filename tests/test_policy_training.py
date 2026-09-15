from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "artifacts/ppo-crude-baseline-20260915/checkpoint-00100.pt"


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional learning dependency")
class PolicyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global TrainingSession, torch, clipped_policy_loss, critic_loss, gae_targets, reward_terms
        import torch
        from spider.policy_training import (TrainingSession, clipped_policy_loss, critic_loss,
                                             gae_targets, reward_terms)

    def test_reward_keeps_raw_and_weighted_rate_terms_separate(self):
        before = object()
        after = type("State", (), {"torso_velocity": (2.0, 0.0, 0.0),
                                    "torso_position": (0.0, 0.0, 0.4)})()
        previous = np.zeros(18)
        target = np.ones(18) * 0.2
        terms = reward_terms(before, target, previous, after, 0.5,
                             {"progress": 1.0, "height": 0.1})
        self.assertAlmostEqual(terms["progress_weighted"], 2.0)
        self.assertAlmostEqual(terms["height_weighted"], 0.04)
        self.assertAlmostEqual(terms["action_rate_raw"], 0.04)
        self.assertAlmostEqual(terms["action_rate_weighted"], -0.02)

    def test_kl_stop_uses_baseline_nonnegative_estimator(self):
        from spider.policy_training import approximate_kl
        old = torch.zeros(2)
        new = torch.tensor([-.2, .2])
        self.assertGreater(float(approximate_kl(new, old)), .01)
        self.assertEqual(float((old - new).mean()), 0.)
        self.assertEqual(float(approximate_kl(old, old)), 0.)

    def test_saved_gae_and_losses_match_direct_equations(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.2, 0.4, 0.6])
        next_values = torch.tensor([0.4, 0.6, 9.0])
        bootstrap = torch.tensor([1.0, 1.0, 0.0])
        ends = torch.tensor([False, False, True])
        advantages, targets = gae_targets(rewards, values, next_values, bootstrap, ends, .99, .95)
        deltas = rewards + .99 * bootstrap * next_values - values
        expected = torch.empty(3)
        expected[2] = deltas[2]
        expected[1] = deltas[1] + .99 * .95 * expected[2]
        expected[0] = deltas[0] + .99 * .95 * expected[1]
        torch.testing.assert_close(advantages, expected)
        torch.testing.assert_close(targets, expected + values)

        old = torch.tensor([-.2, -.4])
        new = torch.tensor([-.1, -.7])
        advantage = torch.tensor([1.0, -2.0])
        ratio = torch.exp(new - old)
        expected_actor = -torch.min(ratio * advantage,
                                    torch.clamp(ratio, .8, 1.2) * advantage).mean()
        torch.testing.assert_close(clipped_policy_loss(new, old, advantage, .2), expected_actor)
        torch.testing.assert_close(critic_loss(torch.tensor([1., 3.]), torch.tensor([2., 1.])),
                                   torch.tensor(2.5))

    def test_resumed_optimizers_apply_a_finite_synthetic_minibatch(self):
        with TemporaryDirectory() as directory:
            session = TrainingSession(CHECKPOINT, Path(directory) / "out", 0.01)
            observations = torch.zeros((16, 47))
            with torch.no_grad():
                means = session.actor(observations)
                actions = means + 0.01
                old_logp = torch.distributions.Normal(
                    means, session.settings["std"]).log_prob(actions).sum(-1)
            report = session._update_minibatch({
                "obs": observations,
                "actions": actions,
                "old_logp": old_logp,
                "advantages": torch.linspace(-1.0, 1.0, 16),
                "targets": torch.zeros(16),
            })
        self.assertTrue(all(np.isfinite(value) for value in report.values()))
        self.assertTrue(session.actor_optimizer.state)
        self.assertTrue(session.critic_optimizer.state)


if __name__ == "__main__":
    unittest.main()
