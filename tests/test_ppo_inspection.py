"""Synthetic record/diagnostic checks; no PPO solution and no robot rollout."""
import importlib.util
import unittest

HAS_LEARNING = all(importlib.util.find_spec(x) for x in ('torch','pandas'))
if HAS_LEARNING:
    import torch
    from lab.ppo_inspection import frozen_batch, policy_diagnostics


@unittest.skipUnless(HAS_LEARNING, 'Optional learning environment')
class PPOInspectionTests(unittest.TestCase):
    def test_records_are_detached_and_finite_horizon_masks_end(self):
        actor = torch.nn.Linear(47,18)
        critic = torch.nn.Linear(47,1)
        obs = torch.zeros(47)
        action = torch.ones(18)*.1
        logp = torch.distributions.Normal(actor(obs), .3).log_prob(action).sum()
        row = dict(observation=obs, next_observation=obs, latent=action,
                   log_prob=logp, reward=1., terminated=False, truncated=True)
        batch = frozen_batch([dict(rows=[row])], critic)
        self.assertEqual(batch['values'].shape, (1,))
        self.assertFalse(batch['bootstrap_allowed'].item())
        self.assertTrue(batch['episode_end'].item())
        self.assertTrue(all(not x.requires_grad for x in batch.values()))
        batch['targets'] = batch['values'].clone()
        diagnostic = policy_diagnostics(actor, critic, batch, .3, .2)
        self.assertAlmostEqual(diagnostic['approx_kl'], 0)
        self.assertEqual(diagnostic['clip_fraction'], 0)
        self.assertEqual(diagnostic['value_mse'], 0)
        self.assertIsNone(diagnostic['explained_variance'])
