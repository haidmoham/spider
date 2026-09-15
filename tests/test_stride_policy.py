from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np

from spider.simulation import measured_state, load_model, reset


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional learning dependency")
class StridePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, JOINT_HIGH, JOINT_LOW, NEUTRAL, SETTINGS, build_actor
        global observation_from, targets_from_latent, gae_episode, phase_stance, reward_terms
        global FreshTrainingSession, PPO, REWARD
        import mujoco
        import torch
        from spider.stride_policy import (JOINT_HIGH, JOINT_LOW, NEUTRAL, SETTINGS,
                                          build_actor, observation_from, targets_from_latent)
        from spider.stride_training import (FreshTrainingSession, PPO, REWARD, gae_episode,
                                            phase_stance, reward_terms)
        cls.model = load_model(); cls.data = mujoco.MjData(cls.model); reset(cls.model, cls.data)
        cls.observed = measured_state(cls.model, cls.data)

    def test_static_ranges_fit_model_and_fresh_mean_is_neutral(self):
        self.assertTrue(np.all(JOINT_LOW >= self.model.actuator_ctrlrange[:, 0]))
        self.assertTrue(np.all(JOINT_HIGH <= self.model.actuator_ctrlrange[:, 1]))
        actor = build_actor(17)
        obs = observation_from(self.observed, NEUTRAL, .25, 0.0)
        self.assertEqual(obs.shape, (68,))
        self.assertLess(np.max(np.abs(targets_from_latent(actor(obs)) - NEUTRAL)), 0.01)

    def test_phase_contact_schedule_alternates_tripods(self):
        np.testing.assert_array_equal(phase_stance(0.3), [True, False, False, True, True, False])
        np.testing.assert_array_equal(phase_stance(0.8), [False, True, True, False, False, True])

    def test_reward_terms_are_finite_and_action_rate_penalizes_change(self):
        contact = phase_stance(0.0)
        state = SimpleNamespace(torso_velocity=(.25, 0., 0.), torso_position=(0., 0., .43),
                                torso_angular_velocity=(0., 0., 0.), torso_orientation=(1.,0.,0.,0.))
        same = reward_terms(state, NEUTRAL, NEUTRAL, 0., np.full(6, .08), contact, np.zeros(6))
        changed = reward_terms(state, NEUTRAL + .1, NEUTRAL, 0., np.full(6, .08), contact, np.zeros(6))
        self.assertTrue(all(np.isfinite(value) for value in same.values()))
        self.assertEqual(same["action_rate_weighted"], 0.0)
        self.assertLess(changed["action_rate_weighted"], 0.0)

    def test_clearance_is_finite_during_all_stance_overlap(self):
        state = SimpleNamespace(torso_velocity=(.25, 0., 0.), torso_position=(0., 0., .43),
                                torso_angular_velocity=(0., 0., 0.), torso_orientation=(1.,0.,0.,0.))
        terms = reward_terms(state, NEUTRAL, NEUTRAL, 0., np.full(6, .045),
                             np.ones(6, dtype=bool), np.zeros(6))
        self.assertEqual(terms["clearance_raw"], 0.0)
        self.assertTrue(all(np.isfinite(value) for value in terms.values()))

    def test_timeout_bootstraps_but_fall_does_not(self):
        rewards = torch.tensor([1., 1.]); values = torch.zeros(2); next_values = torch.tensor([2., 3.])
        timeout, _ = gae_episode(rewards, values, next_values, torch.tensor([1., 1.]), .995, .95)
        fall, _ = gae_episode(rewards, values, next_values, torch.tensor([1., 0.]), .995, .95)
        self.assertGreater(timeout[-1], fall[-1])

    def test_tuning_is_per_session_and_noise_does_not_change_initial_mean(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            default = FreshTrainingSession(f"{directory}/default", 11)
            tuned = FreshTrainingSession(
                f"{directory}/tuned", 11, noise_scale=.4, entropy_coefficient=0,
                target_speed_mps=.4, velocity_weight=4)
            obs = observation_from(self.observed, NEUTRAL, .25, 0.)
            torch.testing.assert_close(default.actor(obs), tuned.actor(obs), rtol=0, atol=0)
            torch.testing.assert_close(
                tuned.actor.log_std, default.actor.log_std + np.log(.4), rtol=0, atol=1e-7)
            self.assertEqual(default.settings["target_speed_mps"], .25)
            self.assertEqual(default.ppo["entropy_coefficient"], .003)
            self.assertEqual(default.reward["velocity"], 1.5)
            self.assertEqual(SETTINGS["target_speed_mps"], .25)
            self.assertEqual(PPO["entropy_coefficient"], .003)
            self.assertEqual(REWARD["velocity"], 1.5)

    def test_forward_reward_uses_session_speed_and_weight(self):
        state = SimpleNamespace(torso_velocity=(.4, 0., 0.), torso_position=(0., 0., .43),
                                torso_angular_velocity=(0., 0., 0.),
                                torso_orientation=(1., 0., 0., 0.))
        terms = reward_terms(
            state, NEUTRAL, NEUTRAL, 0., np.full(6, .045), np.ones(6, dtype=bool),
            np.zeros(6), settings={**SETTINGS, "target_speed_mps": .4},
            reward={**REWARD, "velocity": 4.})
        self.assertAlmostEqual(terms["velocity_raw"], 1.)
        self.assertAlmostEqual(terms["velocity_weighted"], 4.)

    def test_tuned_checkpoint_replays_scaled_noise_and_speed(self):
        from dataclasses import replace
        from tempfile import TemporaryDirectory
        from spider.stride_policy import StridePolicy

        with TemporaryDirectory() as directory:
            session = FreshTrainingSession(
                f"{directory}/branch", 11, noise_scale=.4, entropy_coefficient=0,
                target_speed_mps=.4, velocity_weight=4)
            session.output_directory.mkdir()
            checkpoint = session._save()
            policy = StridePolicy(checkpoint, self.model, seed=201, sampled=True)
            generator = torch.Generator().manual_seed(201)
            previous = NEUTRAL.copy()
            for time in (0., .04, .08):
                observed = replace(self.observed, time=time)
                obs = observation_from(observed, previous, .4, (time * 1.25) % 1.)
                with torch.no_grad():
                    latent = torch.normal(
                        session.actor(obs), session.actor.log_std.clamp(-3.5, -.3).exp(),
                        generator=generator)
                expected = targets_from_latent(latent)
                np.testing.assert_array_equal(policy.targets(observed), expected)
                previous = expected
            self.assertEqual(policy.config["training_tuning"]["noise_scale"], .4)

    def test_tuning_parameters_must_be_finite_and_valid(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            for kwargs in ({"noise_scale": 0}, {"entropy_coefficient": -1},
                           {"target_speed_mps": float("nan")}, {"velocity_weight": 0}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    FreshTrainingSession(f"{directory}/bad", 11, **kwargs)

    def test_saved_runtime_matches_training_sampling_and_history(self):
        from dataclasses import replace
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from spider.stride_policy import StridePolicy

        actor = build_actor(11)
        generator = torch.Generator().manual_seed(201)
        previous = NEUTRAL.copy()
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            torch.save(dict(actor=actor.state_dict(), settings=SETTINGS, updates=0), checkpoint)
            policy = StridePolicy(checkpoint, self.model, seed=201, sampled=True)
            for time in (0., .04, .08, .60):
                observed = replace(self.observed, time=time)
                obs = observation_from(observed, previous, .25, (time * 1.25) % 1.)
                with torch.no_grad():
                    latent = torch.normal(actor(obs), actor.log_std.clamp(-3.5, -.3).exp(),
                                          generator=generator)
                expected = targets_from_latent(latent)
                np.testing.assert_array_equal(policy.targets(observed), expected)
                previous = expected


if __name__ == "__main__": unittest.main()
