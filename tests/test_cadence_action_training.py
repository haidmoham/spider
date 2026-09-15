"""Transfer and timing checks; no training run is started by these tests."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import mujoco
    import torch
    from spider import simulation
    from spider.reference_training import ReferenceResidualPolicy, ReferenceTrainingSession
    from spider.cadence_action_training import CadenceTrainingSession, CadenceResidualPolicy, cadence_from_latent


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional learning dependency")
class CadenceActionTrainingTests(unittest.TestCase):
    parent = Path(__file__).resolve().parents[1] / "artifacts/walk_stable_100/walk_stable_100.pt"

    def test_resume_preserves_trained_heads_adam_and_next_update(self):
        checkpoint = self.parent.parents[1] / (
            "walk_fast_200/experiments/policy-cadence-action-20260915/checkpoint-00200.pt")
        payload = torch.load(checkpoint, weights_only=True)
        def equal(left, right):
            if torch.is_tensor(left):
                self.assertTrue(torch.equal(left, right))
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left: equal(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right): equal(a, b)
            else:
                self.assertEqual(left, right)
        with tempfile.TemporaryDirectory() as directory:
            session = CadenceTrainingSession.from_checkpoint(checkpoint, Path(directory) / "a")
            restored_path = session.prepare()
            restored = torch.load(restored_path, weights_only=True)
            for key in ("actor", "critic", "actor_optimizer", "critic_optimizer",
                        "optimizer_rng_state", "cadence_seed_offset", "updates", "seed",
                        "ppo", "reward", "settings", "model_xml", "reference_config"):
                equal(payload[key], restored[key])
            other = CadenceTrainingSession.from_checkpoint(restored_path, Path(directory) / "b")
            obs = torch.randn(8, 68, generator=torch.Generator().manual_seed(42))
            for candidate in (session, other):
                for network, optimizer in ((candidate.actor, candidate.actor_optimizer),
                                           (candidate.critic, candidate.critic_optimizer)):
                    optimizer.zero_grad()
                    network(obs).square().mean().backward()
                    optimizer.step()
            equal(session.actor.state_dict(), other.actor.state_dict())
            equal(session.critic.state_dict(), other.critic.state_dict())
            self.assertFalse((Path(directory) / "a/steps.csv").exists())

    def test_transfer_and_zero_adjustment_match_locked_mean_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            session = CadenceTrainingSession.from_stable(self.parent, Path(directory) / "prepared")
            self.assertEqual(session.updates, 100)
            old = ReferenceResidualPolicy(self.parent, session.model)
            obs = torch.randn(32, 68, generator=torch.Generator().manual_seed(45))
            with torch.no_grad():
                transferred = session.actor(obs)
                self.assertTrue(torch.equal(transferred[:, :18], old.actor(obs)))
                self.assertTrue(torch.equal(transferred[:, 18], torch.zeros(32)))
            checkpoint = session.prepare()
            payload = torch.load(checkpoint, weights_only=True)
            self.assertEqual(payload["policy_kind"], "cadence_action_v1")
            new = CadenceResidualPolicy(checkpoint, session.model)
            data = mujoco.MjData(session.model)
            simulation.reset(session.model, data)
            data.qpos[7:] = session.neutral
            data.ctrl[:] = session.neutral
            mujoco.mj_forward(session.model, data)
            initial = simulation.measured_state(session.model, data)
            for time_s in np.linspace(0, 5, 251):
                observed = replace(initial, time=float(time_s))
                np.testing.assert_array_equal(new.targets(observed), old.targets(observed))
                self.assertEqual(new.diagnostics["current_cadence_hz"], 1.1)
            new.reset(); old.reset()
            np.testing.assert_array_equal(new.targets(initial), old.targets(initial))
            self.assertFalse((Path(directory) / "prepared/steps.csv").exists())

    def test_legacy_adam_history_is_preserved_and_new_parameters_are_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = ReferenceTrainingSession.from_checkpoint(self.parent, Path(directory) / "legacy")
            candidate = CadenceTrainingSession.from_stable(self.parent, Path(directory) / "candidate")
            old_group = legacy.actor_optimizer.param_groups[0]
            old_candidate_group = candidate.actor_optimizer.param_groups[0]
            self.assertEqual(len(old_group["params"]), len(old_candidate_group["params"]))
            for old_parameter, transferred in zip(old_group["params"], old_candidate_group["params"]):
                self.assertTrue(torch.equal(old_parameter, transferred))
                for key, value in legacy.actor_optimizer.state[old_parameter].items():
                    other = candidate.actor_optimizer.state[transferred][key]
                    self.assertTrue(torch.equal(value, other) if torch.is_tensor(value) else value == other)
            new_parameters = [p for group in candidate.actor_optimizer.param_groups[1:] for p in group["params"]]
            self.assertTrue(new_parameters)
            self.assertTrue(all(p not in candidate.actor_optimizer.state for p in new_parameters))
            self.assertTrue(torch.equal(legacy.generator.get_state(), candidate.generator.get_state()))

    def test_cadence_changes_advance_phase_without_pose_clock_jumps(self):
        with tempfile.TemporaryDirectory() as directory:
            session = CadenceTrainingSession.from_stable(self.parent, Path(directory) / "prepared")
            policy = CadenceResidualPolicy(session.prepare(), session.model)
            data = mujoco.MjData(session.model)
            simulation.reset(session.model, data)
            data.qpos[7:] = session.neutral
            data.ctrl[:] = session.neutral
            mujoco.mj_forward(session.model, data)
            observed = simulation.measured_state(session.model, data)
            previous_phase, previous_frequency = 0., 1.1
            for i in range(100):
                now = replace(observed, time=i * .02)
                policy.controller.observation(now)
                phase = policy.diagnostics["phase_cycles"]
                if i:
                    self.assertAlmostEqual(phase - previous_phase, previous_frequency * .02, places=12)
                action = np.zeros(19)
                action[-1] = 10 if i < 50 else -10
                policy.controller.targets(now, action)
                self.assertEqual(policy.diagnostics["phase_cycles"], phase)
                self.assertTrue(.8 <= policy.diagnostics["current_cadence_hz"] <= 1.8)
                previous_phase = phase
                previous_frequency = policy.diagnostics["current_cadence_hz"]
            self.assertEqual(policy.reference.config["frequency_hz"], 1.1)
            policy.reset()
            self.assertEqual(policy.diagnostics["phase_cycles"], 0)
            self.assertEqual(policy.diagnostics["current_cadence_hz"], 1.1)
            for value in (float("nan"), float("inf")):
                with self.assertRaises(ValueError): cadence_from_latent(value)
            with self.assertRaises(ValueError):
                CadenceResidualPolicy(self.parent, session.model)


if __name__ == "__main__":
    unittest.main()
