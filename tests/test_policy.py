from __future__ import annotations

import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from spider.simulation import MeasuredState, load_model, measured_state, neutral_targets


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "artifacts" / "ppo-crude-baseline-20260915" / "checkpoint-00100.pt"
REPLAYS = ROOT / "artifacts" / "ppo-crude-baseline-20260915" / "replays.zip"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def observed() -> MeasuredState:
    return MeasuredState(
        time=0.0,
        torso_position=(0.1, -0.2, 0.43),
        torso_orientation=(0.99, 0.01, -0.02, 0.03),
        torso_velocity=(0.2, -0.1, 0.05),
        torso_angular_velocity=(-0.03, 0.04, -0.05),
        joint_positions=tuple(np.linspace(-0.3, 0.8, 18)),
        joint_velocities=tuple(np.linspace(0.2, -0.2, 18)),
        actuator_forces=(0.0,) * 18,
        foot_positions={},
        foot_contacts=(),
        com_projection=(0.0, 0.0),
        support_polygon=(),
        support_margin=None,
        foot_normal_loads={},
        foot_position_residual=(),
        foot_position_residual_norm=0.0,
        joint_space_update_direction=(),
        joint_space_update_direction_norm=0.0,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional learning dependency")
class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        global PPOPolicy, torch
        import torch

        from spider.policy import PPOPolicy

        cls.model = load_model()

    def test_baseline_matches_saved_actor_and_original_mapping(self) -> None:
        policy = PPOPolicy(CHECKPOINT, self.model)
        payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        actor = torch.nn.Sequential(
            torch.nn.Linear(47, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 18),
        )
        actor.load_state_dict(payload["actor"])
        state = observed()
        vector = torch.tensor([
            *state.torso_velocity, *state.torso_angular_velocity, *state.torso_orientation,
            *state.joint_positions, *state.joint_velocities, state.torso_position[2],
        ], dtype=torch.float32)
        bound = np.minimum(
            payload["settings"]["bound_rad"],
            np.minimum(
                np.asarray(neutral_targets()) - self.model.actuator_ctrlrange[:, 0],
                self.model.actuator_ctrlrange[:, 1] - np.asarray(neutral_targets()),
            ),
        )
        expected = np.asarray(neutral_targets()) + bound * torch.tanh(actor(vector)).detach().numpy()
        np.testing.assert_allclose(policy.targets(state), expected, rtol=0.0, atol=1e-12)
        self.assertEqual(policy.settings["physics_steps"], 10)
        self.assertEqual(policy.updates, 100)
        self.assertEqual(len(policy.checkpoint_sha256), 64)
        self.assertIn("PPO-100 mean", policy.description)
        self.assertIsNone(policy.config["lower_offset_rad_per_leg"])
        self.assertIsNone(policy.config["smooth_tau_seconds"])

    def test_policy_commands_match_accepted_mean_and_sampled_replays(self) -> None:
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        panes = (("3", False), ("1", True))
        with tempfile.TemporaryDirectory() as directory, zipfile.ZipFile(REPLAYS) as archive:
            required = {
                f"{pane}/{filename}"
                for pane, _ in panes
                for filename in ("model.mjb", "states.npz")
            }
            self.assertTrue(required.issubset(archive.namelist()))
            for member in required:
                path = Path(member)
                self.assertFalse(path.is_absolute())
                self.assertNotIn("..", path.parts)
                archive.extract(member, directory)

            for pane, sampled in panes:
                with self.subTest(pane=pane, sampled=sampled):
                    replay = Path(directory) / pane
                    model = mujoco.MjModel.from_binary_path(str(replay / "model.mjb"))
                    data = mujoco.MjData(model)
                    with np.load(replay / "states.npz", allow_pickle=False) as saved:
                        states = saved["states"]
                    policy = PPOPolicy(CHECKPOINT, model, seed=201, sampled=sampled)
                    for index in range(20):
                        mujoco.mj_setState(model, data, states[index], state_spec)
                        mujoco.mj_forward(model, data)
                        actual = policy.targets(measured_state(model, data))
                        mujoco.mj_setState(model, data, states[index + 1], state_spec)
                        np.testing.assert_array_equal(
                            actual,
                            data.ctrl,
                            err_msg=f"recorded command mismatch at action {index}",
                        )

    def test_sampled_policy_matches_seeded_normal_without_global_rng_effect(self) -> None:
        torch.manual_seed(77)
        expected_global = torch.rand(4)
        torch.manual_seed(77)
        policy = PPOPolicy(CHECKPOINT, self.model, sampled=True, seed=201)
        state = observed()
        actual = np.asarray(policy.targets(state))
        torch.testing.assert_close(torch.rand(4), expected_global)

        actor = policy.actor
        vector = policy._observation(state)
        with torch.random.fork_rng():
            torch.manual_seed(201)
            latent = torch.distributions.Normal(actor(vector), policy.settings["std"]).sample()
        bound = policy._bound
        expected = policy._neutral + bound * torch.tanh(latent).detach().numpy()
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_observation_shape_and_finiteness_are_checked(self) -> None:
        policy = PPOPolicy(CHECKPOINT, self.model)
        bad_shape = observed().__class__(**{**observed().__dict__, "joint_positions": (0.0,) * 17})
        with self.assertRaisesRegex(ValueError, "47 finite"):
            policy.targets(bad_shape)
        bad_finite = observed().__class__(**{**observed().__dict__, "torso_velocity": (np.nan, 0.0, 0.0)})
        with self.assertRaisesRegex(ValueError, "47 finite"):
            policy.targets(bad_finite)

    def test_treatments_are_bounded_and_reset_local_state(self) -> None:
        state = observed()
        baseline = np.asarray(PPOPolicy(CHECKPOINT, self.model).targets(state))
        smooth = PPOPolicy(CHECKPOINT, self.model, treatment="smooth")
        first_smooth = np.asarray(smooth.targets(state))
        self.assertLess(np.linalg.norm(first_smooth - np.asarray(neutral_targets())),
                        np.linalg.norm(baseline - np.asarray(neutral_targets())))
        for _ in range(20):
            smooth.targets(state)
        smooth.reset()
        np.testing.assert_allclose(smooth.targets(state), first_smooth, rtol=0.0, atol=1e-12)

        lower = PPOPolicy(CHECKPOINT, self.model, treatment="lower")
        np.testing.assert_allclose(lower.targets(state), baseline, rtol=0.0, atol=1e-12)
        commands = np.asarray([lower.targets(state) for _ in range(120)])
        self.assertTrue(np.all(commands >= self.model.actuator_ctrlrange[:, 0]))
        self.assertTrue(np.all(commands <= self.model.actuator_ctrlrange[:, 1]))
        lower.reset()
        np.testing.assert_allclose(lower.targets(state), baseline, rtol=0.0, atol=1e-12)

    def test_rejects_wrong_model_shape(self) -> None:
        model = SimpleNamespace(
            actuator_ctrlrange=np.zeros((17, 2)),
            opt=SimpleNamespace(timestep=0.002),
        )
        with self.assertRaisesRegex(ValueError, "18 x 2"):
            PPOPolicy(CHECKPOINT, model)

    def test_rejects_invalid_runtime_settings_and_actor_dtype(self) -> None:
        original = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        cases = (
            ("horizon", 0, "horizon must be a positive integer"),
            ("fall_height_m", -0.1, "fall_height_m must be positive"),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "invalid.pt"
            for key, value, message in cases:
                payload = {**original, "settings": {**original["settings"], key: value}}
                torch.save(payload, checkpoint)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                    PPOPolicy(checkpoint, self.model)

            actor = original["actor"].copy()
            actor["0.weight"] = actor["0.weight"].to(torch.float64)
            torch.save({**original, "actor": actor}, checkpoint)
            with self.assertRaisesRegex(ValueError, "finite float32"):
                PPOPolicy(checkpoint, self.model)


if __name__ == "__main__":
    unittest.main()
