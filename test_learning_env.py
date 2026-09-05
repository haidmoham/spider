"""Checks for the notebook adapter's action and timing boundary."""

import unittest

import mujoco
import numpy as np

import simulation
from learning_env import LearningSimulation


class LearningSimulationTests(unittest.TestCase):
    def test_zero_offsets_match_canonical_rollout(self) -> None:
        env = LearningSimulation()
        model = simulation.load_model()
        data = mujoco.MjData(model)
        simulation.reset(model, data)
        for count in (1, 7, 50):
            actual = env.step(np.zeros(18), physics_steps=count)
            for _ in range(count):
                simulation.step(model, data, simulation.neutral_targets())
            self.assertEqual(actual, simulation.measured_state(model, data))
            np.testing.assert_array_equal(env.data.qpos, data.qpos)
            np.testing.assert_array_equal(env.data.qvel, data.qvel)
            np.testing.assert_array_equal(env.data.ctrl, data.ctrl)

    def test_offsets_clip_absolute_targets_and_advance_exact_count(self) -> None:
        env = LearningSimulation()
        offsets = np.array([100.0, -100.0, 0.1] * 6)
        expected = np.clip(
            np.asarray(simulation.neutral_targets()) + offsets,
            env.model.actuator_ctrlrange[:, 0],
            env.model.actuator_ctrlrange[:, 1],
        )
        data = mujoco.MjData(env.model)
        simulation.reset(env.model, data)
        for _ in range(9):
            simulation.step(env.model, data, expected.tolist())
        actual = env.step(offsets, physics_steps=9)
        self.assertEqual(actual, simulation.measured_state(env.model, data))
        np.testing.assert_array_equal(env.data.ctrl, expected)
        self.assertAlmostEqual(actual.time, 9 * env.model.opt.timestep)

    def test_invalid_inputs_leave_state_and_controls_unchanged(self) -> None:
        env = LearningSimulation()
        env.step([0.1] * 18, physics_steps=4)
        before = simulation.measured_state(env.model, env.data)
        controls = env.data.ctrl.copy()
        cases = [
            ([0.0] * 18, count) for count in (0, -1, 1.5, True, np.bool_(True), "2", None)
        ]
        cases += [(offsets, 1) for offsets in (
            [0.0] * 17, np.zeros((6, 3)), np.zeros((18, 1)),
            [float("nan")] * 18, [float("inf")] * 18, ["invalid"] * 18,
        )]
        for offsets, count in cases:
            with self.subTest(offsets=offsets, physics_steps=count):
                with self.assertRaises(ValueError):
                    env.step(offsets, physics_steps=count)
                self.assertEqual(simulation.measured_state(env.model, env.data), before)
                np.testing.assert_array_equal(env.data.ctrl, controls)

    def test_reset_repeats_initial_state_and_rollout(self) -> None:
        env = LearningSimulation()
        initial = env.reset()
        first = env.step([0.02] * 18, physics_steps=20)
        self.assertEqual(env.reset(), initial)
        np.testing.assert_array_equal(env.data.ctrl, simulation.neutral_targets())
        self.assertEqual(env.step([0.02] * 18, physics_steps=20), first)


if __name__ == "__main__":
    unittest.main()
