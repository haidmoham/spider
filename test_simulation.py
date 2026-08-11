"""Regression checks for C-1N's deterministic simulation boundary."""

from __future__ import annotations

import unittest

import mujoco

from interact import build_simulation, execute
from simulate import run
from simulation import FOOT_NAMES, load_model, measured_state, reset, set_targets, step
from walk import GaitCoordinator, apply_gait_control


class SimulationCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = load_model()
        self.data = mujoco.MjData(self.model)

    def test_reset_is_deterministic_and_has_six_foot_positions(self) -> None:
        reset(self.model, self.data)
        first = measured_state(self.model, self.data)
        step(self.model, self.data)
        reset(self.model, self.data)
        second = measured_state(self.model, self.data)

        self.assertEqual(first, second)
        self.assertEqual(len(first.foot_positions), len(FOOT_NAMES))
        self.assertEqual(first.torso_position, (0.0, 0.0, 0.5))
        self.assertEqual(first.torso_orientation, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(first.torso_angular_velocity, (0.0, 0.0, 0.0))
        self.assertEqual(first.joint_positions, (0.0, 0.8) * 6)

    def test_targets_are_commands_not_measured_state(self) -> None:
        reset(self.model, self.data)
        before = measured_state(self.model, self.data)
        set_targets(self.data, (0.1, 0.7) * 6)
        self.assertEqual(measured_state(self.model, self.data), before)
        step(self.model, self.data)
        self.assertEqual(tuple(self.data.ctrl), (0.1, 0.7) * 6)
        self.assertGreater(measured_state(self.model, self.data).time, 0.0)

    def test_static_and_legacy_shuffle_failure_signatures_are_reproducible(self) -> None:
        """Protect current physics evidence without treating it as a capability claim."""
        reset(self.model, self.data)
        for _ in range(round(1.0 / self.model.opt.timestep)):
            step(self.model, self.data)
        self.assertAlmostEqual(self.data.qpos[2], 0.4925214786895183, places=10)
        self.assertAlmostEqual(self.data.qvel[2], -0.0001325950249269702, places=10)

        reset(self.model, self.data)
        shuffle = GaitCoordinator(self.model, self.data)
        for _ in range(round(2.0 / self.model.opt.timestep)):
            targets, _, _ = apply_gait_control(shuffle, self.data)
            step(self.model, self.data, targets)
        self.assertAlmostEqual(self.data.qpos[0], 1.1371138730204575e-14, places=10)
        self.assertAlmostEqual(self.data.qpos[1], 0.2549283460290788, places=10)
        self.assertAlmostEqual(self.data.qpos[2], 0.5490331532480623, places=10)

    def test_static_and_live_surfaces_share_the_neutral_reset(self) -> None:
        static_model, _ = run(0.002)
        live_model, live_data, power, coordinator = build_simulation("none")
        static_initial = mujoco.MjData(static_model)
        reset(static_model, static_initial)
        self.assertEqual(measured_state(static_model, static_initial), measured_state(live_model, live_data))

        response = execute({"command": "reset"}, live_model, live_data, power, "none", coordinator)
        self.assertEqual(response["torso_position"], [0.0, 0.0, 0.5])
        self.assertEqual(response["torso_orientation"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(response["torso_angular_velocity"], [0.0, 0.0, 0.0])
        self.assertEqual(response["controls"], [0.0, 0.8] * 6)


if __name__ == "__main__":
    unittest.main()
