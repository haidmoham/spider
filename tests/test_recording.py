"""Check the notebook's recorded evidence and preserved viewer diagnostics."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from c1n.learning import record_policy, plot_recordings
from c1n.recording import STATE
from c1n.runtime import advance, build_simulation
from c1n.viewing.live import GaitPlots


class RecordingTests(unittest.TestCase):
    def test_recorded_states_match_measurements_and_action_intervals(self):
        requested = np.array([10.0, -10.0, 0.1] * 6)
        result = record_policy(
            lambda observation: requested, label="test fixture", action_count=3, physics_steps=7
        )
        self.assertEqual(len(result.measurements), 4)
        np.testing.assert_array_equal(result.action_times, result.replay.times[:-1])
        model = result.replay.model
        data = mujoco.MjData(model)
        for state, measurement in zip(result.replay.states, result.measurements):
            mujoco.mj_setState(model, data, state, STATE)
            mujoco.mj_forward(model, data)
            self.assertEqual(data.time, measurement.time)
            np.testing.assert_array_equal(data.qpos[:3], measurement.torso_position)
            np.testing.assert_array_equal(data.qpos[7:], measurement.joint_positions)
            np.testing.assert_array_equal(data.qvel[6:], measurement.joint_velocities)
            # Replay rebuilds display transforms. Dynamics-derived forces must
            # come from the recorded measurement, not a fresh mj_forward solve.
        requested[:] = 0
        self.assertEqual(result.offsets[0, 0], 10.0)
        self.assertTrue(np.all(result.targets <= model.actuator_ctrlrange[:, 1]))
        self.assertTrue(np.all(result.targets >= model.actuator_ctrlrange[:, 0]))
        np.testing.assert_allclose(np.diff(result.replay.times), 7 * model.opt.timestep)

    def test_invalid_timing_does_not_construct_a_simulation(self):
        with patch("c1n.learning.LearningSimulation") as simulation:
            for value in (0, -1, True, 0.5, np.bool_(True)):
                with self.assertRaises(ValueError):
                    record_policy(None, label="invalid", action_count=1, physics_steps=value)
            simulation.assert_not_called()

    def test_policy_plot_uses_applied_targets_and_measured_times(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = record_policy(
            lambda observation: np.zeros(18), label="test fixture", action_count=2, physics_steps=1
        )
        fig, axes = plot_recordings(result, actuator=1)
        np.testing.assert_array_equal(axes[1, 1].lines[0].get_xdata(), result.replay.times)
        np.testing.assert_array_equal(axes[1, 1].lines[1].get_ydata()[:-1], result.targets[:, 1])
        plt.close(fig)

    def test_shared_shuffle_viewer_keeps_six_diagnostic_plots(self):
        model, data, power, coordinator, controller, pulse = build_simulation("shuffle")
        displayed = []
        viewer = SimpleNamespace(
            viewport=SimpleNamespace(width=1200, height=800, left=0, bottom=0),
            set_figures=lambda figures: displayed.append(figures),
        )
        plots = GaitPlots()
        advance(model, data, 1, "shuffle", coordinator, controller, pulse)
        plots.update(viewer, model, data, coordinator)
        self.assertEqual(len(displayed[-1]), 6)
        self.assertEqual(len(plots.samples), 1)
        for _, figure in displayed[-1]:
            self.assertEqual(figure.linepnt[0], 1)


if __name__ == "__main__":
    unittest.main()
