"""Geometry and continuity checks; these do not establish dynamic stability."""

import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from spider import simulation
from spider.gait_reference import GAITS, GaitReference


class GaitReferenceTests(unittest.TestCase):
    def test_does_not_reuse_another_models_neutral_cache(self):
        model = simulation.load_model()
        with patch.object(simulation, "neutral_foot_positions", side_effect=AssertionError("stale cache")):
            reference = GaitReference(model, "tripod")
            self.assertTrue(np.isfinite(reference.pose(0)).all())

    def test_requested_feet_match_canonical_geometry_over_complete_cycles(self):
        model = simulation.load_model()
        data = mujoco.MjData(model)
        for name in GAITS:
            reference = GaitReference(model, name)
            for t in np.linspace(0, 1 / reference.config["frequency_hz"], 201):
                with self.subTest(gait=name, time=t):
                    data.qpos[:3] = (0, 0, reference.height_m)
                    data.qpos[3:7] = (1, 0, 0, 0)
                    data.qpos[7:] = reference.pose(t)
                    mujoco.mj_forward(model, data)
                    actual = np.array([data.geom_xpos[model.geom(n + "_foot").id]
                                       for n in simulation.FOOT_NAMES])
                    np.testing.assert_allclose(actual, reference.feet(t), atol=1e-10)

    def test_periodic_and_velocity_continuous_at_lift_and_plant(self):
        model = simulation.load_model()
        dt = 1e-6
        for name in GAITS:
            r = GaitReference(model, name)
            frequency, duty = r.config["frequency_hz"], r.config["stance_fraction"]
            np.testing.assert_allclose(r.pose(0), r.pose(1 / frequency), atol=1e-10)
            for leg, offset in enumerate(r.config["offsets"]):
                for phase in (0, duty):
                    t = (1 + phase - offset) / frequency
                    before, now, after = (r.feet(t + delta)[leg] for delta in (-dt, 0, dt))
                    np.testing.assert_allclose((now-before)/dt, (after-now)/dt, atol=3e-5)

    def test_stance_foot_stays_fixed_in_translating_world_frame(self):
        model = simulation.load_model()
        for name in GAITS:
            r = GaitReference(model, name)
            # Front-left has phase offset zero for ripple/tripod; account for wave.
            t = (1 + 0.2 - r.config["offsets"][0]) / r.config["frequency_hz"]
            first = r.feet(t)[0] + (r.speed_mps*t, 0, 0)
            second = r.feet(t + .01)[0] + (r.speed_mps*(t+.01), 0, 0)
            np.testing.assert_allclose(first, second, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
