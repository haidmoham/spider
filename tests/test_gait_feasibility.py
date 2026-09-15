import math
import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import mujoco

from spider import gait_feasibility, simulation
from spider.chassis_candidate import LEG_NAMES, load_candidate, reset_candidate


class GaitFeasibilityHelpersTest(unittest.TestCase):
    def test_ramp_starts_at_neutral_without_changing_reference_shape(self) -> None:
        reference = SimpleNamespace(name="test", pose=lambda _time: np.arange(18, dtype=float))
        neutral = tuple(np.linspace(-1.0, 1.0, 18))
        with patch.object(gait_feasibility.simulation, "neutral_targets", return_value=neutral):
            np.testing.assert_array_equal(
                gait_feasibility._ramped_target(reference, 0.0), np.asarray(neutral)
            )

    def test_smoothstep_is_bounded_and_has_zero_endpoint_slope(self) -> None:
        self.assertEqual(gait_feasibility._smoothstep(-1.0), 0.0)
        self.assertEqual(gait_feasibility._smoothstep(0.0), 0.0)
        self.assertEqual(gait_feasibility._smoothstep(1.0), 1.0)
        self.assertEqual(gait_feasibility._smoothstep(2.0), 1.0)
        epsilon = 1e-6
        self.assertLess(gait_feasibility._smoothstep(epsilon) / epsilon, 1e-4)

    def test_tilt_uses_body_up_axis_and_ignores_yaw(self) -> None:
        yaw_90 = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
        roll_60 = (math.cos(math.pi / 6), math.sin(math.pi / 6), 0.0, 0.0)
        self.assertAlmostEqual(gait_feasibility._tilt_deg(yaw_90), 0.0)
        self.assertAlmostEqual(gait_feasibility._tilt_deg(roll_60), 60.0)

    def test_nonfinite_state_is_detected_before_geometric_measurement(self) -> None:
        data = SimpleNamespace(qpos=np.asarray([math.nan]), qvel=np.asarray([0.0]))
        self.assertEqual(
            gait_feasibility._termination(data, height_m=math.nan, tilt_deg=math.inf),
            "non_finite_state",
        )


class ChassisCandidateTest(unittest.TestCase):
    def test_exactly_six_three_joint_legs(self) -> None:
        model = load_candidate()
        feet = [model.geom(i).name for i in range(model.ngeom)
                if model.geom(i).name.endswith("_foot")]
        self.assertEqual(set(feet), {name + "_foot" for name in simulation.FOOT_NAMES})
        self.assertEqual(len(feet), 6)
        self.assertEqual(model.nu, 18)
        for name in simulation.FOOT_NAMES:
            for joint in ("coxa", "hip", "knee"):
                self.assertGreaterEqual(model.joint(f"{name}_{joint}").id, 0)

    def test_reset_preserves_physical_geometry_and_vertical_coxa_axes(self) -> None:
        before = hashlib.sha256(simulation.MODEL_PATH.read_bytes()).hexdigest()
        original = simulation.load_model()
        candidate = load_candidate()
        original_data = mujoco.MjData(original)
        candidate_data = mujoco.MjData(candidate)
        simulation.reset(original, original_data)
        reset_candidate(candidate, candidate_data)

        for geom_id in range(original.ngeom):
            name = original.geom(geom_id).name
            candidate_id = candidate.geom(name).id
            np.testing.assert_allclose(candidate_data.geom_xpos[candidate_id],
                                       original_data.geom_xpos[geom_id], atol=1e-12)
            np.testing.assert_allclose(candidate_data.geom_xmat[candidate_id],
                                       original_data.geom_xmat[geom_id], atol=1e-12)
            np.testing.assert_allclose(candidate.geom_friction[candidate_id],
                                       original.geom_friction[geom_id])
            np.testing.assert_allclose(candidate.geom_size[candidate_id],
                                       original.geom_size[geom_id])
        np.testing.assert_allclose(candidate.body_mass[:original.nbody], original.body_mass)
        for name in LEG_NAMES:
            np.testing.assert_allclose(
                candidate_data.xaxis[candidate.joint(f"{name}_coxa").id], (0.0, 0.0, 1.0),
                atol=1e-12,
            )
        after = hashlib.sha256(simulation.MODEL_PATH.read_bytes()).hexdigest()
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
