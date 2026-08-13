"""Prove that C-1N's redesign changes presentation, not mechanics."""

from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np

from simulation import FOOT_NAMES, MODEL_PATH, reset, step
from visuals import ResponsivePupils


BASELINE_PATH = Path(__file__).resolve().parent / "model" / "spider_physics_baseline.xml"


def _names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> tuple[str, ...]:
    return tuple(mujoco.mj_id2name(model, object_type, index) or "" for index in range(count))


class VisualRedesignInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline = mujoco.MjModel.from_xml_path(str(BASELINE_PATH))
        cls.candidate = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    def assert_array_equal(self, field: str) -> None:
        np.testing.assert_array_equal(getattr(self.baseline, field), getattr(self.candidate, field), err_msg=field)

    def test_kinematics_inertials_actuators_and_options_are_identical(self) -> None:
        baseline, candidate = self.baseline, self.candidate
        self.assertEqual((baseline.nq, baseline.nv, baseline.nu), (candidate.nq, candidate.nv, candidate.nu))
        self.assertEqual(baseline.ntendon, candidate.ntendon)
        self.assertEqual(baseline.neq, candidate.neq)
        self.assertEqual(_names(baseline, mujoco.mjtObj.mjOBJ_BODY, baseline.nbody), _names(candidate, mujoco.mjtObj.mjOBJ_BODY, candidate.nbody))
        self.assertEqual(_names(baseline, mujoco.mjtObj.mjOBJ_JOINT, baseline.njnt), _names(candidate, mujoco.mjtObj.mjOBJ_JOINT, candidate.njnt))
        self.assertEqual(_names(baseline, mujoco.mjtObj.mjOBJ_ACTUATOR, baseline.nu), _names(candidate, mujoco.mjtObj.mjOBJ_ACTUATOR, candidate.nu))

        for field in (
            "body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat",
            "body_subtreemass", "body_invweight0", "jnt_type", "jnt_bodyid", "jnt_pos", "jnt_axis", "jnt_range",
            "qpos0", "qpos_spring", "dof_damping", "dof_armature", "dof_frictionloss", "dof_M0", "dof_invweight0",
            "actuator_trntype", "actuator_trnid", "actuator_ctrlrange", "actuator_forcerange", "actuator_gear",
            "actuator_gainprm", "actuator_biasprm", "actuator_dynprm",
        ):
            self.assert_array_equal(field)

        for field in (
            "timestep", "integrator", "gravity", "solver", "cone", "jacobian", "iterations", "tolerance",
            "ls_iterations", "ls_tolerance", "noslip_iterations", "noslip_tolerance", "ccd_iterations", "ccd_tolerance",
            "impratio", "density", "viscosity", "wind", "disableflags", "enableflags",
        ):
            baseline_value = np.asarray(getattr(baseline.opt, field))
            candidate_value = np.asarray(getattr(candidate.opt, field))
            np.testing.assert_array_equal(baseline_value, candidate_value, err_msg=f"opt.{field}")

    def test_physical_contacts_and_sensors_are_identical(self) -> None:
        baseline, candidate = self.baseline, self.candidate
        self.assertEqual(_names(baseline, mujoco.mjtObj.mjOBJ_GEOM, baseline.ngeom), _names(candidate, mujoco.mjtObj.mjOBJ_GEOM, candidate.ngeom))
        for field in (
            "geom_bodyid", "geom_type", "geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_margin",
            "geom_gap", "geom_condim", "geom_priority", "geom_solmix", "geom_solref", "geom_solimp", "geom_contype",
            "geom_conaffinity", "geom_fluid", "geom_group",
        ):
            self.assert_array_equal(field)

        baseline_sites = _names(baseline, mujoco.mjtObj.mjOBJ_SITE, baseline.nsite)
        candidate_sites = _names(candidate, mujoco.mjtObj.mjOBJ_SITE, candidate.nsite)
        self.assertTrue(set(baseline_sites).issubset(candidate_sites))
        for name in baseline_sites:
            baseline_id = baseline.site(name).id
            candidate_id = candidate.site(name).id
            for field in ("site_bodyid", "site_pos", "site_quat", "site_size", "site_type"):
                np.testing.assert_array_equal(getattr(baseline, field)[baseline_id], getattr(candidate, field)[candidate_id], err_msg=f"{field}:{name}")
        for name in set(candidate_sites) - set(baseline_sites):
            self.assertTrue(name.endswith("_visual"), name)
            self.assertEqual(candidate.site(name).group, 2)

        self.assertEqual(_names(baseline, mujoco.mjtObj.mjOBJ_SENSOR, baseline.nsensor), _names(candidate, mujoco.mjtObj.mjOBJ_SENSOR, candidate.nsensor))
        for field in ("sensor_type", "sensor_objtype", "sensor_reftype", "sensor_refid", "sensor_dim"):
            self.assert_array_equal(field)
        for sensor_id in range(baseline.nsensor):
            object_type = mujoco.mjtObj(int(baseline.sensor_objtype[sensor_id]))
            baseline_object = mujoco.mj_id2name(baseline, object_type, int(baseline.sensor_objid[sensor_id]))
            candidate_object = mujoco.mj_id2name(candidate, object_type, int(candidate.sensor_objid[sensor_id]))
            self.assertEqual(baseline_object, candidate_object)

    def test_reset_dynamics_and_stand_trajectory_are_identical(self) -> None:
        baseline_data = mujoco.MjData(self.baseline)
        candidate_data = mujoco.MjData(self.candidate)
        reset(self.baseline, baseline_data)
        reset(self.candidate, candidate_data)

        for field in ("qpos", "qvel", "ctrl", "subtree_com", "qfrc_bias"):
            np.testing.assert_array_equal(getattr(baseline_data, field), getattr(candidate_data, field), err_msg=field)
        baseline_mass = np.empty((self.baseline.nv, self.baseline.nv))
        candidate_mass = np.empty((self.candidate.nv, self.candidate.nv))
        mujoco.mj_fullM(self.baseline, baseline_data, baseline_mass)
        mujoco.mj_fullM(self.candidate, candidate_data, candidate_mass)
        np.testing.assert_array_equal(baseline_mass, candidate_mass)

        for name in FOOT_NAMES:
            baseline_foot = self.baseline.geom(f"{name}_foot").id
            candidate_foot = self.candidate.geom(f"{name}_foot").id
            np.testing.assert_array_equal(baseline_data.geom_xpos[baseline_foot], candidate_data.geom_xpos[candidate_foot])
            baseline_jacobian = np.empty((3, self.baseline.nv))
            candidate_jacobian = np.empty((3, self.candidate.nv))
            mujoco.mj_jacGeom(self.baseline, baseline_data, baseline_jacobian, None, baseline_foot)
            mujoco.mj_jacGeom(self.candidate, candidate_data, candidate_jacobian, None, candidate_foot)
            np.testing.assert_array_equal(baseline_jacobian, candidate_jacobian)

        for _ in range(round(10.0 / self.baseline.opt.timestep)):
            step(self.baseline, baseline_data)
            step(self.candidate, candidate_data)
        np.testing.assert_array_equal(baseline_data.qpos, candidate_data.qpos)
        np.testing.assert_array_equal(baseline_data.qvel, candidate_data.qvel)

    def test_pupils_fall_under_gravity_and_lag_a_sideways_shove(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        reset(model, data)
        pupils = ResponsivePupils(model)
        pupil_ids = [model.site(f"{name}_pupil_visual").id for name in ("left", "right")]
        initial_positions = model.site_pos[pupil_ids].copy()

        for _ in range(500):
            step(model, data)
            pupils.update(model, data)
        gravity_positions = model.site_pos[pupil_ids].copy()
        self.assertTrue(np.all(gravity_positions[:, 2] < initial_positions[:, 2] - 0.005))

        torso_id = model.body("torso").id
        shove_force = model.body_subtreemass[torso_id] * abs(model.opt.gravity[2])
        for _ in range(100):
            data.xfrc_applied[torso_id, 1] = shove_force
            step(model, data)
            pupils.update(model, data)
        shove_positions = model.site_pos[pupil_ids].copy()
        self.assertTrue(np.all(shove_positions[:, 1] < gravity_positions[:, 1] - 0.003))


if __name__ == "__main__":
    unittest.main()
