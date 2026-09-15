"""Prove that the stalk preset changes presentation fields only."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from spider.recording import STATE
from spider.simulation import MODEL_PATH, reset
from spider.viewing.stalk import apply_stalk_presentation, stalk_camera


_PRESENTATION_ARRAYS = {
    "geom_rgba",
    "light_ambient",
    "light_castshadow",
    "light_diffuse",
    "light_dir",
    "light_pos",
    "light_specular",
}


class StalkPresentationTests(unittest.TestCase):
    def test_stalk_style_and_camera_leave_physics_and_state_unchanged(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        reset(model, data)

        # Snapshot every exposed model array except the fields that define this
        # presentation. This covers transforms, contacts, inertials, joints,
        # actuators, sensors, solver data, and initial state arrays.
        physics_arrays = {
            name: value.copy()
            for name in dir(model)
            if name not in _PRESENTATION_ARRAYS
            and isinstance((value := getattr(model, name)), np.ndarray)
        }
        options = {
            name: np.asarray(getattr(model.opt, name)).copy()
            for name in (
                "timestep",
                "integrator",
                "gravity",
                "solver",
                "cone",
                "jacobian",
                "iterations",
                "tolerance",
                "impratio",
                "density",
                "viscosity",
                "wind",
                "disableflags",
                "enableflags",
            )
        }
        state = np.empty(mujoco.mj_stateSize(model, STATE))
        mujoco.mj_getState(model, data, state, STATE)

        apply_stalk_presentation(model)
        camera = stalk_camera(model, data)

        for name, expected in physics_arrays.items():
            np.testing.assert_array_equal(getattr(model, name), expected, err_msg=name)
        for name, expected in options.items():
            np.testing.assert_array_equal(getattr(model.opt, name), expected, err_msg=f"opt.{name}")
        actual_state = np.empty_like(state)
        mujoco.mj_getState(model, data, actual_state, STATE)
        np.testing.assert_array_equal(actual_state, state)
        np.testing.assert_array_equal(data.qpos, state[1 : 1 + model.nq])
        assert isinstance(camera, mujoco.MjvCamera)
