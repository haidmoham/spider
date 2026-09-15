import unittest

import mujoco
import numpy as np

from spider.recording import STATE
from spider.viewing.googly import apply_googly_frame, apply_life_lights, build_googly_track


def _model(marked=True):
    marker = '<site name="googly_enabled"/>' if marked else ''
    return mujoco.MjModel.from_xml_string(f"""
    <mujoco><option gravity="0 0 -9.81"/><asset>
      <material name="candidate_life_glow" rgba=".95 .015 .025 1"/>
    </asset><worldbody><body name="torso"><freejoint/>{marker}
      <site name="left_eye_visual" pos=".1 .1 0"/>
      <site name="left_pupil_visual" pos=".12 .1 0"/>
      <site name="right_eye_visual" pos=".1 -.1 0"/>
      <site name="right_pupil_visual" pos=".12 -.1 0"/>
      <site name="ruby_accent" rgba=".95 .015 .025 1"/>
      <light name="candidate_underlight" diffuse=".1 .0015 .0025"/>
      <geom type="sphere" size=".01" mass="1" material="candidate_life_glow"/>
    </body></worldbody></mujoco>""")


def _recording(model, quaternion=(1, 0, 0, 0)):
    data, times, states = mujoco.MjData(model), np.arange(8) * 0.02, []
    for time_s in times:
        mujoco.mj_resetData(model, data)
        data.time, data.qpos[3:7] = time_s, quaternion
        state = np.empty(mujoco.mj_stateSize(model, STATE))
        mujoco.mj_getState(model, data, state, STATE)
        states.append(state)
    return np.asarray(states), times


class GooglyTrackTests(unittest.TestCase):
    def test_unmarked_model_is_unchanged(self):
        model = _model(False)
        states, times = _recording(model)
        sites, rgba = model.site_pos.copy(), model.mat_rgba.copy()
        self.assertIsNone(build_googly_track(model, states, times))
        self.assertFalse(apply_life_lights(model, 0.5))
        np.testing.assert_array_equal(model.site_pos, sites)
        np.testing.assert_array_equal(model.mat_rgba, rgba)

    def test_gravity_settle_bounds_and_replay_determinism(self):
        model = _model()
        states, times = _recording(model)
        first = build_googly_track(model, states, times)
        np.testing.assert_array_equal(first, build_googly_track(model, states, times))
        eye = model.site("left_eye_visual").id
        offsets = first[:, 0] - model.site_pos[eye]
        np.testing.assert_allclose(offsets, np.tile(offsets[0], (len(offsets), 1)), atol=1e-15)
        self.assertLess(offsets[0, 2], 0)
        np.testing.assert_allclose(np.linalg.norm(offsets, axis=1), 0.02, atol=1e-14)

    def test_rotated_torso_keeps_gravity_direction_in_world(self):
        model = _model()
        q = (np.sqrt(0.5), np.sqrt(0.5), 0, 0)
        states, times = _recording(model, q)
        local = build_googly_track(model, states, times)[0, 0]
        local -= model.site_pos[model.site("left_eye_visual").id]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(q))
        self.assertLess((rotation.reshape(3, 3) @ local)[2], 0)

    def test_apply_does_not_mutate_recorded_state(self):
        model = _model()
        states, times = _recording(model)
        track, data = build_googly_track(model, states, times), mujoco.MjData(model)
        mujoco.mj_setState(model, data, states[3], STATE)
        before = (data.qpos.copy(), data.qvel.copy(), float(data.time))
        self.assertTrue(apply_googly_frame(model, track, 3))
        np.testing.assert_array_equal(data.qpos, before[0])
        np.testing.assert_array_equal(data.qvel, before[1])
        self.assertEqual(data.time, before[2])

    def test_life_lights_are_time_deterministic_and_red(self):
        model = _model()
        material = model.mat("candidate_life_glow").id
        apply_life_lights(model, 0.0)
        dim = model.mat_emission[material]
        apply_life_lights(model, 0.5)
        bright = model.mat_emission[material]
        rgba = model.mat_rgba[material].copy()
        apply_life_lights(model, 0.5)
        self.assertEqual(model.mat_emission[material], bright)
        self.assertGreater(bright, dim)
        self.assertGreater(rgba[0], 20 * rgba[1])
        self.assertGreater(rgba[0], 20 * rgba[2])


if __name__ == "__main__":
    unittest.main()
