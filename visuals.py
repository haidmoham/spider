"""Presentation-only behaviors for C-1N."""

from __future__ import annotations

import mujoco
import numpy as np


class ResponsivePupils:
    """Model each pupil as a damped loose mass inside its visual eye socket."""

    _EYES = ("left", "right")
    _REST_OFFSETS = {
        "left": np.array((-0.007, 0.003)),
        "right": np.array((0.007, -0.002)),
    }
    _SURFACE_RADIUS = 0.044
    _MAX_OFFSET = 0.025
    _STIFFNESS = {"left": 68.0, "right": 84.0}
    _DAMPING = {"left": 6.5, "right": 7.5}
    _RIM_RESTITUTION = 0.35

    def __init__(self, model: mujoco.MjModel) -> None:
        self._torso_id = model.body("torso").id
        self._eye_ids = {name: model.site(f"{name}_eye_visual").id for name in self._EYES}
        self._pupil_ids = {name: model.site(f"{name}_pupil_visual").id for name in self._EYES}
        self._eye_centers = {name: model.site_pos[site_id].copy() for name, site_id in self._eye_ids.items()}
        self._offsets = {name: offset.copy() for name, offset in self._REST_OFFSETS.items()}
        self._offset_velocities = {name: np.zeros(2) for name in self._EYES}
        self._last_world_velocity = np.zeros(3)
        self._last_time: float | None = None
        self._write_positions(model)

    def reset(self, model: mujoco.MjModel) -> None:
        """Clear pupil momentum after a simulation reset."""
        self._offsets = {name: offset.copy() for name, offset in self._REST_OFFSETS.items()}
        self._offset_velocities = {name: np.zeros(2) for name in self._EYES}
        self._last_world_velocity = np.zeros(3)
        self._last_time = None
        self._write_positions(model)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Apply gravity minus torso acceleration in the torso-local eye plane."""
        world_velocity = np.asarray(data.qvel[:3]).copy()
        if self._last_time is not None and data.time < self._last_time:
            self.reset(model)
        if self._last_time is not None and data.time == self._last_time:
            return
        if self._last_time is None:
            elapsed = float(model.opt.timestep)
            world_acceleration = np.zeros(3)
        else:
            elapsed = min(float(data.time - self._last_time), 1.0 / 30.0)
            world_acceleration = (world_velocity - self._last_world_velocity) / elapsed

        torso_rotation = data.xmat[self._torso_id].reshape(3, 3)
        effective_acceleration = torso_rotation.T @ (np.asarray(model.opt.gravity) - world_acceleration)
        gravity_scale = max(float(np.linalg.norm(model.opt.gravity)), 1.0)
        inertial_offset = self._MAX_OFFSET * np.tanh(effective_acceleration[[1, 2]] / (0.7 * gravity_scale))
        front_back_response = 0.008 * np.tanh(effective_acceleration[0] / gravity_scale)

        for name in self._EYES:
            splay = front_back_response if name == "left" else -front_back_response
            target = self._REST_OFFSETS[name] + inertial_offset + (splay, 0.0)
            target = np.clip(target, -self._MAX_OFFSET, self._MAX_OFFSET)
            spring_acceleration = self._STIFFNESS[name] * (target - self._offsets[name]) - self._DAMPING[name] * self._offset_velocities[name]
            self._offset_velocities[name] += spring_acceleration * elapsed
            self._offsets[name] += self._offset_velocities[name] * elapsed
            for axis in range(2):
                if abs(self._offsets[name][axis]) > self._MAX_OFFSET:
                    self._offsets[name][axis] = np.sign(self._offsets[name][axis]) * self._MAX_OFFSET
                    self._offset_velocities[name][axis] *= -self._RIM_RESTITUTION

        self._last_world_velocity = world_velocity
        self._last_time = float(data.time)
        self._write_positions(model)

    def _write_positions(self, model: mujoco.MjModel) -> None:
        for name in self._EYES:
            lateral, vertical = self._offsets[name]
            forward = np.sqrt(max(0.0, self._SURFACE_RADIUS**2 - lateral**2 - vertical**2))
            model.site_pos[self._pupil_ids[name]] = self._eye_centers[name] + (forward, lateral, vertical)
