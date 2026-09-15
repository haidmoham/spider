"""Deterministic decorative animation for explicitly marked candidate models."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..recording import STATE


@dataclass(frozen=True)
class _Eye:
    eye: int
    pupil: int
    outward: np.ndarray
    radius: float
    tangent_a: np.ndarray
    tangent_b: np.ndarray


MAX_DISK = 0.68
NATURAL_FREQUENCY = 13.0
DAMPING = 8.0


def _id(model, kind, name):
    return int(mujoco.mj_name2id(model, kind, name))


def _marked(model):
    return _id(model, mujoco.mjtObj.mjOBJ_SITE, "googly_enabled") >= 0


def _eyes(model):
    if not _marked(model):
        return None
    result = []
    for side in ("left", "right"):
        eye = _id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_eye_visual")
        pupil = _id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_pupil_visual")
        if eye < 0 or pupil < 0:
            return None
        radial = model.site_pos[pupil].copy() - model.site_pos[eye]
        radius = float(np.linalg.norm(radial))
        if radius <= 0:
            return None
        outward = radial / radius
        reference = np.array((0.0, 0.0, 1.0))
        if abs(float(outward @ reference)) > 0.9:
            reference = np.array((0.0, 1.0, 0.0))
        tangent_a = np.cross(outward, reference)
        tangent_a /= np.linalg.norm(tangent_a)
        tangent_b = np.cross(outward, tangent_a)
        result.append(_Eye(eye, pupil, outward, radius, tangent_a, tangent_b))
    return result[0], result[1]


def _limit(position, velocity):
    length = float(np.linalg.norm(position))
    if length <= MAX_DISK:
        return position, velocity
    normal = position / length
    outward_speed = float(velocity @ normal)
    return normal * MAX_DISK, velocity - max(0.0, outward_speed) * normal


def build_googly_track(model, states, times):
    """Precompute a damped pupil track for every source recording timestamp.

    The apparent force is world gravity minus recorded torso linear acceleration,
    transformed into the eye housing frame. This visual model does not feed back
    into MuJoCo dynamics.
    """
    eyes = _eyes(model)
    if eyes is None:
        return None
    states, times = np.asarray(states), np.asarray(times, dtype=float)
    if len(times) == 0 or states.shape[0] != len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("Googly tracks require ordered, nonempty recorded states")
    data = mujoco.MjData(model)
    rotations = np.empty((len(times), 3, 3))
    velocities = np.empty((len(times), 3))
    torso = int(model.site_bodyid[eyes[0].eye])
    for index, state in enumerate(states):
        mujoco.mj_setState(model, data, state, STATE)
        mujoco.mj_forward(model, data)
        rotations[index] = data.xmat[torso].reshape(3, 3)
        velocities[index] = data.qvel[:3]
    accelerations = np.zeros_like(velocities)
    if len(times) > 1:
        accelerations[0] = (velocities[1] - velocities[0]) / (times[1] - times[0])
        accelerations[-1] = (velocities[-1] - velocities[-2]) / (times[-1] - times[-2])
    if len(times) > 2:
        accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (
            times[2:] - times[:-2]
        )[:, None]
    effective = np.einsum(
        "nji,nj->ni", rotations, np.asarray(model.opt.gravity)[None, :] - accelerations
    )
    track = np.empty((len(times), 2, 3))
    for eye_index, eye in enumerate(eyes):
        force = np.column_stack((effective @ eye.tangent_a, effective @ eye.tangent_b))
        magnitude = np.linalg.norm(force, axis=1)
        targets = np.zeros_like(force)
        valid = magnitude > 1e-12
        targets[valid] = force[valid] / magnitude[valid, None] * MAX_DISK
        position, velocity = targets[0].copy(), np.zeros(2)
        for index in range(len(times)):
            if index:
                dt = float(times[index] - times[index - 1])
                acceleration = NATURAL_FREQUENCY**2 * (targets[index] - position)
                acceleration -= DAMPING * velocity
                velocity += acceleration * dt
                position += velocity * dt
                position, velocity = _limit(position, velocity)
            direction = eye.outward + position[0] * eye.tangent_a + position[1] * eye.tangent_b
            direction /= np.linalg.norm(direction)
            track[index, eye_index] = model.site_pos[eye.eye] + eye.radius * direction
    return track


def apply_googly_frame(model, track, index):
    """Apply one decorative pupil frame; change no simulation state."""
    if track is None:
        return False
    eyes = _eyes(model)
    if eyes is None:
        return False
    for eye_index, eye in enumerate(eyes):
        model.site_pos[eye.pupil] = track[index, eye_index]
    return True


def apply_life_lights(model, recorded_time):
    """Apply deterministic ruby/ember breathing to optional decorative assets."""
    if not _marked(model):
        return False
    time_s = float(recorded_time)
    breath = 0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * time_s - np.pi / 2)
    beat_phase = time_s % 1.0
    heartbeat = np.exp(-((beat_phase - 0.08) / 0.035) ** 2)
    heartbeat += 0.62 * np.exp(-((beat_phase - 0.23) / 0.045) ** 2)
    brightness = float(np.clip(0.20 + 0.48 * breath + 0.25 * heartbeat, 0.0, 1.0))
    material = _id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "candidate_life_glow")
    if material >= 0:
        model.mat_emission[material] = 0.35 + 1.65 * brightness
        model.mat_rgba[material] = (0.50 + 0.45 * brightness, 0.0075, 0.0125, 1.0)
    for site in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site) or ""
        if "accent" in name:
            model.site_rgba[site] = (0.95, 0.015, 0.025, brightness)
    light = _id(model, mujoco.mjtObj.mjOBJ_LIGHT, "candidate_underlight")
    if light >= 0:
        model.light_diffuse[light] = brightness * np.array((0.10, 0.0015, 0.0025))
    return True
