"""Compose controllers, disturbances, and canonical physics steps.

Import this module from notebooks for existing controller rollouts. The core
in simulation.py remains the only owner of reset and physics stepping.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from simulation import FOOT_NAMES, JOINTS_PER_LEG, load_model, measured_state, reset, step
from standing import SupportAwareStanceController
from telemetry import StandTelemetryRecorder, write_rollout_trace
from walk import GaitCoordinator, apply_gait_control

JOINT_NAMES = ("coxa", "hip", "knee")
SHOVE_DURATION_S = 0.2
SHOVE_ANGLES_DEG = tuple(float(angle) for angle in range(0, 360, 45))
SHOVE_MULTIPLES = (0.0, 0.25, 0.5, 0.75, 1.0)


class StancePower:
    """Optional live actuator-gain adjustment; neutral targets remain unchanged."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.base_gainprm = model.actuator_gainprm.copy()
        self.base_biasprm = model.actuator_biasprm.copy()
        self.values = [1.0, 1.0, 1.0]

    def set(self, values: list[float]) -> None:
        if len(values) != 3 or any(value < 0 for value in values):
            raise ValueError("power requires three non-negative values")
        self.values = [float(value) for value in values]
        for pair, value in enumerate(self.values):
            actuator_slice = slice(
                pair * 2 * JOINTS_PER_LEG,
                (pair + 1) * 2 * JOINTS_PER_LEG,
            )
            self.model.actuator_gainprm[actuator_slice] = self.base_gainprm[actuator_slice] * value
            self.model.actuator_biasprm[actuator_slice] = self.base_biasprm[actuator_slice] * value



class TorsoForcePulse:
    """A scheduled world-frame force pulse for a visible disturbance test."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.torso_id = model.body("torso").id
        self.force = (0.0, 0.0, 0.0)
        self.remaining_steps = 0

    def schedule(self, force: list[float], seconds: float, timestep: float) -> None:
        self.force = tuple(float(value) for value in force)
        self.remaining_steps = max(1, round(seconds / timestep))

    def apply_before_step(self, data: mujoco.MjData) -> None:
        data.xfrc_applied[self.torso_id, :3] = self.force if self.remaining_steps else (0.0, 0.0, 0.0)

    def complete_step(self, data: mujoco.MjData) -> None:
        if self.remaining_steps:
            self.remaining_steps -= 1
        if not self.remaining_steps:
            data.xfrc_applied[self.torso_id, :3] = (0.0, 0.0, 0.0)



def state(model: mujoco.MjModel, data: mujoco.MjData, power: StancePower, experiment: str, controller: SupportAwareStanceController | None, perturbation: TorsoForcePulse) -> dict:
    observed = measured_state(model, data)
    legs = {
        name: {
            "foot_position": list(observed.foot_positions[name]),
            "in_ground_contact": name in observed.foot_contacts,
            "joints": {
                joint: {
                    "target": float(data.ctrl[index * JOINTS_PER_LEG + offset]),
                    "position": observed.joint_positions[index * JOINTS_PER_LEG + offset],
                    "velocity": observed.joint_velocities[index * JOINTS_PER_LEG + offset],
                    "actuator_force": observed.actuator_forces[index * JOINTS_PER_LEG + offset],
                }
                for offset, joint in enumerate(JOINT_NAMES)
            },
        }
        for index, name in enumerate(FOOT_NAMES)
    }
    return {
        "time": observed.time,
        "torso_position": list(observed.torso_position),
        "torso_orientation": list(observed.torso_orientation),
        "torso_velocity": list(observed.torso_velocity),
        "torso_angular_velocity": list(observed.torso_angular_velocity),
        "joint_positions": list(observed.joint_positions),
        "joint_velocities": list(observed.joint_velocities),
        "actuator_forces": list(observed.actuator_forces),
        "controls": data.ctrl.tolist(),
        "foot_positions": {name: list(position) for name, position in observed.foot_positions.items()},
        "foot_contacts": list(observed.foot_contacts),
        "support": {
            "com_projection": list(observed.com_projection),
            "support_polygon": [list(point) for point in observed.support_polygon],
            "support_margin_m": observed.support_margin,
            "foot_normal_loads_n": observed.foot_normal_loads,
        },
        "whole_stance_kinematics": {
            "task": "all six foot centres to their neutral world-frame positions",
            "residual_convention": "target_minus_current",
            "foot_position_residual": list(observed.foot_position_residual),
            "foot_position_residual_norm_m": observed.foot_position_residual_norm,
            "joint_space_update_direction": list(observed.joint_space_update_direction),
            "joint_space_update_direction_norm": observed.joint_space_update_direction_norm,
        },
        "legs": legs,
        "pair_powers": power.values,
        "experiment": experiment,
        "standing_controller": None if controller is None or controller.last_telemetry is None else {
            "enabled": controller.last_telemetry.enabled,
            "reason": controller.last_telemetry.reason,
            "manual_override": controller.last_telemetry.manual_override,
            "torso_height_error_m": controller.last_telemetry.torso_height_error,
            "torso_attitude_error": controller.last_telemetry.torso_attitude_error,
            "commanded_targets": list(controller.last_telemetry.commanded_targets),
        },
        "perturbation": {
            "frame": "world",
            "application": "torso centre of mass",
            "force_n": list(perturbation.force) if perturbation.remaining_steps else [0.0, 0.0, 0.0],
            "remaining_steps": perturbation.remaining_steps,
        },
    }



def targets_for_step(experiment: str, coordinator: GaitCoordinator | None, controller: SupportAwareStanceController | None, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, ...] | None:
    if experiment == "stand":
        if controller is None:
            raise RuntimeError("stand experiment requires a stance controller")
        return controller.targets(model, data)
    if experiment == "shuffle":
        if coordinator is None:
            raise RuntimeError("shuffle experiment requires a coordinator")
        targets, _, _ = apply_gait_control(coordinator, data)
        return targets
    return None



def advance(model: mujoco.MjModel, data: mujoco.MjData, count: int, experiment: str, coordinator: GaitCoordinator | None, controller: SupportAwareStanceController | None, perturbation: TorsoForcePulse, after_step: Callable[[], None] | None = None) -> None:
    """Apply force, choose targets, step, sample, then finish the force pulse.

    Keep this order for both headless runs and viewers. The callback observes
    the applied pulse before its remaining-step counter is decremented.
    """
    for _ in range(count):
        perturbation.apply_before_step(data)
        step(model, data, targets_for_step(experiment, coordinator, controller, model, data))
        if after_step is not None:
            after_step()
        perturbation.complete_step(data)



def build_simulation(experiment: str) -> tuple[mujoco.MjModel, mujoco.MjData, StancePower, GaitCoordinator | None, SupportAwareStanceController | None, TorsoForcePulse]:
    model = load_model()
    data = mujoco.MjData(model)
    reset(model, data)
    return (
        model,
        data,
        StancePower(model),
        GaitCoordinator(model, data) if experiment == "shuffle" else None,
        SupportAwareStanceController() if experiment == "stand" else None,
        TorsoForcePulse(model),
    )



def run_headless(seconds: float, experiment: str, trace_path: Path | None = None, shove_force_n: tuple[float, float, float] | None = None, shove_duration_s: float = SHOVE_DURATION_S, trace_metadata: dict | None = None) -> dict:
    model, data, power, coordinator, controller, perturbation = build_simulation(experiment)
    initial_state = state(model, data, power, experiment, controller, perturbation)
    shove_direction = (1.0, 0.0, 0.0)
    direction = shove_direction if shove_force_n is None or np.linalg.norm(shove_force_n) == 0.0 else tuple(np.asarray(shove_force_n) / np.linalg.norm(shove_force_n))
    recorder = StandTelemetryRecorder(direction, tuple(initial_state["torso_position"][:2]))
    recorder.sample_if_due(model, data, perturbation)
    if shove_force_n is not None:
        perturbation.schedule(list(shove_force_n), shove_duration_s, model.opt.timestep)
    advance(
        model,
        data,
        round(seconds / model.opt.timestep),
        experiment,
        coordinator,
        controller,
        perturbation,
        after_step=lambda: recorder.sample_if_due(model, data, perturbation),
    )
    final_state = state(model, data, power, experiment, controller, perturbation)
    if trace_path is not None:
        write_rollout_trace(trace_path, model, experiment, recorder, trace_metadata)
    return final_state



def shove_cases(model: mujoco.MjModel) -> list[tuple[str, tuple[float, float, float], dict]]:
    """Define the eight-direction force grid for visual and headless shove runs."""
    mass_kg = float(model.body_mass[1:].sum())
    gravity_m_per_s2 = abs(float(model.opt.gravity[2]))
    weight_n = mass_kg * gravity_m_per_s2
    cases = []
    for multiple in SHOVE_MULTIPLES:
        angles = (0.0,) if multiple == 0.0 else SHOVE_ANGLES_DEG
        for angle_deg in angles:
            angle_rad = math.radians(angle_deg)
            direction = (math.cos(angle_rad), math.sin(angle_rad), 0.0)
            direction_label = f"{angle_deg:03.0f}deg"
            force = tuple(multiple * weight_n * component for component in direction)
            label = f"{multiple:g}mg"
            cases.append((label, force, {
                "test": "stand_shove",
                "label": label,
                "direction_label": direction_label,
                "case_role": "control" if multiple == 0.0 else "treatment",
                "mass_kg": mass_kg,
                "gravity_m_per_s2": gravity_m_per_s2,
                "weight_n": weight_n,
                "force_multiple_of_mg": multiple,
                "force_angle_deg": angle_deg,
                "angle_reference": "world +X, counter-clockwise toward world +Y",
                "force_direction_unit_vector": list(direction),
                "force_n": list(force),
                "duration_s": SHOVE_DURATION_S,
            }))
    return cases



def run_shove_suite(seconds: float, trace_directory: Path) -> list[dict]:
    """Run the declared eight-direction, 200 ms STAND shove grid and save traces."""
    model = load_model()
    results = []
    for label, force, metadata in shove_cases(model):
        trace_path = trace_directory / metadata["direction_label"] / f"{label}.npz"
        final_state = run_headless(
            seconds,
            "stand",
            trace_path,
            shove_force_n=force,
            trace_metadata=metadata,
        )
        results.append({"label": label, "trace": str(trace_path), "final_time_s": final_state["time"]})
    return results
