"""Run the canonical C-1N simulation with optional viewer and socket control."""

from __future__ import annotations

import argparse
import json
import math
import queue
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from simulation import FOOT_NAMES, JOINTS_PER_LEG, MODEL_PATH, ROOT, load_model, measured_state, reset, step
from standing import SupportAwareStanceController
from walk import GaitCoordinator, apply_gait_control


HOST = "127.0.0.1"
PORT = 5555
JOINT_NAMES = ("coxa", "hip", "knee")
SHOVE_DURATION_S = 0.2
SHOVE_ANGLES_DEG = tuple(float(angle) for angle in range(0, 360, 45))
SHOVE_MULTIPLES = (0.0, 0.25, 0.5, 0.75, 1.0)
TELEMETRY_SAMPLE_INTERVAL_S = 0.02
TELEMETRY_HISTORY_SECONDS = 6.0


@dataclass
class Request:
    command: dict
    done: threading.Event
    response: dict | None = None


@dataclass(frozen=True)
class StandTelemetrySample:
    """Compact STAND evidence used by both the viewer and notebook."""

    time_s: float
    force_along_shove_n: float
    torso_displacement_along_shove_m: float
    torso_height_m: float
    torso_angular_speed_radps: float
    support_margin_m: float
    declared_contact_count: int
    front_pair_load_n: float
    middle_pair_load_n: float
    rear_pair_load_n: float


class StandTelemetryRecorder:
    """Sample the declared STAND evidence at a fixed rate, independent of display."""

    def __init__(self, direction: tuple[float, float, float], origin_xy: tuple[float, float]) -> None:
        self.direction = np.asarray(direction[:2])
        self.origin_xy = np.asarray(origin_xy)
        self.next_sample_time_s = 0.0
        self.samples: list[StandTelemetrySample] = []

    def sample_if_due(self, model: mujoco.MjModel, data: mujoco.MjData, perturbation: "TorsoForcePulse") -> None:
        if data.time + 1e-12 < self.next_sample_time_s:
            return
        observed = measured_state(model, data)
        torso_xy = np.asarray(observed.torso_position[:2])
        loads = observed.foot_normal_loads
        force_xy = np.asarray(perturbation.force[:2]) if perturbation.remaining_steps else np.zeros(2)
        self.samples.append(StandTelemetrySample(
            time_s=float(data.time),
            force_along_shove_n=float(force_xy @ self.direction),
            torso_displacement_along_shove_m=float((torso_xy - self.origin_xy) @ self.direction),
            torso_height_m=observed.torso_position[2],
            torso_angular_speed_radps=float(np.linalg.norm(observed.torso_angular_velocity)),
            support_margin_m=np.nan if observed.support_margin is None else observed.support_margin,
            declared_contact_count=len(observed.foot_contacts),
            front_pair_load_n=loads["front_left"] + loads["front_right"],
            middle_pair_load_n=loads["middle_left"] + loads["middle_right"],
            rear_pair_load_n=loads["rear_left"] + loads["rear_right"],
        ))
        self.next_sample_time_s += TELEMETRY_SAMPLE_INTERVAL_S


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
    for _ in range(count):
        perturbation.apply_before_step(data)
        step(model, data, targets_for_step(experiment, coordinator, controller, model, data))
        if after_step is not None:
            after_step()
        perturbation.complete_step(data)


def execute(command: dict, model: mujoco.MjModel, data: mujoco.MjData, power: StancePower, experiment: str, coordinator: GaitCoordinator | None, controller: SupportAwareStanceController | None = None, perturbation: TorsoForcePulse | None = None) -> dict:
    if perturbation is None:
        perturbation = TorsoForcePulse(model)
    name = command.get("command")
    if name == "state":
        return state(model, data, power, experiment, controller, perturbation)
    if name == "reset":
        reset(model, data)
        perturbation.force = (0.0, 0.0, 0.0)
        perturbation.remaining_steps = 0
        if coordinator is not None:
            coordinator.phase = 0.0
            coordinator.last_time = data.time
        return state(model, data, power, experiment, controller, perturbation)
    if name == "step":
        count = command.get("n", 1)
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("step count must be a positive integer")
        advance(model, data, count, experiment, coordinator, controller, perturbation)
        return state(model, data, power, experiment, controller, perturbation)
    if name == "run":
        seconds = command.get("seconds")
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds <= 0:
            raise ValueError("run duration must be greater than zero")
        advance(model, data, round(seconds / model.opt.timestep), experiment, coordinator, controller, perturbation)
        return state(model, data, power, experiment, controller, perturbation)
    if name == "power":
        values = command.get("values")
        if not isinstance(values, list) or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
            raise ValueError("power requires three numeric values")
        power.set(values)
        return state(model, data, power, experiment, controller, perturbation)
    if name == "perturb":
        force = command.get("force_n")
        seconds = command.get("seconds")
        if not isinstance(force, list) or len(force) != 3 or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in force):
            raise ValueError("perturb force_n requires three numeric world-frame components")
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds <= 0:
            raise ValueError("perturb seconds must be greater than zero")
        perturbation.schedule(force, float(seconds), model.opt.timestep)
        return state(model, data, power, experiment, controller, perturbation)
    raise ValueError("unknown command; use state, step, run, reset, power, or perturb")


def listener(requests: queue.Queue[Request]) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen()
        while True:
            connection, _ = server.accept()
            with connection:
                try:
                    line = connection.makefile("rb").readline()
                    command = json.loads(line) if line else None
                    if not isinstance(command, dict):
                        raise ValueError("command must be a JSON object")
                    request = Request(command, threading.Event())
                    requests.put(request)
                    if not request.done.wait(timeout=60):
                        raise TimeoutError("simulation did not respond within 60 seconds")
                    response = request.response or {"error": "no response"}
                except (json.JSONDecodeError, ValueError, TimeoutError) as error:
                    response = {"error": str(error)}
                connection.sendall((json.dumps(response) + "\n").encode())


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


def write_rollout_trace(path: Path, model: mujoco.MjModel, experiment: str, recorder: StandTelemetryRecorder, metadata: dict | None = None) -> None:
    """Persist compact sampled evidence for the viewer and notebook."""
    if not recorder.samples:
        raise ValueError("trace requires at least one telemetry sample")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {field: np.asarray([getattr(sample, field) for sample in recorder.samples]) for field in StandTelemetrySample.__dataclass_fields__}
    payload["metadata_json"] = np.asarray(json.dumps({
        "schema_version": 2,
        "telemetry_standard": "Telemetry v1",
        "c1n_iteration": "v0.11",
        "experiment": experiment,
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "physics_timestep_s": model.opt.timestep,
        "sample_interval_s": TELEMETRY_SAMPLE_INTERVAL_S,
        **(metadata or {}),
    }))
    np.savez_compressed(path, **payload)


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


def _make_stand_figures(label: str) -> tuple[mujoco.MjvFigure, mujoco.MjvFigure, mujoco.MjvFigure]:
    def figure(title: str, lines: tuple[str, ...]) -> mujoco.MjvFigure:
        result = mujoco.MjvFigure()
        result.title = title
        result.flg_extend = 0
        result.flg_legend = 1
        result.linewidth = 2.0
        for index, line in enumerate(lines):
            result.linename[index] = line
        return result
    return (
        figure(f"{label}: shove and motion", ("force along shove (N)", "displacement along shove (m)")),
        figure("Support state", ("support margin (m)", "declared contacts")),
        figure("Normal load by leg pair", ("front", "middle", "rear")),
    )


def _update_figure(figure: mujoco.MjvFigure, samples: deque[StandTelemetrySample], fields: tuple[str, ...]) -> None:
    if not samples:
        return
    figure.linepnt[:] = 0
    for index, field in enumerate(fields):
        values = np.asarray([getattr(sample, field) for sample in samples])
        figure.linepnt[index] = len(samples)
        figure.linedata[index, : 2 * len(samples)] = np.column_stack(([sample.time_s for sample in samples], values)).reshape(-1)
    values = np.asarray([[getattr(sample, field) for field in fields] for sample in samples])
    finite = values[np.isfinite(values)]
    padding = max(float(finite.max() - finite.min()) * 0.12, 0.01) if finite.size else 0.01
    figure.range[0] = (samples[0].time_s, max(samples[-1].time_s, samples[0].time_s + TELEMETRY_SAMPLE_INTERVAL_S))
    figure.range[1] = ((float(finite.min()) - padding, float(finite.max()) + padding) if finite.size else (-0.01, 0.01))


def _update_stand_figures(viewer: mujoco.viewer.Handle, figures: tuple[mujoco.MjvFigure, mujoco.MjvFigure, mujoco.MjvFigure], samples: deque[StandTelemetrySample]) -> None:
    fields = (
        ("force_along_shove_n", "torso_displacement_along_shove_m"),
        ("support_margin_m", "declared_contact_count"),
        ("front_pair_load_n", "middle_pair_load_n", "rear_pair_load_n"),
    )
    for figure, figure_fields in zip(figures, fields):
        _update_figure(figure, samples, figure_fields)
    viewport = viewer.viewport
    width = max(220, min(420, viewport.width // 3))
    height = max(120, min(240, (viewport.height - 36) // 3))
    viewer.set_figures([
        (mujoco.MjrRect(viewport.left + 10, viewport.bottom + viewport.height - 12 - height * (index + 1), width, height), figure)
        for index, figure in enumerate(figures)
    ])


def run_shove_suite(seconds: float, trace_directory: Path) -> list[dict]:
    """Run the declared 135-degree, 200 ms STAND shove cases and save each trace."""
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


def run_shove_suite_viewer(seconds: float, trace_directory: Path) -> None:
    """Show and record each shove case in order. Close the viewer to stop early."""
    import mujoco.viewer

    model, data, power, coordinator, controller, perturbation = build_simulation("stand")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for label, force, metadata in shove_cases(model):
            reset(model, data)
            perturbation.schedule(list(force), SHOVE_DURATION_S, model.opt.timestep)
            initial_state = state(model, data, power, "stand", controller, perturbation)
            recorder = StandTelemetryRecorder(tuple(metadata["force_direction_unit_vector"]), tuple(initial_state["torso_position"][:2]))
            recorder.sample_if_due(model, data, perturbation)
            displayed_samples: deque[StandTelemetrySample] = deque(maxlen=round(TELEMETRY_HISTORY_SECONDS / TELEMETRY_SAMPLE_INTERVAL_S))
            figures = _make_stand_figures(f"{metadata['direction_label']} {label}")
            print(f"Showing {metadata['direction_label']} {label}: {metadata['force_n']} N for {SHOVE_DURATION_S:.3f} s")
            remaining_steps = round(seconds / model.opt.timestep)
            steps_per_frame = max(1, round((1.0 / 60.0) / model.opt.timestep))
            while remaining_steps:
                if not viewer.is_running():
                    return
                started = time.perf_counter()
                advance(
                    model,
                    data,
                    min(steps_per_frame, remaining_steps),
                    "stand",
                    coordinator,
                    controller,
                    perturbation,
                    after_step=lambda: recorder.sample_if_due(model, data, perturbation),
                )
                remaining_steps -= min(steps_per_frame, remaining_steps)
                if recorder.samples and (not displayed_samples or displayed_samples[-1] is not recorder.samples[-1]):
                    displayed_samples.append(recorder.samples[-1])
                _update_stand_figures(viewer, figures, displayed_samples)
                viewer.sync()
                wait_s = (1.0 / 60.0) - (time.perf_counter() - started)
                if wait_s > 0:
                    time.sleep(wait_s)
            write_rollout_trace(trace_directory / metadata["direction_label"] / f"{label}.npz", model, "stand", recorder, metadata)


def run_viewer(experiment: str) -> None:
    import mujoco.viewer

    model, data, power, coordinator, controller, perturbation = build_simulation(experiment)
    requests: queue.Queue[Request] = queue.Queue()
    threading.Thread(target=listener, args=(requests,), daemon=True).start()
    print(f"Listening on {HOST}:{PORT}; experiment={experiment}. Close the viewer to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            started = time.perf_counter()
            try:
                request = requests.get_nowait()
            except queue.Empty:
                advance(model, data, 1, experiment, coordinator, controller, perturbation)
            else:
                try:
                    request.response = execute(request.command, model, data, power, experiment, coordinator, controller, perturbation)
                except (ValueError, RuntimeError) as error:
                    request.response = {"error": str(error)}
                finally:
                    request.done.set()
            viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    parser.add_argument("--seconds", type=float, default=1.0, help="headless duration in seconds")
    parser.add_argument("--experiment", choices=("none", "stand", "shuffle"), default="stand", help="explicit target generator")
    parser.add_argument("--trace", type=Path, help="write compact 50 Hz Telemetry v1 samples to this .npz file")
    parser.add_argument("--shove-suite", type=Path, metavar="DIRECTORY", help="run 0, 0.25, 0.5, and 1 mg shoves in eight world-frame directions and save traces here")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be greater than zero")
    if args.headless:
        if args.shove_suite is not None:
            print(json.dumps(run_shove_suite(args.seconds, args.shove_suite), indent=2))
        else:
            print(json.dumps(run_headless(args.seconds, args.experiment, args.trace), indent=2))
    elif args.shove_suite is not None:
        run_shove_suite_viewer(args.seconds, args.shove_suite)
    else:
        run_viewer(args.experiment)


if __name__ == "__main__":
    main()
