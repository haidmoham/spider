"""Run the canonical C-1N simulation with optional viewer and socket control."""

from __future__ import annotations

import argparse
import json
import queue
import socket
import threading
import time
from dataclasses import dataclass

import mujoco

from simulation import FOOT_NAMES, JOINTS_PER_LEG, load_model, measured_state, reset, step
from walk import GaitCoordinator, apply_gait_control


HOST = "127.0.0.1"
PORT = 5555
JOINT_NAMES = ("coxa", "hip", "knee")


@dataclass
class Request:
    command: dict
    done: threading.Event
    response: dict | None = None


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


def state(model: mujoco.MjModel, data: mujoco.MjData, power: StancePower, experiment: str) -> dict:
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
        "legs": legs,
        "pair_powers": power.values,
        "experiment": experiment,
    }


def targets_for_step(experiment: str, coordinator: GaitCoordinator | None, data: mujoco.MjData) -> tuple[float, ...] | None:
    if experiment == "shuffle":
        if coordinator is None:
            raise RuntimeError("shuffle experiment requires a coordinator")
        targets, _, _ = apply_gait_control(coordinator, data)
        return targets
    return None


def advance(model: mujoco.MjModel, data: mujoco.MjData, count: int, experiment: str, coordinator: GaitCoordinator | None) -> None:
    for _ in range(count):
        step(model, data, targets_for_step(experiment, coordinator, data))


def execute(command: dict, model: mujoco.MjModel, data: mujoco.MjData, power: StancePower, experiment: str, coordinator: GaitCoordinator | None) -> dict:
    name = command.get("command")
    if name == "state":
        return state(model, data, power, experiment)
    if name == "reset":
        reset(model, data)
        if coordinator is not None:
            coordinator.phase = 0.0
            coordinator.last_time = data.time
        return state(model, data, power, experiment)
    if name == "step":
        count = command.get("n", 1)
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("step count must be a positive integer")
        advance(model, data, count, experiment, coordinator)
        return state(model, data, power, experiment)
    if name == "run":
        seconds = command.get("seconds")
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds <= 0:
            raise ValueError("run duration must be greater than zero")
        advance(model, data, round(seconds / model.opt.timestep), experiment, coordinator)
        return state(model, data, power, experiment)
    if name == "power":
        values = command.get("values")
        if not isinstance(values, list) or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
            raise ValueError("power requires three numeric values")
        power.set(values)
        return state(model, data, power, experiment)
    raise ValueError("unknown command; use state, step, run, reset, or power")


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


def build_simulation(experiment: str) -> tuple[mujoco.MjModel, mujoco.MjData, StancePower, GaitCoordinator | None]:
    model = load_model()
    data = mujoco.MjData(model)
    reset(model, data)
    return model, data, StancePower(model), GaitCoordinator(model, data) if experiment == "shuffle" else None


def run_headless(seconds: float, experiment: str) -> dict:
    model, data, power, coordinator = build_simulation(experiment)
    advance(model, data, round(seconds / model.opt.timestep), experiment, coordinator)
    return state(model, data, power, experiment)


def run_viewer(experiment: str) -> None:
    import mujoco.viewer

    model, data, power, coordinator = build_simulation(experiment)
    requests: queue.Queue[Request] = queue.Queue()
    threading.Thread(target=listener, args=(requests,), daemon=True).start()
    print(f"Listening on {HOST}:{PORT}; experiment={experiment}. Close the viewer to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            started = time.perf_counter()
            try:
                request = requests.get_nowait()
            except queue.Empty:
                advance(model, data, 1, experiment, coordinator)
            else:
                try:
                    request.response = execute(request.command, model, data, power, experiment, coordinator)
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
    parser.add_argument("--experiment", choices=("none", "shuffle"), default="none", help="explicit optional target generator")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be greater than zero")
    if args.headless:
        print(json.dumps(run_headless(args.seconds, args.experiment), indent=2))
    else:
        run_viewer(args.experiment)


if __name__ == "__main__":
    main()
