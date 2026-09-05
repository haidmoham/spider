"""Local JSON commands and socket transport for the live simulation."""
from __future__ import annotations

import json
import queue
import socket
import threading
from dataclasses import dataclass

import mujoco

from runtime import StancePower, TorsoForcePulse, advance, state
from simulation import reset
from standing import SupportAwareStanceController
from walk import GaitCoordinator

HOST = "127.0.0.1"
PORT = 5555


@dataclass
class Request:
    command: dict
    done: threading.Event
    response: dict | None = None



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
