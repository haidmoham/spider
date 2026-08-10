"""Run C-1N v0.1 - SHUFFLE in a live local viewer."""

from __future__ import annotations

import json
import queue
import socket
import sys
import threading
import time
import ctypes
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from simulate import MODEL_PATH, set_standing_pose


HOST = "127.0.0.1"
PORT = 5555
BENCH_EXPERIMENTS = Path(__file__).resolve().parent.parent / "robotics-test-bench" / "experiments"
if not BENCH_EXPERIMENTS.is_dir():
    raise RuntimeError(f"Robotics test-bench telemetry was not found at {BENCH_EXPERIMENTS}")
sys.path.insert(0, str(BENCH_EXPERIMENTS))
from fbd_overlay import draw_force_diagram
from telemetry import PLOT_INTERVAL, make_figure, rolling_samples, update_figure
from viewer_controls import ManualOverride

FOOT_NAMES = (
    "front_left",
    "front_right",
    "middle_left",
    "middle_right",
    "rear_left",
    "rear_right",
)
EUCLIDEAN_SERIES = (("torso", (0.25, 0.80, 0.40)),)
WORLD_AXIS_COLOR = np.array((0.48, 0.52, 0.56))
TORSO_AXIS_COLOR = np.array((0.96, 0.96, 0.96))
SQUAT_TORSO_HEIGHT = 0.245
SQUAT_KNEE_TARGET = -1.0
GAIT_FREQUENCY = 5.0
HIP_SWEEP = 0.22
LATERAL_SWEEP = 0.06
SWING_KNEE_LIFT = 0.15
TRIPOD_A = {0, 3, 4}


def apply_open_loop_gait(data: mujoco.MjData) -> None:
    """Apply an open-loop diagonal tripod scuttle from simulation time only."""
    phase = (data.time * GAIT_FREQUENCY) % 1.0
    for leg in range(6):
        cycle = (phase + (0.0 if leg in TRIPOD_A else 0.5)) % 1.0
        hip_wave = -HIP_SWEEP * np.cos(2.0 * np.pi * cycle)
        swing_progress = max(0.0, 2.0 * cycle - 1.0)
        knee = SQUAT_KNEE_TARGET - SWING_KNEE_LIFT * np.sin(
            np.pi * swing_progress
        ) ** 2
        fore_aft = 1.0 if leg < 2 else -1.0 if leg >= 4 else 0.0
        side = 1.0 if leg % 2 == 0 else -1.0
        data.ctrl[2 * leg] = -fore_aft * hip_wave + side * LATERAL_SWEEP * hip_wave
        data.ctrl[2 * leg + 1] = knee


class StancePower:
    """Scale the existing position-actuator stiffness by symmetric leg pair."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.base_gainprm = model.actuator_gainprm.copy()
        self.base_biasprm = model.actuator_biasprm.copy()
        self.values = np.ones(3)

    def set(self, values: list[float]) -> None:
        if len(values) != 3 or any(value < 0 for value in values):
            raise ValueError("power requires three non-negative values")
        self.values = np.array(values, dtype=float)
        for pair, value in enumerate(self.values):
            actuator_slice = slice(pair * 4, pair * 4 + 4)
            self.model.actuator_gainprm[actuator_slice] = (
                self.base_gainprm[actuator_slice] * value
            )
            self.model.actuator_biasprm[actuator_slice] = (
                self.base_biasprm[actuator_slice] * value
            )


def initialize_stance(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Place the free body in a lower symmetric squat before the open-loop gait."""
    set_standing_pose(model, data)
    data.qpos[2] = SQUAT_TORSO_HEIGHT
    data.qpos[3:7] = (0.0, 0.0, 0.0, 1.0)
    data.qpos[7:] = np.tile((0.0, SQUAT_KNEE_TARGET), 6)
    apply_open_loop_gait(data)
    mujoco.mj_forward(model, data)


@dataclass
class Request:
    command: dict
    done: threading.Event
    response: dict | None = None


class OrientationGizmo:
    """Draw camera-relative world and torso coordinate triads in one figure."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.torso_id = model.body("torso").id
        self.figure = mujoco.MjvFigure()
        self.figure.title = "Orientation compass"
        self.figure.flg_extend = 0
        self.figure.flg_legend = 1
        self.figure.flg_ticklabel[:] = 0
        self.figure.linewidth = 3.0
        self.figure.figurergba = np.array((0.05, 0.05, 0.05, 0.32))
        self.figure.panergba = np.array((0.12, 0.12, 0.12, 0.48))
        self.figure.gridrgb = np.array((0.30, 0.40, 0.52))
        self.figure.range[0] = (-1.0, 1.0)
        self.figure.range[1] = (-1.0, 1.0)
        for index in range(3):
            self.figure.linergb[index] = WORLD_AXIS_COLOR
            self.figure.linergb[index + 3] = TORSO_AXIS_COLOR
            axis_name = "XYZ"[index]
            self.figure.linename[index] = f"world {axis_name}"
            self.figure.linename[index + 3] = f"torso {axis_name}"

    def _project(self, viewer, vector: np.ndarray) -> np.ndarray:
        azimuth = np.deg2rad(viewer.cam.azimuth)
        elevation = np.deg2rad(viewer.cam.elevation)
        screen_right = np.array((np.cos(azimuth), np.sin(azimuth), 0.0))
        screen_up = np.array(
            (-np.sin(elevation) * np.sin(azimuth),
             np.sin(elevation) * np.cos(azimuth), np.cos(elevation))
        )
        return np.array((vector @ screen_right, vector @ screen_up))

    def update(self, viewer, data: mujoco.MjData) -> tuple:
        self.figure.linepnt[:] = 0
        body_rotation = data.xmat[self.torso_id].reshape(3, 3)
        for axis in range(3):
            world_endpoint = self._project(viewer, np.eye(3)[axis]) * 0.72
            torso_endpoint = self._project(viewer, body_rotation @ np.eye(3)[axis])
            for line, endpoint in ((axis, world_endpoint), (axis + 3, torso_endpoint)):
                self.figure.linepnt[line] = 2
                self.figure.linedata[line, :4] = (0.0, 0.0, *endpoint)

        viewport = viewer.viewport
        size = max(145, min(185, viewport.width // 7))
        return (
            mujoco.MjrRect(
                viewport.left + viewport.width - size - 10,
                viewport.bottom + viewport.height - size - 10,
                size,
                size,
            ),
            self.figure,
        )


class LiveTelemetry:
    """Collect the test-bench rolling graph signals for the active MjData."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.figures = (
            make_figure("Torso position |p| (m)", EUCLIDEAN_SERIES),
            make_figure("Torso speed |v| (m/s)", EUCLIDEAN_SERIES),
        )
        self.samples = rolling_samples()
        self.next_sample = 0.0
        self.orientation = OrientationGizmo(model)

    def reset(self, time: float) -> None:
        self.samples.clear()
        self.next_sample = time

    def record(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if data.time >= self.next_sample:
            self.samples.append(
                (data.time, np.array([np.linalg.norm(data.qpos[:3])]),
                 np.array([np.linalg.norm(data.qvel[:3])]))
            )
            self.next_sample += PLOT_INTERVAL

    def update(self, viewer, data: mujoco.MjData) -> None:
        for figure, field in zip(self.figures, (1, 2)):
            update_figure(figure, self.samples, field)
        viewport = viewer.viewport
        margin = 8
        gap = 5
        width = max(160, min(240, viewport.width // 5))
        height = max(70, min(105, viewport.height // 7))
        viewer.set_figures([
            (mujoco.MjrRect(viewport.left + margin, viewport.bottom + margin + height + gap, width, height), self.figures[0]),
            (mujoco.MjrRect(viewport.left + margin, viewport.bottom + margin, width, height), self.figures[1]),
            self.orientation.update(viewer, data),
        ])


def maximize_viewer() -> None:
    """Maximize this process's MuJoCo window on Windows."""
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    process_id = ctypes.windll.kernel32.GetCurrentProcessId()
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def maximize(window, _):
        window_process_id = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(window, ctypes.byref(window_process_id))
        title_length = user32.GetWindowTextLengthW(window)
        title = ctypes.create_unicode_buffer(title_length + 1)
        user32.GetWindowTextW(window, title, title_length + 1)
        if window_process_id.value == process_id and title.value.startswith("MuJoCo"):
            user32.ShowWindow(window, 3)  # SW_MAXIMIZE
        return True

    user32.EnumWindows(callback_type(maximize), 0)


def send_json(connection: socket.socket, response: dict) -> None:
    connection.sendall((json.dumps(response) + "\n").encode())


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
                    send_json(connection, request.response or {"error": "no response"})
                except (json.JSONDecodeError, ValueError, TimeoutError) as error:
                    send_json(connection, {"error": str(error)})


def state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    power: StancePower,
    manual_override: ManualOverride,
) -> dict:
    ground_id = model.geom("ground").id
    foot_ids = {model.geom(f"{name}_foot").id: name for name in FOOT_NAMES}
    contacts = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        foot_id = contact.geom2 if contact.geom1 == ground_id else contact.geom1 if contact.geom2 == ground_id else None
        if foot_id in foot_ids:
            contacts.append(foot_ids[foot_id])
    return {
        "time": float(data.time),
        "torso_position": data.qpos[:3].tolist(),
        "torso_velocity": data.qvel[:3].tolist(),
        "controls": data.ctrl.tolist(),
        "foot_contacts": sorted(set(contacts)),
        "pair_powers": power.values.tolist(),
        "manual_override": manual_override.enabled,
    }


def update_fbd_overlay(model: mujoco.MjModel, data: mujoco.MjData, viewer) -> None:
    """Render stable reference-force arrows without changing simulation state."""
    if viewer.user_scn is None:
        return
    support_points = [
        data.geom_xpos[model.geom(f"{name}_foot").id].copy()
        for name in FOOT_NAMES
    ]
    weight = -model.opt.gravity[2] * model.body_mass.sum()
    draw_force_diagram(
        viewer.user_scn,
        data.xpos[model.body("torso").id].copy(),
        np.array([0.0, 0.0, -weight]),
        support_points,
        np.array([0.0, 0.0, weight / len(support_points)]),
        force_scale=0.0024,
    )


def execute(
    command: dict,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    telemetry: LiveTelemetry,
    power: StancePower,
    manual_override: ManualOverride,
) -> dict:
    name = command.get("command")
    if name == "state":
        return state(model, data, power, manual_override)
    if name == "reset":
        mujoco.mj_resetData(model, data)
        initialize_stance(model, data)
        telemetry.reset(data.time)
        telemetry.record(model, data)
        telemetry.update(viewer, data)
        update_fbd_overlay(model, data, viewer)
        viewer.sync()
        return state(model, data, power, manual_override)
    if name == "step":
        count = command.get("n", 1)
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("step count must be a positive integer")
        for _ in range(count):
            if not manual_override.enabled:
                apply_open_loop_gait(data)
            mujoco.mj_step(model, data)
            telemetry.record(model, data)
        telemetry.update(viewer, data)
        update_fbd_overlay(model, data, viewer)
        viewer.sync()
        return state(model, data, power, manual_override)
    if name == "run":
        seconds = command.get("seconds")
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds <= 0:
            raise ValueError("run duration must be greater than zero")
        end_time = data.time + seconds
        while data.time < end_time:
            if not manual_override.enabled:
                apply_open_loop_gait(data)
            mujoco.mj_step(model, data)
            telemetry.record(model, data)
            if int(data.time / model.opt.timestep) % 20 == 0:
                telemetry.update(viewer, data)
                update_fbd_overlay(model, data, viewer)
                viewer.sync()
        telemetry.update(viewer, data)
        update_fbd_overlay(model, data, viewer)
        viewer.sync()
        return state(model, data, power, manual_override)
    if name == "power":
        values = command.get("values")
        if not isinstance(values, list) or any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in values
        ):
            raise ValueError("power requires three numeric values")
        power.set(values)
        return state(model, data, power, manual_override)
    raise ValueError("unknown command; use state, step, run, reset, or power")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    initialize_stance(model, data)
    power = StancePower(model)
    telemetry = LiveTelemetry(model)
    manual_override = ManualOverride()
    requests: queue.Queue[Request] = queue.Queue()
    threading.Thread(target=listener, args=(requests,), daemon=True).start()
    print(f"Listening on {HOST}:{PORT}. Close the viewer to stop.")

    with mujoco.viewer.launch_passive(
        model, data, key_callback=manual_override.handle_key
    ) as viewer:
        maximize_viewer()
        while viewer.is_running():
            wall_start = time.perf_counter()
            try:
                request = requests.get_nowait()
            except queue.Empty:
                # This gait uses only simulation time. It has no body, contact,
                # or actuator feedback correction.
                if not manual_override.enabled:
                    apply_open_loop_gait(data)
                mujoco.mj_step(model, data)
                telemetry.record(model, data)
                telemetry.update(viewer, data)
                update_fbd_overlay(model, data, viewer)
                viewer.set_texts((
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    manual_override.status_text(),
                    "",
                ))
                viewer.sync()
            else:
                try:
                    request.response = execute(
                        request.command,
                        model,
                        data,
                        viewer,
                        telemetry,
                        power,
                        manual_override,
                    )
                except ValueError as error:
                    request.response = {"error": str(error)}
                except Exception as error:
                    request.response = {"error": f"simulation error: {error}"}
                finally:
                    request.done.set()

            remaining = model.opt.timestep - (time.perf_counter() - wall_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
