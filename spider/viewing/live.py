"""Display existing rollouts and process queued live commands."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from ..runtime import HOST, PORT, Request, execute, listener
from ..runtime import SHOVE_DURATION_S, advance, build_simulation, shove_cases, state
from ..simulation import reset
from ..recording import (
    TELEMETRY_HISTORY_SECONDS,
    TELEMETRY_SAMPLE_INTERVAL_S,
    StandTelemetryRecorder,
    StandTelemetrySample,
    write_rollout_trace,
)
from .appearance import ResponsivePupils


def _make_stand_figures(label: str) -> tuple[mujoco.MjvFigure, mujoco.MjvFigure, mujoco.MjvFigure]:
    def figure(title: str, lines: tuple[str, ...]) -> mujoco.MjvFigure:
        result = mujoco.MjvFigure()
        result.title = title
        result.flg_extend = 0
        result.flg_legend = 1
        result.linewidth = 2.0
        result.figurergba[:] = (0.09, 0.09, 0.14, 0.96)
        result.panergba[:] = (0.12, 0.12, 0.18, 1.0)
        result.legendrgba[:] = (0.09, 0.09, 0.14, 0.94)
        result.textrgb[:] = (0.91, 0.88, 0.84)
        result.gridrgb[:] = (0.27, 0.26, 0.34)
        result.gridwidth = 1.0
        result.xlabel = "Time (s)"
        colors = ((0.96, 0.58, 0.43), (0.59, 0.78, 0.83), (0.76, 0.66, 0.91))
        for index, line in enumerate(lines):
            result.linename[index] = line
            result.linergb[index] = colors[index]
        return result

    return (
        figure(
            f"{label}: shove and motion", ("force along shove (N)", "displacement along shove (m)")
        ),
        figure("Support state", ("support margin (m)", "declared contacts")),
        figure("Normal load by leg pair", ("front", "middle", "rear")),
    )


def _update_figure(
    figure: mujoco.MjvFigure, samples: deque[StandTelemetrySample], fields: tuple[str, ...]
) -> None:
    if not samples:
        return
    figure.linepnt[:] = 0
    for index, field in enumerate(fields):
        values = np.asarray([getattr(sample, field) for sample in samples])
        figure.linepnt[index] = len(samples)
        figure.linedata[index, : 2 * len(samples)] = np.column_stack(
            ([sample.time_s for sample in samples], values)
        ).reshape(-1)
    values = np.asarray([[getattr(sample, field) for field in fields] for sample in samples])
    finite = values[np.isfinite(values)]
    padding = max(float(finite.max() - finite.min()) * 0.12, 0.01) if finite.size else 0.01
    figure.range[0] = (
        samples[0].time_s,
        max(samples[-1].time_s, samples[0].time_s + TELEMETRY_SAMPLE_INTERVAL_S),
    )
    figure.range[1] = (
        (float(finite.min()) - padding, float(finite.max()) + padding)
        if finite.size
        else (-0.01, 0.01)
    )


def _update_stand_figures(
    viewer: mujoco.viewer.Handle,
    figures: tuple[mujoco.MjvFigure, mujoco.MjvFigure, mujoco.MjvFigure],
    samples: deque[StandTelemetrySample],
) -> None:
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
    viewer.set_figures(
        [
            (
                mujoco.MjrRect(
                    viewport.left + 10,
                    viewport.bottom + viewport.height - 12 - height * (index + 1),
                    width,
                    height,
                ),
                figure,
            )
            for index, figure in enumerate(figures)
        ]
    )


def run_shove_suite_viewer(seconds: float, trace_directory: Path) -> None:
    """Show and record each shove case in order. Close the viewer to stop early."""
    import mujoco.viewer

    model, data, power, coordinator, controller, perturbation = build_simulation("stand")
    pupils = ResponsivePupils(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for label, force, metadata in shove_cases(model):
            reset(model, data)
            pupils.reset(model)
            perturbation.schedule(list(force), SHOVE_DURATION_S, model.opt.timestep)
            initial_state = state(model, data, power, "stand", controller, perturbation)
            recorder = StandTelemetryRecorder(
                tuple(metadata["force_direction_unit_vector"]),
                tuple(initial_state["torso_position"][:2]),
            )
            recorder.sample_if_due(model, data, perturbation)
            displayed_samples: deque[StandTelemetrySample] = deque(
                maxlen=round(TELEMETRY_HISTORY_SECONDS / TELEMETRY_SAMPLE_INTERVAL_S)
            )
            figures = _make_stand_figures(f"{metadata['direction_label']} {label}")
            print(
                f"Showing {metadata['direction_label']} {label}: {metadata['force_n']} N for {SHOVE_DURATION_S:.3f} s"
            )
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
                if recorder.samples and (
                    not displayed_samples or displayed_samples[-1] is not recorder.samples[-1]
                ):
                    displayed_samples.append(recorder.samples[-1])
                _update_stand_figures(viewer, figures, displayed_samples)
                pupils.update(model, data)
                viewer.sync()
                wait_s = (1.0 / 60.0) - (time.perf_counter() - started)
                if wait_s > 0:
                    time.sleep(wait_s)
            write_rollout_trace(
                trace_directory / metadata["direction_label"] / f"{label}.npz",
                model,
                "stand",
                recorder,
                metadata,
            )


def run_viewer(experiment: str) -> None:
    import mujoco.viewer

    model, data, power, coordinator, controller, perturbation = build_simulation(experiment)
    pupils = ResponsivePupils(model)
    requests: queue.Queue[Request] = queue.Queue()
    gait_plots = GaitPlots() if experiment == "shuffle" else None
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
                    request.response = execute(
                        request.command,
                        model,
                        data,
                        power,
                        experiment,
                        coordinator,
                        controller,
                        perturbation,
                    )
                except (ValueError, RuntimeError) as error:
                    request.response = {"error": str(error)}
                finally:
                    request.done.set()
            pupils.update(model, data)
            if gait_plots is not None:
                gait_plots.update(viewer, model, data, coordinator)
            viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)


REPORT_INTERVAL = 0.25
PLOT_INTERVAL = 0.02
PLOT_HISTORY_SECONDS = 6.0

PLOT_SERIES = (
    ("A hip", 0, (0.95, 0.25, 0.25)),
    ("A knee", 1, (1.00, 0.65, 0.20)),
    ("B hip", 2, (0.25, 0.55, 1.00)),
    ("B knee", 3, (0.45, 0.85, 1.00)),
)


class GaitPlots:
    """Keep the legacy joint/torque diagnostics separate from physics stepping."""

    def __init__(self):
        self.torque = tuple(
            make_figure(title)
            for title in (
                "Applied actuator torque (N-m)",
                "Torque rate (N-m/s)",
                "Torque acceleration (N-m/s^2)",
            )
        )
        self.position = tuple(
            make_figure(title)
            for title in (
                "Joint position (rad)",
                "Joint velocity (rad/s)",
                "Joint acceleration (rad/s^2)",
            )
        )
        self.samples = deque(maxlen=round(PLOT_HISTORY_SECONDS / PLOT_INTERVAL))
        self.previous_torque = None
        self.previous_rate = None
        self.last_time = None
        self.next_plot = 0.0
        self.next_report = 0.0

    def update(self, viewer, model, data, coordinator):
        if self.last_time is not None and data.time < self.last_time:
            self.__init__()
        if data.time == self.last_time:
            return
        elapsed = model.opt.timestep if self.last_time is None else data.time - self.last_time
        torque = data.qfrc_actuator[[7, 8, 10, 11]].copy()
        rate = (
            np.zeros_like(torque)
            if self.previous_torque is None
            else (torque - self.previous_torque) / elapsed
        )
        acceleration = (
            np.zeros_like(torque)
            if self.previous_rate is None
            else (rate - self.previous_rate) / elapsed
        )
        self.previous_torque, self.previous_rate = torque, rate
        self.last_time = float(data.time)
        if data.time >= self.next_plot:
            self.samples.append(
                (
                    data.time,
                    torque,
                    rate,
                    acceleration,
                    data.qpos[[8, 9, 11, 12]].copy(),
                    data.qvel[[7, 8, 10, 11]].copy(),
                    data.qacc[[7, 8, 10, 11]].copy(),
                )
            )
            self.next_plot = data.time + PLOT_INTERVAL
        if data.time >= self.next_report:
            roll, pitch, _, _ = coordinator.body_errors(data)
            print(
                f"t={data.time:5.2f} x={data.qpos[0]:+.3f} z={data.qpos[2]:.3f} "
                f"contacts={sum(coordinator.contacts(data))}/6 roll={roll:+.3f} pitch={pitch:+.3f}"
            )
            self.next_report = data.time + REPORT_INTERVAL
        update_figures(viewer, self.torque, self.position, self.samples)


def make_figure(title: str) -> mujoco.MjvFigure:
    figure = mujoco.MjvFigure()
    figure.title = title
    figure.xlabel = "simulation time (s)"
    figure.flg_extend = 0
    figure.flg_legend = 1
    figure.flg_ticklabel[:] = 1
    figure.linewidth = 2.0
    figure.figurergba = np.array([0.05, 0.05, 0.05, 0.32])
    figure.panergba = np.array([0.12, 0.12, 0.12, 0.48])
    figure.gridrgb = np.array([0.35, 0.35, 0.35])
    for index, (name, _, color) in enumerate(PLOT_SERIES):
        figure.linename[index] = name
        figure.linergb[index] = np.array(color)
    return figure


def update_figure(figure: mujoco.MjvFigure, samples: deque, value_index: int) -> None:
    if not samples:
        return
    times = np.array([sample[0] for sample in samples])
    values = np.array([sample[value_index] for sample in samples])
    figure.linepnt[:] = 0
    for index in range(values.shape[1]):
        figure.linepnt[index] = len(times)
        figure.linedata[index, : 2 * len(times)] = np.column_stack(
            (times, values[:, index])
        ).reshape(-1)

    figure.range[0] = (times[0], max(times[-1], times[0] + PLOT_INTERVAL))
    lower = float(np.min(values))
    upper = float(np.max(values))
    padding = max((upper - lower) * 0.12, 0.01)
    figure.range[1] = (lower - padding, upper + padding)


def update_figures(viewer, torque_figures: tuple, position_figures: tuple, samples: deque) -> None:
    for figure, value_index in zip(torque_figures, (1, 2, 3)):
        update_figure(figure, samples, value_index)
    for figure, value_index in zip(position_figures, (4, 5, 6)):
        update_figure(figure, samples, value_index)

    viewport = viewer.viewport
    width = min(370, max(250, viewport.width // 4))
    margin = 10
    gap = 6
    height = max(80, min(118, (viewport.height - 2 * margin - 2 * gap) // 3))
    top = viewport.bottom + viewport.height

    def stack_viewports(left: int) -> list:
        return [
            mujoco.MjrRect(
                left,
                top - margin - height * (index + 1) - gap * index,
                width,
                height,
            )
            for index in range(3)
        ]

    left_viewports = stack_viewports(viewport.left + margin)
    right_viewports = stack_viewports(viewport.left + viewport.width - width - margin)
    viewer.set_figures(
        list(zip(left_viewports, position_figures)) + list(zip(right_viewports, torque_figures))
    )
