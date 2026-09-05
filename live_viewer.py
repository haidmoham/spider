"""Display existing rollouts and process queued live commands."""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from commands import HOST, PORT, Request, execute, listener
from runtime import SHOVE_DURATION_S, advance, build_simulation, shove_cases, state
from simulation import reset
from telemetry import (
    TELEMETRY_HISTORY_SECONDS, TELEMETRY_SAMPLE_INTERVAL_S,
    StandTelemetryRecorder, StandTelemetrySample, write_rollout_trace,
)
from visuals import ResponsivePupils


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
                pupils.update(model, data)
                viewer.sync()
                wait_s = (1.0 / 60.0) - (time.perf_counter() - started)
                if wait_s > 0:
                    time.sleep(wait_s)
            write_rollout_trace(trace_directory / metadata["direction_label"] / f"{label}.npz", model, "stand", recorder, metadata)



def run_viewer(experiment: str) -> None:
    import mujoco.viewer

    model, data, power, coordinator, controller, perturbation = build_simulation(experiment)
    pupils = ResponsivePupils(model)
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
            pupils.update(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
