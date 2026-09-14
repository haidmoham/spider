"""Canonical rolling MuJoCo telemetry stacks for test-bench viewers."""

from collections import deque

import mujoco
import numpy as np


PLOT_INTERVAL = 0.02
PLOT_HISTORY_SECONDS = 6.0

# C-1N's readable simulation palette. Experiments may choose different signal
# meanings, but use these named defaults before inventing a new visual system.
PAPER = np.array([0.91, 0.925, 0.945])
INK = np.array([0.01, 0.01, 0.015])
SIGNAL_BLUE = np.array([0.184, 0.412, 0.678])
MUTED_SLATE = np.array([0.302, 0.408, 0.518])
WARM_RED = np.array([0.71, 0.263, 0.247])
WARM_ORANGE = np.array([0.95, 0.55, 0.10])
SIGNAL_GREEN = np.array([0.25, 0.80, 0.40])
OVERLAY_ALPHA = 0.72


class TelemetryPager:
    """Select one bounded telemetry page with a passive-viewer key callback."""

    def __init__(self, page_count, key="T"):
        if page_count <= 0:
            raise ValueError("page count must be positive")
        self.page_count = page_count
        self.key = key.upper()
        self.page = 0

    def handle_key(self, keycode):
        if keycode in (ord(self.key), ord(self.key.lower())):
            self.page = (self.page + 1) % self.page_count


def make_figure(title, series):
    """Create one graph using C-1N's readable viewer styling."""
    figure = mujoco.MjvFigure()
    figure.title = title
    figure.xlabel = ""
    figure.flg_extend = 0
    figure.flg_legend = 1
    figure.flg_ticklabel[:] = 1
    figure.linewidth = 2.0
    figure.figurergba = np.array([0.05, 0.05, 0.05, 0.32])
    figure.panergba = np.array([0.12, 0.12, 0.12, 0.48])
    figure.gridrgb = MUTED_SLATE
    for index, (name, color) in enumerate(series):
        figure.linename[index] = name
        figure.linergb[index] = np.array(color)
    return figure


def rolling_samples(sample_interval=PLOT_INTERVAL):
    """Return a bounded history buffer for the requested viewer sample rate."""
    if sample_interval <= 0.0:
        raise ValueError("sample interval must be positive")
    return deque(maxlen=round(PLOT_HISTORY_SECONDS / sample_interval))


def add_ghost_model_geoms(scene, model, data, rgba, label="reference"):
    """Render a second MuJoCo state as translucent viewer-only geometry."""
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == 0 or scene.ngeom >= len(scene.geoms):
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            model.geom_type[geom_id],
            model.geom_size[geom_id],
            data.geom_xpos[geom_id],
            data.geom_xmat[geom_id],
            rgba,
        )
        geom.label = label
        scene.ngeom += 1


def update_figure(figure, samples, value_index, value_range=None):
    """Draw one vector-valued telemetry field from rolling samples."""
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
    if value_range is None:
        finite_values = values[np.isfinite(values)]
        if not finite_values.size:
            figure.range[1] = (-0.01, 0.01)
            return
        lower = float(np.min(finite_values))
        upper = float(np.max(finite_values))
        padding = max((upper - lower) * 0.12, 0.01)
        figure.range[1] = (lower - padding, upper + padding)
    else:
        figure.range[1] = value_range


def update_stacks(
    viewer,
    left_figures,
    right_figures,
    samples,
    left_fields,
    right_fields,
    left_ranges=None,
    right_ranges=None,
):
    """Place up to two three-panel stacks in the viewer, following C-1N's layout."""
    left_ranges = left_ranges or (None,) * len(left_figures)
    right_ranges = right_ranges or (None,) * len(right_figures)
    for figure, field, value_range in zip(left_figures, left_fields, left_ranges):
        update_figure(figure, samples, field, value_range)
    for figure, field, value_range in zip(right_figures, right_fields, right_ranges):
        update_figure(figure, samples, field, value_range)

    viewport = viewer.viewport
    margin = max(6, min(12, viewport.width // 100))
    gap = max(4, min(8, viewport.height // 120))
    preferred_width = int(viewport.width * 0.32)
    non_overlapping_width = (viewport.width - 3 * margin) // 2
    width = max(180, min(480, preferred_width, non_overlapping_width))
    available_height = viewport.height - 2 * margin - 2 * gap
    height = max(100, min(260, available_height // 3))
    top = viewport.bottom + viewport.height

    def stack_viewports(left):
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
        list(zip(left_viewports, left_figures)) + list(zip(right_viewports, right_figures))
    )


def update_stack(viewer, figures, samples, fields, value_ranges=None, side="left"):
    """Attach one three-panel telemetry page to reduce persistent render cost."""
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    value_ranges = value_ranges or (None,) * len(figures)
    for figure, field, value_range in zip(figures, fields, value_ranges):
        update_figure(figure, samples, field, value_range)

    viewport = viewer.viewport
    margin = max(6, min(12, viewport.width // 100))
    gap = max(4, min(8, viewport.height // 120))
    width = max(180, min(480, int(viewport.width * 0.32)))
    available_height = viewport.height - 2 * margin - 2 * gap
    height = max(100, min(260, available_height // 3))
    left = (
        viewport.left + margin
        if side == "left"
        else viewport.left + viewport.width - width - margin
    )
    top = viewport.bottom + viewport.height
    viewports = [
        mujoco.MjrRect(
            left,
            top - margin - height * (index + 1) - gap * index,
            width,
            height,
        )
        for index in range(len(figures))
    ]
    viewer.set_figures(list(zip(viewports, figures)))
