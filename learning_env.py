"""Notebook boundary for user-written control and learning experiments.

This adapter owns physics state. The notebook owns observations, rewards,
episode boundaries, and the choice of action duration.
"""

from __future__ import annotations

from numbers import Integral

import mujoco
import numpy as np
from numpy.typing import ArrayLike

import simulation
from simulation import MeasuredState


class LearningSimulation:
    """Hold joint-target offsets through an explicit number of physics steps.

    Offsets are radians in actuator order, relative to the neutral stance.
    Absolute targets are clipped to the model's actuator control ranges.
    ``model`` and ``data`` remain public for notebook inspection.
    """

    def __init__(self) -> None:
        self.model = simulation.load_model()
        self.data = mujoco.MjData(self.model)
        self.reset()

    def reset(self) -> MeasuredState:
        """Restore the canonical neutral state and return its measurements."""
        simulation.reset(self.model, self.data)
        return simulation.measured_state(self.model, self.data)

    def step(self, target_offsets_rad: ArrayLike, *, physics_steps: int) -> MeasuredState:
        """Apply 18 finite offsets for a positive integer number of steps.

        The duration is ``physics_steps * model.opt.timestep`` seconds.
        Invalid inputs raise ``ValueError`` before changing simulation state.
        Measurements follow the canonical core's post-step sampling order.
        """
        if isinstance(physics_steps, (bool, np.bool_)) or not isinstance(physics_steps, Integral) or physics_steps <= 0:
            raise ValueError("physics_steps must be a positive integer")
        try:
            offsets = np.asarray(target_offsets_rad, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("target_offsets_rad must contain 18 finite radian offsets") from error
        if offsets.shape != (18,) or not np.all(np.isfinite(offsets)):
            raise ValueError("target_offsets_rad must have shape (18,) and contain only finite values")
        targets = np.clip(
            np.asarray(simulation.neutral_targets()) + offsets,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        ).tolist()
        for _ in range(physics_steps):
            simulation.step(self.model, self.data, targets)
        return simulation.measured_state(self.model, self.data)
