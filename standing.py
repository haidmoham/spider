"""Support-aware, stance-only target generator for C-1N.

This module holds the nominal foot-placement task from issue #31 behind the
contact support measurements from issue #24. It does not claim that the robot
is dynamically stable. In particular, it does not recover from a lost contact
or move a foot to create a new support region.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from simulation import MeasuredState, measured_state, neutral_targets


@dataclass(frozen=True)
class StandingTelemetry:
    """One control decision and the support facts that permitted it."""

    enabled: bool
    reason: str
    torso_height_error: float
    torso_attitude_error: float
    manual_override: bool
    commanded_targets: tuple[float, ...]


class SupportAwareStanceController:
    """Apply a bounded all-feet ``J^T r`` correction only inside support."""

    def __init__(self, support_margin_floor_m: float = 0.01, update_gain: float = 0.5) -> None:
        self.support_margin_floor_m = support_margin_floor_m
        self.update_gain = update_gain
        self.last_telemetry: StandingTelemetry | None = None
        self.last_automatic_targets: tuple[float, ...] | None = None
        self.manual_override = False

    def targets(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, ...]:
        observed = measured_state(model, data)
        torso_height_error = 0.45 - observed.torso_position[2]
        torso_attitude_error = float(np.linalg.norm(observed.torso_orientation[1:]))
        current_controls = tuple(float(value) for value in data.ctrl)
        controls_cleared = np.allclose(current_controls, 0.0, atol=1e-12)
        if self.manual_override and controls_cleared:
            self.manual_override = False
        elif self.last_automatic_targets is not None and not self.manual_override:
            if not np.allclose(current_controls, self.last_automatic_targets, atol=1e-12):
                self.manual_override = True

        support_ok = (
            observed.support_margin is not None
            and observed.support_margin >= self.support_margin_floor_m
            and len(observed.foot_contacts) == 6
        )
        baseline = np.asarray(neutral_targets())
        if self.manual_override:
            reason = "manual Control-panel override; Clear all resumes automatic stance"
            targets = np.asarray(current_controls)
        elif not support_ok:
            reason = "insufficient declared support; hold neutral stance targets"
            targets = baseline
        else:
            reason = "all feet declared and COM projection has positive support margin"
            correction = self.update_gain * np.asarray(observed.joint_space_update_direction)
            targets = baseline + correction
        targets = np.clip(targets, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        command = tuple(float(value) for value in targets)
        if not self.manual_override:
            self.last_automatic_targets = command
        self.last_telemetry = StandingTelemetry(
            enabled=support_ok,
            reason=reason,
            torso_height_error=torso_height_error,
            torso_attitude_error=torso_attitude_error,
            manual_override=self.manual_override,
            commanded_targets=command,
        )
        return self.last_telemetry.commanded_targets
