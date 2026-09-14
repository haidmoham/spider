"""Existing target generators: supported stance and the legacy tripod gait.

STAND holds the foot-placement task while all six feet support the body.
SHUFFLE preserves the old gait for comparison. Neither implements learned walking.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import mujoco
import numpy as np

from .simulation import FOOT_NAMES, measured_state, neutral_targets


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


GAIT_FREQUENCY = 0.65
HIP_SWEEP = 0.32
STANCE_KNEE = 0.8
KNEE_LIFT = 0.55
STARTUP_DURATION = 1.0
BALANCE_HIP_GAIN = 0.10
BALANCE_KNEE_GAIN = 0.06
BALANCE_HIP_DAMPING = 0.01
CONTACT_KNEE_ADJUST = 0.08
TRIPOD_A = {0, 3, 4}


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def gait_target(phase: float, leg: int) -> tuple[float, float]:
    """Return hip and knee targets for one leg in the tripod gait."""
    phase_offset = 0.0 if leg in TRIPOD_A else 0.5
    cycle = (phase + phase_offset) % 1.0

    if cycle < 0.5:
        stance_progress = cycle / 0.5
        hip = HIP_SWEEP * (2.0 * smoothstep(stance_progress) - 1.0)
        knee = STANCE_KNEE
    else:
        swing_progress = (cycle - 0.5) / 0.5
        hip = HIP_SWEEP * (1.0 - 2.0 * smoothstep(swing_progress))
        knee = STANCE_KNEE - KNEE_LIFT * math.sin(math.pi * swing_progress)
    return hip, knee


class GaitCoordinator:
    """Share phase and body-state feedback across all six legs."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.phase = 0.0
        self.last_time = data.time
        self.torso_id = model.body("torso").id
        self.ground_geom_id = model.geom("ground").id
        self.foot_geom_ids = [model.geom(f"{name}_foot").id for name in FOOT_NAMES]
        self.orientation_sensor_address = int(model.sensor("torso_orientation").adr[0])

    def contacts(self, data: mujoco.MjData) -> tuple[bool, ...]:
        contact = [False] * len(FOOT_NAMES)
        foot_index = {geom_id: index for index, geom_id in enumerate(self.foot_geom_ids)}
        for contact_id in range(data.ncon):
            collision = data.contact[contact_id]
            if collision.geom1 == self.ground_geom_id:
                foot = foot_index.get(collision.geom2)
            elif collision.geom2 == self.ground_geom_id:
                foot = foot_index.get(collision.geom1)
            else:
                foot = None
            if foot is not None:
                contact[foot] = True
        return tuple(contact)

    def body_errors(self, data: mujoco.MjData) -> tuple[float, float, float, float]:
        """Return roll, pitch, and their angular rates relative to gravity."""
        quaternion = data.sensordata[
            self.orientation_sensor_address : self.orientation_sensor_address + 4
        ]
        rotation_flat = np.empty(9)
        mujoco.mju_quat2Mat(rotation_flat, quaternion)
        rotation = rotation_flat.reshape(3, 3)
        gravity_body = rotation.T @ self.model.opt.gravity
        roll = math.atan2(gravity_body[1], -gravity_body[2])
        pitch = math.atan2(-gravity_body[0], -gravity_body[2])
        roll_rate = float(data.qvel[3])
        pitch_rate = float(data.qvel[4])
        return roll, pitch, roll_rate, pitch_rate

    def targets(
        self, data: mujoco.MjData
    ) -> tuple[tuple[float, ...], tuple[bool, ...], tuple[float, ...]]:
        dt = max(0.0, data.time - self.last_time)
        self.last_time = data.time
        contact = self.contacts(data)

        starting = data.time < STARTUP_DURATION
        expected_stance = tuple(
            True if starting else ((self.phase + (0.0 if leg in TRIPOD_A else 0.5)) % 1.0) < 0.5
            for leg in range(6)
        )
        # Keep one shared phase. Contact changes target placement, but it does
        # not let one leg drift onto a different clock from the other legs.
        if not starting:
            self.phase = (self.phase + GAIT_FREQUENCY * dt) % 1.0

        roll, pitch, roll_rate, pitch_rate = self.body_errors(data)
        hip_targets = []
        knee_targets = []
        for leg in range(6):
            hip, knee = (0.0, STANCE_KNEE) if starting else gait_target(self.phase, leg)
            side = 1.0 if leg % 2 == 0 else -1.0
            fore_aft = 1.0 if leg < 2 else -1.0 if leg >= 4 else 0.0

            # Use gravity alignment to bias the leg targets back toward a level torso.
            hip += -BALANCE_HIP_GAIN * pitch - BALANCE_HIP_DAMPING * pitch_rate * fore_aft
            knee += BALANCE_KNEE_GAIN * (pitch * fore_aft + roll * side)

            if expected_stance[leg] and not contact[leg]:
                knee += CONTACT_KNEE_ADJUST
            elif not expected_stance[leg] and contact[leg]:
                knee -= CONTACT_KNEE_ADJUST

            hip_targets.append(float(np.clip(hip, -0.8, 0.8)))
            knee_targets.append(float(np.clip(knee, -1.4, 1.4)))

        interleaved_targets = tuple(
            value for pair in zip(hip_targets, knee_targets) for value in pair
        )
        return interleaved_targets, contact, (roll, pitch, roll_rate, pitch_rate)


def apply_gait_control(coordinator: GaitCoordinator, data: mujoco.MjData):
    hip_knee_targets, contact, body_errors = coordinator.targets(data)
    targets = tuple(
        target
        for leg in range(6)
        for target in (0.0, hip_knee_targets[2 * leg], hip_knee_targets[2 * leg + 1])
    )
    return targets, contact, body_errors
