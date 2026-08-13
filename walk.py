"""Run the legacy C-1N tripod gait experiment."""

from __future__ import annotations

import argparse
from collections import deque
import math
import time

import mujoco
import numpy as np

from simulation import JOINTS_PER_LEG, load_model, reset, step
from visuals import ResponsivePupils


GAIT_FREQUENCY = 0.65
HIP_SWEEP = 0.32
STANCE_KNEE = 0.8
KNEE_LIFT = 0.55
STARTUP_DURATION = 1.0
BALANCE_HIP_GAIN = 0.10
BALANCE_KNEE_GAIN = 0.06
BALANCE_HIP_DAMPING = 0.01
CONTACT_KNEE_ADJUST = 0.08
REPORT_INTERVAL = 0.25
PLOT_INTERVAL = 0.02
PLOT_HISTORY_SECONDS = 6.0

TRIPOD_A = {0, 3, 4}
TRIPOD_B = {1, 2, 5}
FOOT_NAMES = (
    "front_left",
    "front_right",
    "middle_left",
    "middle_right",
    "rear_left",
    "rear_right",
)
PLOT_SERIES = (
    ("A hip", 0, (0.95, 0.25, 0.25)),
    ("A knee", 1, (1.00, 0.65, 0.20)),
    ("B hip", 2, (0.25, 0.55, 1.00)),
    ("B knee", 3, (0.45, 0.85, 1.00)),
)


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
        self.orientation_sensor_address = int(
            model.sensor("torso_orientation").adr[0]
        )

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

    def targets(self, data: mujoco.MjData) -> tuple[tuple[float, ...], tuple[bool, ...], tuple[float, ...]]:
        dt = max(0.0, data.time - self.last_time)
        self.last_time = data.time
        contact = self.contacts(data)

        starting = data.time < STARTUP_DURATION
        expected_stance = tuple(
            True
            if starting
            else ((self.phase + (0.0 if leg in TRIPOD_A else 0.5)) % 1.0) < 0.5
            for leg in range(6)
        )
        support_count = sum(
            is_contact for is_contact, is_stance in zip(contact, expected_stance) if is_stance
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
            value
            for pair in zip(hip_targets, knee_targets)
            for value in pair
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
    right_viewports = stack_viewports(
        viewport.left + viewport.width - width - margin
    )
    viewer.set_figures(
        list(zip(left_viewports, position_figures))
        + list(zip(right_viewports, torque_figures))
    )


def run_headless(duration: float) -> None:
    model = load_model()
    data = mujoco.MjData(model)
    reset(model, data)
    coordinator = GaitCoordinator(model, data)
    start_x = float(data.qpos[0])

    while data.time < duration:
        targets, _, _ = apply_gait_control(coordinator, data)
        step(model, data, targets)

    print(f"duration={data.time:.3f}s")
    print(f"displacement_x={data.qpos[0] - start_x:.3f}m")
    print(f"torso_position={data.qpos[:3]}")


def run_viewer() -> None:
    import mujoco.viewer

    model = load_model()
    data = mujoco.MjData(model)
    reset(model, data)
    coordinator = GaitCoordinator(model, data)
    pupils = ResponsivePupils(model)
    figures = (
        make_figure("Applied actuator torque (N-m)"),
        make_figure("Torque rate (N-m/s)"),
        make_figure("Torque acceleration (N-m/s^2)"),
    )
    position_figures = (
        make_figure("Joint position (rad)"),
        make_figure("Joint velocity (rad/s)"),
        make_figure("Joint acceleration (rad/s^2)"),
    )
    samples = deque(maxlen=round(PLOT_HISTORY_SECONDS / PLOT_INTERVAL))
    previous_torque = None
    previous_rate = None
    next_plot = 0.0
    next_report = 0.0
    start_x = float(data.qpos[0])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            wall_start = time.time()
            targets, contact, body_errors = apply_gait_control(coordinator, data)
            step(model, data, targets)

            torque = np.array(
                [
                    data.qfrc_actuator[7],
                    data.qfrc_actuator[8],
                    data.qfrc_actuator[10],
                    data.qfrc_actuator[11],
                ]
            )
            joint_position = np.array(data.qpos[[8, 9, 11, 12]])
            joint_velocity = np.array(data.qvel[[7, 8, 10, 11]])
            joint_acceleration = np.array(data.qacc[[7, 8, 10, 11]])
            if previous_torque is None:
                rate = np.zeros_like(torque)
                acceleration = np.zeros_like(torque)
            else:
                rate = (torque - previous_torque) / model.opt.timestep
                acceleration = (rate - previous_rate) / model.opt.timestep
            previous_torque = torque
            previous_rate = rate

            if data.time >= next_plot:
                samples.append(
                    (
                        data.time,
                        torque,
                        rate,
                        acceleration,
                        joint_position,
                        joint_velocity,
                        joint_acceleration,
                    )
                )
                next_plot += PLOT_INTERVAL
            if data.time >= next_report:
                print(
                    f"t={data.time:5.2f} x={data.qpos[0]:+.3f} "
                    f"z={data.qpos[2]:.3f} dx={data.qpos[0] - start_x:+.3f} "
                    f"contacts={sum(contact)}/6 "
                    f"roll={body_errors[0]:+.3f} pitch={body_errors[1]:+.3f}"
                )
                next_report += REPORT_INTERVAL

            update_figures(viewer, figures, position_figures, samples)
            pupils.update(model, data)
            viewer.sync()

            remaining = model.opt.timestep - (time.time() - wall_start)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be greater than zero")
    if args.headless:
        run_headless(args.duration)
    else:
        run_viewer()


if __name__ == "__main__":
    main()
