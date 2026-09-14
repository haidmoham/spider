"""Notebook recording and display plumbing for user-written policies.

No reward, training update, observation selection, or episode rule lives here.
The caller chooses action timing and the number of calls in an inspection run.
"""

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np

from learning_env import LearningSimulation
from notebook_viewer import TreatmentReplay


@dataclass
class PolicyRecording:
    replay: TreatmentReplay
    measurements: list
    offsets: np.ndarray
    targets: np.ndarray
    action_times: np.ndarray
    physics_steps: int

    def watch(self, directory: Path, *, speed: float = 1.0):
        """Replay measured states; never call the policy or integrate physics."""
        return self.replay.launch(Path(directory), speed=speed)


def record_policy(policy, *, label: str, action_count: int,
                  physics_steps: int) -> PolicyRecording:
    """Record a fresh reset and each action endpoint.

    This executes the supplied policy. Frames and measurements share timestamps.
    The action is held for physics_steps; intermediate physics frames are omitted.
    Use a fresh policy instance for each run if the policy has internal state.
    """
    for name, value in (("action_count", action_count), ("physics_steps", physics_steps)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    sim = LearningSimulation()
    observation = sim.reset()
    dt = physics_steps * sim.model.opt.timestep
    replay = TreatmentReplay(sim.model, f"{label} | action dt={dt:g}s | calls={action_count}")
    replay.capture(sim.data)
    measurements = [observation]
    offsets, targets, action_times = [], [], []
    for _ in range(action_count):
        action = np.asarray(policy(observation), dtype=float).copy()
        action_time = observation.time
        observation = sim.step(action, physics_steps=physics_steps)
        offsets.append(action)
        targets.append(sim.data.ctrl.copy())
        action_times.append(action_time)
        measurements.append(observation)
        replay.capture(sim.data)
    return PolicyRecording(replay, measurements, np.asarray(offsets),
                           np.asarray(targets), np.asarray(action_times), physics_steps)


def plot_recordings(*recordings: PolicyRecording, actuator: int = 0):
    """Plot recorded motion and one actuator's target versus measured angle."""
    import matplotlib.pyplot as plt

    if not recordings:
        raise ValueError("Supply at least one recording")
    if isinstance(actuator, (bool, np.bool_)) or not isinstance(actuator, Integral) or not 0 <= actuator < 18:
        raise ValueError("actuator must be an integer from 0 to 17")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for recording in recordings:
        states = recording.measurements
        times = np.asarray([state.time for state in states])
        xyz = np.asarray([state.torso_position for state in states])
        label = recording.replay.label
        line, = axes[0, 0].plot(times, xyz[:, 0] - xyz[0, 0], label=label)
        color = line.get_color()
        axes[0, 1].plot(times, xyz[:, 2], color=color, label=label)
        axes[1, 0].plot(times, [len(state.foot_contacts) for state in states], color=color)
        axes[1, 1].plot(times, [state.joint_positions[actuator] for state in states],
                        color=color, label=f"{label}: measured")
        axes[1, 1].step(np.r_[recording.action_times, times[-1]],
                        np.r_[recording.targets[:, actuator], recording.targets[-1, actuator]],
                        where="post", linestyle="--", color=color, label=f"{label}: target")
    titles = ("World +X displacement", "Torso height", "Feet in contact", "Joint response")
    units = ("Displacement (m)", "Height (m)", "Contact count", "Joint angle / target (rad)")
    for ax, title, unit in zip(axes.flat, titles, units):
        ax.set(title=title, xlabel="Simulation time (s)", ylabel=unit)
        ax.grid(alpha=0.2)
    axes[1, 0].set(ylim=(-0.2, 6.2), yticks=range(7))
    axes[0, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    name = recordings[0].replay.model.actuator(actuator).name
    fig.suptitle(f"Recorded policy inspection | actuator {actuator}: {name}")
    return fig, axes
