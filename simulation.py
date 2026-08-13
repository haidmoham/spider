"""Canonical deterministic C-1N simulation core.

The core owns model loading, reset, measured state, commanded targets, and
stepping. Viewers and experiments may depend on this module; it never depends
on them.

Future experiment seams (intentionally not implemented here):
- COM/support results belong beside ``MeasuredState`` as measurements, never
  as commands.
- Desired-foot workspace results must be translated into the joint target
  vector before ``set_targets`` or ``step`` receives it.
- A later Jacobian adapter belongs between those Cartesian foot corrections
  and that existing joint-target vector; it must not be folded into stepping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "spider.xml"
FOOT_NAMES = (
    "front_left",
    "front_right",
    "middle_left",
    "middle_right",
    "rear_left",
    "rear_right",
)
NEUTRAL_TORSO_POSITION = (0.0, 0.0, 0.5)
NEUTRAL_QUATERNION = (1.0, 0.0, 0.0, 0.0)
JOINTS_PER_LEG = 3
NEUTRAL_JOINT_TARGETS = (0.0, 0.0, 0.8) * 6


@dataclass(frozen=True)
class MeasuredState:
    """MuJoCo state observed after forward dynamics, separate from commands."""

    time: float
    torso_position: tuple[float, float, float]
    torso_orientation: tuple[float, float, float, float]
    torso_velocity: tuple[float, float, float]
    torso_angular_velocity: tuple[float, float, float]
    joint_positions: tuple[float, ...]
    joint_velocities: tuple[float, ...]
    actuator_forces: tuple[float, ...]
    foot_positions: dict[str, tuple[float, float, float]]
    foot_contacts: tuple[str, ...]


def load_model() -> mujoco.MjModel:
    """Load the one authoritative C-1N model."""
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def neutral_targets() -> tuple[float, ...]:
    """Return the named static baseline targets without reading simulator state."""
    return NEUTRAL_JOINT_TARGETS


def reset(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, ...]:
    """Reset C-1N to its deterministic, geometry-checked neutral baseline."""
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = NEUTRAL_TORSO_POSITION
    data.qpos[3:7] = NEUTRAL_QUATERNION
    data.qpos[7:] = NEUTRAL_JOINT_TARGETS
    set_targets(data, NEUTRAL_JOINT_TARGETS)
    mujoco.mj_forward(model, data)
    return NEUTRAL_JOINT_TARGETS


def set_targets(data: mujoco.MjData, targets: tuple[float, ...] | list[float]) -> None:
    """Set desired actuator targets; this intentionally does not advance physics."""
    if len(targets) != len(data.ctrl):
        raise ValueError(f"expected {len(data.ctrl)} targets, got {len(targets)}")
    data.ctrl[:] = targets


def step(model: mujoco.MjModel, data: mujoco.MjData, targets: tuple[float, ...] | list[float] | None = None) -> None:
    """Optionally apply desired targets, then take exactly one MuJoCo step."""
    if targets is not None:
        set_targets(data, targets)
    mujoco.mj_step(model, data)


def run(model: mujoco.MjModel, data: mujoco.MjData, seconds: float, targets: tuple[float, ...] | list[float] | None = None) -> None:
    """Advance a fixed-duration open-loop rollout using one explicit target vector."""
    if seconds <= 0:
        raise ValueError("seconds must be greater than zero")
    steps = round(seconds / model.opt.timestep)
    for _ in range(steps):
        step(model, data, targets)


def measured_state(model: mujoco.MjModel, data: mujoco.MjData) -> MeasuredState:
    """Read the physical state without exposing it as a desired command."""
    ground_id = model.geom("ground").id
    feet = {model.geom(f"{name}_foot").id: name for name in FOOT_NAMES}
    contacts: set[str] = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        other = contact.geom2 if contact.geom1 == ground_id else contact.geom1 if contact.geom2 == ground_id else None
        if other in feet:
            contacts.add(feet[other])
    return MeasuredState(
        time=float(data.time),
        torso_position=tuple(float(value) for value in data.qpos[:3]),
        torso_orientation=tuple(float(value) for value in data.qpos[3:7]),
        torso_velocity=tuple(float(value) for value in data.qvel[:3]),
        torso_angular_velocity=tuple(float(value) for value in data.qvel[3:6]),
        joint_positions=tuple(float(value) for value in data.qpos[7:]),
        joint_velocities=tuple(float(value) for value in data.qvel[6:]),
        actuator_forces=tuple(float(value) for value in data.actuator_force),
        foot_positions={
            name: tuple(float(value) for value in data.geom_xpos[model.geom(f"{name}_foot").id])
            for name in FOOT_NAMES
        },
        foot_contacts=tuple(sorted(contacts)),
    )
