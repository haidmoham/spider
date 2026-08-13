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
import numpy as np


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
NEUTRAL_TORSO_POSITION = (0.0, 0.0, 0.45)
NEUTRAL_QUATERNION = (1.0, 0.0, 0.0, 0.0)
JOINTS_PER_LEG = 3
# A compact, symmetric stance. The hip and knee flex together so that lowering
# the torso does not drive the foot centres through the ground at reset.
NEUTRAL_JOINT_TARGETS = (0.0, -0.2, 1.1) * 6
_NEUTRAL_FOOT_POSITION_CACHE: dict[int, dict[str, tuple[float, float, float]]] = {}


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
    com_projection: tuple[float, float]
    support_polygon: tuple[tuple[float, float], ...]
    support_margin: float | None
    foot_normal_loads: dict[str, float]
    foot_position_residual: tuple[float, ...]
    foot_position_residual_norm: float
    joint_space_update_direction: tuple[float, ...]
    joint_space_update_direction_norm: float


def load_model() -> mujoco.MjModel:
    """Load the one authoritative C-1N model."""
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def neutral_targets() -> tuple[float, ...]:
    """Return the named static baseline targets without reading simulator state."""
    return NEUTRAL_JOINT_TARGETS


def neutral_foot_positions(model: mujoco.MjModel) -> dict[str, tuple[float, float, float]]:
    """Return world-frame foot targets for the neutral commanded configuration.

    These are kinematic targets. They do not describe a contact equilibrium.
    """
    key = id(model)
    if key in _NEUTRAL_FOOT_POSITION_CACHE:
        return _NEUTRAL_FOOT_POSITION_CACHE[key]
    data = mujoco.MjData(model)
    reset(model, data)
    positions = {
        name: tuple(float(value) for value in data.geom_xpos[model.geom(f"{name}_foot").id])
        for name in FOOT_NAMES
    }
    _NEUTRAL_FOOT_POSITION_CACHE[key] = positions
    return positions


def _convex_hull(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Return the counter-clockwise convex hull without repeated endpoints."""
    ordered = sorted(set(points))
    if len(ordered) <= 1:
        return tuple(ordered)

    def cross(origin: tuple[float, float], first: tuple[float, float], second: tuple[float, float]) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def _support_margin(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> float | None:
    """Return signed planar distance to a convex support boundary in metres."""
    if len(polygon) < 3:
        return None
    distances = []
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        edge_x, edge_y = end[0] - start[0], end[1] - start[1]
        length = float(np.hypot(edge_x, edge_y))
        distances.append((edge_x * (point[1] - start[1]) - edge_y * (point[0] - start[0])) / length)
    return float(min(distances))


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
    """Read physical state and whole-stance kinematic diagnostics.

    The residual stacks target-minus-current XYZ positions for all six feet.
    The update direction is ``J_actuated.T @ residual``. ``J_actuated`` has
    one row per foot coordinate and one column per actuator joint. These are
    diagnostics for the neutral foot-placement task. They are not a standing
    controller or a measure of contact force, slip, or dynamic stability.
    """
    ground_id = model.geom("ground").id
    feet = {model.geom(f"{name}_foot").id: name for name in FOOT_NAMES}
    contacts: set[str] = set()
    normal_loads = {name: 0.0 for name in FOOT_NAMES}
    for index in range(data.ncon):
        contact = data.contact[index]
        other = contact.geom2 if contact.geom1 == ground_id else contact.geom1 if contact.geom2 == ground_id else None
        if other in feet:
            name = feet[other]
            contacts.add(name)
            contact_force = np.empty(6)
            mujoco.mj_contactForce(model, data, index, contact_force)
            normal_loads[name] += max(0.0, float(contact_force[0]))
    foot_positions = {
        name: tuple(float(value) for value in data.geom_xpos[model.geom(f"{name}_foot").id])
        for name in FOOT_NAMES
    }
    targets = neutral_foot_positions(model)
    residual = np.concatenate(
        [np.subtract(targets[name], foot_positions[name]) for name in FOOT_NAMES]
    )
    jacobian = np.empty((len(FOOT_NAMES) * 3, model.nv))
    for index, name in enumerate(FOOT_NAMES):
        position_jacobian = np.empty((3, model.nv))
        rotation_jacobian = np.empty((3, model.nv))
        mujoco.mj_jacGeom(
            model,
            data,
            position_jacobian,
            rotation_jacobian,
            model.geom(f"{name}_foot").id,
        )
        jacobian[index * 3 : (index + 1) * 3] = position_jacobian
    # The free-base coordinates are excluded because translation and rotation
    # use mixed units. This leaves the 18 actuated joints as one coherent
    # update vector while retaining all six feet in the task residual.
    joint_update_direction = jacobian[:, 6:].T @ residual
    torso_id = model.body("torso").id
    com_projection = tuple(float(value) for value in data.subtree_com[torso_id, :2])
    support_polygon = _convex_hull([foot_positions[name][:2] for name in contacts])
    margin = _support_margin(com_projection, support_polygon)
    return MeasuredState(
        time=float(data.time),
        torso_position=tuple(float(value) for value in data.qpos[:3]),
        torso_orientation=tuple(float(value) for value in data.qpos[3:7]),
        torso_velocity=tuple(float(value) for value in data.qvel[:3]),
        torso_angular_velocity=tuple(float(value) for value in data.qvel[3:6]),
        joint_positions=tuple(float(value) for value in data.qpos[7:]),
        joint_velocities=tuple(float(value) for value in data.qvel[6:]),
        actuator_forces=tuple(float(value) for value in data.actuator_force),
        foot_positions=foot_positions,
        foot_contacts=tuple(sorted(contacts)),
        com_projection=com_projection,
        support_polygon=support_polygon,
        support_margin=margin,
        foot_normal_loads=normal_loads,
        foot_position_residual=tuple(float(value) for value in residual),
        foot_position_residual_norm=float(np.linalg.norm(residual)),
        joint_space_update_direction=tuple(float(value) for value in joint_update_direction),
        joint_space_update_direction_norm=float(np.linalg.norm(joint_update_direction)),
    )
