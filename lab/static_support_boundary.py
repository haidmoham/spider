"""Probe the static-support boundary of a three-foot MuJoCo body.

The only swept variable is a payload's forward position.  The feet stay fixed
relative to the torso, so each rollout compares support margin with observed
normal contact load before a gait, reachability, or actuator question enters.
"""

from __future__ import annotations

import argparse
import math

import mujoco
import numpy as np


FEET = np.array(((0.30, 0.0), (-0.20, 0.24), (-0.20, -0.24)))
FOOT_RADIUS = 0.035
BODY_HEIGHT = 0.36
PAYLOAD_SHIFT_LIMIT = 1.25
ATTITUDE_TOLERANCE_RAD = math.radians(5.0)


def make_model() -> mujoco.MjModel:
    foot_geoms = "\n".join(
        f'<geom name="foot{index}" type="sphere" pos="{x} {y} -{BODY_HEIGHT}" '
        f'size="{FOOT_RADIUS}" mass="0.02"/>'
        for index, (x, y) in enumerate(FEET)
    )
    struts = "\n".join(
        f'<geom type="capsule" fromto="0 0 0 {x} {y} -{BODY_HEIGHT}" '
        'size="0.012" mass="0.03" contype="0" conaffinity="0"/>'
        for x, y in FEET
    )
    xml = f"""
    <mujoco model="static_support_boundary">
      <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
      <default>
        <geom friction="1 0.01 0.001" condim="3"/>
        <joint damping="0.2"/>
      </default>
      <worldbody>
        <geom name="ground" type="plane" size="3 3 0.1"/>
        <body name="torso" pos="0 0 {BODY_HEIGHT + FOOT_RADIUS}">
          <freejoint/>
          <geom type="box" size="0.18 0.12 0.06" mass="1.20"/>
          {struts}
          {foot_geoms}
          <body name="payload" pos="0 0 0.05">
            <joint name="payload_x" type="slide" axis="1 0 0" range="-{PAYLOAD_SHIFT_LIMIT} {PAYLOAD_SHIFT_LIMIT}"/>
            <geom type="sphere" size="0.045" mass="0.60" rgba="0.71 0.263 0.247 1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def signed_margin(point: np.ndarray, feet: np.ndarray) -> float:
    """Return distance to the nearest triangle edge; negative means outside."""
    signs = []
    distances = []
    for index in range(3):
        start, end = feet[index], feet[(index + 1) % 3]
        edge = end - start
        cross = edge[0] * (point[1] - start[1]) - edge[1] * (point[0] - start[0])
        signs.append(cross)
        distances.append(abs(cross) / np.linalg.norm(edge))
    inside = all(sign >= -1e-9 for sign in signs) or all(sign <= 1e-9 for sign in signs)
    return min(distances) if inside else -min(distances)


def normal_loads(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    loads = np.zeros(3)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, contact_index, force)
        for foot_index in range(3):
            foot_geom = model.geom(f"foot{foot_index}").id
            if foot_geom in (contact.geom1, contact.geom2):
                loads[foot_index] += force[0]
    return loads


def roll_pitch(data: mujoco.MjData, body_id: int) -> tuple[float, float]:
    """Return torso roll and pitch in radians from the observed world rotation."""
    rotation = data.xmat[body_id].reshape(3, 3)
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    return roll, pitch


def run(payload_shift: float, seconds: float) -> tuple[float, np.ndarray, np.ndarray, float]:
    model = make_model()
    data = mujoco.MjData(model)
    data.qpos[7] = payload_shift
    mujoco.mj_forward(model, data)
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    torso_id = model.body("torso").id
    com_xy = data.subtree_com[torso_id, :2].copy()
    feet_xy = np.array([data.geom_xpos[model.geom(f"foot{index}").id, :2] for index in range(3)])
    margin = signed_margin(com_xy, feet_xy)
    return margin, normal_loads(model, data), com_xy, data.xquat[torso_id, 1]


def observe(payload_shift: float, seconds: float) -> dict[str, np.ndarray | float]:
    """Return the complete #24 observation vector after one headless rollout."""
    model = make_model()
    data = mujoco.MjData(model)
    data.qpos[7] = payload_shift
    mujoco.mj_forward(model, data)
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    torso_id = model.body("torso").id
    feet_xy = np.array(
        [data.geom_xpos[model.geom(f"foot{index}").id, :2] for index in range(3)]
    )
    com_xy = data.subtree_com[torso_id, :2].copy()
    roll, pitch = roll_pitch(data, torso_id)
    loads = normal_loads(model, data)
    return {
        "support_margin_m": signed_margin(com_xy, feet_xy),
        "foot_xy_m": feet_xy,
        "com_world_xy_m": com_xy,
        "foot_normal_loads_n": loads,
        "torso_roll_rad": roll,
        "torso_pitch_rad": pitch,
        "torso_external_wrench": data.cfrc_ext[torso_id].copy(),
        "standing_metric_pass": float(
            min(loads) > 1e-3
            and abs(roll) <= ATTITUDE_TOLERANCE_RAD
            and abs(pitch) <= ATTITUDE_TOLERANCE_RAD
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shift", type=float, default=0.0, help="payload shift in torso +X (m)")
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()
    if not -PAYLOAD_SHIFT_LIMIT <= args.shift <= PAYLOAD_SHIFT_LIMIT:
        parser.error(f"--shift must be within [-{PAYLOAD_SHIFT_LIMIT}, {PAYLOAD_SHIFT_LIMIT}]")
    observation = observe(args.shift, args.seconds)
    print(f"payload_shift_m={args.shift:+.3f}")
    for name, value in observation.items():
        if isinstance(value, np.ndarray):
            print(f"{name}={np.array2string(value, precision=4)}")
        else:
            print(f"{name}={value:+.5f}")


if __name__ == "__main__":
    main()
