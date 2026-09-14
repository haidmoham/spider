"""Inspect three foot Jacobians mapping equal assigned forces into joint torques.

This is intentionally not a standing controller.  Its unconstrained equal-force
allocation may request impossible negative normal forces; preserve that failure
as #6 evidence and defer feasibility/contact control to issue #20.
"""

import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np


BODY_MASS = 1.5
LINK_LENGTH = 0.45
LEG_Q = np.array([0.40, -0.80])
LEG_YAWS = (0.0, 120.0, 240.0)
HIP_RADIUS = 0.18
# Bent legs at LEG_Q place each foot centre 0.829 m below the hips; the
# 55-mm contact sphere should begin just touching the ground.
BODY_HEIGHT = 0.884
REPORT_INTERVAL = 0.25
POSTURE_KP = 45.0
POSTURE_KD = 4.0
BODY_X_KP = 70.0
BODY_X_KD = 12.0
BODY_Z_KP = 90.0
BODY_Z_KD = 16.0


def leg_xml(index, yaw):
    yaw_radians = math.radians(yaw)
    hip_x = HIP_RADIUS * math.cos(yaw_radians)
    hip_y = HIP_RADIUS * math.sin(yaw_radians)
    return f"""
      <body name="hip{index}" pos="{hip_x} {hip_y} 0" euler="0 0 {yaw}">
        <joint name="hip{index}" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -{LINK_LENGTH}" size="0.04" mass="0.18" contype="0" conaffinity="0"/>
        <body name="knee{index}" pos="0 0 -{LINK_LENGTH}">
          <joint name="knee{index}" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -{LINK_LENGTH}" size="0.035" mass="0.14" contype="0" conaffinity="0"/>
          <geom name="foot{index}_contact" type="sphere" pos="0 0 -{LINK_LENGTH}" size="0.055" mass="0.02" rgba="0.71 0.263 0.247 1"/>
          <site name="foot{index}" pos="0 0 -{LINK_LENGTH}" size="0.055" rgba="0.71 0.263 0.247 1"/>
        </body>
      </body>
    """


def make_model():
    legs = "".join(leg_xml(index, yaw) for index, yaw in enumerate(LEG_YAWS))
    motors = "".join(
        f'<motor name="{joint}{index}_motor" joint="{joint}{index}" gear="1" ctrllimited="true" ctrlrange="-20 20"/>'
        for index in range(3)
        for joint in ("hip", "knee")
    )
    xml = f"""
    <mujoco model="tripod_static_support">
      <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
      <default>
        <joint damping="0.8" armature="0.02"/>
        <geom friction="1 0.01 0.001" condim="3" rgba="0.01 0.01 0.015 1"/>
      </default>
      <worldbody>
        <light pos="0 0 3"/>
        <geom name="floor" type="plane" size="3 3 0.1" rgba="0.91 0.925 0.945 1"/>
        <body name="body" pos="0 0 {BODY_HEIGHT}">
          <freejoint/>
          <geom type="box" size="0.25 0.18 0.08" mass="{BODY_MASS}" rgba="0.01 0.01 0.015 1"/>
          {legs}
        </body>
      </worldbody>
      <actuator>{motors}</actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--body-shift",
        type=float,
        default=0.0,
        help="Desired center-of-mass shift in world +X from the centered pose (m).",
    )
    parser.add_argument(
        "--squat-drop",
        type=float,
        default=0.0,
        help="Desired chassis/COM drop created by bending the planted legs (m).",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=6.0)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    return args


def initialize(model, data):
    data.qpos[:7] = (0.0, 0.0, BODY_HEIGHT, 1.0, 0.0, 0.0, 0.0)
    data.qpos[7:] = np.tile(LEG_Q, 3)
    mujoco.mj_forward(model, data)


def squat_posture(drop):
    """Shorten a symmetric two-link leg by bending both planted joints."""
    initial_reach = 2.0 * LINK_LENGTH * math.cos(LEG_Q[0])
    requested_reach = np.clip(initial_reach - drop, 0.0, 2.0 * LINK_LENGTH)
    hip_angle = math.acos(requested_reach / (2.0 * LINK_LENGTH))
    return np.array((hip_angle, -2.0 * hip_angle))


def support_forces(model, data, foot_site_ids, body_id, total_vertical_force):
    """Allocate the requested vertical ground force about the body projection."""
    feet = np.array([data.site_xpos[site_id] for site_id in foot_site_ids])
    support_point = data.subtree_com[body_id, :2]
    system = np.vstack((np.ones(3), feet[:, 0], feet[:, 1]))
    target = np.array(
        (
            total_vertical_force,
            total_vertical_force * support_point[0],
            total_vertical_force * support_point[1],
        )
    )
    return np.linalg.solve(system, target), feet


def apply_support_torques(
    model,
    data,
    foot_site_ids,
    joint_dof_adrs,
    body_id,
    desired_com_x,
    desired_com_z,
    posture_reference,
):
    com = data.subtree_com[body_id]
    body_force_x = BODY_X_KP * (desired_com_x - com[0]) - BODY_X_KD * data.qvel[0]
    body_force_z = BODY_Z_KP * (desired_com_z - com[2]) - BODY_Z_KD * data.qvel[2]
    weight = -model.opt.gravity[2] * model.body_mass.sum()
    vertical_forces, feet = support_forces(
        model, data, foot_site_ids, body_id, max(0.0, weight + body_force_z)
    )
    foot_forces = np.column_stack(
        (
            np.full(3, body_force_x / 3.0),
            np.zeros(3),
            vertical_forces,
        )
    )
    data.ctrl[:] = 0.0
    for index, (site_id, dof_adrs, foot_force) in enumerate(
        zip(foot_site_ids, joint_dof_adrs, foot_forces)
    ):
        jacobian_position = np.zeros((3, model.nv))
        jacobian_rotation = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, site_id)
        # ``foot_force`` is the ground reaction desired *on the robot*.
        # The motors must oppose that external generalized force to create it.
        support_torque = -(jacobian_position[:, dof_adrs].T @ foot_force)
        joint_qpos = data.qpos[[7 + 2 * index, 8 + 2 * index]]
        joint_qvel = data.qvel[[6 + 2 * index, 7 + 2 * index]]
        posture_torque = (
            POSTURE_KP * (posture_reference - joint_qpos) - POSTURE_KD * joint_qvel
        )
        torque = support_torque + posture_torque
        data.ctrl[2 * index : 2 * index + 2] = torque
    return foot_forces, feet


def report(data, body_id, foot_forces, feet, desired_com_x, desired_com_z):
    print(
        f"t={data.time:5.2f} com_xyz={np.array2string(data.subtree_com[body_id], precision=3)} "
        f"target_com_xz=[{desired_com_x:.3f} {desired_com_z:.3f}] "
        f"foot_fx={np.array2string(foot_forces[:, 0], precision=3)} "
        f"foot_fz={np.array2string(foot_forces[:, 2], precision=3)} "
        f"foot_xy={np.array2string(feet[:, :2], precision=3)}"
    )


def run(model, data, args):
    body_id = model.body("body").id
    foot_site_ids = [model.site(f"foot{index}").id for index in range(3)]
    joint_dof_adrs = [
        (model.joint(f"hip{index}").dofadr[0], model.joint(f"knee{index}").dofadr[0])
        for index in range(3)
    ]
    initial_com = data.subtree_com[body_id].copy()
    desired_com_x = initial_com[0] + args.body_shift
    desired_com_z = initial_com[2] - args.squat_drop
    posture_reference = squat_posture(args.squat_drop)
    next_report = 0.0

    def control_step():
        return apply_support_torques(
            model,
            data,
            foot_site_ids,
            joint_dof_adrs,
            body_id,
            desired_com_x,
            desired_com_z,
            posture_reference,
        )

    if args.headless:
        while data.time < args.duration:
            foot_forces, feet = control_step()
            mujoco.mj_step(model, data)
            if data.time >= next_report:
                report(data, body_id, foot_forces, feet, desired_com_x, desired_com_z)
                next_report += REPORT_INTERVAL
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            wall_start = time.perf_counter()
            foot_forces, feet = control_step()
            mujoco.mj_step(model, data)
            if data.time >= next_report:
                report(data, body_id, foot_forces, feet, desired_com_x, desired_com_z)
                next_report += REPORT_INTERVAL
            viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - wall_start)
            if remaining > 0:
                time.sleep(remaining)


def main():
    args = parse_args()
    model = make_model()
    data = mujoco.MjData(model)
    initialize(model, data)
    print(
        f"tripod support: body_shift={args.body_shift:+.3f} m "
        f"squat_drop={args.squat_drop:+.3f} m"
    )
    run(model, data, args)


if __name__ == "__main__":
    main()
