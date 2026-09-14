"""Map one planar joint's motion into shared-frame foot X telemetry."""

import argparse
import math
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.telemetry import (
    OVERLAY_ALPHA,
    SIGNAL_BLUE,
    SIGNAL_GREEN,
    WARM_RED,
    WARM_ORANGE,
    make_figure,
    PLOT_INTERVAL,
    rolling_samples,
    update_stacks,
)


XML_TEMPLATE = """
<mujoco model="jacobians_task_space">
  <option timestep="0.002" gravity="0 0 {gravity_z}"/>
  <worldbody>
    <light pos="0 0 3" diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3"/>
    <geom type="plane" size="3 3 0.1" rgba="0.91 0.925 0.945 1"/>
    <body name="base" pos="0 0 1.5" euler="0 {base_degrees} 0">
      <joint name="joint1" type="hinge" axis="0 1 0" damping="0.02"/>
      <geom name="leg" type="capsule" fromto="0 0 0 0 0 -1" size="0.05" mass="1" rgba="0.01 0.01 0.015 1"/>
      <site name="foot" pos="0 0 -1" size="0.07" rgba="0.71 0.263 0.247 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="joint1_motor" joint="joint1" gear="1"/>
  </actuator>
</mujoco>
"""

TARGET_AMPLITUDE = 0.55
TARGET_FREQUENCY = 1.5
KP = 18.0
KD = 3.0
REPORT_INTERVAL = 0.25
FINITE_DIFFERENCE_STEP = 1e-5
OVERLAY_PLAYBACK_SPEED = 1.0
OVERLAY_ORIENTATIONS = (0.0, 90.0, 180.0)
OVERLAY_COLORS = (
    np.array([*SIGNAL_BLUE, OVERLAY_ALPHA]),
    np.array([*WARM_ORANGE, OVERLAY_ALPHA]),
    np.array([*SIGNAL_GREEN, OVERLAY_ALPHA]),
)
OVERLAY_SERIES = tuple(
    (f"{degrees:g}", color[:3])
    for degrees, color in zip(OVERLAY_ORIENTATIONS, OVERLAY_COLORS)
)
JOINT_SERIES = (
    ("actual", SIGNAL_BLUE),
    ("target", WARM_RED),
)
JOINT_RANGES = ((-0.65, 0.65), (-1.2, 1.2), (-2.5, 2.5))
TASK_RANGES = ((-0.65, 0.65), (-1.2, 1.2), (-2.5, 2.5))
VIEW_CENTER = np.array([0.0, 0.0, 1.5])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-degrees", type=float, default=0.0)
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Render synchronized 0, 90, and 180 degree runs together.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=6.0)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    return args


def x_jacobian(model, data, foot_site_id):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, foot_site_id)
    return jacp[0, 0]


def make_model(base_degrees, gravity_z=-9.81):
    return mujoco.MjModel.from_xml_string(
        XML_TEMPLATE.format(base_degrees=base_degrees, gravity_z=gravity_z)
    )


def finite_difference_x_jacobian(model, data, foot_site_id):
    original_qpos = data.qpos.copy()

    data.qpos[0] = original_qpos[0] + FINITE_DIFFERENCE_STEP
    mujoco.mj_forward(model, data)
    x_plus = data.site_xpos[foot_site_id, 0]

    data.qpos[0] = original_qpos[0] - FINITE_DIFFERENCE_STEP
    mujoco.mj_forward(model, data)
    x_minus = data.site_xpos[foot_site_id, 0]

    data.qpos[:] = original_qpos
    mujoco.mj_forward(model, data)
    return (x_plus - x_minus) / (2.0 * FINITE_DIFFERENCE_STEP)


def target(sim_time):
    return TARGET_AMPLITUDE * math.sin(TARGET_FREQUENCY * sim_time)


def target_state(sim_time):
    phase = TARGET_FREQUENCY * sim_time
    position = TARGET_AMPLITUDE * math.sin(phase)
    velocity = TARGET_AMPLITUDE * TARGET_FREQUENCY * math.cos(phase)
    acceleration = -TARGET_AMPLITUDE * TARGET_FREQUENCY**2 * math.sin(phase)
    return position, velocity, acceleration


def step(model, data, sim_time):
    error = target(sim_time) - data.qpos[0]
    data.ctrl[0] = KP * error - KD * data.qvel[0]
    mujoco.mj_step(model, data)


def task_space_state(model, data, foot_site_id, previous_x_velocity):
    x_position = data.site_xpos[foot_site_id, 0]
    jacobian_x = x_jacobian(model, data, foot_site_id)
    x_velocity = jacobian_x * data.qvel[0]
    x_acceleration = (x_velocity - previous_x_velocity) / model.opt.timestep
    return x_position, x_velocity, x_acceleration, jacobian_x


def report(data, task_state):
    x_position, x_velocity, x_acceleration, jacobian_x = task_state
    print(
        f"t={data.time:5.2f} q={data.qpos[0]:+.4f} qvel={data.qvel[0]:+.4f} "
        f"qacc={data.qacc[0]:+.4f} | x={x_position:+.4f} xvel={x_velocity:+.4f} "
        f"xacc={x_acceleration:+.4f} Jx={jacobian_x:+.4f}"
    )


def add_leg_geoms(scene, model, data, foot_site_id, rgba, label):
    """Render one simulated leg as a viewer-only overlay."""
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == 0:
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
        scene.ngeom += 1

    foot = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        foot,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.08, 0.0, 0.0]),
        data.site_xpos[foot_site_id],
        np.eye(3).reshape(-1),
        rgba,
    )
    foot.label = label
    scene.ngeom += 1


def add_frame_marker(scene, position, label):
    marker = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        marker,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.035, 0.0, 0.0]),
        position,
        np.eye(3).reshape(-1),
        np.array([0.01, 0.01, 0.015, 1.0]),
    )
    marker.label = label
    scene.ngeom += 1


def face_motion_plane(viewer):
    """Show the X-Z motion plane head-on, with world X running across the screen."""
    viewer.cam.lookat[:] = VIEW_CENTER
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 90.0
    viewer.cam.elevation = 0.0


def run_overlay_viewer():
    """Overlay independently simulated base orientations using Experiment 3's scene layer."""
    models = [
        make_model(degrees, gravity_z=0.0)
        for degrees in OVERLAY_ORIENTATIONS
    ]
    data_sets = [mujoco.MjData(model) for model in models]
    foot_site_ids = [model.site("foot").id for model in models]
    for model, data in zip(models, data_sets):
        mujoco.mj_forward(model, data)
    initial_x_positions = np.array(
        [data.site_xpos[foot_site_id, 0] for data, foot_site_id in zip(data_sets, foot_site_ids)]
    )
    for model in models:
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] != 0:
                model.geom_rgba[geom_id, 3] = 0.0

    print("overlay: blue=0 degrees, orange=90 degrees, green=180 degrees")
    joint_figures = (
        make_figure("Joint position q (rad)", JOINT_SERIES),
        make_figure("Joint velocity qvel (rad/s)", JOINT_SERIES),
        make_figure("Joint acceleration qacc (rad/s^2)", JOINT_SERIES),
    )
    task_figures = (
        make_figure("Foot X displacement (m)", OVERLAY_SERIES),
        make_figure("Foot X velocity (m/s)", OVERLAY_SERIES),
        make_figure("Foot X acceleration (m/s^2)", OVERLAY_SERIES),
    )
    samples = rolling_samples()
    previous_x_velocities = np.zeros(len(models))
    next_plot = 0.0
    wall_origin = time.perf_counter()
    task_states = [
        task_space_state(model, data, foot_site_id, 0.0)
        for model, data, foot_site_id in zip(models, data_sets, foot_site_ids)
    ]
    with mujoco.viewer.launch_passive(models[0], data_sets[0]) as viewer:
        face_motion_plane(viewer)
        while viewer.is_running():
            target_sim_time = (time.perf_counter() - wall_origin) * OVERLAY_PLAYBACK_SPEED
            while data_sets[0].time < target_sim_time:
                for model, data in zip(models, data_sets):
                    step(model, data, data.time)

                task_states = [
                    task_space_state(model, data, foot_site_id, previous_x_velocity)
                    for model, data, foot_site_id, previous_x_velocity in zip(
                        models, data_sets, foot_site_ids, previous_x_velocities
                    )
                ]
                previous_x_velocities = np.array([state[1] for state in task_states])
                if data_sets[0].time >= next_plot:
                    desired = target_state(data_sets[0].time)
                    samples.append(
                        (
                            data_sets[0].time,
                            np.array([data_sets[0].qpos[0], desired[0]]),
                            np.array([data_sets[0].qvel[0], desired[1]]),
                            np.array([data_sets[0].qacc[0], desired[2]]),
                            np.array([state[0] for state in task_states])
                            - initial_x_positions,
                            np.array([state[1] for state in task_states]),
                            np.array([state[2] for state in task_states]),
                        )
                    )
                    next_plot += PLOT_INTERVAL

            viewer.user_scn.ngeom = 0
            add_frame_marker(viewer.user_scn, VIEW_CENTER, "shared hinge")
            add_frame_marker(
                viewer.user_scn, VIEW_CENTER + np.array([0.6, 0.0, 0.0]), "+X"
            )
            add_frame_marker(
                viewer.user_scn, VIEW_CENTER + np.array([0.0, 0.0, 0.6]), "+Z"
            )
            for model, data, foot_site_id, color, degrees in zip(
                models, data_sets, foot_site_ids, OVERLAY_COLORS, OVERLAY_ORIENTATIONS
            ):
                add_leg_geoms(
                    viewer.user_scn,
                    model,
                    data,
                    foot_site_id,
                    color,
                    f"{degrees:g} deg base",
                )
            update_stacks(
                viewer,
                joint_figures,
                task_figures,
                samples,
                left_fields=(1, 2, 3),
                right_fields=(4, 5, 6),
                left_ranges=JOINT_RANGES,
                right_ranges=TASK_RANGES,
            )
            viewer.sync()
            time.sleep(0.001)


def main():
    args = parse_args()
    if args.overlay:
        if args.headless:
            raise ValueError("--overlay cannot run headless")
        run_overlay_viewer()
        return

    model = make_model(args.base_degrees)
    data = mujoco.MjData(model)
    foot_site_id = model.site("foot").id
    mujoco.mj_forward(model, data)

    jacobian = x_jacobian(model, data, foot_site_id)
    finite_difference = finite_difference_x_jacobian(model, data, foot_site_id)
    print(
        f"base_degrees={args.base_degrees:g} initial_Jx={jacobian:+.6f} "
        f"finite_difference_Jx={finite_difference:+.6f}"
    )

    previous_x_velocity = 0.0
    next_report = 0.0

    if args.headless:
        while data.time < args.duration:
            step(model, data, data.time)
            task_state = task_space_state(
                model, data, foot_site_id, previous_x_velocity
            )
            previous_x_velocity = task_state[1]
            if data.time >= next_report:
                report(data, task_state)
                next_report += REPORT_INTERVAL
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            wall_start = time.time()
            step(model, data, data.time)
            viewer.sync()
            task_state = task_space_state(
                model, data, foot_site_id, previous_x_velocity
            )
            previous_x_velocity = task_state[1]

            if data.time >= next_report:
                report(data, task_state)
                next_report += REPORT_INTERVAL

            remaining = model.opt.timestep - (time.time() - wall_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
