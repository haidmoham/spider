"""Compare joint PD with the same controller plus gravity compensation."""

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
    MUTED_SLATE,
    PLOT_INTERVAL,
    SIGNAL_BLUE,
    WARM_ORANGE,
    WARM_RED,
    make_figure,
    rolling_samples,
    update_stacks,
)


XML_TEMPLATE = """
<mujoco model="model_based_control_{model_name}">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 4"/>
    <geom type="plane" size="3 3 0.1"/>
    <body name="shoulder" pos="0 0 2.5" euler="0 90 0">
      <joint name="joint1" type="hinge" axis="0 1 0" damping="0.02"/>
      <geom name="link1" type="capsule" fromto="0 0 0 0 0 -1" size="0.05" mass="1" rgba="{rgba}"/>
      <body name="elbow" pos="0 0 -1">
        <joint name="joint2" type="hinge" axis="0 1 0" damping="0.02"/>
      <geom name="link2" type="capsule" fromto="0 0 0 0 0 -0.8" size="0.045" mass="{link2_mass}" rgba="{rgba}"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="joint1" gear="1"/>
    <motor joint="joint2" gear="1"/>
  </actuator>
</mujoco>
"""

START = (0.35, -0.70)
KP = 18.0
KD = 4.0
AMPLITUDE = (0.35, 0.45)
FREQUENCY = 0.8
REPORT_INTERVAL = 0.25
PLANT_LINK2_MASS = 0.7
WRONG_CONTROL_LINK2_MASS = 0.35
COLORS = {
    "pd": ("pd_blue", "0.15 0.40 0.95 1"),
    "gravity-comp": ("gravity_orange", "0.95 0.45 0.08 1"),
    "computed-torque": ("computed_green", "0.10 0.75 0.35 1"),
    "computed-torque-wrong-mass": ("wrong_mass_red", "0.90 0.20 0.25 1"),
}
OVERLAY_ALPHA = 0.5
TURNAROUND_HALF_WIDTH = 0.4
PLOT_SERIES = (
    ("PD joint 1", SIGNAL_BLUE),
    ("PD joint 2", MUTED_SLATE),
    ("gravity joint 1", WARM_ORANGE),
    ("gravity joint 2", WARM_RED),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        choices=(
            "pd",
            "gravity-comp",
            "computed-torque",
            "computed-torque-wrong-mass",
            "overlay",
        ),
        default="pd",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the viewer for deterministic telemetry collection.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=16.0,
        help="Headless simulation duration in seconds.",
    )
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    return args


def make_model(controller, link2_mass=PLANT_LINK2_MASS):
    model_name, rgba = COLORS[controller]
    return mujoco.MjModel.from_xml_string(
        XML_TEMPLATE.format(
            model_name=model_name,
            rgba=rgba,
            link2_mass=link2_mass,
        )
    )


def desired_state(sim_time):
    phase = FREQUENCY * sim_time
    q_des = np.array([START[i] + AMPLITUDE[i] * math.sin(phase) for i in range(2)])
    qvel_des = np.array(
        [AMPLITUDE[i] * FREQUENCY * math.cos(phase) for i in range(2)]
    )
    qacc_des = np.array(
        [-AMPLITUDE[i] * FREQUENCY**2 * math.sin(phase) for i in range(2)]
    )
    return q_des, qvel_des, qacc_des


def inverse_dynamics_torque(model, work_data, qpos, qvel, qacc):
    work_data.qpos[:] = qpos
    work_data.qvel[:] = qvel
    work_data.qacc[:] = qacc
    mujoco.mj_inverse(model, work_data)
    return work_data.qfrc_inverse.copy()


def gravity_torque(model, work_data, qpos):
    return inverse_dynamics_torque(
        model,
        work_data,
        qpos,
        np.zeros_like(qpos),
        np.zeros_like(qpos),
    )


def step_controller(
    controller,
    plant_model,
    data,
    plant_gravity_data,
    control_model,
    control_work_data,
    sim_time,
):
    q_des, qvel_des, qacc_des = desired_state(sim_time)
    tau_feedback = KP * (q_des - data.qpos) + KD * (
        qvel_des - data.qvel
    )
    tau_gravity = gravity_torque(plant_model, plant_gravity_data, data.qpos)
    if controller == "gravity-comp":
        tau_feedforward = gravity_torque(control_model, control_work_data, data.qpos)
    elif controller in ("computed-torque", "computed-torque-wrong-mass"):
        tau_feedforward = inverse_dynamics_torque(
            control_model,
            control_work_data,
            q_des,
            qvel_des,
            qacc_des,
        )
    else:
        tau_feedforward = np.zeros_like(tau_feedback)
    tau_total = tau_feedback + tau_feedforward
    data.ctrl[:] = tau_total
    mujoco.mj_step(plant_model, data)
    error = data.qpos - q_des
    acceleration_error = data.qacc - qacc_des
    return (
        q_des,
        qacc_des,
        error,
        acceleration_error,
        tau_feedback,
        tau_gravity,
        tau_feedforward,
        tau_total,
    )


def report(
    sim_time,
    data,
    target,
    qacc_des,
    error,
    acceleration_error,
    tau_feedback,
    tau_gravity,
    tau_feedforward,
    tau_total,
):
    print(
        f"t={sim_time:5.2f} target={target} qpos={data.qpos[:2]} "
        f"error={error[:2]} qacc_error={acceleration_error[:2]} "
        f"tau_fb={tau_feedback[:2]} tau_g={tau_gravity[:2]} "
        f"tau_ff={tau_feedforward[:2]} tau_total={tau_total[:2]} "
        f"|error|={np.linalg.norm(error):.4f} "
        f"|tau_fb|={np.linalg.norm(tau_feedback):.4f} "
        f"|tau_g|={np.linalg.norm(tau_gravity):.4f} "
        f"|tau_total|={np.linalg.norm(tau_total):.4f}"
    )


def rms(samples):
    return np.sqrt(np.mean(np.square(np.asarray(samples)), axis=0))


def projection_coefficients(feedback, gravity):
    feedback = np.asarray(feedback)
    gravity = np.asarray(gravity)
    return np.sum(feedback * gravity, axis=0) / np.sum(gravity * gravity, axis=0)


def phase_offset_seconds(times, positions):
    """Fit a signed target-frequency phase offset for each joint."""
    phase = FREQUENCY * np.asarray(times)
    basis = np.column_stack((np.sin(phase), np.cos(phase), np.ones_like(phase)))
    coefficients, _, _, _ = np.linalg.lstsq(basis, positions, rcond=None)
    phase_offset = np.arctan2(coefficients[1], coefficients[0])
    return -phase_offset / FREQUENCY


def turnaround_rms(times, values):
    """Return error RMS near target reversals, where desired acceleration is large."""
    phase = FREQUENCY * np.asarray(times)
    mask = np.abs(np.cos(phase)) <= math.sin(TURNAROUND_HALF_WIDTH)
    return rms(np.asarray(values)[mask])


def run_headless(
    args, model, data, plant_gravity_data, control_model, control_work_data
):
    if args.controller == "overlay":
        raise ValueError("--controller overlay is a viewer-only comparison aid")
    period = 2 * math.pi / FREQUENCY
    if args.duration < period:
        raise ValueError("--duration must include at least one full target cycle")
    cycle_start = (math.floor(args.duration / period) - 1) * period
    cycle_end = cycle_start + period
    samples = {
        "time": [],
        "target": [],
        "qpos": [],
        "error": [],
        "qacc_des": [],
        "qacc": [],
        "acceleration_error": [],
        "tau_fb": [],
        "tau_g": [],
        "tau_ff": [],
        "tau_total": [],
    }

    while data.time < args.duration:
        sim_time = data.time
        metrics = step_controller(
            args.controller,
            model,
            data,
            plant_gravity_data,
            control_model,
            control_work_data,
            sim_time,
        )
        if cycle_start <= data.time <= cycle_end + 1e-12:
            (
                target,
                qacc_des,
                error,
                acceleration_error,
                tau_feedback,
                tau_gravity,
                tau_feedforward,
                tau_total,
            ) = metrics
            samples["time"].append(sim_time)
            samples["target"].append(target)
            samples["qpos"].append(data.qpos.copy())
            samples["error"].append(error)
            samples["qacc_des"].append(qacc_des)
            samples["qacc"].append(data.qacc.copy())
            samples["acceleration_error"].append(acceleration_error)
            samples["tau_fb"].append(tau_feedback)
            samples["tau_g"].append(tau_gravity)
            samples["tau_ff"].append(tau_feedforward)
            samples["tau_total"].append(tau_total)

    print(
        f"controller={args.controller} duration={args.duration:g} "
        f"final_cycle=[{cycle_start:.6f}, {cycle_end:.6f}]"
    )
    for name in (
        "error",
        "acceleration_error",
        "tau_fb",
        "tau_g",
        "tau_ff",
        "tau_total",
    ):
        print(f"{name}_rms={rms(samples[name])}")
    print(
        "turnaround_error_rms="
        f"{turnaround_rms(samples['time'], samples['error'])}"
    )
    print(
        "turnaround_acceleration_error_rms="
        f"{turnaround_rms(samples['time'], samples['acceleration_error'])}"
    )
    print(
        "phase_offset_seconds="
        f"{phase_offset_seconds(samples['time'], np.asarray(samples['qpos']))}"
    )
    if args.controller == "pd":
        print(
            "tau_fb_projection_onto_tau_g="
            f"{projection_coefficients(samples['tau_fb'], samples['tau_g'])}"
        )


    return {name: np.asarray(values) for name, values in samples.items()}


def run_viewer(
    args, model, data, plant_gravity_data, control_model, control_work_data
):
    next_report = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            wall_start = time.time()
            sim_time = data.time
            metrics = step_controller(
                args.controller,
                model,
                data,
                plant_gravity_data,
                control_model,
                control_work_data,
                sim_time,
            )
            viewer.sync()

            if sim_time >= next_report:
                report(sim_time, data, *metrics)
                next_report += REPORT_INTERVAL

            remaining = model.opt.timestep - (time.time() - wall_start)
            if remaining > 0:
                time.sleep(remaining)


def add_arm_geoms(scene, model, data, rgba):
    """Add a dynamic arm state to a viewer scene without changing physics."""
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


def make_torque_figure(title):
    return make_figure(title, PLOT_SERIES)


def run_overlay_viewer():
    """Render synchronized PD and gravity-compensation rollouts together.

    The two data objects advance independently. The added orange geoms are a
    visualization layer only; they do not add bodies, contacts, or forces to
    the blue PD simulation rendered by the viewer.
    """
    pd_model = make_model("pd")
    gravity_model = make_model("gravity-comp")
    pd_data = mujoco.MjData(pd_model)
    gravity_data = mujoco.MjData(gravity_model)
    target_data = mujoco.MjData(pd_model)
    pd_gravity_data = mujoco.MjData(pd_model)
    gravity_gravity_data = mujoco.MjData(gravity_model)
    pd_control_data = mujoco.MjData(pd_model)
    gravity_control_data = mujoco.MjData(gravity_model)
    for data, model in ((pd_data, pd_model), (gravity_data, gravity_model)):
        data.qpos[:] = START
        mujoco.mj_forward(model, data)

    for geom_id in range(pd_model.ngeom):
        if pd_model.geom_bodyid[geom_id] != 0:
            pd_model.geom_rgba[geom_id, 3] = OVERLAY_ALPHA
    orange = np.array([0.95, 0.45, 0.08, OVERLAY_ALPHA])
    figures = (
        make_torque_figure("Applied torque (N-m)  P=PD, G=gravity-comp"),
        make_torque_figure("Torque rate (N-m/s)"),
        make_torque_figure("Torque acceleration (N-m/s^2)"),
    )
    samples = rolling_samples()
    previous_torque = None
    previous_rate = None
    next_plot = 0.0
    with mujoco.viewer.launch_passive(pd_model, pd_data) as viewer:
        while viewer.is_running():
            wall_start = time.time()
            sim_time = pd_data.time
            *_, pd_total = step_controller(
                "pd",
                pd_model,
                pd_data,
                pd_gravity_data,
                pd_model,
                pd_control_data,
                sim_time,
            )
            *_, gravity_total = step_controller(
                "gravity-comp",
                gravity_model,
                gravity_data,
                gravity_gravity_data,
                gravity_model,
                gravity_control_data,
                sim_time,
            )
            target_data.qpos[:] = desired_state(sim_time)[0]
            mujoco.mj_forward(pd_model, target_data)
            applied_torque = np.array(
                [pd_total[0], pd_total[1], gravity_total[0], gravity_total[1]]
            )
            if previous_torque is None:
                rate = np.zeros_like(applied_torque)
                acceleration = np.zeros_like(applied_torque)
            else:
                rate = (applied_torque - previous_torque) / pd_model.opt.timestep
                acceleration = (rate - previous_rate) / pd_model.opt.timestep
            previous_torque = applied_torque
            previous_rate = rate
            if sim_time >= next_plot:
                samples.append((sim_time, applied_torque, rate, acceleration))
                next_plot += PLOT_INTERVAL
            viewer.user_scn.ngeom = 0
            add_arm_geoms(
                viewer.user_scn,
                pd_model,
                target_data,
                np.array([0.75, 0.75, 0.75, 0.22]),
            )
            add_arm_geoms(viewer.user_scn, gravity_model, gravity_data, orange)
            update_stacks(
                viewer,
                (),
                figures,
                samples,
                (),
                (1, 2, 3),
            )
            viewer.sync()

            remaining = pd_model.opt.timestep - (time.time() - wall_start)
            if remaining > 0:
                time.sleep(remaining)


def main():
    args = parse_args()
    if args.controller == "overlay":
        if args.headless:
            raise ValueError("--controller overlay cannot run headless")
        print("overlay: blue=PD, orange=gravity-compensation")
        run_overlay_viewer()
        return
    model = make_model(args.controller)
    data = mujoco.MjData(model)
    plant_gravity_data = mujoco.MjData(model)
    control_model = make_model(
        args.controller,
        link2_mass=(
            WRONG_CONTROL_LINK2_MASS
            if args.controller == "computed-torque-wrong-mass"
            else PLANT_LINK2_MASS
        ),
    )
    control_work_data = mujoco.MjData(control_model)
    data.qpos[:] = START
    mujoco.mj_forward(model, data)

    print(f"controller={args.controller} color={COLORS[args.controller][0]}")
    if args.headless:
        run_headless(
            args,
            model,
            data,
            plant_gravity_data,
            control_model,
            control_work_data,
        )
    else:
        run_viewer(
            args,
            model,
            data,
            plant_gravity_data,
            control_model,
            control_work_data,
        )


if __name__ == "__main__":
    main()
