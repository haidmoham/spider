import time

import mujoco
import mujoco.viewer

# Easy-to-edit experiment settings.
CONTROL_MODE = "pd"  # "constant", "p", or "pd"
CONSTANT_CTRL = 1.0
TARGET_ANGLE = 0.0
Kp = 8.0
Kd = 1.5

XML = """
<mujoco model="pendulum">
    <option timestep="0.002" gravity="0 0 -9.81"/>

    <worldbody>
        <light pos="0 0 3"/>

        <geom
            name="floor"
            type="plane"
            size="3 3 0.1"
        />

        <body name="pendulum" pos="0 0 1.5">
            <joint
                name="hinge"
                type="hinge"
                axis="0 1 0"
            />

            <geom
                name="arm"
                type="capsule"
                fromto="0 0 0  0 0 -1"
                size="0.05"
                mass="1"
            />
        </body>
    </worldbody>

    <actuator>
        <motor name="hinge_motor" joint="hinge" gear="1"/>
    </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

data.qpos[0] = 0.2


def compute_control(data):
    if CONTROL_MODE == "constant":
        return CONSTANT_CTRL

    error = TARGET_ANGLE - data.qpos[0]
    position_term = Kp * error

    if CONTROL_MODE == "p":
        return position_term
    if CONTROL_MODE == "pd":
        return position_term - Kd * data.qvel[0]

    raise ValueError(f"Unknown control mode: {CONTROL_MODE}")


def main():
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()

            data.ctrl[0] = compute_control(data)
            mujoco.mj_step(model, data)
            viewer.sync()

            remaining = model.opt.timestep - (time.time() - start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
