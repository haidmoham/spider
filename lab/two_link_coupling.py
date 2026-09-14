"""Minimal 2-DOF arm experiment for observing joint coupling."""

import argparse
import math
import time

import mujoco
import mujoco.viewer


XML_TEMPLATE = """
<mujoco model="two_link_coupling">
  <option timestep="0.002" gravity="0 0 {gravity}"/>
  <worldbody>
    <light pos="0 0 4"/>
    <geom name="floor" type="plane" size="3 3 0.1"/>
    <body name="shoulder" pos="0 0 {base_height}" euler="0 {base_pitch_degrees} 0">
      <joint name="joint1" type="hinge" axis="0 1 0" damping="0.02"/>
      <geom name="link1" type="capsule" fromto="0 0 0 0 0 -1" size="0.05" mass="1"/>
      <body name="elbow" pos="0 0 -1">
        <joint name="joint2" type="hinge" axis="0 1 0" damping="0.02"/>
        <geom name="link2" type="capsule" fromto="0 0 0 0 0 -0.8" size="0.045" mass="0.7"/>
        <geom name="hand_proxy" type="sphere" pos="0 0 -0.8" size="0.11" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="joint1_motor" joint="joint1" gear="1"/>
    <motor name="joint2_motor" joint="joint2" gear="1"/>
  </actuator>
</mujoco>
"""

CONFIGURATIONS = {
    "elbow-down": (0.35, -0.70),
    "elbow-up": (0.35, 0.70),
}

JOINT1_AMPLITUDE = 0.45
WAVE_FREQUENCY = 1.4
WAVE_PHASE = math.pi
CONTROL_COUPLING = 0.35
JOINT2_TARGET_LIMIT = 1.2
READY_DURATION = 1.0
WAVE_DURATION = 8.0
RETURN_DURATION = 1.5
BASE_HEIGHT = 2.5
BASE_PITCH_DEGREES = 90
JOINT1_KP = 20.0
JOINT1_KD = 3.0
JOINT2_KP = 14.0
JOINT2_KD = 1.5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=CONFIGURATIONS, default="elbow-down")
    parser.add_argument("--joint2-mode", choices=("passive", "hold", "wave"), default="passive")
    parser.add_argument("--gravity", choices=("on", "off"), default="on")
    parser.add_argument("--control-coupling", type=float, default=CONTROL_COUPLING)
    return parser.parse_args()


def pd(target, position, velocity, kp, kd):
    return kp * (target - position) - kd * velocity


def smoothstep(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def wave_targets(elapsed, joint1_start, control_coupling):
    joint1_target = joint1_start + JOINT1_AMPLITUDE * math.sin(
        WAVE_FREQUENCY * elapsed
    )
    joint2_wave_target = JOINT2_TARGET_LIMIT * math.sin(
        WAVE_FREQUENCY * elapsed + WAVE_PHASE
    )
    joint2_target = joint2_wave_target + control_coupling * (
        joint1_target - joint1_start
    )
    return joint1_target, joint2_target


def staged_targets(elapsed, joint1_start, joint2_start, control_coupling):
    if elapsed < READY_DURATION:
        return joint1_start, joint2_start, "ready"

    wave_elapsed = elapsed - READY_DURATION
    if wave_elapsed < WAVE_DURATION:
        joint1_target, joint2_target = wave_targets(
            wave_elapsed, joint1_start, control_coupling
        )
        return joint1_target, joint2_target, "wave"

    wave_end = wave_targets(WAVE_DURATION, joint1_start, control_coupling)
    return_elapsed = wave_elapsed - WAVE_DURATION
    if return_elapsed < RETURN_DURATION:
        alpha = smoothstep(return_elapsed / RETURN_DURATION)
        joint1_target = wave_end[0] + alpha * (joint1_start - wave_end[0])
        joint2_target = wave_end[1] + alpha * (joint2_start - wave_end[1])
        return joint1_target, joint2_target, "return"

    return joint1_start, joint2_start, "hold"


def main():
    args = parse_args()
    gravity = "-9.81" if args.gravity == "on" else "0"
    model = mujoco.MjModel.from_xml_string(
        XML_TEMPLATE.format(
            gravity=gravity,
            base_height=BASE_HEIGHT,
            base_pitch_degrees=BASE_PITCH_DEGREES,
        )
    )
    data = mujoco.MjData(model)

    joint1_start, joint2_start = CONFIGURATIONS[args.configuration]
    data.qpos[:] = (joint1_start, joint2_start)
    mujoco.mj_forward(model, data)

    start_time = time.time()
    next_report = 0.0
    print(f"configuration={args.configuration}, joint2_mode={args.joint2_mode}, gravity={args.gravity}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            wall_start = time.time()
            elapsed = wall_start - start_time

            joint1_target, joint2_target, stage = staged_targets(
                elapsed, joint1_start, joint2_start, args.control_coupling
            )
            data.ctrl[0] = pd(
                joint1_target, data.qpos[0], data.qvel[0], kp=JOINT1_KP, kd=JOINT1_KD
            )

            if args.joint2_mode == "passive":
                data.ctrl[1] = 0.0
            elif args.joint2_mode == "hold":
                data.ctrl[1] = pd(joint2_start, data.qpos[1], data.qvel[1], kp=2.0, kd=0.3)
            else:
                data.ctrl[1] = pd(
                    joint2_target, data.qpos[1], data.qvel[1], kp=JOINT2_KP, kd=JOINT2_KD
                )

            mujoco.mj_step(model, data)
            viewer.sync()

            if elapsed >= next_report:
                print(
                    f"t={elapsed:5.2f} stage={stage} qpos={data.qpos[:2]} qvel={data.qvel[:2]} "
                    f"ctrl={data.ctrl[:2]} qacc={data.qacc[:2]}"
                )
                next_report += 0.25

            remaining = model.opt.timestep - (time.time() - wall_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
