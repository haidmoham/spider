# Spider

Spider is a longitudinal MuJoCo spider-robot project. It starts intentionally
simple, and it may be bad at first. The goal is to integrate robotics concepts
as they become understood in
[`robotics-test-bench`](https://github.com/haidmoham/robotics-test-bench), where
individual mechanisms are isolated and tested.

Meaningful versions should preserve either a new robotics capability or an
understood failure. Completed test-bench experiments may suggest targeted
improvements to Spider, but they are optional hooks rather than dependencies or
a fixed roadmap.

Likely integration areas include coupled joints, trajectory-based gait control,
contact and friction, actuator limits, model mismatch, system identification,
and differentiable dynamics.

## v0.1

The first implementation is a deliberately limited baseline:

- a six-legged MuJoCo body with two hinge joints per leg;
- independent joint-position actuators;
- a motor-assisted standing pose;
- a low-clearance, phase-shifted tripod shuffle gait;
- a headless stepping script and interactive viewer with rolling telemetry.

It does not implement a gait controller. The motors currently hold a fixed
joint target to offset gravity. Its purpose is to provide a concrete simulation
and a baseline that can fail in understandable ways.

Run the standing baseline with:

```bash
pip install mujoco
python simulate.py
```

Run the walking viewer with:

```bash
python view.py
```

The viewer includes two rolling telemetry stacks. The right stack shows
representative tripod actuator torque, torque rate, and torque acceleration.
The left stack shows representative joint position, velocity, and acceleration.
Both use the recent simulation window. The stack structure follows the
model-based-control experiment in `robotics-test-bench`; its numeric values are
specific to Spider.

The gait controller shares one phase across the legs. It reads a simulated torso
orientation sensor, compares it with the world gravity vector, and checks
foot-ground contacts. It uses those signals to bias leg targets while keeping
the tripod phases synchronized.

For a headless walking check:

```bash
python walk.py --headless --duration 20
```

Once the simulation has useful behavior and understandable failures, it may
become an interactive WASM portfolio artifact.
