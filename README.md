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
- a headless stepping script with zero control input.

It does not implement a gait controller. Its purpose is to provide a concrete
simulation and a baseline that can fail in understandable ways.

Run it with:

```bash
pip install mujoco
python simulate.py
```

Once the simulation has useful behavior and understandable failures, it may
become an interactive WASM portfolio artifact.
