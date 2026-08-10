# C-1N

C-1N is a longitudinal six-legged MuJoCo robot project. The name is rooted in
*Cien Fleur: Spider Net*: a Straw Hat reference hidden inside a machine-like
model designation.

C-1N starts intentionally simple, and it may be bad at first. The goal is to
integrate robotics concepts as they become understood in
[`robotics-test-bench`](https://github.com/haidmoham/robotics-test-bench), where
individual mechanisms are isolated and tested.

Meaningful checkpoints preserve either a new robotics capability or an
understood failure. Completed test-bench experiments may suggest targeted
improvements to C-1N, but they are optional hooks rather than dependencies or a
fixed roadmap.

## Checkpoint grammar

Public checkpoints use:

```text
C-1N // NN · CODENAME
```

The number preserves chronology. The codename records the capability or
understanding gained at that boundary. Do not use semantic versioning as the
primary public identity.

Current and planned lineage:

```text
C-1N // 00 · POSE     motor-assisted static-pose baseline
C-1N // 01 · SHUFFLE  current coordinated gait failure
C-1N // 02 · STAND    reserved for the first support-aware stable stance
C-1N // 03 · STRIDE   reserved for the first materially better walk
```

Only `// 01 · SHUFFLE` is the current public checkpoint. `POSE` is historical.
`STAND` and `STRIDE` are names for future boundaries, not completed releases.

`POSE` does not claim that C-1N learned to stand. It records only that the
simulator can initialize a static pose and that the position actuators can hold
that pose under the baseline conditions. `STAND` is reserved for a later
checkpoint that demonstrates understood support and stable equilibrium.

Bench integrations do not require a public checkpoint. In particular, issue #6
may add task-space telemetry and improve the explanation of the current failure
without creating a `FRAME` release. A new checkpoint is cut only when the robot
capability or understood failure itself is worth preserving.

## C-1N // 01 · SHUFFLE

The current implementation is a deliberately limited baseline:

- a six-legged MuJoCo body with two hinge joints per leg;
- independent joint-position actuators;
- a motor-assisted static pose baseline;
- a low-clearance, phase-shifted tripod shuffle gait;
- a headless stepping script and interactive viewer with rolling telemetry.

The gait shares one phase across all six legs. It reads torso orientation and
foot-ground contacts and uses those signals to bias joint targets. It does not
produce sustained walking. That failure is preserved as part of the project.

Run the static pose baseline with:

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
specific to C-1N.

For a headless walking check:

```bash
python walk.py --headless --duration 20
```

The current route is bench issue #6, Jacobians and task space, then bench issue
#20, static support and equilibrium. Task-space telemetry from #6 may integrate
into C-1N without a public checkpoint. If the later #20 integration demonstrates
understood support-aware stable equilibrium, that boundary becomes
`C-1N // 02 · STAND`.

The browser artifact should preserve historical checkpoints so the robot's
behavior and the reasoning that changed it can be compared over time.
